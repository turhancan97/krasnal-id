"""The first stage's recall, and the two standard fixes that fail on it."""

import json
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pytest
from typer.testing import CliRunner

from helpers import FAKE_BACKBONE, seed_embedding_cache, synthetic_manifest
from krasnal_id.cli import app
from krasnal_id.config import RecallCurveConfig, load_config
from krasnal_id.data_pipeline.build_split import build_evaluation_split, write_evaluation_split
from krasnal_id.embeddings.store import load_embedding_matrix
from krasnal_id.experiments.recall_curve import (
    ABSENT,
    ANSWERABLE,
    DISJOINT,
    FULL,
    Arm,
    RecallCurveError,
    class_rank,
    expanded_query,
    measure_arm,
    run_recall_curve,
    summarize,
)
from krasnal_id.models import DatasetManifest

runner = CliRunner()


def _with_two_photographers(manifest: DatasetManifest) -> DatasetManifest:
    return manifest.model_copy(
        update={
            "images": tuple(
                image.model_copy(
                    update={"author": "Bruno" if image.image_id.endswith("-2") else "Ada"}
                )
                for image in manifest.images
            )
        }
    )


def test_a_rank_is_the_statues_position_not_the_photographs() -> None:
    """Candidates collapse to statues by their best photograph, as elsewhere."""
    scores = np.asarray([0.9, 0.8, 0.7, 0.6], dtype=np.float32)
    dwarfs = np.asarray(["Q2", "Q2", "Q1", "Q3"])

    # Q2 takes the first two places but counts once, so Q1 is second.
    assert class_rank(scores, dwarfs, "Q2") == 1
    assert class_rank(scores, dwarfs, "Q1") == 2
    assert class_rank(scores, dwarfs, "Q3") == 3
    # A statue with no photograph in the candidate set is absent, not last.
    assert class_rank(scores, dwarfs, "Q9") == ABSENT


def test_expansion_moves_the_query_toward_its_neighbours() -> None:
    vector = np.asarray([1.0, 0.0], dtype=np.float32)
    # A partly similar neighbour, so it carries some weight, and an identical one.
    references = np.asarray([[0.6, 0.8], [1.0, 0.0]], dtype=np.float32)
    scores = references @ vector

    moved = expanded_query(vector, references, scores, neighbours=2, alpha=1.0)

    assert moved[1] > 0.0, "a similar neighbour should pull the query toward it"
    assert np.isclose(np.linalg.norm(moved), 1.0)
    # Weighting is by similarity, so a neighbour orthogonal to the query
    # contributes nothing — which is the property that makes expansion safe when
    # the top results are unrelated, and useless when they are wrong lookalikes.
    orthogonal = np.asarray([[0.0, 1.0]], dtype=np.float32)
    unmoved = expanded_query(vector, orthogonal, orthogonal @ vector, neighbours=1, alpha=1.0)
    assert np.array_equal(unmoved, vector)
    # No neighbours means no expansion, which is how the variant is switched off.
    assert np.array_equal(expanded_query(vector, references, scores, 0, 1.0), vector)
    # A degenerate sum leaves the query alone rather than dividing by zero.
    opposite = np.asarray([[-1.0, 0.0]], dtype=np.float32)
    assert np.array_equal(
        expanded_query(vector, opposite, np.asarray([1.0], dtype=np.float32), 1, 1.0), vector
    )


def test_each_arm_asks_the_queries_it_should(tmp_path: Path) -> None:
    manifest = _with_two_photographers(synthetic_manifest(dwarf_count=3, per_dwarf=3))
    seed_embedding_cache(tmp_path, manifest)
    matrices = {"dinov2": load_embedding_matrix(manifest, FAKE_BACKBONE, tmp_path)}
    split = build_evaluation_split(manifest, datetime.now(UTC))
    config = load_config(["experiment=recall_curve"]).experiment
    assert isinstance(config, RecallCurveConfig)

    counts = {}
    for arm in (
        Arm(FULL, only_answerable=False, photographer_disjoint=False),
        Arm(ANSWERABLE, only_answerable=True, photographer_disjoint=False),
        Arm(DISJOINT, only_answerable=True, photographer_disjoint=True),
    ):
        ranks, total = measure_arm(arm, split, manifest, matrices, "dinov2", config)
        counts[arm.name] = total
        assert set(ranks) >= {"plain"}
        assert all(len(values) == total for values in ranks.values())

    # Every class here has two photographers, so nothing is dropped.
    assert counts[FULL] == counts[ANSWERABLE] == counts[DISJOINT] == 9


def test_a_single_photographer_dataset_cannot_fill_the_disjoint_arm(tmp_path: Path) -> None:
    manifest = synthetic_manifest(dwarf_count=3, per_dwarf=3)  # one shared author
    seed_embedding_cache(tmp_path, manifest)
    matrices = {"dinov2": load_embedding_matrix(manifest, FAKE_BACKBONE, tmp_path)}
    split = build_evaluation_split(manifest, datetime.now(UTC))
    config = load_config(["experiment=recall_curve"]).experiment
    assert isinstance(config, RecallCurveConfig)

    with pytest.raises(RecallCurveError, match="disjoint arm scored no queries"):
        measure_arm(
            Arm(DISJOINT, only_answerable=True, photographer_disjoint=True),
            split,
            manifest,
            matrices,
            "dinov2",
            config,
        )


def test_recall_is_monotone_and_reported_per_variant() -> None:
    measurements = {
        FULL: ({"plain": [1, 2, 6, ABSENT], "fused": [1, 1, 8, ABSENT]}, 4),
        DISJOINT: ({"plain": [3, ABSENT, ABSENT, ABSENT]}, 4),
    }

    metrics = {m.name: m.value for m in summarize(measurements, (1, 5, 10))}

    assert metrics["full_r_at_1"] == pytest.approx(0.25)
    assert metrics["full_r_at_5"] == pytest.approx(0.5)
    assert metrics["full_r_at_10"] == pytest.approx(0.75)
    # Recall can only rise with k, which is what makes the curve readable.
    assert metrics["full_r_at_1"] <= metrics["full_r_at_5"] <= metrics["full_r_at_10"]
    assert metrics["full_fused_r_at_1"] == pytest.approx(0.5)
    assert metrics["disjoint_r_at_5"] == pytest.approx(0.25)
    # An absent statue never counts, at any cut-off.
    assert metrics["disjoint_r_at_10"] == pytest.approx(0.25)
    assert metrics["full_queries"] == 4

    with pytest.raises(RecallCurveError, match="must be positive"):
        summarize(measurements, (0,))


def test_run_requires_the_recall_experiment_group() -> None:
    with pytest.raises(RecallCurveError, match="requires experiment=recall_curve"):
        run_recall_curve(load_config(["experiment=baseline"]))


def test_packaged_defaults_measure_both_rejected_fixes() -> None:
    experiment = load_config(["experiment=recall_curve"]).experiment
    assert isinstance(experiment, RecallCurveConfig)
    assert len(experiment.fuse_backbones) == 2, "fusion needs two backbones to fuse"
    assert experiment.expansion_neighbours, "query expansion is one of the measured fixes"
    assert 1 in experiment.top_k and 50 in experiment.top_k

    for broken, message in (
        ({"top_k": (0,)}, "must be positive"),
        ({"expansion_neighbours": (0,)}, "must be positive"),
        ({"fuse_backbones": ("clip", "clip")}, "cannot contain duplicates"),
    ):
        with pytest.raises(ValueError, match=message):
            RecallCurveConfig.model_validate(
                {
                    "kind": "recall_curve",
                    "seed": 1,
                    "top_k": (1, 5),
                    "fuse_backbones": ("dinov2", "clip"),
                    "expansion_neighbours": (2,),
                    "expansion_alpha": 3.0,
                    **broken,
                }
            )


def test_cli_reports_the_curve_and_writes_the_artifact(tmp_path: Path) -> None:
    manifest = _with_two_photographers(synthetic_manifest(dwarf_count=3, per_dwarf=3))
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(manifest.model_dump(mode="json")), encoding="utf-8")
    split_path = tmp_path / "splits" / "leave-one-out.json"
    write_evaluation_split(split_path, build_evaluation_split(manifest, datetime.now(UTC)))
    seed_embedding_cache(tmp_path / "embeddings", manifest)

    result = runner.invoke(
        app,
        [
            "experiment",
            "recall",
            *[
                f"-o{value}"
                for value in (
                    f"paths.manifest_path={manifest_path}",
                    f"paths.evaluation_split_path={split_path}",
                    f"paths.embeddings_dir={tmp_path / 'embeddings'}",
                    f"paths.results_dir={tmp_path / 'results'}",
                    f"backbone.model_id={FAKE_BACKBONE.model_id}",
                    f"backbone.revision={FAKE_BACKBONE.revision}",
                    f"backbone.preprocessing_id={FAKE_BACKBONE.preprocessing_id}",
                    # Only one backbone's vectors are cached here, so fusion is off.
                    "experiment.fuse_backbones=[]",
                    "logging.json_output=false",
                )
            ],
        ],
    )

    assert result.exit_code == 0, result.output
    for expected in ("full_r_at_1", "answerable_r_at_1", "disjoint_r_at_1", "full_queries"):
        assert expected in result.output
    written = json.loads((tmp_path / "results" / "recall_curve-dinov2.json").read_text())
    assert written["experiment"] == "recall_curve"


def test_a_comparison_counts_only_the_queries_that_separate_two_backbones() -> None:
    """The paired comparison is the whole reason a second backbone is scored.

    Four queries: one both get right, one both get wrong, one only the comparison
    gets right, one only the selected one does. Only the last two are evidence, and
    they cancel -- so the delta is zero and nothing is significant, which two
    separate recall figures of 0.5 and 0.5 could not have told apart from agreement.
    """
    measurements = {
        FULL: (
            {
                "plain": [1, ABSENT, ABSENT, 1],
                "against_dinov2-large": [1, ABSENT, 1, ABSENT],
            },
            4,
        )
    }

    metrics = {m.name: m.value for m in summarize(measurements, (1,))}

    assert metrics["full_r_at_1"] == pytest.approx(0.5)
    assert metrics["full_against_dinov2-large_r_at_1"] == pytest.approx(0.5)
    assert metrics["full_dinov2-large_vs_selected_at_1_wins"] == 1
    assert metrics["full_dinov2-large_vs_selected_at_1_losses"] == 1
    assert metrics["full_dinov2-large_vs_selected_at_1_delta"] == pytest.approx(0.0)
    assert metrics["full_dinov2-large_vs_selected_at_1_p_value"] == pytest.approx(1.0)


def test_a_one_sided_comparison_is_reported_as_significant() -> None:
    """Eight queries gained and none lost is not a coin, and must not read as one."""
    measurements = {
        DISJOINT: (
            {
                "plain": [ABSENT] * 8,
                "against_dinov2-large": [1] * 8,
            },
            8,
        )
    }

    metrics = {m.name: m.value for m in summarize(measurements, (1,))}

    assert metrics["disjoint_dinov2-large_vs_selected_at_1_wins"] == 8
    assert metrics["disjoint_dinov2-large_vs_selected_at_1_losses"] == 0
    assert metrics["disjoint_dinov2-large_vs_selected_at_1_delta"] == pytest.approx(1.0)
    assert metrics["disjoint_dinov2-large_vs_selected_at_1_p_value"] < 0.01


def test_an_unpaired_comparison_is_refused_rather_than_averaged() -> None:
    """Two backbones scored on different query sets cannot be differenced."""
    measurements = {FULL: ({"plain": [1, 2, 3], "against_clip": [1, 2]}, 3)}

    with pytest.raises(RecallCurveError, match="would not be paired"):
        summarize(measurements, (1,))


def test_a_compared_backbone_is_scored_on_the_selected_backbones_folds(tmp_path: Path) -> None:
    """Pairing is by construction: the same folds, and the same candidate sets."""
    manifest = _with_two_photographers(synthetic_manifest(dwarf_count=3, per_dwarf=3))
    seed_embedding_cache(tmp_path, manifest)
    split = build_evaluation_split(manifest, datetime.now(UTC))
    experiment = load_config(
        [
            "experiment=recall_curve",
            "experiment.fuse_backbones=[]",
            "experiment.expansion_neighbours=[]",
            "experiment.compare_backbones=[clip]",
        ]
    ).experiment
    assert isinstance(experiment, RecallCurveConfig)

    matrix = load_embedding_matrix(manifest, FAKE_BACKBONE, tmp_path)
    ranks, total = measure_arm(
        Arm(FULL, only_answerable=False, photographer_disjoint=False),
        split,
        manifest,
        {"dinov2": matrix, "clip": matrix},
        "dinov2",
        experiment,
    )

    # The same vectors under two names must rank identically, query for query.
    assert ranks["against_clip"] == ranks["plain"]
    assert len(ranks["plain"]) == total


def test_a_comparison_against_the_selected_backbone_itself_is_dropped(tmp_path: Path) -> None:
    """Otherwise every artifact would carry a comparison of a backbone with itself."""
    manifest = _with_two_photographers(synthetic_manifest(dwarf_count=3, per_dwarf=3))
    seed_embedding_cache(tmp_path, manifest)
    experiment = load_config(
        [
            "experiment=recall_curve",
            "experiment.fuse_backbones=[]",
            "experiment.expansion_neighbours=[]",
            "experiment.compare_backbones=[dinov2]",
        ]
    ).experiment
    assert isinstance(experiment, RecallCurveConfig)

    ranks, _ = measure_arm(
        Arm(FULL, only_answerable=False, photographer_disjoint=False),
        build_evaluation_split(manifest, datetime.now(UTC)),
        manifest,
        {"dinov2": load_embedding_matrix(manifest, FAKE_BACKBONE, tmp_path)},
        "dinov2",
        experiment,
    )

    assert set(ranks) == {"plain"}


def test_duplicate_comparisons_are_refused() -> None:
    with pytest.raises(ValueError, match="compare_backbones cannot contain duplicates"):
        RecallCurveConfig.model_validate(
            {
                "kind": "recall_curve",
                "seed": 1,
                "top_k": (1, 5),
                "compare_backbones": ("clip", "clip"),
            }
        )

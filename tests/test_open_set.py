"""Open-set rejection of queries whose dwarf is absent from the reference set."""

import json
from datetime import UTC, datetime
from itertools import pairwise
from pathlib import Path

import numpy as np
import numpy.typing as npt
import pytest
from typer.testing import CliRunner

from helpers import FAKE_BACKBONE, seed_embedding_cache, synthetic_manifest
from krasnal_id.cli import app
from krasnal_id.config import OpenSetExperimentConfig, load_config
from krasnal_id.data_pipeline.build_split import build_evaluation_split, write_evaluation_split
from krasnal_id.embeddings.store import load_embedding_matrix
from krasnal_id.experiments.contracts import DwarfRejection, OpenSetRejectionResult
from krasnal_id.experiments.open_set import (
    OpenSetExperimentError,
    ScoredQuery,
    best_balanced_accuracy,
    calibrate_leave_one_class_out,
    rank_rejections,
    rejection_curve,
    run_open_set_rejection,
    score_known_queries,
    score_unknown_queries,
    separability_auroc,
    summarize_open_set,
    threshold_for_target,
)


def _config(**overrides: object) -> OpenSetExperimentConfig:
    values: dict[str, object] = {
        "kind": "open_set",
        "seed": 42,
        "target_known_acceptance": (0.9, 0.95),
        "top_rejections": 20,
    }
    values.update(overrides)
    return OpenSetExperimentConfig.model_validate(values)


def _twin_cluster_vector(
    dwarf_index: int, position: int, dwarf_count: int
) -> npt.NDArray[np.float32]:
    """Put dwarves 0 and 1 in one tight cluster and the rest on their own axes.

    The twins are the interesting case: they sit closer to each other than two
    images of the same dwarf do, which is the real dataset's confusable-cluster
    shape. Remove either twin and the other still covers for it at the same
    similarity, so no threshold can reject its queries.
    """
    vector = np.zeros(dwarf_count + 2, dtype=np.float32)
    if dwarf_index < 2:
        vector[0] = 1.0
        vector[1] = 0.02 * dwarf_index
    else:
        vector[dwarf_index] = 1.0
    vector[dwarf_count + 1] = 0.05 * (position + 1)
    return np.asarray(vector / np.linalg.norm(vector), dtype=np.float32)


def _query(
    dwarf_id: str, similarity: float, present: bool, top_dwarf_id: str | None = None
) -> ScoredQuery:
    return ScoredQuery(
        query_image_id=f"image-{dwarf_id}-{similarity}",
        query_dwarf_id=dwarf_id,
        top_similarity=similarity,
        top_dwarf_id=top_dwarf_id if top_dwarf_id is not None else dwarf_id,
        present=present,
    )


def test_config_rejects_impossible_or_repeated_targets() -> None:
    with pytest.raises(ValueError, match="strictly between 0 and 1"):
        _config(target_known_acceptance=(1.0,))
    with pytest.raises(ValueError, match="strictly between 0 and 1"):
        _config(target_known_acceptance=(0.0,))
    with pytest.raises(ValueError, match="cannot contain duplicates"):
        _config(target_known_acceptance=(0.9, 0.9))
    with pytest.raises(ValueError, match="greater than 0"):
        _config(top_rejections=0)


def test_packaged_open_set_defaults_are_ordered_operating_points() -> None:
    experiment = load_config(["experiment=open_set"]).experiment
    assert isinstance(experiment, OpenSetExperimentConfig)
    # The per-dwarf rows describe the first target, so it must be a real one.
    assert experiment.primary_target == experiment.target_known_acceptance[0]
    assert 0.0 < experiment.primary_target < 1.0


def test_auroc_separates_populations_and_halves_ties() -> None:
    perfect = separability_auroc(
        np.asarray([0.9, 0.8], dtype=np.float64), np.asarray([0.1, 0.2], dtype=np.float64)
    )
    assert perfect == pytest.approx(1.0)
    # Identical populations carry no information about presence.
    tied = separability_auroc(
        np.asarray([0.5, 0.5], dtype=np.float64), np.asarray([0.5, 0.5], dtype=np.float64)
    )
    assert tied == pytest.approx(0.5)
    inverted = separability_auroc(
        np.asarray([0.1], dtype=np.float64), np.asarray([0.9], dtype=np.float64)
    )
    assert inverted == pytest.approx(0.0)
    # One tie against one clean win sits halfway between them.
    partial = separability_auroc(
        np.asarray([0.5], dtype=np.float64), np.asarray([0.5, 0.1], dtype=np.float64)
    )
    assert partial == pytest.approx(0.75)

    with pytest.raises(OpenSetExperimentError, match="at least one query in each"):
        separability_auroc(np.asarray([], dtype=np.float64), np.asarray([0.1], dtype=np.float64))


def test_a_threshold_always_reaches_its_target_acceptance() -> None:
    scores = np.asarray([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0], dtype=np.float64)

    for target in (0.5, 0.9, 0.95, 0.99):
        threshold = threshold_for_target(scores, target)
        achieved = float((scores >= threshold).mean())
        assert achieved >= target
        # The threshold is an observed score, never an interpolated one.
        assert threshold in set(scores.tolist())

    with pytest.raises(OpenSetExperimentError, match="must lie in"):
        threshold_for_target(scores, 1.0)
    with pytest.raises(OpenSetExperimentError, match="zero known queries"):
        threshold_for_target(np.asarray([], dtype=np.float64), 0.9)


def test_calibration_never_uses_the_class_it_judges() -> None:
    known = (
        _query("Q1", 0.10, present=True),
        _query("Q1", 0.11, present=True),
        _query("Q2", 0.90, present=True),
        _query("Q2", 0.91, present=True),
    )

    thresholds = calibrate_leave_one_class_out(known, 0.5)

    # Q1's threshold is set by Q2's high scores alone, and vice versa.
    assert thresholds["Q1"] >= 0.90
    assert thresholds["Q2"] <= 0.11
    with pytest.raises(OpenSetExperimentError, match="at least two dwarves"):
        calibrate_leave_one_class_out((_query("Q1", 0.5, present=True),), 0.9)


def test_best_balanced_accuracy_finds_a_separating_threshold() -> None:
    score, threshold = best_balanced_accuracy(
        np.asarray([0.8, 0.9], dtype=np.float64), np.asarray([0.1, 0.2], dtype=np.float64)
    )

    assert score == pytest.approx(1.0)
    assert 0.2 < threshold <= 0.8
    # Fully overlapping populations cannot beat a coin flip.
    overlapping, _ = best_balanced_accuracy(
        np.asarray([0.5], dtype=np.float64), np.asarray([0.5], dtype=np.float64)
    )
    assert overlapping == pytest.approx(0.5)


def test_unknown_queries_search_a_gallery_without_their_own_dwarf(tmp_path: Path) -> None:
    manifest = synthetic_manifest(dwarf_count=4)
    seed_embedding_cache(tmp_path, manifest)
    matrix = load_embedding_matrix(manifest, FAKE_BACKBONE, tmp_path)

    unknown = score_unknown_queries(matrix)

    assert len(unknown) == len(manifest.images)
    for query in unknown:
        assert not query.present
        # Removing the whole class is what makes the query genuinely unknown.
        assert query.top_dwarf_id != query.query_dwarf_id
        assert not query.identified


def test_a_single_class_leaves_no_gallery_to_search(tmp_path: Path) -> None:
    manifest = synthetic_manifest(dwarf_count=1, per_dwarf=3)
    seed_embedding_cache(tmp_path, manifest)
    matrix = load_embedding_matrix(manifest, FAKE_BACKBONE, tmp_path)

    with pytest.raises(OpenSetExperimentError, match="at least two dwarves"):
        score_unknown_queries(matrix)


def test_known_queries_reach_their_own_dwarf_on_separable_data(tmp_path: Path) -> None:
    manifest = synthetic_manifest(dwarf_count=4)
    seed_embedding_cache(tmp_path, manifest)
    matrix = load_embedding_matrix(manifest, FAKE_BACKBONE, tmp_path)
    split = build_evaluation_split(manifest, datetime.now(UTC))

    known = score_known_queries(split, matrix)

    assert len(known) == len(split.folds)
    assert all(query.present and query.identified for query in known)
    with pytest.raises(OpenSetExperimentError, match="no folds"):
        score_known_queries(split.model_copy(update={"folds": ()}), matrix)


def test_separable_classes_are_rejected_and_twins_are_not(tmp_path: Path) -> None:
    manifest = synthetic_manifest(dwarf_count=4)
    seed_embedding_cache(tmp_path, manifest, vector_for=_twin_cluster_vector)
    matrix = load_embedding_matrix(manifest, FAKE_BACKBONE, tmp_path)
    split = build_evaluation_split(manifest, datetime.now(UTC))

    known = score_known_queries(split, matrix)
    unknown = score_unknown_queries(matrix)
    metrics = {metric.name: metric for metric in summarize_open_set(known, unknown, _config())}
    rejections = rank_rejections(known, unknown, manifest, _config())

    # The twins score high even when removed, so separability is imperfect.
    assert 0.5 < metrics["auroc"].value < 1.0
    # The unrejectable dwarves are exactly the twins, each covered by the other.
    assert {row.dwarf_id for row in rejections if row.false_accepts} == {"Q0", "Q1"}
    assert {row.nearest_dwarf_id for row in rejections[:2]} == {"Q0", "Q1"}
    assert metrics["known_queries"].value == pytest.approx(float(len(known)))
    assert metrics["unknown_queries"].value == pytest.approx(float(len(unknown)))
    assert metrics["mean_similarity_gap"].value > 0.0
    # A higher acceptance target means a lower threshold and more false accepts.
    assert metrics["target_95_threshold"].value <= metrics["target_90_threshold"].value
    assert metrics["target_95_false_acceptance"].value >= (
        metrics["target_90_false_acceptance"].value
    )
    for target in ("90", "95"):
        assert metrics[f"target_{target}_known_acceptance"].value >= float(target) / 100.0
    # The in-sample sweep cannot be beaten by an out-of-sample threshold.
    assert metrics["in_sample_best_balanced_accuracy"].value >= 0.5


def test_summary_requires_both_populations() -> None:
    with pytest.raises(OpenSetExperimentError, match="must be non-empty"):
        summarize_open_set((), (_query("Q1", 0.5, present=False),), _config())
    with pytest.raises(OpenSetExperimentError, match="must be non-empty"):
        summarize_open_set((_query("Q1", 0.5, present=True),), (), _config())


def test_rejection_rows_name_what_covered_for_a_removed_dwarf() -> None:
    manifest = synthetic_manifest(dwarf_count=3, per_dwarf=3)
    known = tuple(
        _query(f"Q{index}", 0.5 + 0.01 * position, present=True)
        for index in range(3)
        for position in range(3)
    )
    unknown = (
        # Q0 is covered by Q1 at a similarity no threshold from the others rejects.
        _query("Q0", 0.99, present=False, top_dwarf_id="Q1"),
        _query("Q0", 0.98, present=False, top_dwarf_id="Q1"),
        _query("Q1", 0.10, present=False, top_dwarf_id="Q0"),
        _query("Q2", 0.10, present=False, top_dwarf_id="Q0"),
    )

    rejections = rank_rejections(known, unknown, manifest, _config())

    assert [row.dwarf_id for row in rejections] == ["Q0", "Q1", "Q2"]
    assert rejections[0].false_accepts == 2
    assert rejections[0].unknown_queries == 2
    assert rejections[0].nearest_dwarf_id == "Q1"
    assert rejections[0].nearest_display_name == "Dwarf 1"
    assert rejections[1].false_accepts == 0
    # The cap keeps a large dataset's artifact readable.
    assert len(rank_rejections(known, unknown, manifest, _config(top_rejections=1))) == 1


def test_rejection_rows_reject_a_dwarf_outside_the_manifest() -> None:
    manifest = synthetic_manifest(dwarf_count=2, per_dwarf=3)
    known = tuple(_query(f"Q{index}", 0.5, present=True) for index in range(2) for _ in range(3))

    with pytest.raises(OpenSetExperimentError, match="Q9 is absent from the manifest"):
        rank_rejections(
            known, (_query("Q9", 0.5, present=False, top_dwarf_id="Q0"),), manifest, _config()
        )
    with pytest.raises(OpenSetExperimentError, match="Q8 is absent from the manifest"):
        rank_rejections(
            known, (_query("Q0", 0.5, present=False, top_dwarf_id="Q8"),), manifest, _config()
        )


def test_a_removed_dwarf_cannot_be_its_own_cover() -> None:
    with pytest.raises(ValueError, match="cannot be its own nearest neighbour"):
        DwarfRejection(
            dwarf_id="Q1",
            display_name="Dwarf 1",
            unknown_queries=2,
            false_accepts=1,
            mean_top_similarity=0.5,
            nearest_dwarf_id="Q1",
            nearest_display_name="Dwarf 1",
        )
    with pytest.raises(ValueError, match="cannot exceed the queries"):
        DwarfRejection(
            dwarf_id="Q1",
            display_name="Dwarf 1",
            unknown_queries=2,
            false_accepts=3,
            mean_top_similarity=0.5,
            nearest_dwarf_id="Q2",
            nearest_display_name="Dwarf 2",
        )


def test_run_requires_the_open_set_experiment_group() -> None:
    with pytest.raises(OpenSetExperimentError, match="requires experiment=open_set"):
        run_open_set_rejection(load_config(["experiment=baseline"]))


def test_run_reports_a_missing_split(tmp_path: Path) -> None:
    config = load_config(
        [
            "experiment=open_set",
            f"paths.manifest_path={tmp_path / 'manifest.json'}",
            f"paths.evaluation_split_path={tmp_path / 'absent.json'}",
        ]
    )
    with pytest.raises(OpenSetExperimentError, match="invalid"):
        run_open_set_rejection(config)


def test_cli_open_set_reports_rejections_and_fails_without_embeddings(tmp_path: Path) -> None:
    manifest = synthetic_manifest(dwarf_count=4)
    manifest_path = tmp_path / "manifest.json"
    split_path = tmp_path / "splits" / "leave-one-out.json"
    manifest_path.write_text(json.dumps(manifest.model_dump(mode="json")), encoding="utf-8")
    write_evaluation_split(split_path, build_evaluation_split(manifest, datetime.now(UTC)))
    overrides = [
        f"paths.manifest_path={manifest_path}",
        f"paths.evaluation_split_path={split_path}",
        f"paths.embeddings_dir={tmp_path / 'embeddings'}",
        f"paths.results_dir={tmp_path / 'results'}",
        f"backbone.model_id={FAKE_BACKBONE.model_id}",
        f"backbone.revision={FAKE_BACKBONE.revision}",
        f"backbone.preprocessing_id={FAKE_BACKBONE.preprocessing_id}",
        "logging.json_output=false",
    ]
    arguments = ["experiment", "open-set", *[f"-o{value}" for value in overrides]]
    runner = CliRunner()

    missing = runner.invoke(app, arguments)
    assert missing.exit_code == 2
    assert "Open-set rejection error" in missing.output

    seed_embedding_cache(tmp_path / "embeddings", manifest, vector_for=_twin_cluster_vector)
    result = runner.invoke(app, arguments)
    assert result.exit_code == 0, result.output
    assert "auroc" in result.output
    assert "target_90_false_acceptance" in result.output
    assert "hardest dwarves to reject" in result.output

    written = OpenSetRejectionResult.model_validate(
        json.loads((tmp_path / "results" / "open_set-dinov2.json").read_text(encoding="utf-8"))
    )
    assert written.experiment == "open_set"
    assert written.backbone == "dinov2"
    assert len(written.rejections) == 4
    assert {metric.name for metric in written.metrics} >= {
        "auroc",
        "closed_set_top_1",
        "target_90_open_set_accuracy",
        "target_90_false_acceptance",
        "in_sample_best_balanced_accuracy",
        "known_queries",
    }


def test_the_rejection_curve_spans_both_extremes() -> None:
    known = np.asarray([0.8, 0.9], dtype=np.float64)
    unknown = np.asarray([0.1, 0.85], dtype=np.float64)

    curve = rejection_curve(known, unknown)

    # The endpoints accept everything and reject everything.
    assert curve[0].known_acceptance == pytest.approx(1.0)
    assert curve[0].false_acceptance == pytest.approx(1.0)
    assert curve[-1].known_acceptance == pytest.approx(0.0)
    assert curve[-1].false_acceptance == pytest.approx(0.0)
    # Both rates fall as the threshold rises, and every value stays finite so the
    # curve survives a JSON round trip.
    for earlier, later in pairwise(curve):
        assert earlier.threshold < later.threshold
        assert earlier.known_acceptance >= later.known_acceptance
        assert earlier.false_acceptance >= later.false_acceptance
    assert all(np.isfinite(point.threshold) for point in curve)

    with pytest.raises(OpenSetExperimentError, match="needs both populations"):
        rejection_curve(known, np.asarray([], dtype=np.float64))

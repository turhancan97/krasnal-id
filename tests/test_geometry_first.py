"""Local features promoted from re-ranker to first stage."""

import json
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pytest
from typer.testing import CliRunner

from helpers import FAKE_BACKBONE, seed_embedding_cache, synthetic_manifest
from krasnal_id.cli import app
from krasnal_id.config import GeometryFirstConfig, RerankAblationConfig, load_config
from krasnal_id.data_pipeline.build_split import build_evaluation_split, write_evaluation_split
from krasnal_id.embeddings.store import load_embedding_matrix
from krasnal_id.experiments.geometry_first import (
    ANSWERABLE,
    APPEARANCE,
    DISJOINT,
    GEOMETRY,
    GeometryFirstError,
    QueryRanks,
    measure,
    rank_one_query,
    run_geometry_first,
    summarize,
)
from krasnal_id.experiments.recall_curve import ABSENT
from krasnal_id.models import DatasetManifest

runner = CliRunner()


def _textured(path: Path, seed: int, size: int = 160) -> None:
    """Write an image with enough texture for a keypoint detector to work on."""
    from PIL import Image

    rng = np.random.default_rng(seed)
    pixels = rng.integers(0, 256, size=(size, size, 3), dtype=np.uint8)
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(pixels).save(path)


def _manifest_with_images(tmp_path: Path, dwarfs: int = 3) -> DatasetManifest:
    """A manifest whose images exist on disk, one distinct texture per class."""
    manifest = synthetic_manifest(dwarf_count=dwarfs, per_dwarf=3)
    records = []
    for image in manifest.images:
        path = tmp_path / "images" / image.dwarf_id / f"{image.image_id}.png"
        _textured(path, seed=int(image.dwarf_id.removeprefix("Q")))
        records.append(image.model_copy(update={"local_path": path, "width": 160, "height": 160}))
    return manifest.model_copy(update={"images": tuple(records)})


def _by_two_photographers(manifest: DatasetManifest) -> DatasetManifest:
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


def _experiment(**overrides: object) -> GeometryFirstConfig:
    config = load_config(
        ["experiment=geometry_first", *(f"experiment.{k}={v}" for k, v in overrides.items())]
    ).experiment
    assert isinstance(config, GeometryFirstConfig)
    return config


def test_geometry_ranks_by_inliers_and_appearance_by_cosine() -> None:
    """The two signals are read off the same candidates and must not be mixed."""
    dwarfs = np.asarray(["Q1", "Q2", "Q3"])
    # Appearance likes Q1 most; geometry likes Q3 most.
    cosine = np.asarray([0.9, 0.5, 0.1], dtype=np.float32)
    inliers = np.asarray([2.0, 5.0, 40.0], dtype=np.float32)

    ranks = rank_one_query(inliers, cosine, dwarfs, "Q3", np.ones(3, dtype=bool))

    assert ranks.answerable[APPEARANCE] == 3
    assert ranks.answerable[GEOMETRY] == 1


def test_the_disjoint_arm_is_a_mask_over_the_same_inliers() -> None:
    """Two arms for the price of one matching pass is the whole cost argument."""
    dwarfs = np.asarray(["Q1", "Q2", "Q3"])
    cosine = np.asarray([0.9, 0.5, 0.1], dtype=np.float32)
    inliers = np.asarray([40.0, 5.0, 2.0], dtype=np.float32)
    # Withhold the reference that was carrying the right answer.
    mask = np.asarray([False, True, True], dtype=bool)

    ranks = rank_one_query(inliers, cosine, dwarfs, "Q1", mask)

    assert ranks.answerable[GEOMETRY] == 1
    # Q1 has no photograph left in the candidate set, so it is absent, not last.
    assert ranks.disjoint[GEOMETRY] == ABSENT


def test_an_empty_candidate_set_is_absent_rather_than_an_error() -> None:
    """A query whose every reference shares its photographer still has a row."""
    ranks = rank_one_query(
        np.asarray([3.0], dtype=np.float32),
        np.asarray([0.5], dtype=np.float32),
        np.asarray(["Q1"]),
        "Q1",
        np.zeros(1, dtype=bool),
    )

    assert ranks.disjoint == {APPEARANCE: ABSENT, GEOMETRY: ABSENT}


def _ranks(appearance: int, geometry: int) -> QueryRanks:
    row = {APPEARANCE: appearance, GEOMETRY: geometry}
    return QueryRanks(answerable=dict(row), disjoint=dict(row))


def test_the_rescue_rate_counts_only_what_appearance_lost() -> None:
    """The decisive number: a first stage has to find what the other one misses.

    Four queries at k=1 -- both right, both wrong, appearance-only, geometry-only.
    Appearance misses two of them and geometry retrieves one, so the rescue rate is
    one half, and that is a different question from geometry's own recall of 0.5.
    """
    measured = (_ranks(1, 1), _ranks(9, 9), _ranks(1, 9), _ranks(9, 1))

    metrics = {m.name: m.value for m in summarize(measured, (1,))}

    assert metrics[f"{ANSWERABLE}_appearance_r_at_1"] == pytest.approx(0.5)
    assert metrics[f"{ANSWERABLE}_geometry_r_at_1"] == pytest.approx(0.5)
    assert metrics[f"{ANSWERABLE}_appearance_misses_at_1"] == 2
    assert metrics[f"{ANSWERABLE}_rescued_at_1"] == 1
    assert metrics[f"{ANSWERABLE}_rescue_rate_at_1"] == pytest.approx(0.5)
    # Equal recall hides one win and one loss, which is why the pairing is reported.
    assert metrics[f"{ANSWERABLE}_geometry_vs_appearance_at_1_wins"] == 1
    assert metrics[f"{ANSWERABLE}_geometry_vs_appearance_at_1_losses"] == 1
    assert metrics[f"{ANSWERABLE}_geometry_vs_appearance_at_1_p_value"] == pytest.approx(1.0)
    assert metrics[f"{ANSWERABLE}_queries"] == 4


def test_a_rescue_rate_is_omitted_when_appearance_missed_nothing() -> None:
    """Zero of zero is not zero percent, and must not be published as one."""
    metrics = {m.name: m.value for m in summarize((_ranks(1, 1),), (1,))}

    assert f"{ANSWERABLE}_rescue_rate_at_1" not in metrics
    assert metrics[f"{ANSWERABLE}_appearance_misses_at_1"] == 0
    assert metrics[f"{ANSWERABLE}_rescued_at_1"] == 0


def test_both_arms_are_reported_and_cut_offs_must_be_positive() -> None:
    measured = (
        QueryRanks(answerable={APPEARANCE: 1, GEOMETRY: 5}, disjoint={APPEARANCE: 9, GEOMETRY: 2}),
    )

    metrics = {m.name: m.value for m in summarize(measured, (1, 10))}

    assert metrics[f"{ANSWERABLE}_appearance_r_at_1"] == pytest.approx(1.0)
    assert metrics[f"{DISJOINT}_appearance_r_at_1"] == pytest.approx(0.0)
    assert metrics[f"{DISJOINT}_geometry_r_at_10"] == pytest.approx(1.0)

    with pytest.raises(GeometryFirstError, match="must be positive"):
        summarize(measured, (0,))


def test_a_class_verifies_against_itself_rather_than_its_neighbours(tmp_path: Path) -> None:
    """The end-to-end path, on images whose classes are genuinely distinguishable."""
    manifest = _by_two_photographers(_manifest_with_images(tmp_path))
    seed_embedding_cache(tmp_path / "embeddings", manifest)
    matrix = load_embedding_matrix(manifest, FAKE_BACKBONE, tmp_path / "embeddings")
    split = build_evaluation_split(manifest, datetime.now(UTC))

    measured = measure(split, manifest, matrix, _experiment(max_keypoints=200))

    assert len(measured) == len(split.folds)
    # Each class's photographs share a texture and no two classes share one, so
    # geometry alone should put the right statue first.
    assert all(query.answerable[GEOMETRY] == 1 for query in measured)


def test_single_photographer_classes_are_skipped(tmp_path: Path) -> None:
    """The same answerable subset as sections 9 and 11, so the columns line up."""
    manifest = _manifest_with_images(tmp_path)  # one shared author throughout
    seed_embedding_cache(tmp_path / "embeddings", manifest)
    matrix = load_embedding_matrix(manifest, FAKE_BACKBONE, tmp_path / "embeddings")
    split = build_evaluation_split(manifest, datetime.now(UTC))

    with pytest.raises(GeometryFirstError, match="no answerable query"):
        measure(split, manifest, matrix, _experiment(max_keypoints=200))


def test_max_queries_stops_early_without_changing_what_is_measured(tmp_path: Path) -> None:
    """The knob that makes the path runnable without paying for the full sweep."""
    manifest = _by_two_photographers(_manifest_with_images(tmp_path))
    seed_embedding_cache(tmp_path / "embeddings", manifest)
    matrix = load_embedding_matrix(manifest, FAKE_BACKBONE, tmp_path / "embeddings")
    split = build_evaluation_split(manifest, datetime.now(UTC))

    everything = measure(split, manifest, matrix, _experiment(max_keypoints=200))
    limited = measure(split, manifest, matrix, _experiment(max_keypoints=200, max_queries=2))

    assert len(limited) == 2
    assert limited == everything[:2]


def test_run_requires_the_geometry_first_experiment_group() -> None:
    with pytest.raises(GeometryFirstError, match="requires experiment=geometry_first"):
        run_geometry_first(load_config(["experiment=baseline"]))


def test_packaged_defaults_sweep_everything_at_the_reranker_s_keypoints() -> None:
    experiment = _experiment()

    assert experiment.max_queries == 0, "the published run asks every answerable query"
    # Matched to the re-ranking sweep, or the two experiments describe different
    # keypoints and their inlier counts stop being comparable.
    rerank = load_config(["experiment=rerank_ablation"]).experiment
    assert isinstance(rerank, RerankAblationConfig)
    assert experiment.max_keypoints == rerank.max_keypoints

    with pytest.raises(ValueError, match="must be positive"):
        GeometryFirstConfig.model_validate(
            {"kind": "geometry_first", "seed": 1, "top_k": (0,), "max_keypoints": 200}
        )


def test_cli_ranks_by_geometry_and_writes_the_artifact(tmp_path: Path) -> None:
    manifest = _by_two_photographers(_manifest_with_images(tmp_path))
    seed_embedding_cache(tmp_path / "embeddings", manifest)
    split = build_evaluation_split(manifest, datetime.now(UTC))
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(manifest.model_dump_json(), encoding="utf-8")
    split_path = tmp_path / "split.json"
    write_evaluation_split(split_path, split)

    result = runner.invoke(
        app,
        [
            "experiment",
            "geometry-first",
            "-o",
            f"paths.manifest_path={manifest_path}",
            "-o",
            f"paths.evaluation_split_path={split_path}",
            "-o",
            f"paths.embeddings_dir={tmp_path / 'embeddings'}",
            "-o",
            f"paths.results_dir={tmp_path / 'results'}",
            "-o",
            f"backbone.model_id={FAKE_BACKBONE.model_id}",
            "-o",
            f"backbone.revision={FAKE_BACKBONE.revision}",
            "-o",
            f"backbone.preprocessing_id={FAKE_BACKBONE.preprocessing_id}",
            "-o",
            "experiment.max_keypoints=200",
        ],
    )

    assert result.exit_code == 0, result.output
    written = json.loads((tmp_path / "results" / "geometry_first-dinov2.json").read_text())
    assert written["experiment"] == "geometry_first"
    names = {metric["name"] for metric in written["metrics"]}
    assert f"{DISJOINT}_geometry_r_at_1" in names
    assert f"{ANSWERABLE}_appearance_r_at_1" in names
    # The configuration is recorded, which is what arms the overwrite guard.
    assert written["configuration"]["max_keypoints"] == 200

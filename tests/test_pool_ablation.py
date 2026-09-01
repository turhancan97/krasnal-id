"""Candidate-pool-size ablation protocol, determinism, and reporting."""

import json
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pytest
from typer.testing import CliRunner

from helpers import FAKE_BACKBONE, seed_embedding_cache, synthetic_manifest
from krasnal_id.cli import app
from krasnal_id.config import load_config
from krasnal_id.data_pipeline.build_split import build_evaluation_split, write_evaluation_split
from krasnal_id.embeddings.store import EmbeddingMatrix, load_embedding_matrix
from krasnal_id.experiments.contracts import ExperimentResult
from krasnal_id.experiments.pool_size_ablation import (
    PoolAblationError,
    PoolMeasurement,
    measure_pool_size,
    points_per_doubling,
    resolve_pool_sizes,
    run_pool_size_ablation,
    summarize_measurements,
)
from krasnal_id.models import DatasetManifest, EvaluationSplit

SEEDS = (11, 23, 37)


def _setup(
    tmp_path: Path, dwarf_count: int = 6
) -> tuple[DatasetManifest, EvaluationSplit, EmbeddingMatrix, tuple[str, ...]]:
    manifest = synthetic_manifest(dwarf_count=dwarf_count)
    seed_embedding_cache(tmp_path, manifest)
    matrix = load_embedding_matrix(manifest, FAKE_BACKBONE, tmp_path)
    split = build_evaluation_split(manifest, datetime.now(UTC))
    return manifest, split, matrix, tuple(sorted(set(matrix.dwarf_ids)))


def test_configured_pool_sizes_are_filtered_to_what_the_dataset_supports() -> None:
    # 50 and 100 cannot be sampled from 23 dwarves; the full pool is always measured.
    assert resolve_pool_sizes((2, 3, 5, 8, 10, 15, 20, 50, 100), 23) == (
        2,
        3,
        5,
        8,
        10,
        15,
        20,
        23,
    )
    assert resolve_pool_sizes((2, 5), 5) == (2, 5)
    # An already-present full pool is not duplicated.
    assert resolve_pool_sizes((2, 4), 4) == (2, 4)

    with pytest.raises(PoolAblationError, match="at least two dwarves"):
        resolve_pool_sizes((2,), 1)
    with pytest.raises(PoolAblationError, match="at least two"):
        resolve_pool_sizes((1, 5), 10)


def test_a_pool_restricts_which_dwarves_can_be_confused(tmp_path: Path) -> None:
    _, split, matrix, dwarf_ids = _setup(tmp_path)

    small = measure_pool_size(split, matrix, 2, SEEDS, dwarf_ids)
    full = measure_pool_size(split, matrix, len(dwarf_ids), SEEDS, dwarf_ids)

    # Separated clusters score perfectly at any pool size, but the protocol must
    # still have narrowed the candidate set rather than scoring the full pool twice.
    assert small.pool_size == 2
    assert full.pool_size == 6
    assert small.top_1 == pytest.approx(1.0)
    assert full.top_1 == pytest.approx(1.0)
    # Sampling cannot vary at the full pool, so every seed must agree exactly.
    assert len(set(full.top_1_per_seed)) == 1
    assert len(full.top_1_per_seed) == len(SEEDS)


def test_measurements_are_reproducible_and_seed_dependent(tmp_path: Path) -> None:
    # Indistinguishable vectors force the ranking onto the tie-break, which makes
    # the measurement sensitive to which distractors each seed draws.
    manifest = synthetic_manifest(dwarf_count=6)
    seed_embedding_cache(tmp_path, manifest, lambda *_: np.asarray([1.0, 0.0], dtype=np.float32))
    matrix = load_embedding_matrix(manifest, FAKE_BACKBONE, tmp_path)
    split = build_evaluation_split(manifest, datetime.now(UTC))
    dwarf_ids = tuple(sorted(set(matrix.dwarf_ids)))

    first = measure_pool_size(split, matrix, 3, SEEDS, dwarf_ids)
    again = measure_pool_size(split, matrix, 3, SEEDS, dwarf_ids)
    other_seeds = measure_pool_size(split, matrix, 3, (99, 100, 101), dwarf_ids)

    assert first == again
    assert first.top_1_per_seed != other_seeds.top_1_per_seed
    # Requesting one pool size must not change the draw for another.
    assert measure_pool_size(split, matrix, 3, SEEDS[:1], dwarf_ids).top_1_per_seed == (
        first.top_1_per_seed[0],
    )


def test_accuracy_falls_as_the_pool_grows_when_classes_overlap(tmp_path: Path) -> None:
    # Dwarves are paired onto a shared axis with interleaved offsets, so each one's
    # nearest neighbour is its partner rather than its own other images. A larger
    # pool is then more likely to contain that partner and lose the top rank to it.
    manifest = synthetic_manifest(dwarf_count=8)

    def interleaved(dwarf_index: int, position: int, dwarf_count: int) -> np.ndarray:
        vector = np.zeros(dwarf_count // 2 + 2, dtype=np.float32)
        vector[dwarf_index // 2] = 1.0
        vector[-1] = 0.02 * position + 0.01 * (dwarf_index % 2)
        return np.asarray(vector / np.linalg.norm(vector), dtype=np.float32)

    seed_embedding_cache(tmp_path, manifest, interleaved)
    matrix = load_embedding_matrix(manifest, FAKE_BACKBONE, tmp_path)
    split = build_evaluation_split(manifest, datetime.now(UTC))
    dwarf_ids = tuple(sorted(set(matrix.dwarf_ids)))

    small = measure_pool_size(split, matrix, 2, SEEDS, dwarf_ids)
    large = measure_pool_size(split, matrix, 8, SEEDS, dwarf_ids)

    assert small.top_1 > large.top_1
    assert points_per_doubling((small, large)) < 0.0


def test_points_per_doubling_recovers_a_known_slope() -> None:
    # A clean two-point-per-doubling decline from 1.00 at N=2.
    measurements = tuple(
        PoolMeasurement(
            pool_size=size,
            top_1_per_seed=(1.0 - 0.02 * index,),
            mrr_per_seed=(1.0,),
        )
        for index, size in enumerate((2, 4, 8, 16))
    )

    assert points_per_doubling(measurements) == pytest.approx(-2.0)
    with pytest.raises(PoolAblationError, match="at least two measured"):
        points_per_doubling(measurements[:1])


def test_summary_reports_a_curve_with_seed_spread_as_error_bars() -> None:
    measurements = (
        PoolMeasurement(pool_size=2, top_1_per_seed=(0.9, 1.0), mrr_per_seed=(0.95, 1.0)),
        PoolMeasurement(pool_size=4, top_1_per_seed=(0.8, 0.9), mrr_per_seed=(0.85, 0.9)),
    )

    metrics = {metric.name: metric for metric in summarize_measurements(measurements)}

    assert metrics["top_1_pool_2"].value == pytest.approx(0.95)
    assert metrics["top_1_pool_2"].lower_bound == pytest.approx(0.9)
    assert metrics["top_1_pool_2"].upper_bound == pytest.approx(1.0)
    assert metrics["mrr_pool_4"].value == pytest.approx(0.875)
    assert metrics["top_1_points_per_doubling"].value == pytest.approx(-10.0)
    assert metrics["candidate_dwarfs"].value == pytest.approx(4.0)
    assert metrics["evaluated_pool_sizes"].value == pytest.approx(2.0)


def test_run_requires_the_ablation_experiment_group() -> None:
    with pytest.raises(PoolAblationError, match="requires experiment=pool_size_ablation"):
        run_pool_size_ablation(load_config(["experiment=baseline"]))


def test_cli_ablation_writes_a_curve_and_fails_without_embeddings(tmp_path: Path) -> None:
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
    arguments = ["experiment", "pool-ablation", *[f"-o{value}" for value in overrides]]
    runner = CliRunner()

    missing = runner.invoke(app, arguments)
    assert missing.exit_code == 2
    assert "Pool-size ablation error" in missing.output

    seed_embedding_cache(tmp_path / "embeddings", manifest)
    result = runner.invoke(app, arguments)
    assert result.exit_code == 0, result.output
    assert "top_1_pool_2" in result.output
    assert "seeds" in result.output

    written = ExperimentResult.model_validate(
        json.loads(
            (tmp_path / "results" / "pool_size_ablation-dinov2.json").read_text(encoding="utf-8")
        )
    )
    assert written.experiment == "pool_size_ablation"
    assert written.seed == 11
    names = {metric.name for metric in written.metrics}
    # Configured sizes above the four available dwarves are skipped, not clamped.
    assert "top_1_pool_4" in names
    assert not {name for name in names if name.startswith("top_1_pool_") and name > "top_1_pool_4"}
    assert "top_1_points_per_doubling" in names

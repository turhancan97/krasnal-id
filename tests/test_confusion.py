"""Most-confused-pair analysis over the full candidate pool."""

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
from krasnal_id.experiments.confusion_analysis import (
    Competition,
    ConfusionAnalysisError,
    analyze_confusion,
    find_strongest_competitor,
    rank_confusion_pairs,
    run_confusion_analysis,
)
from krasnal_id.experiments.contracts import ConfusionAnalysisResult, ConfusionPair


def _competition(true_id: str, competitor: str, margin: float, query: str = "q") -> Competition:
    return Competition(
        query_image_id=query,
        true_dwarf_id=true_id,
        competitor_dwarf_id=competitor,
        margin=margin,
    )


def test_competitor_is_the_best_wrong_dwarf_and_the_margin_signs_the_outcome() -> None:
    matrix = EmbeddingMatrix(
        image_ids=("own", "near-wrong", "far-wrong", "query"),
        dwarf_ids=("Q0", "Q1", "Q2", "Q0"),
        vectors=np.asarray(
            [[0.8, 0.6], [0.9, 0.1], [0.0, 1.0], [1.0, 0.0]],
            dtype=np.float32,
        ),
    )

    lost = find_strongest_competitor("query", "Q0", ("own", "near-wrong", "far-wrong"), matrix)

    # Q1 is nearer the query than the query's own reference, so Q0 was misidentified.
    assert lost.competitor_dwarf_id == "Q1"
    assert lost.margin < 0.0
    assert lost.misidentified

    won = find_strongest_competitor("query", "Q0", ("own", "far-wrong"), matrix)
    assert won.competitor_dwarf_id == "Q2"
    assert won.margin > 0.0
    assert not won.misidentified


def test_a_query_needs_its_own_dwarf_and_a_competitor() -> None:
    matrix = EmbeddingMatrix(
        image_ids=("own", "other", "query"),
        dwarf_ids=("Q0", "Q1", "Q0"),
        vectors=np.asarray([[0.8, 0.6], [0.0, 1.0], [1.0, 0.0]], dtype=np.float32),
    )

    with pytest.raises(ConfusionAnalysisError, match="no reference image"):
        find_strongest_competitor("query", "Q9", ("own", "other"), matrix)
    with pytest.raises(ConfusionAnalysisError, match="no competing dwarf"):
        find_strongest_competitor("query", "Q0", ("own",), matrix)


def test_pairs_rank_misidentifications_first_then_frequency_then_margin() -> None:
    competitions = (
        # One outright error, seen once.
        _competition("Q0", "Q1", -0.01),
        # No errors, but competes constantly at a tight margin.
        *[_competition("Q2", "Q3", 0.02) for _ in range(9)],
        # No errors, competes as often but with more headroom.
        *[_competition("Q4", "Q5", 0.40) for _ in range(9)],
    )
    names = {f"Q{index}": f"Dwarf {index}" for index in range(6)}

    pairs = rank_confusion_pairs(competitions, names, top_pairs=10)

    assert [(pair.true_dwarf_id, pair.confused_dwarf_id) for pair in pairs] == [
        ("Q0", "Q1"),
        ("Q2", "Q3"),
        ("Q4", "Q5"),
    ]
    assert pairs[0].misidentifications == 1
    assert pairs[0].queries == 1
    assert pairs[1].queries == 9
    assert pairs[1].mean_margin == pytest.approx(0.02)
    assert pairs[0].true_display_name == "Dwarf 0"
    # An unknown dwarf falls back to its QID rather than failing validation.
    assert rank_confusion_pairs(competitions[:1], {}, 1)[0].true_display_name == "Q0"


def test_pair_reporting_is_directed_and_truncated() -> None:
    competitions = (
        _competition("Q0", "Q1", -0.01),
        _competition("Q1", "Q0", -0.02),
        _competition("Q2", "Q3", 0.5),
    )
    names = {f"Q{index}": f"Dwarf {index}" for index in range(4)}

    both = rank_confusion_pairs(competitions, names, top_pairs=10)
    assert len(both) == 3
    # A mutual confusion appears once per direction rather than being averaged away.
    assert ("Q0", "Q1") in [(pair.true_dwarf_id, pair.confused_dwarf_id) for pair in both]
    assert ("Q1", "Q0") in [(pair.true_dwarf_id, pair.confused_dwarf_id) for pair in both]

    assert len(rank_confusion_pairs(competitions, names, top_pairs=1)) == 1
    with pytest.raises(ConfusionAnalysisError, match="top_pairs must be positive"):
        rank_confusion_pairs(competitions, names, top_pairs=0)


def test_pair_contract_rejects_impossible_counts() -> None:
    with pytest.raises(ValueError, match="cannot exceed the queries"):
        ConfusionPair(
            true_dwarf_id="Q0",
            true_display_name="A",
            confused_dwarf_id="Q1",
            confused_display_name="B",
            queries=1,
            misidentifications=2,
            mean_margin=0.0,
        )
    with pytest.raises(ValueError, match="cannot be confused with itself"):
        ConfusionPair(
            true_dwarf_id="Q0",
            true_display_name="A",
            confused_dwarf_id="Q0",
            confused_display_name="A",
            queries=1,
            misidentifications=0,
            mean_margin=0.0,
        )


def test_analysis_summarizes_a_separable_dataset(tmp_path: Path) -> None:
    manifest = synthetic_manifest(dwarf_count=4)
    seed_embedding_cache(tmp_path, manifest)
    matrix = load_embedding_matrix(manifest, FAKE_BACKBONE, tmp_path)
    split = build_evaluation_split(manifest, datetime.now(UTC))

    metrics, pairs = analyze_confusion(manifest, split, matrix, top_pairs=5)
    named = {metric.name: metric.value for metric in metrics}

    assert named["top_1_errors"] == pytest.approx(0.0)
    assert named["top_1_error_rate"] == pytest.approx(0.0)
    assert named["reported_pairs"] == pytest.approx(len(pairs))
    assert named["competing_pairs"] > 0
    # Separated clusters leave a healthy positive margin on every query.
    assert named["mean_margin"] > 0.0
    assert named["median_margin"] > 0.0
    assert all(pair.misidentifications == 0 for pair in pairs)

    with pytest.raises(ConfusionAnalysisError, match="no folds"):
        analyze_confusion(manifest, split.model_copy(update={"folds": ()}), matrix, 5)


def test_analysis_finds_a_planted_confusion(tmp_path: Path) -> None:
    # Q1 and Q2 share an axis with interleaved offsets, so each is the other's
    # nearest neighbour and both directions should be misidentified.
    manifest = synthetic_manifest(dwarf_count=4)

    def interleaved(dwarf_index: int, position: int, dwarf_count: int) -> np.ndarray:
        vector = np.zeros(dwarf_count + 1, dtype=np.float32)
        vector[0 if dwarf_index < 2 else dwarf_index] = 1.0
        vector[-1] = 0.02 * position + 0.01 * (dwarf_index % 2) if dwarf_index < 2 else 0.0
        return np.asarray(vector / np.linalg.norm(vector), dtype=np.float32)

    seed_embedding_cache(tmp_path, manifest, interleaved)
    matrix = load_embedding_matrix(manifest, FAKE_BACKBONE, tmp_path)
    split = build_evaluation_split(manifest, datetime.now(UTC))

    metrics, pairs = analyze_confusion(manifest, split, matrix, top_pairs=20)
    named = {metric.name: metric.value for metric in metrics}
    confused = {
        (pair.true_dwarf_id, pair.confused_dwarf_id) for pair in pairs if pair.misidentifications
    }

    assert named["top_1_errors"] > 0
    assert ("Q1", "Q2") in confused or ("Q2", "Q1") in confused
    assert pairs[0].mean_margin < 0.1


def test_run_requires_the_confusion_experiment_group() -> None:
    with pytest.raises(ConfusionAnalysisError, match="requires experiment=confusion"):
        run_confusion_analysis(load_config(["experiment=baseline"]))


def test_cli_confusion_reports_pairs_and_fails_without_embeddings(tmp_path: Path) -> None:
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
    arguments = ["experiment", "confusion", *[f"-o{value}" for value in overrides]]
    runner = CliRunner()

    missing = runner.invoke(app, arguments)
    assert missing.exit_code == 2
    assert "Confusion analysis error" in missing.output

    seed_embedding_cache(tmp_path / "embeddings", manifest)
    result = runner.invoke(app, arguments)
    assert result.exit_code == 0, result.output
    assert "top_1_error_rate" in result.output
    assert "most-confused pairs" in result.output
    assert "mean margin" in result.output

    written = ConfusionAnalysisResult.model_validate(
        json.loads((tmp_path / "results" / "confusion-dinov2.json").read_text(encoding="utf-8"))
    )
    assert written.experiment == "confusion"
    assert written.pairs
    assert all(pair.true_dwarf_id != pair.confused_dwarf_id for pair in written.pairs)

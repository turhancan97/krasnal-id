"""Trained-classifier comparison against raw cosine retrieval."""

import json
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pytest
from typer.testing import CliRunner

from helpers import FAKE_BACKBONE, seed_embedding_cache, synthetic_manifest
from krasnal_id.cli import app
from krasnal_id.config import ProbeExperimentConfig, load_config
from krasnal_id.data_pipeline.build_split import build_evaluation_split, write_evaluation_split
from krasnal_id.embeddings.store import load_embedding_matrix
from krasnal_id.experiments.contracts import ExperimentResult
from krasnal_id.experiments.probe_baseline import (
    MethodOutcome,
    ProbeExperimentError,
    class_prototypes,
    evaluate_methods,
    import_optional_analysis,
    prototype_rank,
    run_probe_comparison,
    single_threaded_math,
    summarize_methods,
)


def _config(**overrides: object) -> ProbeExperimentConfig:
    values: dict[str, object] = {
        "kind": "probe",
        "seed": 42,
        "methods": ("retrieval", "prototype"),
        "top_k": (1, 5),
        "max_iterations": 200,
        "regularization": 100.0,
    }
    values.update(overrides)
    return ProbeExperimentConfig.model_validate(values)


def test_config_rejects_duplicate_methods_and_bad_cut_offs() -> None:
    with pytest.raises(ValueError, match="cannot contain duplicates"):
        _config(methods=("prototype", "prototype"))
    with pytest.raises(ValueError, match="top_k values must be positive"):
        _config(top_k=(1, 0))
    with pytest.raises(ValueError, match="greater than 0"):
        _config(regularization=0.0)


def test_packaged_probe_defaults_use_weak_regularization() -> None:
    # Unit-norm embeddings need a large C; a small one silently underfits.
    experiment = load_config(["experiment=probe"]).experiment
    assert isinstance(experiment, ProbeExperimentConfig)
    assert experiment.regularization >= 100.0
    assert "retrieval" in experiment.methods


def test_prototypes_are_class_means_on_the_unit_sphere() -> None:
    vectors = np.asarray(
        [[1.0, 0.0], [0.0, 1.0], [0.0, 1.0], [0.0, -1.0]],
        dtype=np.float32,
    )
    labels = ("Q1", "Q1", "Q2", "Q3")

    names, prototypes = class_prototypes(vectors, labels)

    assert names == ("Q1", "Q2", "Q3")
    np.testing.assert_allclose(np.linalg.norm(prototypes, axis=1), 1.0, atol=1e-6)
    # Q1 averages two orthogonal vectors, so its prototype bisects them.
    np.testing.assert_allclose(prototypes[0], [2**-0.5, 2**-0.5], atol=1e-6)
    np.testing.assert_allclose(prototypes[1], [0.0, 1.0], atol=1e-6)


def test_a_cancelling_class_mean_is_rejected() -> None:
    # Opposite vectors average to zero, which has no direction to compare against.
    with pytest.raises(ProbeExperimentError, match="zero or invalid norm"):
        class_prototypes(
            np.asarray([[1.0, 0.0], [-1.0, 0.0]], dtype=np.float32),
            ("Q1", "Q1"),
        )


def test_prototype_ranking_prefers_the_nearest_class_mean() -> None:
    vectors = np.asarray(
        [[1.0, 0.0], [0.99, 0.14], [0.0, 1.0], [0.14, 0.99]],
        dtype=np.float32,
    )
    labels = ("Q1", "Q1", "Q2", "Q2")
    query = np.asarray([1.0, 0.0], dtype=np.float32)

    assert prototype_rank(query, vectors, labels, "Q1") == 1
    assert prototype_rank(query, vectors, labels, "Q2") == 2
    with pytest.raises(ProbeExperimentError, match="absent from the classifier's classes"):
        prototype_rank(query, vectors, labels, "Q9")


def test_method_outcome_summarizes_ranks() -> None:
    outcome = MethodOutcome(method="prototype", ranks=(1, 1, 2, 4))

    assert outcome.top_k_hits(1) == 2
    assert outcome.top_k_hits(2) == 3
    assert outcome.top_k_hits(5) == 4
    assert outcome.mrr == pytest.approx((1.0 + 1.0 + 0.5 + 0.25) / 4)


def test_summary_records_the_gap_against_retrieval() -> None:
    outcomes = (
        MethodOutcome(method="retrieval", ranks=(1, 1, 2, 2)),
        MethodOutcome(method="prototype", ranks=(1, 1, 1, 2)),
    )

    metrics = {metric.name: metric.value for metric in summarize_methods(outcomes, (1, 5))}

    assert metrics["retrieval_top_1"] == pytest.approx(0.5)
    assert metrics["prototype_top_1"] == pytest.approx(0.75)
    assert metrics["prototype_top_1_gain_over_retrieval"] == pytest.approx(0.25)
    assert metrics["evaluated_folds"] == pytest.approx(4.0)
    # Without a retrieval arm there is nothing to compare against.
    solo = {
        metric.name
        for metric in summarize_methods((MethodOutcome(method="prototype", ranks=(1,)),), (1,))
    }
    assert not {name for name in solo if name.endswith("_gain_over_retrieval")}

    with pytest.raises(ProbeExperimentError, match="no methods were evaluated"):
        summarize_methods((), (1,))


def test_thread_limiting_is_optional() -> None:
    # Purely a performance measure, so it must never break the run.
    with single_threaded_math():
        assert True

    with pytest.raises(ProbeExperimentError, match="uv sync --extra analysis"):
        import_optional_analysis("krasnal_id_absent_probe_module")


def test_methods_agree_on_a_separable_dataset(tmp_path: Path) -> None:
    manifest = synthetic_manifest(dwarf_count=4)
    seed_embedding_cache(tmp_path, manifest)
    matrix = load_embedding_matrix(manifest, FAKE_BACKBONE, tmp_path)
    split = build_evaluation_split(manifest, datetime.now(UTC))

    outcomes = evaluate_methods(split, matrix, _config())

    assert [outcome.method for outcome in outcomes] == ["retrieval", "prototype"]
    for outcome in outcomes:
        assert len(outcome.ranks) == len(split.folds)
        # Cleanly separated clusters are trivial for both methods.
        assert outcome.top_k_hits(1) == len(split.folds)

    with pytest.raises(ProbeExperimentError, match="no folds"):
        evaluate_methods(split.model_copy(update={"folds": ()}), matrix, _config())


def test_the_linear_probe_runs_on_real_vectors(tmp_path: Path) -> None:
    pytest.importorskip("sklearn.linear_model")
    manifest = synthetic_manifest(dwarf_count=4)
    seed_embedding_cache(tmp_path, manifest)
    matrix = load_embedding_matrix(manifest, FAKE_BACKBONE, tmp_path)
    split = build_evaluation_split(manifest, datetime.now(UTC))

    outcomes = evaluate_methods(split, matrix, _config(methods=("linear_probe",)))
    again = evaluate_methods(split, matrix, _config(methods=("linear_probe",)))

    assert outcomes[0].method == "linear_probe"
    assert outcomes[0].ranks == again[0].ranks
    assert outcomes[0].top_k_hits(1) == len(split.folds)


def test_run_requires_the_probe_experiment_group() -> None:
    with pytest.raises(ProbeExperimentError, match="requires experiment=probe"):
        run_probe_comparison(load_config(["experiment=baseline"]))


def test_cli_probe_compares_methods_and_fails_without_embeddings(tmp_path: Path) -> None:
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
        "experiment.methods=[retrieval,prototype]",
        "logging.json_output=false",
    ]
    arguments = ["experiment", "probe", *[f"-o{value}" for value in overrides]]
    runner = CliRunner()

    missing = runner.invoke(app, arguments)
    assert missing.exit_code == 2
    assert "Probe comparison error" in missing.output

    seed_embedding_cache(tmp_path / "embeddings", manifest)
    result = runner.invoke(app, arguments)
    assert result.exit_code == 0, result.output
    assert "retrieval_top_1" in result.output
    assert "prototype_top_1" in result.output
    assert "gain_over_retrieval" in result.output

    written = ExperimentResult.model_validate(
        json.loads((tmp_path / "results" / "probe-dinov2.json").read_text(encoding="utf-8"))
    )
    assert written.experiment == "probe"
    assert {metric.name for metric in written.metrics} >= {
        "retrieval_top_1",
        "prototype_mrr",
        "prototype_top_1_gain_over_retrieval",
    }

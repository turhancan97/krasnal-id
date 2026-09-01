"""Ablation curve extraction and rendering from saved results."""

import json
from datetime import UTC, datetime
from pathlib import Path

import pytest
from typer.testing import CliRunner

from krasnal_id.cli import app
from krasnal_id.config import load_config
from krasnal_id.experiments.artifacts import write_experiment_result
from krasnal_id.experiments.contracts import ExperimentResult, MetricSummary
from krasnal_id.viz.ablation_plot import (
    create_ablation_plot,
    curve_from_result,
    find_ablation_results,
    read_ablation_result,
    render_curves,
)
from krasnal_id.viz.embedding_plot import VisualizationError


def _result(backbone: str = "dinov2", experiment: str = "pool_size_ablation") -> ExperimentResult:
    return ExperimentResult(
        experiment=experiment,
        backbone=backbone,
        created_at=datetime.now(UTC),
        seed=11,
        metrics=(
            MetricSummary(name="top_1_pool_10", value=0.9, lower_bound=0.88, upper_bound=0.93),
            MetricSummary(name="top_1_pool_2", value=0.99, lower_bound=0.98, upper_bound=1.0),
            MetricSummary(name="mrr_pool_2", value=0.995),
            MetricSummary(name="top_1_points_per_doubling", value=-1.25),
        ),
    )


def test_curve_is_ordered_by_pool_size_regardless_of_metric_order() -> None:
    curve = curve_from_result(_result())

    # Metrics arrive in whatever order they were emitted; the curve must be sorted.
    assert curve.pool_sizes == (2, 10)
    assert curve.top_1 == pytest.approx((0.99, 0.9))
    assert curve.lower == pytest.approx((0.98, 0.88))
    assert curve.upper == pytest.approx((1.0, 0.93))
    assert curve.points_per_doubling == pytest.approx(-1.25)
    assert curve.backbone == "dinov2"


def test_a_missing_slope_and_missing_bounds_are_tolerated() -> None:
    result = ExperimentResult(
        experiment="pool_size_ablation",
        backbone="clip",
        created_at=datetime.now(UTC),
        seed=11,
        metrics=(MetricSummary(name="top_1_pool_4", value=0.5),),
    )

    curve = curve_from_result(result)

    assert curve.points_per_doubling is None
    # Without a recorded spread the error bar collapses onto the value.
    assert curve.lower == pytest.approx((0.5,))
    assert curve.upper == pytest.approx((0.5,))


def test_a_result_without_pool_metrics_is_rejected() -> None:
    barren = ExperimentResult(
        experiment="pool_size_ablation",
        backbone="dinov2",
        created_at=datetime.now(UTC),
        seed=11,
        metrics=(MetricSummary(name="evaluated_folds", value=146.0),),
    )

    with pytest.raises(VisualizationError, match="no top_1_pool_"):
        curve_from_result(barren)


def test_reading_rejects_unreadable_and_mismatched_artifacts(tmp_path: Path) -> None:
    with pytest.raises(VisualizationError, match="invalid ablation result"):
        read_ablation_result(tmp_path / "absent.json")

    broken = tmp_path / "broken.json"
    broken.write_text("{}", encoding="utf-8")
    with pytest.raises(VisualizationError, match="invalid ablation result"):
        read_ablation_result(broken)

    wrong = tmp_path / "wrong.json"
    write_experiment_result(wrong, _result(experiment="baseline"))
    with pytest.raises(VisualizationError, match="holds a baseline result"):
        read_ablation_result(wrong)


def test_results_are_discovered_in_a_stable_order(tmp_path: Path) -> None:
    write_experiment_result(tmp_path / "pool_size_ablation-dinov2.json", _result("dinov2"))
    write_experiment_result(tmp_path / "pool_size_ablation-clip.json", _result("clip"))
    write_experiment_result(tmp_path / "baseline-dinov2.json", _result(experiment="baseline"))

    found = find_ablation_results(tmp_path)

    assert [path.name for path in found] == [
        "pool_size_ablation-clip.json",
        "pool_size_ablation-dinov2.json",
    ]


def test_a_figure_is_rendered_for_every_curve(tmp_path: Path) -> None:
    pytest.importorskip("matplotlib")
    curves = (curve_from_result(_result("dinov2")), curve_from_result(_result("clip")))

    path = render_curves(curves, tmp_path / "curve.png")

    assert path.is_file()
    assert path.stat().st_size > 1000

    with pytest.raises(VisualizationError, match="no ablation curves"):
        render_curves((), tmp_path / "empty.png")


def test_create_requires_results_and_the_visualization_group(tmp_path: Path) -> None:
    with pytest.raises(VisualizationError, match="requires experiment=visualization"):
        create_ablation_plot(load_config(["experiment=baseline"]))

    empty = load_config(["experiment=visualization", f"paths.results_dir={tmp_path / 'nothing'}"])
    with pytest.raises(VisualizationError, match="run krasnal-id experiment pool-ablation"):
        create_ablation_plot(empty)


def test_cli_ablation_figure_is_written_and_failures_are_reported(tmp_path: Path) -> None:
    pytest.importorskip("matplotlib")
    results_dir = tmp_path / "results"
    runner = CliRunner()
    arguments = [
        "visualize",
        "ablation",
        f"-opaths.results_dir={results_dir}",
        "-ologging.json_output=false",
    ]

    missing = runner.invoke(app, arguments)
    assert missing.exit_code == 2
    assert "Ablation visualization error" in missing.output

    write_experiment_result(results_dir / "pool_size_ablation-dinov2.json", _result("dinov2"))
    result = runner.invoke(app, arguments)

    assert result.exit_code == 0, result.output
    assert (results_dir / "pool-size-ablation.png").is_file()
    assert (
        json.loads((results_dir / "pool_size_ablation-dinov2.json").read_text(encoding="utf-8"))[
            "experiment"
        ]
        == "pool_size_ablation"
    )

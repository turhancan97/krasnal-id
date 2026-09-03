"""Rejection tradeoff extraction and rendering from saved open-set results."""

from datetime import UTC, datetime
from pathlib import Path

import pytest
from typer.testing import CliRunner

from krasnal_id.cli import app
from krasnal_id.config import load_config
from krasnal_id.experiments.artifacts import write_experiment_result
from krasnal_id.experiments.contracts import (
    DwarfRejection,
    MetricSummary,
    OpenSetRejectionResult,
    RejectionOperatingPoint,
)
from krasnal_id.viz.embedding_plot import VisualizationError
from krasnal_id.viz.open_set_plot import (
    create_open_set_plot,
    curve_from_result,
    find_open_set_results,
    read_open_set_result,
    render_curves,
)


def _result(
    backbone: str = "dinov2",
    experiment: str = "open_set",
    curve: tuple[RejectionOperatingPoint, ...] | None = None,
    metrics: tuple[MetricSummary, ...] | None = None,
) -> OpenSetRejectionResult:
    return OpenSetRejectionResult(
        experiment=experiment,
        backbone=backbone,
        created_at=datetime.now(UTC),
        seed=11,
        metrics=metrics
        if metrics is not None
        else (
            MetricSummary(name="auroc", value=0.96),
            MetricSummary(name="target_90_known_acceptance", value=0.9),
            MetricSummary(name="target_90_false_acceptance", value=0.04),
            MetricSummary(name="target_95_known_acceptance", value=0.95),
            MetricSummary(name="target_95_false_acceptance", value=0.24),
            # The in-sample reference must not be drawn as an achievable point.
            MetricSummary(name="in_sample_target_90_known_acceptance", value=0.9),
            MetricSummary(name="in_sample_target_90_false_acceptance", value=0.04),
        ),
        rejections=(
            DwarfRejection(
                dwarf_id="Q1",
                display_name="Dwarf 1",
                unknown_queries=3,
                false_accepts=1,
                mean_top_similarity=0.7,
                nearest_dwarf_id="Q2",
                nearest_display_name="Dwarf 2",
            ),
        ),
        curve=curve
        if curve is not None
        else (
            RejectionOperatingPoint(threshold=0.1, known_acceptance=1.0, false_acceptance=1.0),
            RejectionOperatingPoint(threshold=0.8, known_acceptance=0.9, false_acceptance=0.04),
            RejectionOperatingPoint(threshold=0.99, known_acceptance=0.0, false_acceptance=0.0),
        ),
    )


def test_curve_is_ordered_and_carries_its_calibrated_points() -> None:
    curve = curve_from_result(_result())

    # Drawn in one direction regardless of the order the points were stored in.
    assert curve.false_acceptance == pytest.approx((0.0, 0.04, 1.0))
    assert curve.known_acceptance == pytest.approx((0.0, 0.9, 1.0))
    assert curve.auroc == pytest.approx(0.96)
    assert [point.target for point in curve.calibrated] == pytest.approx([0.9, 0.95])
    assert curve.calibrated[0].false_acceptance == pytest.approx(0.04)
    # Only the two calibrated targets: the in-sample pair carries the same numbers
    # and would otherwise be drawn a second time.
    assert len(curve.calibrated) == 2


def test_a_missing_auroc_and_half_a_target_are_tolerated() -> None:
    curve = curve_from_result(
        _result(
            metrics=(
                MetricSummary(name="target_90_known_acceptance", value=0.9),
                MetricSummary(name="known_queries", value=146.0),
            )
        )
    )

    assert curve.auroc is None
    # A target with only one of its two rates cannot be placed on the axes.
    assert curve.calibrated == ()


def test_a_result_without_a_curve_asks_for_a_rerun() -> None:
    with pytest.raises(VisualizationError, match="predates the curve field"):
        curve_from_result(_result(curve=()))


def test_reading_rejects_unreadable_and_mismatched_artifacts(tmp_path: Path) -> None:
    with pytest.raises(VisualizationError, match="invalid open-set result"):
        read_open_set_result(tmp_path / "absent.json")

    broken = tmp_path / "broken.json"
    broken.write_text("{}", encoding="utf-8")
    with pytest.raises(VisualizationError, match="invalid open-set result"):
        read_open_set_result(broken)

    # An artifact of the right shape but the wrong experiment: the name is what
    # decides, so a renamed or mis-copied file is caught rather than drawn.
    wrong = tmp_path / "wrong.json"
    write_experiment_result(wrong, _result(experiment="baseline"))
    with pytest.raises(VisualizationError, match="holds a baseline result"):
        read_open_set_result(wrong)


def test_results_are_discovered_in_a_stable_order(tmp_path: Path) -> None:
    write_experiment_result(tmp_path / "open_set-dinov2.json", _result("dinov2"))
    write_experiment_result(tmp_path / "open_set-clip.json", _result("clip"))
    write_experiment_result(tmp_path / "baseline-dinov2.json", _result(experiment="baseline"))

    found = find_open_set_results(tmp_path)

    assert [path.name for path in found] == ["open_set-clip.json", "open_set-dinov2.json"]


def test_a_figure_is_rendered_for_every_curve(tmp_path: Path) -> None:
    pytest.importorskip("matplotlib")
    curves = (curve_from_result(_result("dinov2")), curve_from_result(_result("clip")))

    path = render_curves(curves, tmp_path / "rejection.png")

    assert path.is_file()
    assert path.stat().st_size > 1000

    with pytest.raises(VisualizationError, match="no rejection curves"):
        render_curves((), tmp_path / "empty.png")


def test_create_requires_results_and_the_visualization_group(tmp_path: Path) -> None:
    with pytest.raises(VisualizationError, match="requires experiment=visualization"):
        create_open_set_plot(load_config(["experiment=baseline"]))

    empty = load_config(["experiment=visualization", f"paths.results_dir={tmp_path / 'nothing'}"])
    with pytest.raises(VisualizationError, match="run krasnal-id experiment open-set"):
        create_open_set_plot(empty)


def test_cli_open_set_figure_is_written_and_failures_are_reported(tmp_path: Path) -> None:
    pytest.importorskip("matplotlib")
    results_dir = tmp_path / "results"
    runner = CliRunner()
    arguments = [
        "visualize",
        "open-set",
        f"-opaths.results_dir={results_dir}",
        "-ologging.json_output=false",
    ]

    missing = runner.invoke(app, arguments)
    assert missing.exit_code == 2
    assert "Open-set visualization error" in missing.output

    write_experiment_result(results_dir / "open_set-dinov2.json", _result("dinov2"))
    result = runner.invoke(app, arguments)

    assert result.exit_code == 0, result.output
    assert (results_dir / "open-set-rejection.png").is_file()

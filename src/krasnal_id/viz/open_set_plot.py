"""Rejection tradeoff curves rendered from saved open-set results."""

import json
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from krasnal_id.config import AppConfig, VisualizationExperimentConfig
from krasnal_id.experiments.contracts import OpenSetRejectionResult
from krasnal_id.viz.embedding_plot import VisualizationError, import_optional_analysis

_TARGET_METRIC = re.compile(r"^target_([0-9.]+)_(known_acceptance|false_acceptance)$")
# Shared with the ablation figure, so a reader carries one color mapping between them.
_BACKBONE_COLORS = {"dinov2": "#1b6ca8", "clip": "#d1620a"}
_FALLBACK_COLORS = ("#4c8c2b", "#8f3f97", "#a3372f")


@dataclass(frozen=True, slots=True)
class CalibratedPoint:
    """One leave-one-class-out operating point, named by its acceptance target."""

    target: float
    known_acceptance: float
    false_acceptance: float


@dataclass(frozen=True, slots=True)
class RejectionCurve:
    """One backbone's full rejection tradeoff, plus its calibrated points."""

    backbone: str
    false_acceptance: tuple[float, ...]
    known_acceptance: tuple[float, ...]
    auroc: float | None
    calibrated: tuple[CalibratedPoint, ...]


def read_open_set_result(path: Path) -> OpenSetRejectionResult:
    """Read one saved open-set artifact."""
    try:
        result = OpenSetRejectionResult.model_validate(json.loads(path.read_text(encoding="utf-8")))
    except (OSError, json.JSONDecodeError, TypeError, ValueError) as error:
        raise VisualizationError(f"invalid open-set result {path}: {error}") from error
    if result.experiment != "open_set":
        raise VisualizationError(f"{path} holds a {result.experiment} result, not an open_set one")
    return result


def curve_from_result(result: OpenSetRejectionResult) -> RejectionCurve:
    """Extract the tradeoff curve and the calibrated points from one result."""
    if not result.curve:
        raise VisualizationError(
            f"{result.backbone} open-set result carries no curve; it predates the "
            "curve field and must be regenerated with krasnal-id experiment open-set"
        )

    auroc: float | None = None
    targets: dict[float, dict[str, float]] = {}
    for metric in result.metrics:
        if metric.name == "auroc":
            auroc = metric.value
            continue
        # Anchored, so the in-sample reference metrics are correctly ignored here.
        matched = _TARGET_METRIC.match(metric.name)
        if matched:
            targets.setdefault(float(matched.group(1)) / 100.0, {})[matched.group(2)] = metric.value

    calibrated = tuple(
        CalibratedPoint(
            target=target,
            known_acceptance=values["known_acceptance"],
            false_acceptance=values["false_acceptance"],
        )
        for target, values in sorted(targets.items())
        if {"known_acceptance", "false_acceptance"} <= values.keys()
    )
    # Ascending false acceptance, so the line is drawn in one direction.
    ordered = sorted(
        result.curve, key=lambda point: (point.false_acceptance, point.known_acceptance)
    )
    return RejectionCurve(
        backbone=result.backbone,
        false_acceptance=tuple(point.false_acceptance for point in ordered),
        known_acceptance=tuple(point.known_acceptance for point in ordered),
        auroc=auroc,
        calibrated=calibrated,
    )


def find_open_set_results(results_dir: Path) -> tuple[Path, ...]:
    """Find every saved open-set artifact, in a stable order."""
    return tuple(sorted(results_dir.glob("open_set-*.json")))


def render_curves(curves: tuple[RejectionCurve, ...], path: Path) -> Path:
    """Draw every backbone's rejection tradeoff on one axis."""
    if not curves:
        raise VisualizationError("no rejection curves to draw")

    matplotlib = import_optional_analysis("matplotlib")
    matplotlib.use("Agg")
    pyplot = import_optional_analysis("matplotlib.pyplot")

    figure, axes = pyplot.subplots(figsize=(8, 7))
    # Chance: a threshold that carries no information about presence.
    axes.plot([0, 100], [0, 100], color="#999999", linewidth=1.0, linestyle=":", label="chance")

    for index, curve in enumerate(curves):
        color = _BACKBONE_COLORS.get(
            curve.backbone, _FALLBACK_COLORS[index % len(_FALLBACK_COLORS)]
        )
        label = curve.backbone
        if curve.auroc is not None:
            label += f" (AUROC {curve.auroc:.3f})"
        axes.plot(
            np.asarray(curve.false_acceptance, dtype=float) * 100.0,
            np.asarray(curve.known_acceptance, dtype=float) * 100.0,
            color=color,
            linewidth=1.9,
            label=label,
        )
        for point in curve.calibrated:
            axes.plot(
                point.false_acceptance * 100.0,
                point.known_acceptance * 100.0,
                marker="o",
                markersize=7,
                color=color,
                markeredgecolor="white",
                markeredgewidth=1.2,
                linestyle="none",
            )
            # Just the percentage: two backbones' points sit close together on
            # this axis, and a longer label collides with its neighbour.
            axes.annotate(
                f"{point.target:.0%}",
                (point.false_acceptance * 100.0, point.known_acceptance * 100.0),
                textcoords="offset points",
                xytext=(8, -4),
                fontsize=8,
                color=color,
            )

    # One neutral entry explains every marked point, so the dots need no per-point
    # prose beside their target percentage.
    axes.plot(
        [],
        [],
        marker="o",
        markersize=7,
        color="#555555",
        markeredgecolor="white",
        markeredgewidth=1.2,
        linestyle="none",
        label="calibrated target acceptance",
    )

    axes.set_xlabel("unknown statues wrongly accepted (%)")
    axes.set_ylabel("known statues accepted (%)")
    axes.set_title("Rejecting a dwarf that is not in the reference set")
    axes.set_xlim(-2, 102)
    axes.set_ylim(-2, 102)
    axes.grid(True, alpha=0.25, linewidth=0.6)
    axes.legend(frameon=False, loc="lower right")
    axes.spines["top"].set_visible(False)
    axes.spines["right"].set_visible(False)
    axes.set_aspect("equal", adjustable="box")
    figure.tight_layout()

    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        figure.savefig(path, dpi=200)
    except OSError as error:
        raise VisualizationError(f"could not write figure {path}: {error}") from error
    finally:
        pyplot.close(figure)
    return path


def create_open_set_plot(config: AppConfig) -> Path:
    """Draw every saved open-set tradeoff into one comparable figure.

    The marked points are the calibrated ones, so the figure shows the achievable
    operating points against the descriptive curve they sit on rather than
    inviting a reader to pick a point off the curve directly.
    """
    if not isinstance(config.experiment, VisualizationExperimentConfig):
        raise VisualizationError(
            f"visualization requires experiment=visualization, got {config.experiment.kind}"
        )

    paths = find_open_set_results(config.paths.results_dir)
    if not paths:
        raise VisualizationError(
            f"no open_set results in {config.paths.results_dir}; "
            "run krasnal-id experiment open-set first"
        )
    curves = tuple(curve_from_result(read_open_set_result(path)) for path in paths)
    return render_curves(curves, config.paths.results_dir / "open-set-rejection.png")

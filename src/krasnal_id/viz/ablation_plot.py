"""Accuracy-versus-pool-size curve rendered from saved ablation results."""

import json
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from krasnal_id.config import AppConfig, VisualizationExperimentConfig
from krasnal_id.experiments.contracts import ExperimentResult
from krasnal_id.viz.embedding_plot import VisualizationError, import_optional_analysis

_POOL_METRIC = re.compile(r"^top_1_pool_(\d+)$")
# One color per backbone, distinguishable in print and for common color-vision
# deficiencies.
_BACKBONE_COLORS = {"dinov2": "#1b6ca8", "clip": "#d1620a"}
_FALLBACK_COLORS = ("#4c8c2b", "#8f3f97", "#a3372f")


@dataclass(frozen=True, slots=True)
class AblationCurve:
    """One backbone's measured accuracy against candidate-pool size."""

    backbone: str
    pool_sizes: tuple[int, ...]
    top_1: tuple[float, ...]
    lower: tuple[float, ...]
    upper: tuple[float, ...]
    points_per_doubling: float | None


def read_ablation_result(path: Path) -> ExperimentResult:
    """Read one saved ablation artifact."""
    try:
        result = ExperimentResult.model_validate(json.loads(path.read_text(encoding="utf-8")))
    except (OSError, json.JSONDecodeError, TypeError, ValueError) as error:
        raise VisualizationError(f"invalid ablation result {path}: {error}") from error
    if result.experiment != "pool_size_ablation":
        raise VisualizationError(
            f"{path} holds a {result.experiment} result, not a pool_size_ablation one"
        )
    return result


def curve_from_result(result: ExperimentResult) -> AblationCurve:
    """Extract the ordered accuracy curve from a result's flat metrics."""
    points: dict[int, tuple[float, float, float]] = {}
    slope: float | None = None
    for metric in result.metrics:
        matched = _POOL_METRIC.match(metric.name)
        if matched:
            pool = int(matched.group(1))
            lower = metric.lower_bound if metric.lower_bound is not None else metric.value
            upper = metric.upper_bound if metric.upper_bound is not None else metric.value
            points[pool] = (metric.value, lower, upper)
        elif metric.name == "top_1_points_per_doubling":
            slope = metric.value

    if not points:
        raise VisualizationError(
            f"{result.backbone} ablation result carries no top_1_pool_* metrics"
        )
    sizes = tuple(sorted(points))
    return AblationCurve(
        backbone=result.backbone,
        pool_sizes=sizes,
        top_1=tuple(points[size][0] for size in sizes),
        lower=tuple(points[size][1] for size in sizes),
        upper=tuple(points[size][2] for size in sizes),
        points_per_doubling=slope,
    )


def find_ablation_results(results_dir: Path) -> tuple[Path, ...]:
    """Find every saved ablation artifact, in a stable order."""
    return tuple(sorted(results_dir.glob("pool_size_ablation-*.json")))


def render_curves(curves: tuple[AblationCurve, ...], path: Path) -> Path:
    """Draw every backbone's curve on one log-scaled pool-size axis."""
    if not curves:
        raise VisualizationError("no ablation curves to draw")

    matplotlib = import_optional_analysis("matplotlib")
    matplotlib.use("Agg")
    pyplot = import_optional_analysis("matplotlib.pyplot")

    figure, axes = pyplot.subplots(figsize=(9, 6))
    for index, curve in enumerate(curves):
        color = _BACKBONE_COLORS.get(
            curve.backbone, _FALLBACK_COLORS[index % len(_FALLBACK_COLORS)]
        )
        sizes = np.asarray(curve.pool_sizes, dtype=float)
        values = np.asarray(curve.top_1, dtype=float) * 100.0
        # Error bars are the observed spread across seeds, so they are asymmetric.
        errors = np.vstack(
            [
                values - np.asarray(curve.lower, dtype=float) * 100.0,
                np.asarray(curve.upper, dtype=float) * 100.0 - values,
            ]
        )
        label = curve.backbone
        if curve.points_per_doubling is not None:
            label += f" ({curve.points_per_doubling:+.2f} pts/doubling)"
        axes.errorbar(
            sizes,
            values,
            yerr=np.clip(errors, 0.0, None),
            color=color,
            marker="o",
            markersize=5,
            capsize=3,
            linewidth=1.8,
            label=label,
        )

    axes.set_xscale("log", base=2)
    axes.set_xlabel("candidate pool size (number of dwarves, log scale)")
    axes.set_ylabel("top-1 accuracy (%)")
    axes.set_title("Identification accuracy against candidate-pool size")
    axes.grid(True, which="both", axis="y", alpha=0.25, linewidth=0.6)
    axes.legend(frameon=False, loc="lower left")
    axes.spines["top"].set_visible(False)
    axes.spines["right"].set_visible(False)
    all_sizes = sorted({size for curve in curves for size in curve.pool_sizes})
    axes.set_xticks(all_sizes)
    axes.set_xticklabels([str(size) for size in all_sizes])
    figure.tight_layout()

    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        figure.savefig(path, dpi=200)
    except OSError as error:
        raise VisualizationError(f"could not write figure {path}: {error}") from error
    finally:
        pyplot.close(figure)
    return path


def create_ablation_plot(config: AppConfig) -> Path:
    """Draw every saved ablation curve into one comparable figure."""
    if not isinstance(config.experiment, VisualizationExperimentConfig):
        raise VisualizationError(
            f"visualization requires experiment=visualization, got {config.experiment.kind}"
        )

    paths = find_ablation_results(config.paths.results_dir)
    if not paths:
        raise VisualizationError(
            f"no pool_size_ablation results in {config.paths.results_dir}; "
            "run krasnal-id experiment pool-ablation first"
        )
    curves = tuple(curve_from_result(read_ablation_result(path)) for path in paths)
    return render_curves(curves, config.paths.results_dir / "pool-size-ablation.png")

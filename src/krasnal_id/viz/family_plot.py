"""The pretraining-family figure (AGENTS.md section 7.17): one forest plot.

Thirteen backbones, grouped by pretraining family, one row each, and one panel per
measurement: closed-set accuracy, how fast it decays with pool size, what location
narrowing costs, what the photographer is worth, and how well it rejects. Every
interval is the statue-level bootstrap the validation artifact carries.

Rows are labelled directly, so colour carries no identity here. The four backbones
the rest of the paper draws keep their own colours, because colour follows the
entity (``figure_style``); the nine that appear only in this figure wear ink.
"""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

from krasnal_id.viz import figure_style as style
from krasnal_id.viz.embedding_plot import VisualizationError, import_optional_analysis

# Families in the order the paper discusses them; models within a family by date.
FAMILIES: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("Joint-embedding", ("dino", "dinov2", "dinov3")),
    ("JEPA", ("ijepa",)),
    ("Masked image modelling", ("mae", "beit", "eva02-mim")),
    ("Vision-language", ("clip", "clip-b16", "siglip2")),
    ("Agglomerative", ("radio",)),
    ("Autoregressive", ("aimv2",)),
    ("Supervised", ("vit-supervised",)),
)
MODEL_LABEL = {
    "dino": "DINO B/16",
    "dinov2": "DINOv2 B/14",
    "dinov3": "DINOv3 B/16",
    "ijepa": "I-JEPA H/14",
    "mae": "MAE B/16",
    "beit": "BEiT B/16",
    "eva02-mim": "EVA-02 B/14",
    "clip": "CLIP B/32",
    "clip-b16": "CLIP B/16",
    "siglip2": "SigLIP 2 B/16",
    "radio": "C-RADIOv3 B/16",
    "aimv2": "AIMv2 L/14",
    "vit-supervised": "ViT B/16 (IN-21k)",
}


# What the roster table states about each checkpoint beside its pinned identity:
# the pooling rule applied (section 7.17), the input size its released processor
# produces, the embedding width (read off the extracted vectors), and the licence
# its Hub card declares. Kept here, beside the labels, so the table and the figure
# cannot disagree about which model is which.
MODEL_DETAILS: dict[str, tuple[str, str, int, str]] = {
    "dino": ("CLS", "224", 768, "Apache-2.0"),
    "dinov2": ("CLS", "224", 768, "Apache-2.0"),
    "dinov3": ("CLS", "224", 768, "DINOv3 Licence"),
    "ijepa": ("patch mean", "224", 1280, "CC-BY-NC-4.0"),
    "mae": ("patch mean", "224", 768, "CC-BY-NC-4.0"),
    "beit": ("pooler (mean)", "224", 768, "Apache-2.0"),
    "eva02-mim": ("patch mean", "224", 768, "MIT"),
    "clip": ("projection", "224", 512, "MIT"),
    "clip-b16": ("projection", "224", 512, "MIT"),
    "siglip2": ("attention pool", "224", 768, "Apache-2.0"),
    "radio": ("summary", "432", 2304, "NVIDIA Open Model"),
    "aimv2": ("patch mean", "224", 1024, "Apple ASCL"),
    "vit-supervised": ("CLS", "224", 768, "Apache-2.0"),
}


@dataclass(frozen=True, slots=True)
class Panel:
    """One measurement: the metric suffix, its axis label, and how to scale it."""

    suffix: str
    label: str
    scale: float = 100.0
    zero_line: bool = False


PANELS: tuple[Panel, ...] = (
    Panel("top_1", "Top-1\n(%)"),
    Panel("points_per_doubling", "Change per\ndoubling (points)", scale=1.0, zero_line=True),
    Panel("geo_minus_random_pool_10", "Location \u2212 random,\npool 10 (points)", zero_line=True),
    Panel("photographer_attributable", "Photographer\neffect (points)", zero_line=True),
    Panel("reject_margin_auroc", "Rejection AUROC\n(margin score)", scale=1.0),
)


@dataclass(frozen=True, slots=True)
class Estimate:
    """A point estimate and its interval, already scaled for the axis."""

    value: float
    low: float | None
    high: float | None


def collect(
    metrics: Mapping[str, Mapping[str, float | None]],
    models: Sequence[str],
    panels: Sequence[Panel] = PANELS,
) -> dict[tuple[str, str], Estimate]:
    """Every (model, panel) estimate, refusing a missing one rather than leaving a gap."""
    found: dict[tuple[str, str], Estimate] = {}
    for model in models:
        for panel in panels:
            name = f"{model}_{panel.suffix}"
            record = metrics.get(name)
            if record is None or record.get("value") is None:
                raise VisualizationError(f"the validation artifact has no {name}")
            value = float(record["value"]) * panel.scale  # type: ignore[arg-type]
            low, high = record.get("lower_bound"), record.get("upper_bound")
            found[(model, panel.suffix)] = Estimate(
                value,
                None if low is None else float(low) * panel.scale,
                None if high is None else float(high) * panel.scale,
            )
    return found


def rows() -> list[tuple[str | None, str | None]]:
    """The plotted rows top to bottom: a family heading, then its models."""
    laid_out: list[tuple[str | None, str | None]] = []
    for family, models in FAMILIES:
        laid_out.append((family, None))
        laid_out.extend((None, model) for model in models)
    return laid_out


def render_family_figure(
    metrics: Mapping[str, Mapping[str, float | None]],
    path: Path,
    width: float,
    dpi: int = style.PUBLICATION_DPI,
) -> Path:
    """Draw the forest plot at the given print width."""
    matplotlib = import_optional_analysis("matplotlib")
    matplotlib.use("Agg")
    plt = import_optional_analysis("matplotlib.pyplot")
    ticker = import_optional_analysis("matplotlib.ticker")
    style.apply(plt)
    laid_out = rows()
    models = [model for _, model in laid_out if model is not None]
    estimates = collect(metrics, models)
    height = 0.115 * len(laid_out) + 0.45
    label_share = 0.17
    figure = plt.figure(figsize=(width, height), layout="constrained")
    grid = figure.add_gridspec(
        1,
        len(PANELS) + 1,
        width_ratios=[label_share * len(PANELS) / (1 - label_share)] + [1.0] * len(PANELS),
    )
    labels = figure.add_subplot(grid[0, 0])
    y = list(range(len(laid_out)))[::-1]
    labels.set_ylim(-0.6, len(laid_out) - 0.4)
    labels.set_xlim(0, 1)
    labels.set_axis_off()
    for row, (family, model) in zip(y, laid_out, strict=True):
        if family is not None:
            labels.text(
                0,
                row,
                family,
                fontsize=style.SIZE_LABEL,
                fontweight="bold",
                color=style.INK,
                va="center",
                ha="left",
            )
        else:
            assert model is not None
            labels.text(
                0.06,
                row,
                MODEL_LABEL[model],
                fontsize=style.SIZE_TICK,
                color=style.INK,
                va="center",
                ha="left",
            )
    for column, panel in enumerate(PANELS, start=1):
        axes = figure.add_subplot(grid[0, column], sharey=labels)
        axes.set_ylim(-0.6, len(laid_out) - 0.4)
        axes.tick_params(left=False, labelleft=False)
        for side in ("left", "right", "top"):
            axes.spines[side].set_visible(False)
        axes.grid(False)
        axes.grid(axis="x", color=style.GRID, linewidth=style.AXIS_LINE_WIDTH)
        axes.set_axisbelow(True)
        if panel.zero_line:
            axes.axvline(0, color=style.INK_MUTED, linewidth=style.REFERENCE_LINE_WIDTH)
        for row, (_, model) in zip(y, laid_out, strict=True):
            if model is None:
                continue
            estimate = estimates[(model, panel.suffix)]
            colour = style.BACKBONE_COLOUR.get(model, style.INK)
            marker = style.BACKBONE_MARKER.get(model, "o")
            if estimate.low is not None and estimate.high is not None:
                axes.plot(
                    [estimate.low, estimate.high],
                    [row, row],
                    color=colour,
                    linewidth=style.LINE_WIDTH * 0.8,
                    solid_capstyle="butt",
                )
            axes.plot(
                estimate.value,
                row,
                marker=marker,
                color=colour,
                markersize=style.MARKER_SIZE,
                markeredgecolor="white",
                markeredgewidth=0.6,
                linestyle="none",
            )
        axes.set_xlabel(panel.label, fontsize=style.SIZE_LABEL, color=style.INK_SECONDARY)
        axes.xaxis.set_major_formatter(ticker.FuncFormatter(lambda v, _: style.minus(f"{v:g}")))
        style.panel_tag(axes, "abcdefgh"[column - 1])
    written = style.save(figure, path, dpi)
    plt.close(figure)
    return written


__all__ = [
    "FAMILIES",
    "MODEL_DETAILS",
    "MODEL_LABEL",
    "PANELS",
    "collect",
    "render_family_figure",
    "rows",
]

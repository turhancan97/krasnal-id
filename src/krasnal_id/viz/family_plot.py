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


POOL_SIZES = (2, 3, 5, 8, 10, 15, 20, 50, 100, 200, 306)
# A tick at every measured size would collide at this width; the curves keep them all.
_TICKS = (2, 5, 10, 20, 50, 100, 306)
GEO_POOLS = (2, 3, 5, 8, 10, 15, 20, 50, 100, 200)
# Inside a paradigm the members that have no colour of their own are told apart by
# line style and marker, in ink, so the four colours stay with their backbones.
_UNCOLOURED_STYLES = (("-", "o"), ("--", "s"), (":", "^"))


def _member_styles(models: tuple[str, ...]) -> dict[str, tuple[str, str, str]]:
    """Colour, line style and marker for each model in one paradigm's panels."""
    styles: dict[str, tuple[str, str, str]] = {}
    spare = iter(_UNCOLOURED_STYLES)
    for model in models:
        if model in style.BACKBONE_COLOUR:
            styles[model] = (style.BACKBONE_COLOUR[model], "-", style.BACKBONE_MARKER[model])
        else:
            line, marker = next(spare)
            styles[model] = (style.INK, line, marker)
    return styles


def render_family_curves(
    root: Path,
    validation: Mapping[str, Mapping[str, float | None]],
    path: Path,
    width: float,
    dpi: int = style.PUBLICATION_DPI,
) -> Path:
    """The full pool-size and location curves of all thirteen backbones, by paradigm.

    One row per paradigm, so each panel holds at most three models and keeps its own
    y-range: on one shared axis the reconstruction models' 50% would flatten the
    differences among the 90%+ ones. Left, top-1 against pool size with the spread
    across seeds; right, location minus random with statue intervals.
    """
    from krasnal_id.viz.workshop_tables import load_metrics

    matplotlib = import_optional_analysis("matplotlib")
    matplotlib.use("Agg")
    plt = import_optional_analysis("matplotlib.pyplot")
    style.apply(plt)
    rows = len(FAMILIES)
    figure, grid = plt.subplots(
        rows,
        2,
        figsize=(width, 0.84 * rows + 0.5),
        layout="constrained",
        sharex="col",
        squeeze=False,
    )
    for row, (family, models) in enumerate(FAMILIES):
        pool_axes, geo_axes = grid[row]
        styles = _member_styles(models)
        for model in models:
            colour, line, marker = styles[model]
            pool = load_metrics(root / model / f"pool_size_ablation-{model}.json")
            xs = [n for n in POOL_SIZES if f"top_1_pool_{n}" in pool]
            if not xs:
                raise VisualizationError(f"pool_size_ablation-{model} has no pool sizes")
            values = [100 * (pool[f"top_1_pool_{n}"]["value"] or 0.0) for n in xs]
            low = [100 * (pool[f"top_1_pool_{n}"]["lower"] or 0.0) for n in xs]
            high = [100 * (pool[f"top_1_pool_{n}"]["upper"] or 0.0) for n in xs]
            pool_axes.fill_between(xs, low, high, color=colour, alpha=0.14, linewidth=0)
            pool_axes.plot(
                xs,
                values,
                color=colour,
                linestyle=line,
                marker=marker,
                markersize=style.MARKER_SIZE * 0.7,
                linewidth=style.LINE_WIDTH * 0.8,
                label=MODEL_LABEL[model],
            )
            names = [f"{model}_geo_minus_random_pool_{n}" for n in GEO_POOLS]
            present = [
                (n, validation[k]) for n, k in zip(GEO_POOLS, names, strict=True) if k in validation
            ]
            if not present:
                raise VisualizationError(f"the family validation has no geography for {model}")
            gx = [n for n, _ in present]
            geo_axes.fill_between(
                gx,
                [100 * (m["lower_bound"] or 0.0) for _, m in present],
                [100 * (m["upper_bound"] or 0.0) for _, m in present],
                color=colour,
                alpha=0.14,
                linewidth=0,
            )
            geo_axes.plot(
                gx,
                [100 * (m["value"] or 0.0) for _, m in present],
                color=colour,
                linestyle=line,
                marker=marker,
                markersize=style.MARKER_SIZE * 0.7,
                linewidth=style.LINE_WIDTH * 0.8,
            )
        geo_axes.axhline(0, color=style.INK_MUTED, linewidth=style.REFERENCE_LINE_WIDTH)
        # A legend needs empty space, and in a panel of falling curves every corner
        # is crossed by one. Rows with several models get headroom below the curves
        # instead, and the legend sits in it.
        if len(models) > 1:
            low, high = pool_axes.get_ylim()
            pool_axes.set_ylim(low - 0.55 * (high - low), high)
        pool_axes.legend(
            loc="lower left" if len(models) > 1 else "best",
            fontsize=style.SIZE_ANNOTATION,
            frameon=False,
            handlelength=1.8,
            borderaxespad=0.2,
            ncol=len(models) if len(models) > 1 else 1,
        )
        pool_axes.set_ylabel(f"{family}\nTop-1 (%)", fontsize=style.SIZE_ANNOTATION)
        geo_axes.set_ylabel("Location \u2212\nrandom (pts)", fontsize=style.SIZE_ANNOTATION)
        for axes in (pool_axes, geo_axes):
            axes.set_xscale("log")
            axes.minorticks_off()
            axes.tick_params(labelsize=style.SIZE_ANNOTATION)
            axes.yaxis.set_major_formatter(
                import_optional_analysis("matplotlib.ticker").FuncFormatter(
                    lambda v, _: style.minus(f"{v:g}")
                )
            )
    grid[-1][0].set_xticks(_TICKS, [str(n) for n in _TICKS])
    grid[-1][1].set_xticks((*_TICKS[:-1], 200), [str(n) for n in (*_TICKS[:-1], 200)])
    grid[-1][0].set_xlabel("Candidate statues in the pool")
    grid[-1][1].set_xlabel("Candidate pool size")
    written = style.save(figure, path, dpi)
    plt.close(figure)
    return written


__all__ = [
    "FAMILIES",
    "MODEL_DETAILS",
    "MODEL_LABEL",
    "PANELS",
    "collect",
    "render_family_curves",
    "render_family_figure",
    "rows",
]

"""Publication figures for the manuscript, built from the result artifacts.

These exist because the venue allows eight display items and `RESULTS.md` holds
twenty tables. Turning tables into figures is what makes the material fit, so
each function here consolidates several source tables into one panel set.

The code is tracked even though the manuscript is not. A figure that cannot be
regenerated from a committed command is a figure nobody can check, and §6.4's
rule -- every published number traceable to a `results/*.json` artifact -- applies
to the paper at least as strongly as it applies to the repository.

Every figure is drawn by `krasnal_id.viz.figure_style`: one palette in which colour
follows the entity, one sans face (TeX Gyre Heros, vendored), one type scale, and
widths that match the page it is printed on, so no figure is ever scaled and every
7.5 pt label prints at 7.5 pt. A `Layout` picks those widths and which backbones a
figure shows; the journal draft keeps its own, unchanged.
"""

import json
import textwrap
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from krasnal_id.models import DatasetManifest
from krasnal_id.viz import figure_style as style
from krasnal_id.viz.figure_style import (
    BACKBONE_COLOUR,
    BACKBONE_LABEL,
    BACKBONE_MARKER,
    JOURNAL,
    PUBLICATION_DPI,
    Layout,
)

__all__ = ["BACKBONE_COLOUR", "BACKBONE_LABEL", "PUBLICATION_DPI"]

SEPARATION_BANDS = (50, 100, 200, 300, 500, 1000)


class PaperFigureError(ValueError):
    """Raised when an artifact a figure needs is missing or lacks a metric."""


@dataclass(frozen=True, slots=True)
class Artifact:
    """One experiment's metrics, addressed by name."""

    name: str
    values: dict[str, float]
    # Confidence bounds, where the metric carries them. The pool-size ablation's
    # spread across seeds lives here rather than in separate metric names, so a
    # loader that kept only values would silently draw a curve with no band.
    bounds: dict[str, tuple[float, float]]

    def get(self, key: str) -> float:
        """Return one metric, naming the artifact if it is absent."""
        if key not in self.values:
            raise PaperFigureError(f"{self.name} has no metric {key!r}")
        return self.values[key]

    def maybe(self, key: str) -> float | None:
        """Return one metric, or None where its absence is meaningful."""
        return self.values.get(key)

    def interval(self, key: str) -> tuple[float, float] | None:
        """Return one metric's confidence bounds, if it records them."""
        return self.bounds.get(key)


def read_artifact(results_dir: Path, stem: str) -> dict[str, Any]:
    """Read one result artifact whole, for the records metrics cannot hold."""
    path = results_dir / f"{stem}.json"
    if not path.is_file():
        raise PaperFigureError(f"missing artifact: {path}")
    return dict(json.loads(path.read_text(encoding="utf-8")))


def load(results_dir: Path, stem: str) -> Artifact:
    """Read one result artifact's metrics."""
    metrics = read_artifact(results_dir, stem)["metrics"]
    return Artifact(
        stem,
        {m["name"]: float(m["value"]) for m in metrics},
        {
            m["name"]: (float(m["lower_bound"]), float(m["upper_bound"]))
            for m in metrics
            if m.get("lower_bound") is not None and m.get("upper_bound") is not None
        },
    )


@dataclass(frozen=True, slots=True)
class Sources:
    """Where a figure's evidence comes from.

    Most figures read nothing but `results/*.json`. Figure 1 describes the corpus
    itself, so it also needs the manifest and the photographs the manifest points
    at -- which is why a figure is handed a source set rather than a directory.
    """

    results_dir: Path
    manifest: Path
    # Backbones whose artifacts live in `results_dir/<name>/` rather than in
    # `results_dir/`, so that no figure glob over `results/` picks them up (§7.12).
    separate: tuple[str, ...] = ("dinov3", "siglip2")
    # The workshop validation artifact, for figures that show a section 7.15 result.
    # Named explicitly rather than found by a glob, because several revisions sit
    # side by side and a figure must not quietly draw a superseded one.
    validation: Path | None = None

    def validation_metrics(self) -> dict[str, float]:
        """The validation artifact's metrics by name, or an error naming the flag."""
        if self.validation is None or not self.validation.is_file():
            raise PaperFigureError(
                f"the short-paper figures need the validation artifact; pass --validation "
                f"(got {self.validation})"
            )
        payload = json.loads(self.validation.read_text(encoding="utf-8"))
        return {str(m["name"]): float(m["value"]) for m in payload["metrics"]}

    def _directory(self, stem: str) -> Path:
        for name in self.separate:
            if stem.endswith(f"-{name}"):
                return self.results_dir / name
        return self.results_dir

    def has(self, stem: str) -> bool:
        """Whether one artifact exists, for figures that draw only what was measured."""
        return (self._directory(stem) / f"{stem}.json").is_file()

    def load(self, stem: str) -> Artifact:
        """Read one result artifact's metrics."""
        return load(self._directory(stem), stem)

    def records(self, stem: str, key: str) -> list[dict[str, Any]]:
        """Read one artifact's per-item records, such as the confused pairs."""
        payload = read_artifact(self._directory(stem), stem)
        if key not in payload:
            raise PaperFigureError(f"{stem} has no {key!r} records")
        return list(payload[key])

    def dataset(self) -> "DatasetManifest":
        """Read the manifest, validated against the project's own schema."""
        if not self.manifest.is_file():
            raise PaperFigureError(f"missing manifest: {self.manifest}")
        return DatasetManifest.model_validate(json.loads(self.manifest.read_text(encoding="utf-8")))


def _pyplot() -> Any:
    """Import matplotlib with the shared publication style, only when a figure is drawn."""
    try:
        import matplotlib
    except ImportError as error:  # pragma: no cover - exercised by the extra's absence
        raise PaperFigureError(
            "drawing needs the analysis extra; run uv sync --extra analysis"
        ) from error
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    style.apply(plt)
    return plt


def _save(figure: Any, path: Path) -> Path:
    """Write one figure at publication density, at the size it was drawn."""
    return style.save(figure, path)


def _measured(sources: Sources, layout: Layout, stem: str) -> tuple[str, ...]:
    """The layout's backbones that have the artifact `stem` names, in layout order.

    A figure draws what was measured and nothing else: DINOv3 has no confusion or
    camera-origin artifact, so it is simply absent from those panels rather than
    an error.
    """
    present = tuple(b for b in layout.backbones if sources.has(stem.format(backbone=b)))
    if not present:
        raise PaperFigureError(f"no backbone in the {layout.name} layout has {stem}")
    return present


def _line(axes: Any, x: Any, y: Any, backbone: str, label: str | None = None) -> None:
    """One backbone's series, in its colour and with its marker."""
    axes.plot(
        x,
        y,
        marker=BACKBONE_MARKER[backbone],
        color=BACKBONE_COLOUR[backbone],
        label=label or BACKBONE_LABEL[backbone],
        markeredgecolor="white",
        markeredgewidth=0.5,
    )


def _reference(axes: Any, value: float, horizontal: bool = True) -> None:
    """A recessive dashed reference line: zero, parity, chance."""
    draw = axes.axhline if horizontal else axes.axvline
    draw(
        value,
        color=style.INK_MUTED,
        linewidth=style.REFERENCE_LINE_WIDTH,
        linestyle=(0, (3, 2)),
        zorder=1,
    )


def _grouped(count: int, total: float = 0.78) -> tuple[float, list[float]]:
    """Bar width and centred offsets for `count` bars per group."""
    width = total / count
    return width, [(index - (count - 1) / 2) * width for index in range(count)]


def _bar_labels(axes: Any, bars: Any, fmt: str) -> None:
    axes.bar_label(
        bars, fmt=fmt, padding=1.5, fontsize=style.SIZE_ANNOTATION, color=style.INK_SECONDARY
    )


# Panel (c) bins classes by how many photographs they have. The bins are uneven
# on purpose: more than half the corpus sits at three, four or five images, and
# equal-width bins would hide that under one tall bar.
IMAGE_COUNT_BINS = ((3, 3), (4, 4), (5, 5), (6, 7), (8, 11), (12, 1000))

# How many of the most-confused pairs figure 1(a) shows. Three fits the column at
# a size where two near-identical dwarves are still tellable apart.
CONFUSED_PAIRS_SHOWN = 3

# Pool sizes the ablation evaluated. 306 is the whole corpus, so its band has no
# width -- there is only one way to draw every dwarf.
POOL_SIZES = (2, 3, 5, 8, 10, 15, 20, 50, 100, 200, 306)

# The two radii figure 1(b) draws, and the pool size each one reaches. They are
# read from the ablation rather than written here; these name which to read.
INSET_POOLS = (5, 10)


def _square_thumbnail(path: Path, pixels: int) -> Any:
    """Centre-crop one photograph to a square, so a row shares one scale.

    Two dwarves photographed at different aspect ratios look different for a
    reason that has nothing to do with the statues. Cropping to a common square
    is what makes "these are near identical" a claim the reader can check.
    """
    try:
        image_module = __import__("PIL.Image", fromlist=["Image"])
    except ImportError as error:  # pragma: no cover - exercised by the extra's absence
        raise PaperFigureError(
            "drawing needs the analysis extra; run uv sync --extra analysis"
        ) from error
    if not path.is_file():
        raise PaperFigureError(f"missing photograph: {path}")
    with image_module.open(path) as handle:
        frame = handle.convert("RGB")
        side = min(frame.size)
        left = (frame.width - side) // 2
        top = (frame.height - side) // 2
        square = frame.crop((left, top, left + side, top + side))
        return np.asarray(square.resize((pixels, pixels)))


def _local_metres(
    points: Any,
    origin: tuple[float, float],
) -> tuple[Any, Any]:
    """Project latitude and longitude to metres east and north of one origin.

    An equirectangular projection about the corpus's own centre. Over a city the
    distortion is well under the precision of a Commons coordinate, and it keeps
    the panel's axes in the same unit as the pool radii drawn on it.
    """
    latitude, longitude = origin
    metres_per_degree = 111_320.0
    east = (points[:, 1] - longitude) * metres_per_degree * np.cos(np.radians(latitude))
    north = (points[:, 0] - latitude) * metres_per_degree
    return east, north


def dataset_figure(sources: Sources, output: Path, layout: Layout = JOURNAL) -> Path:
    """Fig. 1 -- what the corpus is, and the three things that make it hard.

    The three panels are the paper's three standing difficulties in the order the
    Results meet them: statues that were sculpted alike, statues that stand close
    together, and classes whose photographs mostly come from one person. None of
    them is a backbone, so nothing here wears a backbone's colour.
    """
    plt = _pyplot()
    figure = plt.figure(figsize=(layout.full, layout.full * 0.47), layout="constrained")
    panels = figure.subfigures(1, 3, width_ratios=(1.2, 1.2, 1.0), wspace=0.03)
    manifest = sources.dataset()

    arrows = _confused_pairs_panel(panels[0], sources, manifest)
    _placement_panel(panels[1], sources, manifest)
    _photographer_panel(panels[2], manifest)

    # The arrows' labels are placed once the layout is final: the gap between a
    # pair's photographs is only known after constrained layout has sized them.
    figure.canvas.draw()
    to_figure = figure.transFigure.inverted()
    for left, right, above, below in arrows:
        # Through display coordinates: an axes in a subfigure reports its position
        # relative to that subfigure, not to the figure the text is placed on.
        box_l = left.get_window_extent().transformed(to_figure)
        box_r = right.get_window_extent().transformed(to_figure)
        x = (box_l.x1 + box_r.x0) / 2
        y = (box_l.y0 + box_l.y1) / 2
        for text, offset, colour in (
            (above, 0.022, style.INK),
            (below, -0.022, style.INK_SECONDARY),
        ):
            figure.text(
                x,
                y + offset,
                text,
                ha="center",
                va="bottom" if offset > 0 else "top",
                fontsize=style.SIZE_ANNOTATION,
                color=colour,
                in_layout=False,
            )

    written = _save(figure, output)
    plt.close(figure)
    return written


def _subfigure_tag(panel: Any, letter: str) -> None:
    """The bold (a)-style tag for a panel built as a subfigure.

    Set as the subfigure's suptitle, so constrained layout reserves room for it
    instead of letting it overprint the first photograph.
    """
    panel.suptitle(
        f"({letter})",
        x=0.0,
        ha="left",
        fontsize=style.SIZE_PANEL,
        fontweight="bold",
        color=style.INK,
    )


def group_links() -> set[frozenset[str]]:
    """Every (group class, own member) pair, from the packaged group definitions.

    A group statue confused with one of its own members is a labelling overlap,
    not two lookalike statues, so panel (a) never shows one.
    """
    from krasnal_id.config import WorkshopValidationConfig, load_config

    settings = load_config(["experiment=workshop_validation"]).experiment
    assert isinstance(settings, WorkshopValidationConfig)
    return {
        frozenset((group, member))
        for group, members in settings.groups.items()
        for member in members
    }


def shown_confusions(
    pairs: Sequence[Mapping[str, Any]], groups: set[frozenset[str]]
) -> list[Mapping[str, Any]]:
    """The most frequent confusions, as recorded, that panel (a) draws.

    In the artifact's order (most misidentifications first), skipping a group
    and its own member, and any pair naming a statue already shown, so each
    photograph appears once and every pair is two different sculptures.
    """
    shown: list[Mapping[str, Any]] = []
    seen: set[str] = set()
    for pair in pairs:
        ids = {str(pair["true_dwarf_id"]), str(pair["confused_dwarf_id"])}
        if frozenset(ids) in groups or ids & seen:
            continue
        shown.append(pair)
        seen |= ids
        if len(shown) == CONFUSED_PAIRS_SHOWN:
            return shown
    raise PaperFigureError(f"fewer than {CONFUSED_PAIRS_SHOWN} distinct confusions to show")


def _confused_pairs_panel(
    panel: Any, sources: Sources, manifest: DatasetManifest
) -> list[tuple[Any, Any, str, str]]:
    """Panel (a) -- the pairs the retrieval actually confuses, photographed.

    Returns each pair's two axes and the arrow's labels, which are placed after
    the layout is final.
    """
    names = {dwarf.dwarf_id: dwarf.display_name for dwarf in manifest.dwarfs}
    # The class's widest-framed photograph, ties broken by image ID: landscape
    # first, then nearest to square. A centre crop of a portrait frame returns
    # the plinth or the pillar a dwarf sits on, which would make two statues look
    # different for a reason that is about the photograph and not the sculpture.
    chosen: dict[str, Path] = {}
    for image in sorted(
        manifest.images,
        key=lambda record: (
            record.width < record.height,
            max(record.width, record.height) / min(record.width, record.height),
            record.image_id,
        ),
    ):
        chosen.setdefault(image.dwarf_id, Path(image.local_path))

    from matplotlib.patches import ConnectionPatch

    pairs = shown_confusions(sources.records("confusion-dinov2", "pairs"), group_links())
    axes = panel.subplots(CONFUSED_PAIRS_SHOWN, 2, gridspec_kw={"wspace": 0.12, "hspace": 0.22})
    arrows: list[tuple[Any, Any, str, str]] = []
    # The denominator is every query of the left statue: leave-one-out queries each
    # of its photographs once. The artifact's own `queries` counts only the queries
    # whose strongest rival was the right statue, which reads as an impossible
    # "2 of 2" for a statue with seven photographs.
    photographs: dict[str, int] = {}
    for image in manifest.images:
        photographs[image.dwarf_id] = photographs.get(image.dwarf_id, 0) + 1
    for row, pair in zip(axes, pairs, strict=True):
        separation = float(pair["separation_metres"])
        total = photographs[str(pair["true_dwarf_id"])]
        arrows.append(
            (
                row[0],
                row[1],
                f"{pair['misidentifications']} of {total} queries",
                "same plinth" if separation < 1.0 else f"{separation:.0f} m apart",
            )
        )
        ids = (str(pair["true_dwarf_id"]), str(pair["confused_dwarf_id"]))
        for cell, dwarf_id in zip(row, ids, strict=True):
            if dwarf_id not in chosen:
                raise PaperFigureError(f"confusion names an unknown dwarf: {dwarf_id}")
            cell.imshow(_square_thumbnail(chosen[dwarf_id], 360))
            cell.set_xticks([])
            cell.set_yticks([])
            cell.grid(False)
            # A hairline frame, so a pale sky or wall does not dissolve into the page.
            for spine in cell.spines.values():
                spine.set_visible(True)
                spine.set_edgecolor(style.NEUTRAL_FILL)
                spine.set_linewidth(0.6)
            name = names.get(dwarf_id, dwarf_id)
            # Wrapped rather than trimmed: the pairs differ in exactly the words a
            # truncation drops (Solne, Oławskie), and one line overflows a thumbnail.
            cell.set_xlabel(
                "\n".join(textwrap.wrap(name, 13)),
                fontsize=style.SIZE_ANNOTATION,
                color=style.INK_SECONDARY,
                labelpad=1.5,
            )
        # The arrow reads in the direction the errors run: queries of the left
        # statue are returned as the right one.
        # Anchored to both photographs, so it spans the gap whatever width
        # the layout gives it.
        panel.add_artist(
            ConnectionPatch(
                xyA=(1.06, 0.5),
                coordsA="axes fraction",
                axesA=row[0],
                xyB=(-0.06, 0.5),
                coordsB="axes fraction",
                axesB=row[1],
                arrowstyle="-|>,head_length=0.4,head_width=0.22",
                color=style.CORPUS_ACCENT,
                linewidth=1.0,
                shrinkA=0,
                shrinkB=0,
                # Out of the layout, or constrained layout shrinks the photographs
                # to make room for an arrow that only spans the gap between them.
                in_layout=False,
            )
        )
    _subfigure_tag(panel, "a")
    return arrows


def _placement_panel(panel: Any, sources: Sources, manifest: DatasetManifest) -> None:
    """Panel (b) -- where the statues stand, and what a pool radius covers."""
    located = [
        (dwarf, dwarf.coordinates) for dwarf in manifest.dwarfs if dwarf.coordinates is not None
    ]
    placed = np.array(
        [(point.latitude, point.longitude) for _, point in located],
        dtype=np.float64,
    )
    # Where a position came from is the one property of a point the reader should
    # see: Wikidata's are authoritative, the rest are the median of the camera
    # positions in a statue's own photographs (section 3).
    from_wikidata = np.array([dwarf.coordinate_source == "wikidata" for dwarf, _ in located])
    origin = (float(np.median(placed[:, 0])), float(np.median(placed[:, 1])))
    east, north = _local_metres(placed, origin)

    axes = panel.subplots()
    axes.grid(False)

    def draw(target: Any, size: float) -> None:
        target.scatter(
            east[~from_wikidata] / 1000,
            north[~from_wikidata] / 1000,
            s=size,
            color=style.CORPUS_MUTED,
            alpha=0.75,
            linewidths=0,
        )
        target.scatter(
            east[from_wikidata] / 1000,
            north[from_wikidata] / 1000,
            s=size * 2.2,
            color=style.CORPUS_ACCENT,
            edgecolors="white",
            linewidths=0.4,
            zorder=3,
        )

    draw(axes, 4.0)
    from matplotlib.lines import Line2D

    axes.legend(
        handles=[
            Line2D(
                [],
                [],
                marker="o",
                linestyle="none",
                markersize=3.6,
                color=style.CORPUS_ACCENT,
                label=f"Wikidata ({int(from_wikidata.sum())})",
            ),
            Line2D(
                [],
                [],
                marker="o",
                linestyle="none",
                markersize=2.6,
                color=style.CORPUS_MUTED,
                label=f"photo positions ({int((~from_wikidata).sum())})",
            ),
        ],
        loc="lower right",
        fontsize=style.SIZE_ANNOTATION,
        frameon=False,
        handletextpad=0.2,
        borderaxespad=0.1,
    )
    axes.set_aspect("equal")
    axes.set_xlabel("km east of the corpus centre")
    axes.set_ylabel("km north")

    # The densest 300 m neighbourhood, found rather than chosen, is where the
    # radii mean the most: this is the old town, where the statues crowd.
    separations = np.hypot(east[:, None] - east[None, :], north[:, None] - north[None, :])
    centre = int(np.argmax((separations <= 300.0).sum(axis=1)))

    art = sources.load("geo_ablation-dinov2")
    radii = [(pool, art.get(f"geo_radius_metres_pool_{pool}")) for pool in INSET_POOLS]
    span = max(radius for _, radius in radii) * 1.9
    inset = axes.inset_axes(
        (0.0, 0.0, 0.46, 0.46),
        xlim=((east[centre] - span) / 1000, (east[centre] + span) / 1000),
        ylim=((north[centre] - span) / 1000, (north[centre] + span) / 1000),
        xticks=[],
        yticks=[],
    )
    # An inset that shares the panel's frameless style reads as a second cloud of
    # statues rather than as a magnification of the first.
    inset.set_facecolor("white")
    inset.grid(False)
    for spine in inset.spines.values():
        spine.set_visible(True)
        spine.set_edgecolor(style.INK_SECONDARY)
        spine.set_linewidth(0.6)
    draw(inset, 7.0)
    inset.set_aspect("equal")
    for _, radius in radii:
        inset.add_patch(
            plt_circle(east[centre] / 1000, north[centre] / 1000, radius / 1000, style.INK)
        )
    # One caption for both circles. Labelling each at its own arc puts two lines
    # of text inside a 780 m box, where they collide.
    # Set in the map's empty top-left corner, not in the inset, where it
    # overprints the circles.
    axes.text(
        0.02,
        0.98,
        "\n".join(f"{radius:.0f} m \N{RIGHTWARDS ARROW} {pool} statues" for pool, radius in radii),
        transform=axes.transAxes,
        va="top",
        fontsize=style.SIZE_ANNOTATION,
        color=style.INK,
        bbox={"facecolor": "white", "edgecolor": "none", "pad": 0.8, "alpha": 0.85},
    )
    axes.indicate_inset_zoom(inset, edgecolor=style.INK_MUTED, linewidth=0.6, alpha=0.9)
    _subfigure_tag(panel, "b")


def plt_circle(x: float, y: float, radius: float, colour: str) -> Any:
    """One unfilled circle, for the pool radii drawn on the inset."""
    from matplotlib.patches import Circle

    # Above the scatter: the inner radius is small enough that the statues it
    # contains would otherwise draw over it.
    return Circle(
        (x, y),
        radius,
        fill=False,
        edgecolor=colour,
        linewidth=0.8,
        linestyle=(0, (3, 2)),
        zorder=5,
    )


def _photographer_panel(panel: Any, manifest: DatasetManifest) -> None:
    """Panel (c) -- small classes, and how much of each is one person's work.

    One axis. The share was once a line on a second y-axis, which invites reading
    the two scales against each other; it is now the label on each bar.
    """
    per_class: dict[str, list[str]] = {}
    for image in manifest.images:
        per_class.setdefault(image.dwarf_id, []).append(image.author)

    counts, shares, labels = [], [], []
    for low, high in IMAGE_COUNT_BINS:
        members = [authors for authors in per_class.values() if low <= len(authors) <= high]
        if not members:
            raise PaperFigureError(f"no class holds {low}-{high} images")
        counts.append(len(members))
        shares.append(
            float(np.mean([max(a.count(x) for x in set(a)) / len(a) * 100 for a in members]))
        )
        if low == high:
            labels.append(str(low))
        elif high < 1000:
            # An en dash, which is what a printed range takes.
            labels.append(f"{low}\N{EN DASH}{high}")
        else:
            labels.append(f"{low}+")

    axes = panel.subplots()
    positions = np.arange(len(IMAGE_COUNT_BINS), dtype=float)
    # A white edge is the surface gap between neighbouring fills.
    bars = axes.bar(
        positions, counts, 0.72, color=style.CORPUS_FILL, edgecolor="white", linewidth=0.8
    )
    for bar, share in zip(bars, shares, strict=True):
        axes.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + max(counts) * 0.02,
            f"{share:.0f}%",
            ha="center",
            va="bottom",
            fontsize=style.SIZE_ANNOTATION,
            color=style.INK,
        )
    axes.set_ylim(0, max(counts) * 1.24)
    # Ticks stop at the tallest bar, so the note above them has clear space.
    axes.set_yticks([tick for tick in range(0, max(counts) + 1, 20)])
    axes.set_xticks(positions)
    axes.set_xticklabels(labels)
    axes.set_xlabel("Photographs of the statue")
    axes.set_ylabel("Statues")
    axes.text(
        0.98,
        0.97,
        "% = main photographer's share",
        transform=axes.transAxes,
        ha="right",
        va="top",
        fontsize=style.SIZE_ANNOTATION,
        color=style.INK_MUTED,
    )
    _subfigure_tag(panel, "c")


def pool_size_figure(sources: Sources, output: Path, layout: Layout = JOURNAL) -> Path:
    """Fig. 3 -- accuracy falls as the candidate pool grows.

    One curve per backbone, with its slope per doubling in the legend. The band is
    the spread across seeds, and it closes at 306 because there is only one way to
    draw the whole corpus.
    """
    plt = _pyplot()
    figure, axes = plt.subplots(figsize=(layout.column, layout.column * 0.72), layout="constrained")

    for backbone in _measured(sources, layout, "pool_size_ablation-{backbone}"):
        art = sources.load(f"pool_size_ablation-{backbone}")
        pools, values, lower, upper = [], [], [], []
        for pool in POOL_SIZES:
            value = art.maybe(f"top_1_pool_{pool}")
            if value is None:
                continue
            bounds = art.interval(f"top_1_pool_{pool}") or (value, value)
            pools.append(pool)
            values.append(value * 100)
            lower.append(bounds[0] * 100)
            upper.append(bounds[1] * 100)
        if not pools:
            raise PaperFigureError(f"pool_size_ablation-{backbone} evaluated no pool sizes")
        # The slope rides in the legend rather than beside its curve: the curves are
        # close over most of the axis, so a label against one would cross another.
        slope = style.minus(f"{art.get('top_1_points_per_doubling'):.2f}")
        axes.fill_between(
            pools, lower, upper, color=BACKBONE_COLOUR[backbone], alpha=0.16, linewidth=0
        )
        _line(axes, pools, values, backbone, f"{BACKBONE_LABEL[backbone]}  ({slope} / doubling)")

    axes.set_xscale("log")
    axes.set_xticks(POOL_SIZES)
    axes.set_xticklabels([str(pool) for pool in POOL_SIZES])
    axes.minorticks_off()
    axes.set_xlabel("Candidate statues in the pool")
    axes.set_ylabel("Top-1 accuracy (%)")
    axes.legend(loc="lower left")

    written = _save(figure, output)
    plt.close(figure)
    return written


def geography_figure(sources: Sources, output: Path, layout: Layout = JOURNAL) -> Path:
    """Fig. 4 -- location narrowing underperforms random narrowing, and why.

    Panel (a) is the penalty against pool size; panel (b) the co-location
    enrichment behind it, which has to decay with distance for the explanation to
    hold, and does.
    """
    plt = _pyplot()
    figure, (left, right) = plt.subplots(
        1, 2, figsize=(layout.full, layout.full * 0.31), layout="constrained"
    )

    pools = (2, 3, 5, 8, 10, 15, 20, 50, 100, 200)
    present_geo = _measured(sources, layout, "geo_ablation-{backbone}")
    for backbone in present_geo:
        art = sources.load(f"geo_ablation-{backbone}")
        diffs, seen = [], []
        for pool in pools:
            geo = art.maybe(f"geo_top_1_pool_{pool}")
            rnd = art.maybe(f"random_top_1_pool_{pool}")
            if geo is None or rnd is None:
                continue
            diffs.append((geo - rnd) * 100)
            seen.append(pool)
        _line(left, seen, diffs, backbone)
    _reference(left, 0.0)
    left.set_xscale("log")
    left.set_xticks(pools)
    left.set_xticklabels([str(pool) for pool in pools])
    left.minorticks_off()
    left.set_xlabel("Candidate pool size")
    left.set_ylabel(style.minus("Geographic - random (points)"))
    # Above the plot: the three curves cross the whole plotting area between them,
    # so no corner inside it is clear of a line.
    # Anchored at the left edge, clear of the (a) tag, which the centred legend overprinted
    # once the type scale grew; tighter columns keep four entries inside the panel.
    left.legend(
        loc="lower left",
        bbox_to_anchor=(0.03, 1.0),
        ncol=len(present_geo),
        columnspacing=1.0,
        handlelength=1.6,
    )
    style.panel_tag(left, "a")

    enriched = _measured(sources, layout, "confusion-{backbone}")
    width, offsets = _grouped(len(enriched))
    positions = np.arange(len(SEPARATION_BANDS), dtype=float)
    for offset, backbone in zip(offsets, enriched, strict=True):
        art = sources.load(f"confusion-{backbone}")
        ratios = []
        for band in SEPARATION_BANDS:
            confused = art.maybe(f"confused_pairs_within_{band}m")
            competing = art.maybe(f"competing_pairs_within_{band}m")
            ratios.append(confused / competing if confused and competing else np.nan)
        right.bar(
            positions + offset,
            ratios,
            width * 0.92,
            color=BACKBONE_COLOUR[backbone],
            label=BACKBONE_LABEL[backbone],
        )
    _reference(right, 1.0)
    right.set_xticks(positions)
    right.set_xticklabels([f"{b} m" for b in SEPARATION_BANDS])
    right.set_xlabel("Statues separated by at most")
    right.set_ylabel("Confused \N{DIVISION SIGN} competing pairs")
    right.legend(loc="upper right")
    style.panel_tag(right, "b")

    written = _save(figure, output)
    plt.close(figure)
    return written


def photographer_figure(sources: Sources, output: Path, layout: Layout = JOURNAL) -> Path:
    """Fig. 5 -- what a change of photographer costs, and what a phone costs.

    Panel (a) is the decomposition, not the raw drop: the size-matched control is
    what separates "fewer references" from "a different photographer".
    """
    plt = _pyplot()
    figure, (left, right) = plt.subplots(
        1,
        2,
        figsize=(layout.full, layout.full * 0.33),
        layout="constrained",
        gridspec_kw={"width_ratios": (1.5, 1.0)},
    )

    arms = ("standard_top_1", "control_top_1", "disjoint_top_1")
    labels = ("Standard\nleave-one-out", "Size-matched\nrandom control", "Photographer\nwithheld")
    present = _measured(sources, layout, "photographer_gap-{backbone}")
    width, offsets = _grouped(len(present))
    positions = np.arange(len(arms), dtype=float)
    for offset, backbone in zip(offsets, present, strict=True):
        art = sources.load(f"photographer_gap-{backbone}")
        bars = left.bar(
            positions + offset,
            [art.get(a) * 100 for a in arms],
            width * 0.92,
            color=BACKBONE_COLOUR[backbone],
            label=BACKBONE_LABEL[backbone],
        )
        _bar_labels(left, bars, "%.1f")
    left.set_xticks(positions)
    left.set_xticklabels(labels)
    left.set_ylabel("Top-1 accuracy (%)")
    left.set_ylim(0, 108)
    left.legend(loc="upper center", ncol=len(present), bbox_to_anchor=(0.5, 1.14))
    style.panel_tag(left, "a")

    conditions = ("camera_top_1", "phone_top_1")
    phone = _measured(sources, layout, "camera_gap-{backbone}")
    width, offsets = _grouped(len(phone))
    positions = np.arange(len(conditions), dtype=float)
    for offset, backbone in zip(offsets, phone, strict=True):
        art = sources.load(f"camera_gap-{backbone}")
        bars = right.bar(
            positions + offset,
            [art.get(c) * 100 for c in conditions],
            width * 0.92,
            color=BACKBONE_COLOUR[backbone],
            label=BACKBONE_LABEL[backbone],
        )
        _bar_labels(right, bars, "%.1f")
    right.set_xticks(positions)
    right.set_xticklabels(("Camera", "Phone"))
    right.set_ylabel("Top-1 accuracy (%)")
    right.set_ylim(0, 108)
    right.legend(loc="upper center", ncol=len(phone), bbox_to_anchor=(0.5, 1.14))
    style.panel_tag(right, "b")

    written = _save(figure, output)
    plt.close(figure)
    return written


def matcher_figure(sources: Sources, output: Path, layout: Layout = JOURNAL) -> Path:
    """Fig. 6 -- geometric re-ranking, the matcher that widens it, and the ceiling.

    The ceiling line is the point of the figure: both curves rise, and neither can
    pass the first stage's recall, because re-ranking only reorders what was
    already proposed.
    """
    plt = _pyplot()
    figure, axes = plt.subplots(figsize=(layout.column, layout.column * 0.72), layout="constrained")

    art = sources.load("matcher_rerank-disjoint-dinov2")
    weights = (0.0, 0.01, 0.02, 0.05, 0.1, 0.2, 0.3, 0.5, 0.75, 1.0, 1.5, 2.0)
    for matcher in ("sift", "disk-lightglue"):
        values = []
        for w in weights:
            v = art.maybe(f"{matcher}_w{w:g}_top_1")
            values.append(v * 100 if v is not None else np.nan)
        axes.plot(
            weights,
            values,
            marker=style.MATCHER_MARKER[matcher],
            color=style.MATCHER_COLOUR[matcher],
            label=style.MATCHER_LABEL[matcher],
            markeredgecolor="white",
            markeredgewidth=0.5,
        )

    ceiling = sources.load("recall_curve-dinov2").get("disjoint_r_at_10") * 100
    _reference(axes, ceiling)
    axes.text(
        0.012,
        ceiling + 0.4,
        f"first stage's recall at 10 ({ceiling:.1f}%)",
        color=style.INK_SECONDARY,
        fontsize=style.SIZE_ANNOTATION,
    )

    axes.set_ylim(top=ceiling + 2.2)
    axes.set_xscale("symlog", linthresh=0.01)
    # Weights are non-negative; symlog would otherwise draw a mirrored negative
    # decade and invite the reader to wonder what a negative weight means.
    axes.set_xlim(left=0.0, right=2.4)
    axes.set_xlabel("Weight given to geometry")
    axes.set_ylabel("Top-1, photographer withheld (%)")
    axes.legend(loc="lower left")

    written = _save(figure, output)
    plt.close(figure)
    return written


def rejection_figure(sources: Sources, output: Path, layout: Layout = JOURNAL) -> Path:
    """Fig. 7 -- the system cannot decline to answer, on either signal."""
    plt = _pyplot()
    if layout.name == "cvww":
        # One column: the paper shows the margin comparison and states the
        # geometry result in a sentence (section 7.15).
        figure, left = plt.subplots(
            figsize=(layout.column, layout.column * 0.62), layout="constrained"
        )
        right = None
    else:
        figure, (left, right) = plt.subplots(
            1, 2, figsize=(layout.full, layout.full * 0.33), layout="constrained"
        )

    present = _measured(sources, layout, "open_set-{backbone}")
    if layout.name == "cvww":
        # Section 7.15: cosine against the margin score at 90% known acceptance.
        # Bars, not curves: every operating point shares the 90% row, where markers
        # and arrows overprinted one another; and at 95% and 99% every backbone
        # accepts nearly every unknown statue, which says nothing.
        validation = sources.validation_metrics()
        width, offsets = _grouped(2)
        positions = np.arange(len(present), dtype=float)
        for index, backbone in enumerate(present):
            colour = BACKBONE_COLOUR[backbone]
            values = (
                sources.load(f"open_set-{backbone}").get("target_90_false_acceptance") * 100,
                validation[f"{backbone}_reject_margin_false_acceptance_90"] * 100,
            )
            for offset, value, hatch in zip(offsets, values, ("", "////"), strict=True):
                bars = left.bar(
                    [positions[index] + offset],
                    [value],
                    width * 0.92,
                    color="white" if hatch else colour,
                    edgecolor=colour,
                    hatch=hatch,
                    linewidth=0.9,
                )
                _bar_labels(left, bars, "%.1f")
        left.set_xticks(positions)
        left.set_xticklabels([BACKBONE_LABEL[b] for b in present])
        left.set_ylabel("Unknown statues accepted (%)")
        left.set_ylim(0, 80)
        # The legend names the encoding, in ink: colour already names the backbone.
        from matplotlib.patches import Patch

        left.legend(
            handles=[
                Patch(facecolor=style.INK_MUTED, edgecolor=style.INK_MUTED, label="Cosine"),
                Patch(facecolor="white", edgecolor=style.INK_MUTED, hatch="////", label="Margin"),
            ],
            loc="upper right",
            title="At 90% known acceptance",
            title_fontsize=style.SIZE_LEGEND,
        )
    else:
        targets = (90, 95, 99)
        for backbone in present:
            art = sources.load(f"open_set-{backbone}")
            known = [art.get(f"target_{t}_known_acceptance") * 100 for t in targets]
            false = [art.get(f"target_{t}_false_acceptance") * 100 for t in targets]
            _line(
                left,
                false,
                known,
                backbone,
                f"{BACKBONE_LABEL[backbone]} (AUROC {art.get('auroc'):.3f})",
            )
        left.set_xlabel("Unknown statues wrongly accepted (%)")
        left.set_ylabel("Known statues accepted (%)")
        # The three operating points sit on the tick marks, so the axis labels them.
        left.set_yticks(targets)
        left.legend(loc="upper left")
    if right is None:
        written = _save(figure, output)
        plt.close(figure)
        return written
    style.panel_tag(left, "a")

    # Two regimes of one backbone: DINOv2's blue, full and as a light tint.
    signals = ("cosine", "inliers_best", "blended")
    width, offsets = _grouped(2)
    positions = np.arange(len(signals), dtype=float)
    art = sources.load("open_set_geometry-dinov2")
    for offset, regime, colour, label in (
        (offsets[0], "standard", BACKBONE_COLOUR["dinov2"], "All references"),
        (offsets[1], "photographer_disjoint", "#B9CAFF", "Photographer withheld"),
    ):
        bars = right.bar(
            positions + offset,
            [art.get(f"{regime}_{s}_auroc") for s in signals],
            width * 0.92,
            color=colour,
            label=label,
        )
        _bar_labels(right, bars, "%.3f")
    _reference(right, 0.5)
    right.set_xticks(positions)
    right.set_xticklabels(("Appearance", "Geometry", "Blended"))
    right.set_ylabel("DINOv2 AUROC, known vs unknown")
    right.set_ylim(0.4, 1.02)
    right.legend(loc="upper center", ncol=2, bbox_to_anchor=(0.5, 1.14))
    style.panel_tag(right, "b")

    written = _save(figure, output)
    plt.close(figure)
    return written


FIGURES = {
    "fig1-dataset": dataset_figure,
    "fig3-pool-size": pool_size_figure,
    "fig4-geography": geography_figure,
    "fig5-photographer": photographer_figure,
    "fig6-matcher": matcher_figure,
    "fig7-rejection": rejection_figure,
}


def draw_all(sources: Sources, output_dir: Path, layout: Layout = JOURNAL) -> tuple[Path, ...]:
    """Draw every data-derived paper figure at one layout's sizes."""
    return tuple(
        draw(sources, output_dir / f"{name}.png", layout) for name, draw in FIGURES.items()
    )

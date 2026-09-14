"""Publication figures for the manuscript, built from the result artifacts.

These exist because the venue allows eight display items and `RESULTS.md` holds
twenty tables. Turning tables into figures is what makes the material fit, so
each function here consolidates several source tables into one panel set.

The code is tracked even though the manuscript is not. A figure that cannot be
regenerated from a committed command is a figure nobody can check, and §6.4's
rule -- every published number traceable to a `results/*.json` artifact -- applies
to the paper at least as strongly as it applies to the repository.

Every figure is written at `PUBLICATION_DPI` in a sans-serif face, which is what
Nature asks for.
"""

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from krasnal_id.models import DatasetManifest

# Nature asks for 300 DPI minimum for publication figures. The repository's own
# figures are drawn at a lower density for the web, so these are separate files
# rather than the same ones reused.
PUBLICATION_DPI = 300

# Both backbones, everywhere. Colour-blind safe and distinguishable in greyscale,
# which a print reviewer may be reading.
BACKBONE_COLOUR = {"dinov2": "#1b6ca8", "clip": "#c05640"}
BACKBONE_LABEL = {"dinov2": "DINOv2", "clip": "CLIP"}

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

    def load(self, stem: str) -> Artifact:
        """Read one result artifact's metrics."""
        return load(self.results_dir, stem)

    def records(self, stem: str, key: str) -> list[dict[str, Any]]:
        """Read one artifact's per-item records, such as the confused pairs."""
        payload = read_artifact(self.results_dir, stem)
        if key not in payload:
            raise PaperFigureError(f"{stem} has no {key!r} records")
        return list(payload[key])

    def dataset(self) -> "DatasetManifest":
        """Read the manifest, validated against the project's own schema."""
        if not self.manifest.is_file():
            raise PaperFigureError(f"missing manifest: {self.manifest}")
        return DatasetManifest.model_validate(json.loads(self.manifest.read_text(encoding="utf-8")))


def _pyplot() -> Any:
    """Import matplotlib configured for print, only when a figure is drawn."""
    try:
        import matplotlib
    except ImportError as error:  # pragma: no cover - exercised by the extra's absence
        raise PaperFigureError(
            "drawing needs the analysis extra; run uv sync --extra analysis"
        ) from error
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # Sans-serif throughout, as the journal requires. DejaVu Sans ships with
    # matplotlib, so this does not depend on host fonts.
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["DejaVu Sans"],
            "font.size": 8,
            "axes.titlesize": 9,
            "axes.labelsize": 8,
            "legend.fontsize": 7.5,
            "xtick.labelsize": 7.5,
            "ytick.labelsize": 7.5,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )
    return plt


def _save(figure: Any, path: Path) -> Path:
    """Write one figure at publication density."""
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=PUBLICATION_DPI, bbox_inches="tight")
    return path


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


def dataset_figure(sources: Sources, output: Path) -> Path:
    """Fig. 1 -- what the corpus is, and the two things that make it hard.

    The three panels are the paper's three standing difficulties in the order the
    Results meet them: statues that were sculpted alike, statues that stand close
    together, and classes whose photographs mostly come from one person.
    """
    plt = _pyplot()
    figure = plt.figure(figsize=(7.2, 3.4))
    panels = figure.subfigures(1, 3, width_ratios=(1.0, 1.15, 1.3), wspace=0.02)
    manifest = sources.dataset()

    _confused_pairs_panel(panels[0], sources, manifest)
    _placement_panel(panels[1], sources, manifest)
    _photographer_panel(panels[2], manifest)

    written = _save(figure, output)
    plt.close(figure)
    return written


def _confused_pairs_panel(panel: Any, sources: Sources, manifest: DatasetManifest) -> None:
    """Panel (a) -- the pairs the retrieval actually confuses, photographed."""
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

    pairs = sources.records("confusion-dinov2", "pairs")[:CONFUSED_PAIRS_SHOWN]
    axes = panel.subplots(CONFUSED_PAIRS_SHOWN, 2)
    for row, pair in zip(axes, pairs, strict=True):
        ids = (str(pair["true_dwarf_id"]), str(pair["confused_dwarf_id"]))
        for cell, dwarf_id in zip(row, ids, strict=True):
            if dwarf_id not in chosen:
                raise PaperFigureError(f"confusion names an unknown dwarf: {dwarf_id}")
            cell.imshow(_square_thumbnail(chosen[dwarf_id], 320))
            cell.set_xticks([])
            cell.set_yticks([])
            for spine in cell.spines.values():
                spine.set_visible(False)
            cell.set_xlabel(names.get(dwarf_id, dwarf_id), fontsize=6, labelpad=1.5)
        # The arrow reads in the direction the errors run: queries of the left
        # statue are returned as the right one.
        row[0].annotate(
            "",
            xy=(1.15, 0.5),
            xytext=(1.02, 0.5),
            xycoords="axes fraction",
            arrowprops={"arrowstyle": "-|>", "color": "#cf4832", "linewidth": 0.9},
        )
    panel.subplots_adjust(left=0.02, right=0.98, top=0.88, bottom=0.03, hspace=0.45, wspace=0.12)
    panel.suptitle("(a) Most-confused pairs", x=0.02, ha="left", fontsize=9)


def _placement_panel(panel: Any, sources: Sources, manifest: DatasetManifest) -> None:
    """Panel (b) -- where the statues stand, and what a pool radius covers."""
    placed = np.array(
        [
            (dwarf.coordinates.latitude, dwarf.coordinates.longitude)
            for dwarf in manifest.dwarfs
            if dwarf.coordinates is not None
        ],
        dtype=float,
    )
    origin = (float(np.median(placed[:, 0])), float(np.median(placed[:, 1])))
    east, north = _local_metres(placed, origin)

    axes = panel.subplots()
    axes.scatter(east / 1000, north / 1000, s=4, color="#1b6ca8", alpha=0.65, linewidths=0)
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
    for spine in inset.spines.values():
        spine.set_visible(True)
        spine.set_edgecolor("#cf4832")
        spine.set_linewidth(0.7)
    inset.scatter(east / 1000, north / 1000, s=6, color="#1b6ca8", linewidths=0)
    inset.set_aspect("equal")
    for _, radius in radii:
        inset.add_patch(
            plt_circle(east[centre] / 1000, north[centre] / 1000, radius / 1000, "#cf4832")
        )
    # One caption for both circles. Labelling each at its own arc puts two lines
    # of text inside a 780 m box, where they collide.
    inset.text(
        0.03,
        0.97,
        "\n".join(f"{radius:.0f} m → {pool} dwarves" for pool, radius in radii),
        transform=inset.transAxes,
        va="top",
        fontsize=5.5,
        color="#cf4832",
        bbox={"facecolor": "white", "edgecolor": "none", "pad": 1.0},
    )
    axes.indicate_inset_zoom(inset, edgecolor="#cf4832", linewidth=0.7, alpha=0.9)

    panel.subplots_adjust(left=0.20, right=0.99, top=0.88, bottom=0.16)
    panel.suptitle(
        f"(b) {placed.shape[0]} of {len(manifest.dwarfs)} are placed",
        x=0.02,
        ha="left",
        fontsize=9,
    )


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
        linewidth=0.9,
        linestyle="--",
        zorder=5,
    )


def _photographer_panel(panel: Any, manifest: DatasetManifest) -> None:
    """Panel (c) -- small classes, and how much of each is one person's work."""
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
    axes.bar(positions, counts, 0.68, color="#b9c6d1", label="Classes")
    axes.set_xticks(positions)
    axes.set_xticklabels(labels)
    axes.set_xlabel("Photographs of the statue")
    axes.set_ylabel("Classes")

    concentration = axes.twinx()
    concentration.plot(
        positions, shares, "o-", color="#c05640", linewidth=1.4, markersize=3.5, label="One person"
    )
    concentration.set_ylabel("Share from the class's\nmost prolific photographer (%)", fontsize=7)
    concentration.set_ylim(0, 100)
    concentration.spines["right"].set_visible(True)
    concentration.spines["top"].set_visible(False)

    photographers = len({image.author for image in manifest.images})
    panel.subplots_adjust(left=0.16, right=0.78, top=0.88, bottom=0.16)
    panel.suptitle(
        f"(c) {len(manifest.images):,} photographs, {photographers} people",
        x=0.02,
        ha="left",
        fontsize=9,
    )


def pool_size_figure(sources: Sources, output: Path) -> Path:
    """Fig. 2 -- accuracy falls log-linearly as the candidate pool grows.

    The straight line on a log axis is the result: every doubling of the pool
    costs a fixed number of points, which is what lets the curve be extrapolated
    past the 306 dwarves that exist. The band is the spread across seeds, and it
    closes at 306 because there is only one way to draw the whole corpus.
    """
    plt = _pyplot()
    figure, axes = plt.subplots(figsize=(4.8, 3.3))

    for backbone in ("dinov2", "clip"):
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
        colour = BACKBONE_COLOUR[backbone]
        # The slope rides in the legend rather than beside the curve it describes.
        # Both curves are straight and close over most of the axis, so any label
        # placed against one of them crosses the other.
        slope = art.get("top_1_points_per_doubling")
        # A true minus sign: this is set in the figure's sans face, where a hyphen
        # is visibly shorter.
        printed = f"{slope:.2f}".replace("-", "\N{MINUS SIGN}")
        axes.fill_between(pools, lower, upper, color=colour, alpha=0.20, linewidth=0)
        axes.plot(
            pools,
            values,
            "o-",
            color=colour,
            label=f"{BACKBONE_LABEL[backbone]}, {printed} points per doubling",
            linewidth=1.4,
            markersize=3.5,
        )

    axes.set_xscale("log")
    axes.set_xticks(POOL_SIZES)
    axes.set_xticklabels([str(pool) for pool in POOL_SIZES])
    axes.minorticks_off()
    axes.set_xlabel("Candidate dwarves in the pool")
    axes.set_ylabel("Top-1 accuracy (%)")
    axes.set_title("Every doubling of the pool costs a fixed amount", loc="left")
    axes.legend(frameon=False, loc="lower left")

    figure.tight_layout()
    written = _save(figure, output)
    plt.close(figure)
    return written


def geography_figure(sources: Sources, output: Path) -> Path:
    """Fig. 3 — location narrowing hurts, and the mechanism decays with distance.

    Two panels replacing thirteen table rows: the penalty against pool size, and
    the co-location enrichment that explains it. Panel (b) is what makes the
    explanation falsifiable rather than a story -- the enrichment has to decay,
    and it does.
    """
    plt = _pyplot()
    figure, (left, right) = plt.subplots(1, 2, figsize=(7.2, 2.9))

    pools = (2, 3, 5, 8, 10, 15, 20, 50, 100, 200)
    for backbone in ("dinov2", "clip"):
        art = sources.load(f"geo_ablation-{backbone}")
        diffs, seen = [], []
        for pool in pools:
            geo = art.maybe(f"geo_top_1_pool_{pool}")
            rnd = art.maybe(f"random_top_1_pool_{pool}")
            if geo is None or rnd is None:
                continue
            diffs.append((geo - rnd) * 100)
            seen.append(pool)
        left.plot(
            seen,
            diffs,
            "o-",
            color=BACKBONE_COLOUR[backbone],
            label=BACKBONE_LABEL[backbone],
            linewidth=1.4,
            markersize=3.5,
        )
    left.axhline(0, color="#555555", linewidth=0.8, linestyle="--")
    left.set_xscale("log")
    left.set_xlabel("Candidate pool size")
    # A true minus sign rather than a hyphen: this is a printed axis label, and
    # the journal sets figures in a sans face where the two are visibly different.
    left.set_ylabel("Geographic − random (points)")  # noqa: RUF001
    left.set_title("(a) Narrowing by location costs accuracy", loc="left")
    left.legend(frameon=False)

    width = 0.38
    positions = np.arange(len(SEPARATION_BANDS), dtype=float)
    for offset, backbone in zip((-width / 2, width / 2), ("dinov2", "clip"), strict=True):
        art = sources.load(f"confusion-{backbone}")
        ratios = []
        for band in SEPARATION_BANDS:
            confused = art.maybe(f"confused_pairs_within_{band}m")
            competing = art.maybe(f"competing_pairs_within_{band}m")
            ratios.append(confused / competing if confused and competing else np.nan)
        right.bar(
            positions + offset,
            ratios,
            width,
            color=BACKBONE_COLOUR[backbone],
            label=BACKBONE_LABEL[backbone],
        )
    right.axhline(1.0, color="#555555", linewidth=0.8, linestyle="--")
    right.set_xticks(positions)
    right.set_xticklabels([f"{b} m" for b in SEPARATION_BANDS])
    right.set_xlabel("Statues separated by at most")
    right.set_ylabel("Confused ÷ competing")
    right.set_title("(b) Confusion is enriched only at very short range", loc="left")
    right.legend(frameon=False)

    figure.tight_layout()
    written = _save(figure, output)
    plt.close(figure)
    return written


def photographer_figure(sources: Sources, output: Path) -> Path:
    """Fig. 5 — what a change of photographer costs, and what a phone costs.

    Panel (a) is the decomposition, not the raw drop: the size-matched control is
    what separates "fewer references" from "a different photographer", and
    without it the 12-point fall reads as a 12-point photographer effect.
    """
    plt = _pyplot()
    figure, (left, right) = plt.subplots(1, 2, figsize=(7.2, 2.9))

    arms = ("standard_top_1", "control_top_1", "disjoint_top_1")
    labels = ("Standard\nleave-one-out", "Size-matched\nrandom control", "Photographer\nwithheld")
    width = 0.38
    positions = np.arange(len(arms), dtype=float)
    for offset, backbone in zip((-width / 2, width / 2), ("dinov2", "clip"), strict=True):
        art = sources.load(f"photographer_gap-{backbone}")
        values = [art.get(a) * 100 for a in arms]
        bars = left.bar(
            positions + offset,
            values,
            width,
            color=BACKBONE_COLOUR[backbone],
            label=BACKBONE_LABEL[backbone],
        )
        left.bar_label(bars, fmt="%.1f", padding=1.5, fontsize=6.5)
    left.set_xticks(positions)
    left.set_xticklabels(labels)
    left.set_ylabel("Top-1 accuracy (%)")
    left.set_ylim(0, 105)
    left.set_title("(a) Only part of the drop is the photographer", loc="left")
    left.legend(frameon=False, loc="upper center", bbox_to_anchor=(0.5, 1.20), ncol=2)

    conditions = ("camera_top_1", "phone_top_1")
    cond_labels = ("Camera", "Phone")
    positions = np.arange(len(conditions), dtype=float)
    for offset, backbone in zip((-width / 2, width / 2), ("dinov2", "clip"), strict=True):
        art = sources.load(f"camera_gap-{backbone}")
        values = [art.get(c) * 100 for c in conditions]
        bars = right.bar(
            positions + offset,
            values,
            width,
            color=BACKBONE_COLOUR[backbone],
            label=BACKBONE_LABEL[backbone],
        )
        right.bar_label(bars, fmt="%.1f", padding=1.5, fontsize=6.5)
    right.set_xticks(positions)
    right.set_xticklabels(cond_labels)
    right.set_ylabel("Top-1 accuracy (%)")
    right.set_ylim(0, 105)
    right.set_title("(b) Phone-shot queries are harder", loc="left")
    right.legend(frameon=False, loc="upper center", bbox_to_anchor=(0.5, 1.20), ncol=2)

    figure.tight_layout()
    written = _save(figure, output)
    plt.close(figure)
    return written


def matcher_figure(sources: Sources, output: Path) -> Path:
    """Fig. 6 — geometric re-ranking, the matcher that widens it, and the ceiling.

    The ceiling line is the point of the figure: both curves rise, and neither can
    pass the first stage's recall, because re-ranking only reorders what was
    already proposed.
    """
    plt = _pyplot()
    figure, axes = plt.subplots(figsize=(4.6, 3.2))

    art = sources.load("matcher_rerank-disjoint-dinov2")
    weights = (0.0, 0.01, 0.02, 0.05, 0.1, 0.2, 0.3, 0.5, 0.75, 1.0, 1.5, 2.0)
    style = {
        "sift": ("#7a7a7a", "o", "SIFT"),
        "disk-lightglue": ("#1b6ca8", "s", "DISK + LightGlue"),
    }
    for matcher, (colour, marker, label) in style.items():
        values = []
        for w in weights:
            v = art.maybe(f"{matcher}_w{w:g}_top_1")
            values.append(v * 100 if v is not None else np.nan)
        axes.plot(
            weights, values, marker=marker, color=colour, label=label, linewidth=1.4, markersize=3.5
        )

    ceiling = sources.load("recall_curve-dinov2").get("disjoint_r_at_10") * 100
    axes.axhline(ceiling, color="#c05640", linewidth=1.0, linestyle="--")
    axes.text(
        0.012,
        ceiling + 0.35,
        f"first stage's recall@10 ({ceiling:.1f}%)",
        color="#c05640",
        fontsize=7,
    )

    axes.set_xscale("symlog", linthresh=0.01)
    # Weights are non-negative; symlog would otherwise draw a mirrored negative
    # decade and invite the reader to wonder what a negative weight means.
    axes.set_xlim(left=0.0, right=2.4)
    axes.set_xlabel("Weight given to geometry in the blended score")
    axes.set_ylabel("Top-1 accuracy, photographer-disjoint (%)")
    axes.set_title("Verifying geometry, and the ceiling it runs into", loc="left")
    axes.legend(frameon=False, loc="lower left")

    figure.tight_layout()
    written = _save(figure, output)
    plt.close(figure)
    return written


def rejection_figure(sources: Sources, output: Path) -> Path:
    """Fig. 7 — the system cannot decline to answer, on either signal.

    Panel (b) is why the negative result is credible: appearance and geometry are
    both measured, on both regimes, and neither separates known from unknown.
    """
    plt = _pyplot()
    figure, (left, right) = plt.subplots(1, 2, figsize=(7.2, 2.9))

    targets = (90, 95, 99)
    for backbone in ("dinov2", "clip"):
        art = sources.load(f"open_set-{backbone}")
        known = [art.get(f"target_{t}_known_acceptance") * 100 for t in targets]
        false = [art.get(f"target_{t}_false_acceptance") * 100 for t in targets]
        left.plot(
            false,
            known,
            "o-",
            color=BACKBONE_COLOUR[backbone],
            label=f"{BACKBONE_LABEL[backbone]} (AUROC {art.get('auroc'):.3f})",
            linewidth=1.4,
            markersize=4,
        )
        # The backbones converge at the 99% target, so one series is labelled to the
        # right of its markers and the other to the left. Offsetting vertically is
        # not enough when the two points nearly coincide.
        dx = 4 if backbone == "dinov2" else -24
        for t, f, k in zip(targets, false, known, strict=True):
            left.annotate(
                f"{t}%",
                (f, k),
                textcoords="offset points",
                xytext=(dx, -7),
                fontsize=6.5,
                color="#444444",
            )
    left.set_xlabel("Unknown statues wrongly accepted (%)")
    left.set_ylabel("Known statues accepted (%)")
    left.set_title("(a) No operating point is usable", loc="left")
    left.legend(frameon=False, loc="lower right")

    signals = ("cosine", "inliers_best", "blended")
    signal_labels = ("Appearance", "Geometry", "Blended")
    width = 0.38
    positions = np.arange(len(signals), dtype=float)
    for offset, regime, colour, label in (
        (-width / 2, "standard", "#1b6ca8", "All references"),
        (width / 2, "photographer_disjoint", "#c05640", "Photographer withheld"),
    ):
        art = sources.load("open_set_geometry-dinov2")
        values = [art.get(f"{regime}_{s}_auroc") for s in signals]
        bars = right.bar(positions + offset, values, width, color=colour, label=label)
        right.bar_label(bars, fmt="%.3f", padding=1.5, fontsize=6.5)
    right.axhline(0.5, color="#555555", linewidth=0.8, linestyle="--")
    right.set_xticks(positions)
    right.set_xticklabels(signal_labels)
    right.set_ylabel("AUROC, known vs unknown")
    right.set_ylim(0.4, 1.0)
    right.set_title("(b) Geometry adds almost nothing", loc="left")
    right.legend(frameon=False, loc="upper center", bbox_to_anchor=(0.5, 1.20), ncol=2)

    figure.tight_layout()
    written = _save(figure, output)
    plt.close(figure)
    return written


FIGURES = {
    "fig1-dataset": dataset_figure,
    "fig2-pool-size": pool_size_figure,
    "fig3-geography": geography_figure,
    "fig5-photographer": photographer_figure,
    "fig6-matcher": matcher_figure,
    "fig7-rejection": rejection_figure,
}


def draw_all(sources: Sources, output_dir: Path) -> tuple[Path, ...]:
    """Draw every data-derived paper figure."""
    return tuple(draw(sources, output_dir / f"{name}.png") for name, draw in FIGURES.items())

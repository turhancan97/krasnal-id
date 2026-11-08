"""Draw what a matcher actually matched, between two photographs.

Every other number in this project is an aggregate: a percentage over 1,157
queries, a count of wins against losses. Those are the right units for a claim
and the wrong ones for understanding, because none of them shows *why* one
matcher beats another. A correspondence figure does: it puts the two
photographs side by side and draws a line for every point the matcher believes
is the same physical point on the same statue.

It is also the smallest way into this code. Section 15's comparison needs a
manifest, cached embeddings and about two hours; this needs two image files.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt

from krasnal_id.retrieval.matchers import create_matcher
from krasnal_id.retrieval.rerank import Correspondences, RerankError

# Lines drawn per panel. Every inlier is counted in the caption, but a figure
# with 166 lines across it is a colour wash rather than a picture, so the drawn
# ones are sampled deterministically from the strongest.
MAX_DRAWN = 60


class MatchPlotError(ValueError):
    """Raised when a match figure cannot be drawn."""


@dataclass(frozen=True, slots=True)
class MatchPanel:
    """One matcher's result on one pair, ready to draw."""

    matcher: str
    correspondences: Correspondences

    @property
    def inliers(self) -> int:
        """Return how many matches survived the homography."""
        return self.correspondences.inliers


def _load_rgb(path: Path, long_side: int) -> npt.NDArray[np.uint8]:
    """Read one image at the scale the matchers describe it at.

    The same scale, or the drawn keypoints would not sit on the features they
    were detected from.
    """
    import cv2

    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise MatchPlotError(f"could not read {path} as an image")
    height, width = image.shape[:2]
    longest = max(height, width)
    if longest > long_side:
        scale = long_side / longest
        image = cv2.resize(
            image, (int(width * scale), int(height * scale)), interpolation=cv2.INTER_AREA
        )
    return np.asarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB), dtype=np.uint8)


def match_pair(
    query: Path,
    candidate: Path,
    matcher_name: str,
    keypoints: int,
    device: str = "auto",
) -> MatchPanel:
    """Run one matcher over one pair of photographs."""
    matcher = create_matcher(matcher_name, keypoints, device)
    try:
        described_query = matcher.get(f"query:{query}", query)
        described_candidate = matcher.get(f"candidate:{candidate}", candidate)
    except RerankError as error:
        raise MatchPlotError(str(error)) from error
    return MatchPanel(
        matcher=matcher.name,
        correspondences=matcher.correspondences(described_query, described_candidate),
    )


def _draw_panel(
    axes: Any,
    left: npt.NDArray[np.uint8],
    right: npt.NDArray[np.uint8],
    panel: MatchPanel,
) -> None:
    """Draw one matcher's correspondences over a side-by-side pair."""
    height = max(left.shape[0], right.shape[0])
    canvas = np.full((height, left.shape[1] + right.shape[1], 3), 255, dtype=np.uint8)
    canvas[: left.shape[0], : left.shape[1]] = left
    canvas[: right.shape[0], left.shape[1] :] = right
    axes.imshow(canvas)
    axes.set_axis_off()

    kept = np.flatnonzero(panel.correspondences.inlier_mask)
    # Deterministic subsample: evenly spaced through the kept matches, so the
    # figure is reproducible and not a lucky draw.
    if kept.size > MAX_DRAWN:
        kept = kept[np.linspace(0, kept.size - 1, MAX_DRAWN).astype(int)]
    source = panel.correspondences.source[kept]
    target = panel.correspondences.target[kept]
    for (x0, y0), (x1, y1) in zip(source, target, strict=True):
        axes.plot(
            [x0, x1 + left.shape[1]],
            [y0, y1],
            color="#1b9e77",
            linewidth=0.6,
            alpha=0.65,
            solid_capstyle="round",
        )
    axes.scatter(source[:, 0], source[:, 1], s=3, color="#1b9e77", linewidths=0)
    axes.scatter(target[:, 0] + left.shape[1], target[:, 1], s=3, color="#1b9e77", linewidths=0)
    drawn = "" if panel.inliers <= MAX_DRAWN else f", {MAX_DRAWN} drawn"
    axes.set_title(
        f"{panel.matcher} — {panel.inliers} verified matches{drawn}",
        fontsize=11,
        pad=8,
    )


def draw_match_figure(
    query: Path,
    candidate: Path,
    panels: tuple[MatchPanel, ...],
    output_path: Path,
    long_side: int,
    caption: str | None = None,
    dpi: int = 150,
) -> Path:
    """Write a figure comparing what each matcher found on one pair.

    The density is a parameter for the same reason the contact sheet's is: the
    repository reads this on a screen and Supplementary S12 needs it at the
    density the journal asks for.
    """
    if not panels:
        raise MatchPlotError("at least one matcher panel is required")
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as error:  # pragma: no cover - exercised by the extra's absence
        raise MatchPlotError(
            "drawing needs the analysis extra; run uv sync --extra analysis"
        ) from error

    left, right = _load_rgb(query, long_side), _load_rgb(candidate, long_side)
    figure, axes = plt.subplots(len(panels), 1, figsize=(11, 4.2 * len(panels)))
    for panel, axis in zip(panels, np.atleast_1d(axes), strict=True):
        _draw_panel(axis, left, right, panel)
    if caption:
        figure.suptitle(caption, fontsize=12, y=0.995)
    figure.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(figure)
    return output_path


def _square_window(
    shape: tuple[int, ...], points: list[npt.NDArray[np.float32]]
) -> tuple[int, int, int]:
    """The largest square in an image, centred on where its matches fall.

    Every panel in the grid becomes the same square, as the teaser and retrieval
    sheets are, so rows line up; centring on the matches keeps the statue that the
    lines point at inside the crop rather than whatever the frame happened to centre.
    """
    height, width = shape[0], shape[1]
    side = min(height, width)
    stacked = np.concatenate([p for p in points if p.size]) if any(p.size for p in points) else None
    centre_x, centre_y = (
        (float(np.median(stacked[:, 0])), float(np.median(stacked[:, 1])))
        if stacked is not None
        else (width / 2, height / 2)
    )
    left = int(np.clip(round(centre_x - side / 2), 0, width - side))
    top = int(np.clip(round(centre_y - side / 2), 0, height - side))
    return left, top, side


def _draw_pair(
    axes: Any,
    left: npt.NDArray[np.uint8],
    right: npt.NDArray[np.uint8],
    windows: tuple[tuple[int, int, int], tuple[int, int, int]],
    panel: MatchPanel,
    colour: str,
    size: int = 600,
    gap: int = 14,
) -> None:
    """Query and candidate as equal squares, their verified matches drawn between.

    Each line wears the matcher's colour over a thin white casing, and ends in a dot,
    so it stays legible over dark bronze and pale sky alike.
    """
    import cv2
    from matplotlib import patheffects
    from matplotlib.patches import Rectangle

    from krasnal_id.viz import figure_style as style

    def crop(image: npt.NDArray[np.uint8], window: tuple[int, int, int]) -> Any:
        x0, y0, side = window
        return cv2.resize(
            image[y0 : y0 + side, x0 : x0 + side], (size, size), interpolation=cv2.INTER_AREA
        )

    canvas = np.full((size, 2 * size + gap, 3), 255, np.uint8)
    canvas[:, :size] = crop(left, windows[0])
    canvas[:, size + gap :] = crop(right, windows[1])
    axes.imshow(canvas)
    axes.set_axis_off()
    kept = np.flatnonzero(panel.correspondences.inlier_mask)
    if kept.size > MAX_DRAWN:
        kept = kept[np.linspace(0, kept.size - 1, MAX_DRAWN).astype(int)]
    (lx, ly, lside), (rx, ry, rside) = windows
    casing = [
        patheffects.Stroke(linewidth=1.5, foreground="white", alpha=0.75),
        patheffects.Normal(),
    ]
    ends_x: list[float] = []
    ends_y: list[float] = []
    for (x0, y0), (x1, y1) in zip(
        panel.correspondences.source[kept], panel.correspondences.target[kept], strict=True
    ):
        ax0, ay0 = (x0 - lx) * size / lside, (y0 - ly) * size / lside
        ax1, ay1 = (x1 - rx) * size / rside + size + gap, (y1 - ry) * size / rside
        if not (
            0 <= ax0 < size
            and 0 <= ay0 < size
            and size + gap <= ax1 < 2 * size + gap
            and 0 <= ay1 < size
        ):
            continue  # an end outside the crop would be drawn into the wrong photograph
        axes.plot(
            [ax0, ax1],
            [ay0, ay1],
            color=colour,
            linewidth=0.7,
            alpha=0.95,
            solid_capstyle="round",
            path_effects=casing,
        )
        ends_x += [ax0, ax1]
        ends_y += [ay0, ay1]
    axes.scatter(ends_x, ends_y, s=4, color=colour, edgecolors="white", linewidths=0.4, zorder=3)
    for spine_x in (0, size + gap):
        axes.add_patch(
            Rectangle(
                (spine_x - 0.5, -0.5),
                size,
                size,
                fill=False,
                edgecolor=style.NEUTRAL_FILL,
                linewidth=0.6,
            )
        )
    axes.text(
        0.012,
        0.03,
        f"{panel.inliers} verified matches",
        transform=axes.transAxes,
        ha="left",
        va="bottom",
        fontsize=style.SIZE_ANNOTATION,
        color=style.INK,
        bbox={
            "boxstyle": "round,pad=0.25",
            "facecolor": "white",
            "edgecolor": "none",
            "alpha": 0.88,
        },
    )


def draw_match_grid(
    rows: tuple[tuple[str, Path, Path], ...],
    matchers: tuple[str, ...],
    output_path: Path,
    long_side: int,
    width: float,
    keypoints: int = 1024,
    device: str = "auto",
    dpi: int = 300,
) -> tuple[Path, dict[tuple[int, str], int]]:
    """Correspondences in a grid: one row per pair, one column per matcher.

    The publication version of `draw_match_figure`, in the shared figure style and
    drawn at the width it prints. Every photograph is the same square, matchers are
    named once as column headings, rows carry a panel letter and a two-line label
    ("Kind, detail" is split at its first comma), and each panel's count sits in a
    badge. Returns the inlier count of every panel, so the caller can report it.
    """
    if not rows or not matchers:
        raise MatchPlotError("a match grid needs at least one pair and one matcher")
    from krasnal_id.viz import figure_style as style

    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as error:  # pragma: no cover - exercised by the extra's absence
        raise MatchPlotError(
            "drawing needs the analysis extra; run uv sync --extra analysis"
        ) from error
    style.apply(plt)

    counts: dict[tuple[int, str], int] = {}
    label_width = 0.42  # inches for the row labels
    panel_width = (width - label_width) / len(matchers)
    panel_height = panel_width * 600 / (2 * 600 + 14)
    header = 0.2
    figure = plt.figure(figsize=(width, header + len(rows) * (panel_height + 0.06)))
    grid = figure.add_gridspec(
        len(rows),
        len(matchers) + 1,
        width_ratios=[label_width, *([panel_width] * len(matchers))],
        left=0.0,
        right=1.0,
        bottom=0.0,
        top=1 - header / (header + len(rows) * (panel_height + 0.06)),
        wspace=0.04,
        hspace=0.06,
    )
    for row, (label, query, candidate) in enumerate(rows):
        left, right = _load_rgb(query, long_side), _load_rgb(candidate, long_side)
        panels = {name: match_pair(query, candidate, name, keypoints, device) for name in matchers}

        kept = [np.flatnonzero(p.correspondences.inlier_mask) for p in panels.values()]
        windows = (
            _square_window(
                left.shape,
                [p.correspondences.source[k] for p, k in zip(panels.values(), kept, strict=True)],
            ),
            _square_window(
                right.shape,
                [p.correspondences.target[k] for p, k in zip(panels.values(), kept, strict=True)],
            ),
        )
        kind, _, detail = label.partition(", ")
        tag = figure.add_subplot(grid[row, 0])
        tag.set_axis_off()
        tag.text(
            0.0,
            0.98,
            f"({'abcdefgh'[row]})",
            transform=tag.transAxes,
            ha="left",
            va="top",
            fontsize=style.SIZE_PANEL,
            fontweight="bold",
            color=style.INK,
        )
        tag.text(
            0.62,
            0.5,
            f"{kind}\n{detail}" if detail else kind,
            transform=tag.transAxes,
            rotation=90,
            ha="center",
            va="center",
            fontsize=style.SIZE_ANNOTATION,
            color=style.INK_SECONDARY,
            linespacing=1.3,
        )
        for column, name in enumerate(matchers, start=1):
            axes = figure.add_subplot(grid[row, column])
            counts[(row, name)] = panels[name].inliers
            _draw_pair(
                axes, left, right, windows, panels[name], style.MATCHER_COLOUR.get(name, style.INK)
            )
            if row == 0:
                axes.set_title(
                    style.MATCHER_LABEL.get(name, name),
                    fontsize=style.SIZE_LABEL,
                    fontweight="bold",
                    color=style.INK,
                    pad=4,
                )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    style.save(figure, output_path, dpi)
    plt.close(figure)
    return output_path, counts


def _side_by_side(
    left: npt.NDArray[np.uint8], right: npt.NDArray[np.uint8], gap: int = 12
) -> tuple[npt.NDArray[np.uint8], float]:
    """Both photographs at the query's height, and the scale applied to the right one.

    Matching each pair at a common height keeps every panel the same height in
    print, so a portrait candidate cannot leave a landscape query floating in white.
    """
    import cv2

    scale = left.shape[0] / right.shape[0]
    resized = cv2.resize(
        right,
        (max(1, round(right.shape[1] * scale)), left.shape[0]),
        interpolation=cv2.INTER_AREA,
    )
    canvas = np.full((left.shape[0], left.shape[1] + gap + resized.shape[1], 3), 255, np.uint8)
    canvas[:, : left.shape[1]] = left
    canvas[:, left.shape[1] + gap :] = resized
    return canvas, scale


def _draw_styled(
    axes: Any,
    left: npt.NDArray[np.uint8],
    right: npt.NDArray[np.uint8],
    panel: MatchPanel,
    colour: str,
    gap: int = 12,
) -> None:
    """One pair side by side, its verified matches drawn in the matcher's colour."""
    canvas, scale = _side_by_side(left, right, gap)
    axes.imshow(canvas)
    axes.set_axis_off()
    kept = np.flatnonzero(panel.correspondences.inlier_mask)
    if kept.size > MAX_DRAWN:
        kept = kept[np.linspace(0, kept.size - 1, MAX_DRAWN).astype(int)]
    offset = left.shape[1] + gap
    for (x0, y0), (x1, y1) in zip(
        panel.correspondences.source[kept], panel.correspondences.target[kept], strict=True
    ):
        axes.plot(
            [x0, x1 * scale + offset], [y0, y1 * scale], color=colour, linewidth=0.5, alpha=0.8
        )

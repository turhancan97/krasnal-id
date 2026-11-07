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
    drawn at the width it prints. Each matcher's lines wear that matcher's colour,
    as everywhere else in the paper. Returns the inlier count of every panel too,
    so the caller can report what was drawn.
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
    images = [
        (_load_rgb(query, long_side), _load_rgb(candidate, long_side))
        for _, query, candidate in rows
    ]
    aspects = [_side_by_side(left, right)[0].shape for left, right in images]
    column = width / len(matchers)
    heading = 0.17  # inches for each row's heading and its panel titles
    heights = [column * shape[0] / shape[1] + heading for shape in aspects]
    figure = plt.figure(figsize=(width, sum(heights)), layout="constrained")
    subfigures = figure.subfigures(len(rows), 1, height_ratios=heights, squeeze=False)
    for row, ((label, query, candidate), (left, right)) in enumerate(
        zip(rows, images, strict=True)
    ):
        subfigure = subfigures[row][0]
        subfigure.suptitle(
            label,
            x=0.0,
            ha="left",
            fontsize=style.SIZE_ANNOTATION,
            fontweight="bold",
            color=style.INK,
        )
        axes = subfigure.subplots(1, len(matchers), squeeze=False)[0]
        for axis, name in zip(axes, matchers, strict=True):
            panel = match_pair(query, candidate, name, keypoints, device)
            counts[(row, name)] = panel.inliers
            _draw_styled(axis, left, right, panel, style.MATCHER_COLOUR.get(name, style.INK))
            axis.set_title(
                f"{style.MATCHER_LABEL.get(name, name)}: {panel.inliers} verified matches",
                fontsize=style.SIZE_ANNOTATION,
                color=style.INK_SECONDARY,
                pad=2,
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

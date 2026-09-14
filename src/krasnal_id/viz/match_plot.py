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

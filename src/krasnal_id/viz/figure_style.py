"""One visual system for every publication figure: palette, type, sizes and marks.

Figures used to be styled one function at a time, so each picked its own width,
title wording and red. This module is the single place those decisions live, and
every figure function reads from it rather than choosing.

**Colour follows the entity, never the figure.** DINOv2 is the same blue in every
figure it appears in; a map of statues is not DINOv2 and therefore is not blue. The
hues are the author's palette (2026-11-05). The categorical subsets actually used
were checked with the dataviz skill's validator against a white page, for lightness,
chroma, colour-vision-deficiency separation and normal-vision separation:

* backbones -- blue, orange, teal, deep purple: all checks pass. SigLIP 2's
  ``#7B3FA0`` is one step deeper than the author's purple, which sits 9.2 from
  blue under full colour vision (floor 15) and so cannot stand beside it; found by
  running candidates through the validator, worst pair 19.4;
* matchers  -- pink, purple: all checks pass.

Teal is one step deeper than the author's ``#55A398``, which reads grey on paper
(chroma 0.079 against a 0.10 floor); ``#2EA28F`` is the nearest step that passes.
Mustard and yellow fail the lightness band on white and are kept for fills and
highlights, never for thin lines or small points.

**Figures are drawn at the size they are printed.** The CVWW kit's text block is
17 cm wide with a 0.8 cm gutter, so a column is 8.1 cm and the full width 17 cm.
A figure drawn at those widths and placed at ``\\linewidth`` is never scaled, so
the 7 pt tick labels below really are 7 pt on the page, in every figure.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Any

# --- the author's palette ---------------------------------------------------
APPLE_BLUE = "#507AFF"
APPLE_PINK = "#D97FAE"
APPLE_TEAL = "#2EA28F"  # one step deeper than #55A398, which fails the chroma floor
APPLE_PURPLE = "#8A6BDD"
APPLE_MUSTARD = "#EAB34D"  # fills and highlights only: too light for lines on white
APPLE_ORANGE = "#FF7F50"
APPLE_GREEN = "#34C759"
APPLE_YELLOW = "#FFCC00"  # fills and highlights only
DEEP_PURPLE = "#7B3FA0"  # SigLIP 2: the author's purple, deepened until it parts from blue

# --- ink: text and structure wear neutrals, never a series colour -----------
INK = "#1F2328"  # titles, values, direct labels
INK_SECONDARY = "#57606A"  # axis labels, annotations
INK_MUTED = "#8C959F"  # reference lines, "wrong" frames
GRID = "#E6E8EB"
NEUTRAL_FILL = "#C9D1D9"  # data that is not an entity: corpus statistics, statues on a map

# --- entities ---------------------------------------------------------------
BACKBONE_COLOUR = {
    "dinov2": APPLE_BLUE,
    "clip": APPLE_ORANGE,
    "dinov3": APPLE_TEAL,
    "siglip2": DEEP_PURPLE,
}
BACKBONE_LABEL = {"dinov2": "DINOv2", "clip": "CLIP", "dinov3": "DINOv3", "siglip2": "SigLIP 2"}
# A second encoding beside colour, so identity never rests on hue alone.
BACKBONE_MARKER = {"dinov2": "o", "clip": "s", "dinov3": "^", "siglip2": "D"}
MATCHER_COLOUR = {"sift": APPLE_PINK, "disk-lightglue": APPLE_PURPLE}
MATCHER_LABEL = {"sift": "SIFT", "disk-lightglue": "DISK + LightGlue"}
MATCHER_MARKER = {"sift": "o", "disk-lightglue": "s"}
CORRECT = APPLE_GREEN  # a correct retrieval, always with a check mark and a heavier frame
WRONG = INK_MUTED  # a wrong one, always with a cross

# --- type scale (points) ----------------------------------------------------
FONT_FAMILY = "TeX Gyre Heros"
SIZE_TICK = 7.0
SIZE_LABEL = 7.5
SIZE_LEGEND = 7.0
SIZE_ANNOTATION = 6.5
SIZE_PANEL = 8.5  # the bold (a), (b) tags

# --- marks ------------------------------------------------------------------
LINE_WIDTH = 1.3
MARKER_SIZE = 3.6
REFERENCE_LINE_WIDTH = 0.8
AXIS_LINE_WIDTH = 0.6

PUBLICATION_DPI = 300
FONT_DIRECTORY = Path(__file__).with_name("fonts")


@dataclass(frozen=True, slots=True)
class Layout:
    """The widths a figure may be drawn at, in inches, and which backbones it shows."""

    name: str
    column: float
    full: float
    backbones: tuple[str, ...]


# The CVWW 2027 kit: 17 cm text block, 0.8 cm gutter.
CVWW = Layout(
    "cvww", column=8.1 / 2.54, full=17.0 / 2.54, backbones=("dinov2", "clip", "dinov3", "siglip2")
)
# The journal draft, unchanged in size and content: it never introduced DINOv3.
JOURNAL = Layout("journal", column=4.8, full=7.2, backbones=("dinov2", "clip"))
LAYOUTS = {layout.name: layout for layout in (CVWW, JOURNAL)}


class FigureStyleError(ValueError):
    """Raised when the vendored font cannot be found or an unknown layout is asked for."""


def layout_named(name: str) -> Layout:
    """Return one layout by name, refusing an unknown one."""
    if name not in LAYOUTS:
        raise FigureStyleError(f"unknown figure layout {name!r}; choose from {sorted(LAYOUTS)}")
    return LAYOUTS[name]


def register_fonts() -> None:
    """Make the vendored TeX Gyre Heros available to matplotlib, once per process."""
    from matplotlib import font_manager

    faces = sorted(FONT_DIRECTORY.glob("texgyreheros-*.otf"))
    if not faces:
        raise FigureStyleError(f"the figure font is missing from {FONT_DIRECTORY}")
    known = {entry.fname for entry in font_manager.fontManager.ttflist}
    for face in faces:
        if str(face) not in known:
            font_manager.fontManager.addfont(str(face))


def rc_parameters() -> dict[str, Any]:
    """The matplotlib settings every publication figure shares."""
    return {
        "font.family": "sans-serif",
        "font.sans-serif": [FONT_FAMILY, "DejaVu Sans"],
        "font.size": SIZE_LABEL,
        "axes.titlesize": SIZE_LABEL,
        "axes.labelsize": SIZE_LABEL,
        "axes.labelcolor": INK_SECONDARY,
        "axes.edgecolor": INK_SECONDARY,
        "axes.linewidth": AXIS_LINE_WIDTH,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "axes.grid.axis": "y",
        "axes.axisbelow": True,
        "grid.color": GRID,
        "grid.linewidth": 0.6,
        "xtick.labelsize": SIZE_TICK,
        "ytick.labelsize": SIZE_TICK,
        "xtick.color": INK_SECONDARY,
        "ytick.color": INK_SECONDARY,
        "xtick.major.width": AXIS_LINE_WIDTH,
        "ytick.major.width": AXIS_LINE_WIDTH,
        "xtick.major.size": 2.5,
        "ytick.major.size": 2.5,
        "legend.fontsize": SIZE_LEGEND,
        "legend.frameon": False,
        "legend.handlelength": 1.6,
        "legend.borderaxespad": 0.3,
        "lines.linewidth": LINE_WIDTH,
        "lines.markersize": MARKER_SIZE,
        "patch.linewidth": 0.0,
        "text.color": INK,
        "savefig.facecolor": "white",
        # A true minus sign in tick labels, in the figure's own face.
        "axes.unicode_minus": True,
    }


def apply(plt: Any) -> None:
    """Register the font and set the shared parameters on a pyplot module."""
    register_fonts()
    plt.rcParams.update(rc_parameters())


def panel_tag(axes: Any, letter: str) -> None:
    """Put a bold (a)-style tag at the top left of one panel, outside the plot area."""
    axes.text(
        -0.02,
        1.04,
        f"({letter})",
        transform=axes.transAxes,
        ha="right",
        va="bottom",
        fontsize=SIZE_PANEL,
        fontweight="bold",
        color=INK,
    )


def minus(text: str) -> str:
    """Replace hyphen-minus with a true minus sign, for numbers set in text."""
    return text.replace("-", "\N{MINUS SIGN}")


def save(figure: Any, path: Path, dpi: int = PUBLICATION_DPI) -> Path:
    """Write one figure at publication density, at exactly the size it was drawn.

    No ``bbox_inches="tight"``: it crops to the drawn content, so a figure laid out
    at one column's width would come out narrower and be scaled back up by
    ``\\linewidth`` -- changing every font size it was drawn at. Figures use
    ``layout="constrained"`` for their margins instead.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    options = {"quality": 92} if path.suffix.lower() in {".jpg", ".jpeg"} else {}
    figure.savefig(path, dpi=dpi, pil_kwargs=options)
    return path

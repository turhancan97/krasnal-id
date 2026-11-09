"""Generate the Doppeldwarf-306 logo assets as SVG with an outlined wordmark.

The mark is two identical gnomes, the second a colour-swapped twin standing behind
the first: a lookalike, which is what the dataset is about. It is drawn in the
corpus colours every figure uses (`src/krasnal_id/viz/figure_style.py`): deep
mustard, mustard and ink.

The wordmark is outlined from TeX Gyre Heros, which is vendored with the figure
style under the GUST Font License, so the logo rebuilds from this repository alone
and matches the figures' typeface. Edit the geometry or palette here and
regenerate; never patch the SVG by hand.

    uv run --with matplotlib --with cairosvg python docs/brand/doppeldwarf.py

cairosvg is optional: without it the SVGs are still written and the raster preview
is skipped. The older Krasnal-ID mark is produced by `generate.py` beside this file.
"""

from pathlib import Path

from matplotlib.font_manager import FontProperties
from matplotlib.ft2font import FT2Font
from matplotlib.path import Path as MPath
from matplotlib.textpath import TextPath

OUT = Path("docs/brand")
FONTS = Path("src/krasnal_id/viz/fonts")
BOLD = str(FONTS / "texgyreheros-bold.otf")
REGULAR = str(FONTS / "texgyreheros-regular.otf")

NAME = "Doppeldwarf-306"
TAGLINE = "Near-identical instance retrieval"

# Corpus colours, as in figure_style: CORPUS_ACCENT, CORPUS_FILL, INK, INK_SECONDARY.
DEEP = "#B87A12"
MUSTARD = "#EAB34D"

# Per theme: the front gnome's beard, the twin's beard, the wordmark, the tagline,
# and the background the mark sits on, which the ring around the front gnome takes
# so that the twin appears to stand behind it.
THEMES = {
    "light": {
        "beard": "#1F2328",
        "twin": "#57606A",
        "ink": "#1F2328",
        "tag": "#57606A",
        "ground": "#FFFFFF",
    },
    # GitHub's dark background, where the README shows this variant.
    "dark": {
        "beard": "#E6EDF3",
        "twin": "#8C959F",
        "ink": "#E6EDF3",
        "tag": "#8C959F",
        "ground": "#0D1117",
    },
}
PLATE = "#12181C"  # the dark plate behind the social preview

# One gnome on a 48-unit grid: beard first, so the brim and nose sit on top of it.
SHAPES = (
    ("path", 'd="M11 29 Q11 46 24 47.5 Q37 46 37 29 Z"', "beard"),
    ("path", 'd="M14.5 26 C16.5 17 20 10 25.5 3.5 C27.5 11 31 18 33.5 26 Z"', "hat"),
    ("rect", 'x="8.5" y="24.5" width="31" height="5.6" rx="2.8"', "hat"),
    ("circle", 'cx="24" cy="33" r="3.6"', "nose"),
)
# The pair's bounding box on that grid, as the paper's mark uses it.
MARK_X, MARK_Y, MARK_W, MARK_H = 6.5, 1.5, 51.5, 47.5
RING = 4.8


def gnome(dx: float, colours: dict[str, str], ring: str | None = None) -> str:
    """One gnome shifted right by `dx`; with `ring`, an outline in that colour first."""
    halo = (
        "".join(
            f'<{tag} {geometry} fill="{ring}" stroke="{ring}" stroke-width="{RING}" '
            'stroke-linejoin="round"/>'
            for tag, geometry, _ in SHAPES
        )
        if ring
        else ""
    )
    body = "".join(f'<{tag} {geometry} fill="{colours[part]}"/>' for tag, geometry, part in SHAPES)
    return f'<g transform="translate({dx:g} 0)">{halo}{body}</g>'


def mark(theme: dict[str, str]) -> str:
    """The pair, in mark coordinates (origin at the bounding box's corner)."""
    twin = gnome(17, {"hat": MUSTARD, "beard": theme["twin"], "nose": DEEP})
    front = gnome(0, {"hat": DEEP, "beard": theme["beard"], "nose": MUSTARD}, ring=theme["ground"])
    return f'<g transform="translate({-MARK_X:g} {-MARK_Y:g})">{twin}{front}</g>'


def _to_svg_path(path: MPath, dx: float, dy: float, scale: float) -> str:
    """Convert a matplotlib text path to SVG path data, flipping the y axis."""
    out: list[str] = []
    verts, codes = path.vertices, path.codes
    i = 0

    def pt(k: int) -> str:
        x, y = verts[k]
        return f"{x * scale + dx:.2f} {-y * scale + dy:.2f}"

    while i < len(verts):
        code = codes[i]
        if code == MPath.MOVETO:
            out.append(f"M{pt(i)}")
            i += 1
        elif code == MPath.LINETO:
            out.append(f"L{pt(i)}")
            i += 1
        elif code == MPath.CURVE3:
            out.append(f"Q{pt(i)} {pt(i + 1)}")
            i += 2
        elif code == MPath.CURVE4:
            out.append(f"C{pt(i)} {pt(i + 1)} {pt(i + 2)}")
            i += 3
        elif code == MPath.CLOSEPOLY:
            out.append("Z")
            i += 1
        else:
            i += 1
    return "".join(out)


_SIZE = 100.0


def _advance(text: str, font_file: str) -> float:
    font = FT2Font(font_file)
    font.set_size(_SIZE, 72)
    font.set_text(text)
    return font.get_width_height()[0] / 64.0


def outlined(
    text: str, size: float, x: float, baseline: float, fill: str, font_file: str
) -> tuple[str, float]:
    """Outlined path data for one run of text, and its advance width."""
    scale = size / _SIZE
    path = TextPath((0, 0), text, size=_SIZE, prop=FontProperties(fname=font_file))
    return (
        f'<path d="{_to_svg_path(path, x, baseline, scale)}" fill="{fill}"/>',
        _advance(text, font_file) * scale,
    )


def svg(width: float, height: float, body: str, title: str) -> str:
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width:.2f} {height:.2f}" '
        f'width="{width:.2f}" height="{height:.2f}" role="img" aria-label="{title}">'
        f"<title>{title}</title>{body}</svg>\n"
    )


def write(name: str, content: str) -> None:
    (OUT / name).write_text(content, encoding="utf-8")
    print(f"  {name}  {len(content):>6} bytes")


def horizontal(theme: dict[str, str]) -> str:
    """Mark on the left; the name and, beneath it, the tagline, block-centred on the mark."""
    gap, name_size, tag_size = 12.0, 26.0, 10.5
    x = MARK_W + gap
    name, name_w = outlined(NAME, name_size, x, 25.0, theme["ink"], BOLD)
    tag, tag_w = outlined(TAGLINE, tag_size, x + 0.6, 41.0, theme["tag"], REGULAR)
    width = x + max(name_w, tag_w) + 2
    return svg(width, MARK_H, mark(theme) + name + tag, NAME)


def stacked(theme: dict[str, str]) -> str:
    """Mark centred above the name, the tagline under it."""
    name_size, tag_size = 24.0, 9.5
    _, name_w = outlined(NAME, name_size, 0, 0, theme["ink"], BOLD)
    _, tag_w = outlined(TAGLINE, tag_size, 0, 0, theme["tag"], REGULAR)
    width = max(name_w, tag_w, MARK_W) + 4
    name, _ = outlined(NAME, name_size, (width - name_w) / 2, MARK_H + 30, theme["ink"], BOLD)
    tag, _ = outlined(TAGLINE, tag_size, (width - tag_w) / 2, MARK_H + 46, theme["tag"], REGULAR)
    art = f'<g transform="translate({(width - MARK_W) / 2:.2f} 0)">{mark(theme)}</g>'
    return svg(width, MARK_H + 52, art + name + tag, NAME)


def social_preview(width: int = 1280, height: int = 640) -> None:
    """The stacked dark lockup centred on the dark plate, for link previews."""
    try:
        import cairosvg
    except ImportError:
        print("  doppeldwarf-social-preview.png  skipped (install cairosvg)")
        return
    plate = dict(THEMES["dark"], ground=PLATE)
    lockup = stacked(plate)
    box = lockup.split('viewBox="')[1].split('"')[0].split()
    lw, lh = float(box[2]), float(box[3])
    factor = 560 / lw
    art = lockup.split("</title>")[1].replace("</svg>\n", "")
    x, y = (width - lw * factor) / 2, (height - lh * factor) / 2
    page = (
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" '
        f'width="{width}" height="{height}"><rect width="{width}" height="{height}" '
        f'fill="{PLATE}"/><g transform="translate({x:.2f} {y:.2f}) scale({factor:.4f})">'
        f"{art}</g></svg>"
    )
    target = OUT / "doppeldwarf-social-preview.png"
    cairosvg.svg2png(bytestring=page.encode(), write_to=str(target))
    print(f"  {target.name}  {target.stat().st_size:>6} bytes")


if __name__ == "__main__":
    for path in (BOLD, REGULAR):
        if not Path(path).is_file():
            raise SystemExit(f"font not found: {path}; run from the repository root")
    OUT.mkdir(parents=True, exist_ok=True)
    for name, theme in THEMES.items():
        write(f"doppeldwarf-mark-{name}.svg", svg(MARK_W, MARK_H, mark(theme), f"{NAME} mark"))
        write(f"doppeldwarf-lockup-horizontal-{name}.svg", horizontal(theme))
        write(f"doppeldwarf-lockup-stacked-{name}.svg", stacked(theme))
    social_preview()

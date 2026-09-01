"""Generate the Krasnal-ID logo assets as SVG with an outlined wordmark.

The wordmark is converted to outlines rather than left as live `<text>`, so the
files render identically everywhere and need no font installed. That makes the
path data unreadable by hand, which is why this generator is tracked: edit the
geometry or the palette here and regenerate, never patch the SVG by hand.

Requires IBM Plex Sans SemiBold as a TTF. Fetch the URL that Google Fonts serves
for `https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@600` and point
FONT_FILE at it:

    uv run --with matplotlib --with cairosvg python docs/brand/generate.py

cairosvg is optional: without it the SVGs are still written and the raster plates
are skipped.
"""

import os
from pathlib import Path

from matplotlib.font_manager import FontProperties
from matplotlib.ft2font import FT2Font
from matplotlib.path import Path as MPath
from matplotlib.textpath import TextPath

OUT = Path("docs/brand")
# IBM Plex Sans SemiBold. Override with KRASNAL_BRAND_FONT; see the module docstring.
FONT_FILE = os.environ.get("KRASNAL_BRAND_FONT", "plex600.ttf")

# The tagline is set in the regular weight; without it the plate simply omits it.
TAGLINE_FONT = os.environ.get("KRASNAL_BRAND_FONT_REGULAR", "plex400.ttf")
TAGLINE = "Fine-grained visual instance recognition"
TAGLINE_INK = "#8A9499"

if not Path(FONT_FILE).is_file():
    raise SystemExit(
        f"font file not found: {FONT_FILE}\n"
        "The wordmark is outlined from IBM Plex Sans SemiBold. Download the TTF that "
        "https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@600 points to, then either "
        "place it beside this script as plex600.ttf or set KRASNAL_BRAND_FONT to its path."
    )

# Brand palette, as supplied.
RED = "#cf4832"
BLUE = "#3a7eab"
GREY = "#d1d3d4"

# Dark-theme steps: the supplied blue is too dim against a dark ground, so it is
# lightened. The red holds as given.
RED_D = "#e05a45"
BLUE_D = "#6aa8cf"
INK = "#16242c"

# The mark: a gnome hat (crown plus brim) above a location point, on a 48 grid.
HAT = "M14.5 27C16.5 18 20 11 24 6C28 11 31.5 18 33.5 27Z"


def mark(red: str, blue: str) -> str:
    return (
        f'<path d="{HAT}" fill="{red}"/>'
        f'<rect x="8" y="26" width="32" height="6.6" rx="3.3" fill="{red}"/>'
        f'<circle cx="24" cy="38.6" r="5.2" fill="{blue}"/>'
    )


def _to_svg_path(path: MPath, dx: float, dy: float, scale: float) -> str:
    """Convert a matplotlib text path to SVG path data, flipping the y axis."""
    out: list[str] = []
    verts = path.vertices
    codes = path.codes
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
_fp = FontProperties(fname=FONT_FILE)


def _advance(text: str, font_file: str = FONT_FILE) -> float:
    font = FT2Font(font_file)
    font.set_size(_SIZE, 72)
    font.set_text(text)
    return font.get_width_height()[0] / 64.0


def outlined(
    text: str, size: float, x: float, baseline: float, fill: str, font_file: str = FONT_FILE
) -> tuple[str, float]:
    """Return outlined path data for one run of text, and its advance width."""
    scale = size / _SIZE
    path = TextPath((0, 0), text, size=_SIZE, prop=FontProperties(fname=font_file))
    body = f'<path d="{_to_svg_path(path, x, baseline, scale)}" fill="{fill}"/>'
    return body, _advance(text, font_file) * scale


def wordmark(x: float, baseline: float, size: float, ink: str, blue: str) -> tuple[str, float]:
    """Return outlined 'Krasnal-ID' path data and its advance width."""
    scale = size / _SIZE
    head = TextPath((0, 0), "Krasnal", size=_SIZE, prop=_fp)
    tail = TextPath((0, 0), "-ID", size=_SIZE, prop=_fp)
    offset = _advance("Krasnal") * scale
    body = (
        f'<path d="{_to_svg_path(head, x, baseline, scale)}" fill="{ink}"/>'
        f'<path d="{_to_svg_path(tail, x + offset, baseline, scale)}" fill="{blue}"/>'
    )
    return body, offset + _advance("-ID") * scale


def svg(width: float, height: float, body: str, title: str) -> str:
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width:g} {height:g}" '
        f'width="{width:g}" height="{height:g}" role="img" aria-label="{title}">'
        f"<title>{title}</title>{body}</svg>\n"
    )


def write(name: str, content: str) -> None:
    (OUT / name).write_text(content, encoding="utf-8")
    print(f"  {name}  {len(content):>5} bytes")


OUT.mkdir(parents=True, exist_ok=True)
for theme, red, blue, ink in (("light", RED, BLUE, INK), ("dark", RED_D, BLUE_D, GREY)):
    write(f"krasnal-mark-{theme}.svg", svg(48, 48, mark(red, blue), "Krasnal-ID mark"))

    # Horizontal lockup: wordmark cap-centred on the mark.
    size = 27.0
    text, width = wordmark(62, 34.5, size, ink, blue)
    body = f'<g transform="translate(0,1)">{mark(red, blue)}</g>{text}'
    write(f"krasnal-lockup-horizontal-{theme}.svg", svg(62 + width + 2, 50, body, "Krasnal-ID"))

    # Stacked lockup: mark centred over the wordmark, exact centring from the advance.
    size = 25.0
    _, width = wordmark(0, 0, size, ink, blue)
    total = max(width, 48)
    text, _ = wordmark((total - width) / 2, 80, size, ink, blue)
    body = f'<g transform="translate({(total - 48) / 2:.2f},0)">{mark(red, blue)}</g>{text}'
    write(f"krasnal-lockup-stacked-{theme}.svg", svg(total, 90, body, "Krasnal-ID"))


def render_plate(
    source: str,
    width: float,
    height: float,
    art_width: float,
    name: str,
    scale: int = 1,
    tagline: str | None = None,
) -> None:
    """Render one lockup centred on a dark plate, for places that need a raster."""
    try:
        import cairosvg
    except ImportError:
        print(f"  {name}.png  skipped (install cairosvg to render plates)")
        return

    svg_text = (OUT / f"{source}.svg").read_text(encoding="utf-8")
    art = svg_text.split("</title>")[1].replace("</svg>\n", "").replace("</svg>", "")
    box = svg_text.split('viewBox="')[1].split('"')[0].split()
    aw, ah = float(box[2]), float(box[3])

    factor = art_width / aw
    caption = ""
    caption_size = width * 0.0215
    # The tagline sits below the lockup, so the pair is centred as one block.
    block = ah * factor + (caption_size * 2.9 if tagline else 0)
    x, y = (width - aw * factor) / 2, (height - block) / 2
    if tagline:
        if not Path(TAGLINE_FONT).is_file():
            print(f"  {name}.png  tagline skipped (no {TAGLINE_FONT})")
        else:
            _, run = outlined(tagline, caption_size, 0, 0, TAGLINE_INK, TAGLINE_FONT)
            caption, _ = outlined(
                tagline,
                caption_size,
                (width - run) / 2,
                y + ah * factor + caption_size * 2.0,
                TAGLINE_INK,
                TAGLINE_FONT,
            )
    plate = (
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width:g} {height:g}" '
        f'width="{width:g}" height="{height:g}">'
        f'<rect width="{width:g}" height="{height:g}" fill="#12181C"/>'
        f'<g transform="translate({x:.2f},{y:.2f}) scale({factor:.4f})">{art}</g>{caption}</svg>'
    )
    cairosvg.svg2png(
        bytestring=plate.encode("utf-8"),
        write_to=str(OUT / f"{name}.png"),
        output_width=int(width * scale),
        output_height=int(height * scale),
    )
    size = (OUT / f"{name}.png").stat().st_size / 1024
    print(f"  {name}.png  {int(width * scale)}x{int(height * scale)}  {size:.0f} KB")


# GitHub renders the social preview at 1280x640.
render_plate("krasnal-lockup-stacked-dark", 1280, 640, 400, "social-preview", tagline=TAGLINE)
render_plate("krasnal-lockup-stacked-dark", 1280, 640, 430, "social-preview-plain")
render_plate("krasnal-lockup-stacked-dark", 720, 420, 300, "krasnal-lockup-stacked-dark", scale=2)

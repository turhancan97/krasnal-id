# Krasnal-ID brand assets

A red gnome hat above a location point: the subject and the mechanism in three shapes, minimal
enough to hold at 16 pixels.

| File | Use |
|---|---|
| `krasnal-mark-light.svg` / `krasnal-mark-dark.svg` | The mark alone — favicons, avatars, tight spaces. |
| `krasnal-lockup-horizontal-light.svg` / `-dark.svg` | Name to the right of the mark — headers, README, slides. |
| `krasnal-lockup-stacked-light.svg` / `-dark.svg` | Name below the mark — square spaces, title cards. |
| `social-preview.png` | 1280×640 card on a dark ground, for the GitHub social preview (Settings → Social preview). |
| `krasnal-lockup-stacked-dark.png` | The stacked lockup on a dark plate, 2×, for slides and anywhere SVG is not accepted. |

## Palette

| Role | Light | Dark |
|---|---|---|
| Hat | `#cf4832` | `#e05a45` |
| Point and “ID” | `#3a7eab` | `#6aa8cf` |
| Wordmark | `#16242c` | `#d1d3d4` |

The dark steps are lightened because `#3a7eab` sits close to the contrast floor on a dark ground.

## Rules

- Keep clear space of at least the hat brim's width on every side.
- Minimum size: 16px for the mark, 100px wide for the horizontal lockup.
- Do not recolour the hat, add effects, or place the mark over a busy photograph.

## Regenerating

The wordmark is outlined, not live text, so the files render identically everywhere without a font
installed. That also means the path data cannot be edited by hand — change `generate.py` and
regenerate instead.

    KRASNAL_BRAND_FONT=/path/to/IBMPlexSans-SemiBold.ttf \
      uv run --with matplotlib --with cairosvg python docs/brand/generate.py

The font is not vendored here. `cairosvg` is optional — without it the SVGs are still written and
the PNG plates are skipped.

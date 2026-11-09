# Brand assets

Two marks live here. **Doppeldwarf-306**, the current one, is used by the README and the website.
**Krasnal-ID**, the earlier one, is kept as an archive below.

## Doppeldwarf-306

Two identical gnomes, the second a colour-swapped twin standing behind the first: a lookalike,
which is what the dataset is about. Drawn in the corpus colours every figure uses.

| File | Use |
|---|---|
| `doppeldwarf-mark-light.svg` / `-dark.svg` | The mark alone: favicon, avatars, tight spaces. |
| `doppeldwarf-lockup-horizontal-light.svg` / `-dark.svg` | Name and tagline to the right of the mark: README, headers, slides. |
| `doppeldwarf-lockup-stacked-light.svg` / `-dark.svg` | Name and tagline below the mark: square spaces, title cards. |
| `doppeldwarf-social-preview.png` | 1280×640 card on a dark plate, for link previews and the GitHub social preview (Settings → Social preview). |

| Role | Light | Dark |
|---|---|---|
| Front hat, twin nose | `#B87A12` | `#B87A12` |
| Twin hat, front nose | `#EAB34D` | `#EAB34D` |
| Front beard, wordmark | `#1F2328` | `#E6EDF3` |
| Twin beard, tagline | `#57606A` | `#8C959F` |

Use the light files on light backgrounds and the dark files on dark ones; the README picks the right
one from the viewer's system theme. The gap that makes the twin read as standing behind is cut out
of the twin, not painted, so it is transparent and both variants work on any background colour.
The website's inline header mark paints that gap in the page's `--ground` variable instead, which
follows the theme.

- Keep clear space of at least one hat brim's width on every side.
- Minimum size: 16px tall for the mark, 160px wide for the horizontal lockup (the tagline needs it).
- Do not recolour the gnomes or separate the twin from the front gnome.

Regenerate with the vendored TeX Gyre Heros, so no font needs installing:

    uv run --with matplotlib --with cairosvg python docs/brand/doppeldwarf.py

## Krasnal-ID (archive)

A red gnome hat above a location point: the subject and the mechanism in three shapes, minimal
enough to hold at 16 pixels.

| File | Use |
|---|---|
| `krasnal-mark-light.svg` / `krasnal-mark-dark.svg` | The mark alone — favicons, avatars, tight spaces. |
| `krasnal-lockup-horizontal-light.svg` / `-dark.svg` | Name to the right of the mark — headers, README, slides. |
| `krasnal-lockup-stacked-light.svg` / `-dark.svg` | Name below the mark — square spaces, title cards. |
| `social-preview.png` | 1280×640 card on a dark ground with the tagline, for the GitHub social preview (Settings → Social preview). |
| `social-preview-plain.png` | The same card without the tagline. |
| `krasnal-lockup-stacked-dark.png` | The stacked lockup on a dark plate, 2×, for slides and anywhere SVG is not accepted. |

### Palette

| Role | Light | Dark |
|---|---|---|
| Hat | `#cf4832` | `#e05a45` |
| Point and “ID” | `#3a7eab` | `#6aa8cf` |
| Wordmark | `#16242c` | `#d1d3d4` |

The dark steps are lightened because `#3a7eab` sits close to the contrast floor on a dark ground.

### Rules

- Keep clear space of at least the hat brim's width on every side.
- Minimum size: 16px for the mark, 100px wide for the horizontal lockup.
- Do not recolour the hat, add effects, or place the mark over a busy photograph.

### Regenerating

The wordmark is outlined, not live text, so the files render identically everywhere without a font
installed. That also means the path data cannot be edited by hand — change `generate.py` and
regenerate instead.

    KRASNAL_BRAND_FONT=/path/to/IBMPlexSans-SemiBold.ttf \
      uv run --with matplotlib --with cairosvg python docs/brand/generate.py

The tagline is set in IBM Plex Sans Regular, so a second file is needed for it; without it the
plates render and the tagline is skipped. Neither font is vendored here. `cairosvg` is optional — without it the SVGs are still written and
the PNG plates are skipped.

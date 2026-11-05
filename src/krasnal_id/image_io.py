"""Turning a decoded file into the pixels a person looking at it would see.

A phone does not rotate the pixels of a portrait photograph. It stores them in
the sensor's landscape order and writes an EXIF `Orientation` tag saying how to
turn them for display. Every viewer honours that tag, so the photographer never
sees the problem; a model fed `Image.convert("RGB")` does not, and receives the
statue lying on its side.

That is why this lives in one place that both the extraction loop and
single-image retrieval call. `AGENTS.md` section 5.8 measures the field
photographs as a *difference* against Commons queries, so anything that degrades
only phone photographs would be reported as domain gap when it is a decoding bug.

Applying it to the references changes nothing: all 1,691 carry either
`Orientation = 1` or no tag at all (measured 2026-11-05: 1,657 and 34), because
Commons serves its derivatives with the rotation already applied. The transpose
is therefore an identity on every cached reference vector, and no result is
invalidated by it.
"""

from PIL import Image, ImageOps


def upright_rgb(image: Image.Image) -> Image.Image:
    """Return the image as RGB, rotated and mirrored as its EXIF tag says to display it."""
    return ImageOps.exif_transpose(image).convert("RGB")

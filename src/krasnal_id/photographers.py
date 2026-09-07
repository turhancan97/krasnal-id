"""Who took which photograph, shared by the experiments that ask.

Two experiments need the same primitives — the photographer-disjoint evaluation
and the geometric re-ranking sweep run under it — so they live here rather than
one importing the other. `krasnal_id.geometry` and `krasnal_id.statistics` exist
for the same reason.

The `author` string is Commons' own `Artist` field, verbatim and unnormalised.
One value credits a sculptor alongside the photographer and another is a username
rather than a legal name. Treating two spellings of one person as two people
splits their work and *understates* any leakage measured from it, so the strings
are compared exactly and callers report how many distinct values there were.
"""

from krasnal_id.models import DatasetManifest


def authors_by_image(manifest: DatasetManifest) -> dict[str, str]:
    """Map each image to the photographer credited for it."""
    return {image.image_id: image.author for image in manifest.images}


def photographers_by_class(manifest: DatasetManifest) -> dict[str, set[str]]:
    """Map each statue to the distinct photographers who documented it."""
    grouped: dict[str, set[str]] = {}
    for image in manifest.images:
        grouped.setdefault(image.dwarf_id, set()).add(image.author)
    return grouped


def disjoint_references(
    reference_image_ids: tuple[str, ...],
    authors: dict[str, str],
    photographer: str,
) -> tuple[str, ...]:
    """Return the references not taken by the query's own photographer."""
    return tuple(image_id for image_id in reference_image_ids if authors[image_id] != photographer)


def is_answerable(dwarf_id: str, by_class: dict[str, set[str]]) -> bool:
    """Say whether a statue can be identified without its own photographer.

    A statue documented by one person has no correct reference once that person is
    withheld. Its queries are then unanswerable rather than hard, and scoring them
    as failures would measure the dataset's coverage and report it as a model's
    weakness.
    """
    return len(by_class.get(dwarf_id, set())) >= 2

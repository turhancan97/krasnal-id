"""Turning manifest records into publishable rows.

Everything the export adds beyond the manifest lives here: the normalised licence
URL, the SPDX identifier, the per-file modification flag, and the pre-rendered
credit line. All of it is derived at export time rather than added to
`ImageRecord`, because anything that changes the staging chain invalidates the
manifest, the split, twelve result artifacts and the published demo — the trade
`AGENTS.md` section 5.9 already priced for camera metadata.
"""

import hashlib
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import unquote

from krasnal_id.data_pipeline.license_templates import basis_only
from krasnal_id.models import DatasetManifest, DwarfRecord, ImageRecord

# Commons short names to SPDX identifiers. A public-domain *label* has no SPDX
# identifier because it is not a licence and grants nothing, so it maps to None
# and the basis is carried by `license_template` instead.
SPDX_BY_LICENSE = {
    "CC BY-SA 4.0": "CC-BY-SA-4.0",
    "CC BY-SA 3.0": "CC-BY-SA-3.0",
    "CC BY-SA 3.0 pl": "CC-BY-SA-3.0-PL",
    "CC BY-SA 2.5": "CC-BY-SA-2.5",
    "CC BY-SA 2.0": "CC-BY-SA-2.0",
    "CC BY 4.0": "CC-BY-4.0",
    "CC BY 3.0": "CC-BY-3.0",
    "CC BY 2.0": "CC-BY-2.0",
    "CC0": "CC0-1.0",
    "Public domain": None,
}

MODIFICATION_NOTE = (
    "downscaled to at most 2000 px on the long edge and re-encoded; EXIF not preserved"
)


class RowError(ValueError):
    """Raised when a manifest record cannot be turned into a publishable row."""


def normalise_license_url(url: str) -> str:
    """Collapse the Commons licence-URL variants onto one form per licence.

    The manifest holds twelve distinct URLs for ten licences, because Commons
    supplies `by-sa/3.0` beside `by-sa/3.0/` and `by-sa/4.0/deed.en` beside
    `by-sa/4.0`. Left alone, a reader counting distinct URLs concludes there are
    twelve licences.
    """
    trimmed = url.split("?", 1)[0].rstrip("/")
    for suffix in ("/deed.en", "/deed", "/legalcode"):
        if trimmed.endswith(suffix):
            trimmed = trimmed[: -len(suffix)]
    return f"{trimmed.rstrip('/')}/"


def commons_filename(source_url: str) -> str:
    """Return the Commons file title, which is the licences' "title, if supplied"."""
    _, _, tail = source_url.rpartition("/")
    name = tail.removeprefix("File:")
    if not name:
        raise RowError(f"cannot read a Commons filename from {source_url}")
    return unquote(name).replace("_", " ")


def sha1_file(path: Path) -> str:
    """Hash one stored file the way Commons hashes its originals."""
    # SHA-1 because that is what Commons publishes; this is a comparison, not a
    # security boundary.
    digest = hashlib.sha1()
    try:
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
    except OSError as error:
        raise RowError(f"could not read {path}: {error}") from error
    return digest.hexdigest()


def is_modified(record: ImageRecord, stored_sha1: str | None) -> bool:
    """Say whether the stored copy differs from the Commons original.

    The fetcher only resizes what exceeds the long-side cap, so 153 of the 1,691
    files are byte-identical to their originals. Asserting modification over
    those would be a false statement in a rights field, and it would destroy the
    one signal telling a downstream user which files are exact copies.
    """
    if record.commons_sha1 is None or stored_sha1 is None:
        # Nothing to compare against. Say modified: the pipeline may have resized
        # it, and over-declaring a change is the safe direction for the reuser.
        return True
    return stored_sha1 != record.commons_sha1


def attribution_line(
    *,
    commons_file: str,
    author: str,
    license_name: str,
    license_url: str,
    source_url: str,
    modified: bool,
) -> str:
    """Render one ready-to-paste credit.

    Title, author, source, licence — and the modification, which the licences
    require to be indicated. Downstream compliance then costs a user one column
    rather than a rights analysis, which is the only form of attribution that
    survives contact with a research pipeline.
    """
    grant = (
        f"licensed {license_name} ({license_url})"
        if SPDX_BY_LICENSE.get(license_name, "") is not None
        else license_name
    )
    credit = f'"{commons_file}" by {author}, via Wikimedia Commons, {grant} — {source_url}'
    return f"{credit} — modified: {MODIFICATION_NOTE}" if modified else credit


@dataclass(frozen=True, slots=True)
class ImageRow:
    """One reference photograph, ready to write."""

    record: ImageRecord
    dwarf: DwarfRecord
    label: int
    commons_file: str
    license_url: str
    license_spdx: str | None
    license_template: str | None
    modified: bool
    modification: str
    attribution_text: str
    stored_sha256: str

    @property
    def image_path(self) -> str:
        """Return the path an extracted file keeps, traceable to its Commons page."""
        return f"{self.dwarf.dwarf_id}/{self.commons_file}"


def class_names(manifest: DatasetManifest) -> tuple[str, ...]:
    """Return the label vocabulary: every dwarf ID, sorted."""
    return tuple(sorted(dwarf.dwarf_id for dwarf in manifest.dwarfs))


def build_image_rows(
    manifest: DatasetManifest,
    templates: dict[str, tuple[str, ...]] | None = None,
) -> tuple[ImageRow, ...]:
    """Describe every reference photograph, verifying its bytes on the way.

    The stored digest is recomputed rather than trusted: these bytes are about to
    be published under a licence whose attribution names a specific photographer,
    so publishing bytes that disagree with the manifest's record of them would be
    unrecoverable after the fact.

    The bytes themselves are not retained. Holding the whole corpus would cost
    hundreds of megabytes of resident memory for no gain; the writer re-reads each
    file when it fills a row group, by which point its digest is already known
    good.
    """
    labels = {name: index for index, name in enumerate(class_names(manifest))}
    dwarfs = {dwarf.dwarf_id: dwarf for dwarf in manifest.dwarfs}
    basis = templates or {}

    rows: list[ImageRow] = []
    for record in sorted(manifest.images, key=lambda image: image.image_id):
        dwarf = dwarfs.get(record.dwarf_id)
        if dwarf is None:
            raise RowError(f"image {record.image_id} names no dwarf in the manifest")
        if not record.local_path.is_file():
            raise RowError(f"image {record.image_id} is missing: {record.local_path}")

        payload = record.local_path.read_bytes()
        stored_sha256 = hashlib.sha256(payload).hexdigest()
        if stored_sha256 != record.sha256:
            raise RowError(
                f"image {record.image_id} checksum mismatch: manifest records "
                f"{record.sha256[:12]}, the file on disk is {stored_sha256[:12]}"
            )

        commons_file = commons_filename(str(record.source_url))
        license_url = normalise_license_url(str(record.license_url))
        modified = is_modified(record, hashlib.sha1(payload).hexdigest())
        # Filtered again on read, so an artifact fetched before the formatting
        # templates were excluded does not need re-fetching to be clean.
        page_templates = basis_only(basis.get(str(record.commons_page_id or ""), ()))
        rows.append(
            ImageRow(
                record=record,
                dwarf=dwarf,
                label=labels[record.dwarf_id],
                commons_file=commons_file,
                license_url=license_url,
                license_spdx=SPDX_BY_LICENSE.get(record.license),
                license_template=", ".join(page_templates) or None,
                modified=modified,
                modification=MODIFICATION_NOTE if modified else "",
                attribution_text=attribution_line(
                    commons_file=commons_file,
                    author=record.author,
                    license_name=record.license,
                    license_url=license_url,
                    source_url=str(record.source_url),
                    modified=modified,
                ),
                stored_sha256=stored_sha256,
            )
        )
    return tuple(rows)

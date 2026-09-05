"""Build and validate the audited local dataset manifest."""

from __future__ import annotations

import hashlib
import json
import logging
import os
from datetime import UTC, datetime
from pathlib import Path
from statistics import median
from tempfile import NamedTemporaryFile
from typing import Final, cast

from pydantic import ValidationError

from krasnal_id.geometry import haversine_metres
from krasnal_id.models import (
    CategoryReviewFile,
    CategoryReviewRecord,
    CategoryReviewStatus,
    Coordinates,
    CoordinateSource,
    DatasetManifest,
    DwarfDiscoveryFile,
    DwarfRecord,
    FetchedImagesFile,
    ImageRecord,
    ImageReviewFile,
    ImageReviewStatus,
)

LOGGER = logging.getLogger(__name__)

SCHEMA_VERSION: Final = "1.0"


class ManifestConfigurationError(ValueError):
    """Raised when manifest input artifacts are missing or inconsistent."""


def canonical_json_sha256(payload: object) -> str:
    """Hash JSON independently of formatting or mapping key order."""
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def category_review_sha256(review: CategoryReviewFile) -> str:
    """Hash acquisition-relevant category review decisions."""
    payload = review.model_dump(mode="json")
    for record in payload["records"]:
        record.pop("display_name_override", None)
    return canonical_json_sha256(payload)


def place_dwarf(dwarf: DwarfRecord, images: tuple[ImageRecord, ...]) -> DwarfRecord:
    """Give a dwarf coordinates derived from its photographs, if it has none.

    A Wikidata `P625` statement is authoritative and always wins. Otherwise the
    position is the **median** of wherever its photographs were taken — the median
    rather than the mean because one mis-tagged photograph should not drag a class
    across the city, and a handful of points is exactly where that matters. The
    source is recorded either way, so a derived position is never mistaken for a
    stated one. See `AGENTS.md` section 5.7.
    """
    if dwarf.coordinates is not None:
        return dwarf.model_copy(update={"coordinate_source": CoordinateSource.WIKIDATA})
    placed = [image.coordinates for image in images if image.coordinates is not None]
    if not placed:
        return dwarf
    return dwarf.model_copy(
        update={
            "coordinates": Coordinates(
                latitude=median(point.latitude for point in placed),
                longitude=median(point.longitude for point in placed),
            ),
            "coordinate_source": CoordinateSource.COMMONS_CAMERA,
        }
    )


def reject_outlying_coordinates(
    dwarfs: tuple[DwarfRecord, ...],
    max_drift_km: float,
) -> tuple[tuple[DwarfRecord, ...], tuple[str, ...]]:
    """Drop derived coordinates too far from the dataset to be a real position.

    These are Wrocław statues, so a position tens of kilometres outside the city is
    a mis-tagged photograph rather than a statue — the observed case is a
    one-degree latitude typo that lands 111 km away, on a class whose only
    geotagged photograph carried it.

    Only *derived* positions are subject to this. A Wikidata `P625` statement is
    authoritative and is never second-guessed by a heuristic. The reference point
    is the median of the placed dwarves themselves rather than a hardcoded city
    centre, so the check needs no knowledge of where Wrocław is.
    """
    placed = [d for d in dwarfs if d.coordinates is not None]
    if len(placed) < 3:
        return dwarfs, ()

    centre = (
        median(d.coordinates.latitude for d in placed if d.coordinates),
        median(d.coordinates.longitude for d in placed if d.coordinates),
    )
    limit_metres = max_drift_km * 1000.0
    kept: list[DwarfRecord] = []
    dropped: list[str] = []
    for dwarf in dwarfs:
        if (
            dwarf.coordinates is not None
            and dwarf.coordinate_source is CoordinateSource.COMMONS_CAMERA
            and haversine_metres(centre, (dwarf.coordinates.latitude, dwarf.coordinates.longitude))
            > limit_metres
        ):
            dropped.append(dwarf.display_name)
            kept.append(dwarf.model_copy(update={"coordinates": None, "coordinate_source": None}))
            continue
        kept.append(dwarf)
    return tuple(kept), tuple(dropped)


def build_dataset_manifest(
    dwarfs: tuple[DwarfRecord, ...],
    images: tuple[ImageRecord, ...],
    generated_at: datetime,
    minimum_images_per_dwarf: int,
    *,
    source_query_sha256: str,
    staging_sha256: str,
    image_review_sha256: str,
    max_coordinate_drift_km: float = 25.0,
) -> DatasetManifest:
    """Build a manifest from validated records without filesystem access."""
    if minimum_images_per_dwarf < 1:
        raise ValueError("minimum_images_per_dwarf must be positive")

    dwarf_ids = [record.dwarf_id for record in dwarfs]
    if len(dwarf_ids) != len(set(dwarf_ids)):
        raise ValueError("dwarf IDs must be unique")

    image_ids = [record.image_id for record in images]
    if len(image_ids) != len(set(image_ids)):
        raise ValueError("image IDs must be unique")

    known_dwarf_ids = set(dwarf_ids)
    unknown_image_dwarfs = {
        record.dwarf_id for record in images if record.dwarf_id not in known_dwarf_ids
    }
    if unknown_image_dwarfs:
        unknown = ", ".join(sorted(unknown_image_dwarfs))
        raise ValueError(f"images reference unknown dwarf IDs: {unknown}")

    counts: dict[str, int] = {dwarf_id: 0 for dwarf_id in dwarf_ids}
    for image in images:
        counts[image.dwarf_id] += 1

    admitted_dwarf_ids = {
        dwarf_id for dwarf_id, count in counts.items() if count >= minimum_images_per_dwarf
    }
    admitted_images = tuple(image for image in images if image.dwarf_id in admitted_dwarf_ids)
    images_by_dwarf: dict[str, list[ImageRecord]] = {}
    for image in admitted_images:
        images_by_dwarf.setdefault(image.dwarf_id, []).append(image)
    admitted_dwarfs = tuple(
        place_dwarf(dwarf, tuple(images_by_dwarf.get(dwarf.dwarf_id, ())))
        for dwarf in dwarfs
        if dwarf.dwarf_id in admitted_dwarf_ids
    )
    admitted_dwarfs, misplaced = reject_outlying_coordinates(
        admitted_dwarfs, max_coordinate_drift_km
    )
    if misplaced:
        LOGGER.warning(
            "dropped %d derived coordinate(s) further than %.0f km from the dataset: %s",
            len(misplaced),
            max_coordinate_drift_km,
            ", ".join(misplaced),
        )

    return DatasetManifest(
        schema_version=SCHEMA_VERSION,
        source_query_sha256=source_query_sha256,
        staging_sha256=staging_sha256,
        image_review_sha256=image_review_sha256,
        generated_at=generated_at,
        minimum_images_per_dwarf=minimum_images_per_dwarf,
        dwarfs=admitted_dwarfs,
        images=admitted_images,
    )


def _read_json(path: Path) -> object:
    try:
        with path.open(encoding="utf-8") as handle:
            return cast(object, json.load(handle))
    except OSError as error:
        raise ManifestConfigurationError(f"required artifact is unreadable: {path}") from error
    except json.JSONDecodeError as error:
        raise ManifestConfigurationError(f"required artifact is invalid JSON: {path}") from error


def _validate_category_review(
    discovery: DwarfDiscoveryFile,
    review: CategoryReviewFile,
) -> dict[str, CategoryReviewRecord]:
    discovery_by_id = {record.dwarf_id: record for record in discovery.records}
    review_by_id = {record.dwarf_id: record for record in review.records}

    missing = sorted(set(discovery_by_id) - set(review_by_id))
    extra = sorted(set(review_by_id) - set(discovery_by_id))
    pending = sorted(
        dwarf_id
        for dwarf_id, record in review_by_id.items()
        if record.status is CategoryReviewStatus.PENDING
    )
    mismatched = sorted(
        dwarf_id
        for dwarf_id, dwarf in discovery_by_id.items()
        if dwarf_id in review_by_id
        and review_by_id[dwarf_id].discovered_category != dwarf.commons_category
    )
    problems: list[str] = []
    if missing:
        problems.append(f"missing category decisions: {', '.join(missing)}")
    if extra:
        problems.append(f"category decisions reference unknown dwarfs: {', '.join(extra)}")
    if pending:
        problems.append(f"pending category decisions: {', '.join(pending)}")
    if mismatched:
        problems.append(f"changed category mappings: {', '.join(mismatched)}")
    if problems:
        raise ManifestConfigurationError("; ".join(problems))

    return review_by_id


def build_manifest_from_artifacts(
    discovery_path: Path,
    fetched_images_path: Path,
    category_review_path: Path,
    image_review_path: Path,
    minimum_images_per_dwarf: int,
    *,
    generated_at: datetime | None = None,
    max_coordinate_drift_km: float = 25.0,
) -> DatasetManifest:
    """Load audited artifacts, validate provenance, and build the manifest."""
    discovery_raw = _read_json(discovery_path)
    fetched_raw = _read_json(fetched_images_path)
    category_raw = _read_json(category_review_path)
    image_review_raw = _read_json(image_review_path)

    try:
        discovery = DwarfDiscoveryFile.model_validate(discovery_raw)
        fetched = FetchedImagesFile.model_validate(fetched_raw)
        category_review = CategoryReviewFile.model_validate(category_raw)
        image_review = ImageReviewFile.model_validate(image_review_raw)
    except ValidationError as error:
        raise ManifestConfigurationError(
            "one or more manifest artifacts failed schema validation"
        ) from error

    if fetched.source_query_sha256 != discovery.query_sha256:
        raise ManifestConfigurationError(
            "fetched-images.json and dwarfs.json use different discovery query hashes"
        )
    if image_review.source_query_sha256 != discovery.query_sha256:
        raise ManifestConfigurationError(
            "image-review.json and dwarfs.json use different discovery query hashes"
        )
    acquisition_review_sha256 = category_review_sha256(category_review)
    if fetched.review_sha256 != acquisition_review_sha256:
        raise ManifestConfigurationError(
            "fetched-images.json does not match the current category-review.json hash"
        )

    staging_sha256 = canonical_json_sha256(fetched_raw)
    if image_review.staging_sha256 != staging_sha256:
        raise ManifestConfigurationError(
            "image-review.json does not match the current fetched-images.json staging hash"
        )
    image_review_sha256 = canonical_json_sha256(image_review_raw)
    review_by_id = _validate_category_review(discovery, category_review)

    discovery_ids = {record.dwarf_id for record in discovery.records}
    image_ids = [record.image_id for record in fetched.records]
    if len(image_ids) != len(set(image_ids)):
        raise ManifestConfigurationError("fetched-images.json contains duplicate image IDs")

    page_keys = [
        (record.dwarf_id, record.commons_page_id)
        for record in fetched.records
        if record.commons_page_id is not None
    ]
    if len(page_keys) != len(set(page_keys)):
        raise ManifestConfigurationError(
            "fetched-images.json contains duplicate dwarf/Commons-page keys"
        )
    if len(page_keys) != len(fetched.records):
        raise ManifestConfigurationError(
            "every fetched image must have a Commons page ID for image review"
        )

    review_keys = {(record.dwarf_id, record.commons_page_id) for record in image_review.records}
    staged_keys = set(page_keys)
    unknown_review_keys = sorted(review_keys - staged_keys)
    if unknown_review_keys:
        formatted = ", ".join(f"{dwarf_id}/{page_id}" for dwarf_id, page_id in unknown_review_keys)
        raise ManifestConfigurationError(
            f"image-review.json references unstaged images: {formatted}"
        )

    unknown_image_dwarfs = sorted({record.dwarf_id for record in fetched.records} - discovery_ids)
    if unknown_image_dwarfs:
        raise ManifestConfigurationError(
            "fetched-images.json references unknown dwarfs: " + ", ".join(unknown_image_dwarfs)
        )

    rejected_image_dwarfs = sorted(
        {
            record.dwarf_id
            for record in fetched.records
            if review_by_id[record.dwarf_id].status is not CategoryReviewStatus.APPROVED
        }
    )
    if rejected_image_dwarfs:
        raise ManifestConfigurationError(
            "fetched-images.json contains images for non-approved categories: "
            + ", ".join(rejected_image_dwarfs)
        )

    excluded_keys = {
        (record.dwarf_id, record.commons_page_id)
        for record in image_review.records
        if record.status is ImageReviewStatus.EXCLUDE
    }
    admitted_images = tuple(
        image
        for image in fetched.records
        if (image.dwarf_id, image.commons_page_id) not in excluded_keys
    )
    reviewed_dwarfs = tuple(
        dwarf.model_copy(
            update={"display_name": review_by_id[dwarf.dwarf_id].selected_display_name}
        )
        for dwarf in discovery.records
    )

    try:
        return build_dataset_manifest(
            reviewed_dwarfs,
            admitted_images,
            generated_at or datetime.now(UTC),
            minimum_images_per_dwarf,
            source_query_sha256=discovery.query_sha256,
            staging_sha256=staging_sha256,
            image_review_sha256=image_review_sha256,
            max_coordinate_drift_km=max_coordinate_drift_km,
        )
    except ValueError as error:
        raise ManifestConfigurationError(str(error)) from error


def write_dataset_manifest(path: Path, manifest: DatasetManifest) -> None:
    """Write a validated manifest atomically as formatted JSON."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    try:
        with NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temporary_path = Path(handle.name)
            json.dump(
                manifest.model_dump(mode="json"),
                handle,
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
            )
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        temporary_path.replace(path)
    except OSError as error:
        raise ManifestConfigurationError(f"could not write manifest: {path}") from error
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)

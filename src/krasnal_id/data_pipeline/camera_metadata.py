"""Commons EXIF camera models, used to ask which photographs are harder queries.

Kept outside the staging chain on purpose. Anything added to `fetched-images.json`
changes the staging hash and invalidates the manifest, the split, every result
artifact and the published demo; a camera model is read by one analysis and does
not build the dataset. `AGENTS.md` section 5.9 records that trade and when to
reverse it.
"""

import json
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field, HttpUrl, ValidationError

from krasnal_id.config import WikimediaDataConfig
from krasnal_id.models import DatasetManifest

CAMERA_METADATA_FILENAME = "camera-metadata.json"
SCHEMA_VERSION = "1.0"
# The API accepts fifty page IDs per request.
PAGE_ID_BATCH = 50


class CameraMetadataError(RuntimeError):
    """Raised when camera metadata cannot be retrieved or is unusable."""


class CameraMetadataFile(BaseModel):
    """Camera model per Commons page, with the provenance to detect staleness."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: str = Field(pattern=r"^\d+\.\d+$")
    endpoint: HttpUrl
    retrieved_at: datetime
    # Page IDs with no recorded camera map to None, so a missing key means "never
    # asked" while a null means "asked, and Commons has nothing".
    cameras: dict[str, str | None]


def camera_metadata_path(discovery_dir: Path) -> Path:
    """Return the artifact path for the camera metadata."""
    return discovery_dir / CAMERA_METADATA_FILENAME


def request_parameters(page_ids: tuple[int, ...]) -> dict[str, str]:
    """Build the parameters for one batch of page IDs."""
    return {
        "action": "query",
        "format": "json",
        "formatversion": "2",
        "prop": "imageinfo",
        "iiprop": "commonmetadata",
        "pageids": "|".join(str(page_id) for page_id in page_ids),
    }


def camera_from_page(page: object) -> tuple[int, str | None] | None:
    """Extract one page's camera description, joining make and model."""
    if not isinstance(page, dict) or "pageid" not in page:
        return None
    page_id = page["pageid"]
    if not isinstance(page_id, int):
        return None
    for info in page.get("imageinfo") or []:
        if not isinstance(info, dict):
            continue
        fields = {
            entry.get("name"): entry.get("value")
            for entry in info.get("commonmetadata") or []
            if isinstance(entry, dict)
        }
        make = str(fields.get("Make") or "").strip()
        model = str(fields.get("Model") or "").strip()
        combined = " ".join(part for part in (make, model) if part)
        return page_id, combined or None
    return page_id, None


def collect_cameras(payloads: tuple[object, ...]) -> dict[str, str | None]:
    """Fold every response batch into one page-to-camera mapping."""
    cameras: dict[str, str | None] = {}
    for payload in payloads:
        if not isinstance(payload, dict):
            raise CameraMetadataError("Commons returned a response that was not an object")
        if "error" in payload:
            raise CameraMetadataError(f"Commons API error: {payload['error']}")
        pages = (
            payload.get("query", {}).get("pages", [])
            if isinstance(payload.get("query"), dict)
            else []
        )
        for page in pages:
            extracted = camera_from_page(page)
            if extracted is not None:
                cameras[str(extracted[0])] = extracted[1]
    return cameras


def fetch_camera_metadata(
    manifest: DatasetManifest,
    config: WikimediaDataConfig,
    session: Callable[[dict[str, str]], object],
) -> CameraMetadataFile:
    """Retrieve the camera model behind every manifest image that has a page ID."""
    page_ids = tuple(
        sorted({image.commons_page_id for image in manifest.images if image.commons_page_id})
    )
    if not page_ids:
        raise CameraMetadataError("no manifest image carries a Commons page ID")

    payloads = tuple(
        session(request_parameters(page_ids[start : start + PAGE_ID_BATCH]))
        for start in range(0, len(page_ids), PAGE_ID_BATCH)
    )
    return CameraMetadataFile(
        schema_version=SCHEMA_VERSION,
        endpoint=config.commons_api_endpoint,
        retrieved_at=datetime.now(UTC),
        cameras=collect_cameras(payloads),
    )


def load_camera_metadata(path: Path) -> CameraMetadataFile:
    """Read a camera metadata artifact, refusing anything malformed."""
    try:
        return CameraMetadataFile.model_validate(json.loads(path.read_text(encoding="utf-8")))
    except (OSError, json.JSONDecodeError, ValidationError) as error:
        raise CameraMetadataError(
            f"invalid camera metadata {path}: {error}; rebuild it with "
            "krasnal-id data camera-metadata"
        ) from error

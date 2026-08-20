"""Fetch reviewed, licensed image metadata and files from Wikimedia Commons."""

from __future__ import annotations

import hashlib
import html
import json
import logging
import os
import re
import time
from collections import defaultdict
from collections.abc import Mapping
from contextlib import nullcontext
from dataclasses import dataclass
from datetime import UTC, datetime
from email.utils import parsedate_to_datetime
from html.parser import HTMLParser
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Final, cast
from urllib.parse import urlparse

import httpx
from PIL import Image, UnidentifiedImageError
from pydantic import BaseModel, ConfigDict, Field, HttpUrl, ValidationError

from krasnal_id.config import WikimediaDataConfig
from krasnal_id.data_pipeline.build_manifest import category_review_sha256
from krasnal_id.models import (
    AuditDisposition,
    CategoryReviewFile,
    CategoryReviewRecord,
    CategoryReviewStatus,
    CommonsCacheMetadata,
    DiscoveryAuditFile,
    DwarfDiscoveryFile,
    DwarfRecord,
    FetchAuditDisposition,
    FetchAuditFile,
    FetchAuditReason,
    FetchAuditRecord,
    FetchedImagesFile,
    FetchResult,
    ImageRecord,
)

LOGGER = logging.getLogger(__name__)

SCHEMA_VERSION: Final = "1.0"
USER_AGENT_ENV_VAR: Final = "KRASNAL_ID_USER_AGENT"
FETCHED_IMAGES_FILENAME: Final = "fetched-images.json"
FETCH_AUDIT_FILENAME: Final = "fetch-audit.json"
COMMONS_CACHE_DIRECTORY: Final = "commons"
RETRYABLE_STATUS_CODES: Final = frozenset({429, 502, 503, 504})
_CONTACT_PATTERN: Final = re.compile(
    r"(?:mailto:|https?://)[^\s)]+|[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}",
    re.IGNORECASE,
)
_MIME_EXTENSIONS: Final = {
    "image/gif": ".gif",
    "image/jpeg": ".jpg",
    "image/png": ".png",
    "image/tiff": ".tif",
    "image/webp": ".webp",
}
_PUBLIC_DOMAIN_URL: Final = "https://creativecommons.org/publicdomain/mark/1.0/"
_CC0_URL: Final = "https://creativecommons.org/publicdomain/zero/1.0/"


class CommonsConfigurationError(ValueError):
    """Raised when local input or fetch configuration is invalid."""


class CommonsFetchError(RuntimeError):
    """Raised when a Commons API request or image download fails."""


class _MetadataValue(BaseModel):
    model_config = ConfigDict(extra="ignore", frozen=True)
    value: str | int | float | bool | None


class _ImageInfo(BaseModel):
    model_config = ConfigDict(extra="ignore", frozen=True)
    timestamp: datetime
    sha1: str = Field(pattern=r"^[0-9a-z]{31,40}$")
    url: HttpUrl
    descriptionurl: HttpUrl
    thumburl: HttpUrl | None = None
    width: int = Field(gt=0)
    height: int = Field(gt=0)
    mime: str = Field(min_length=1)
    extmetadata: dict[str, _MetadataValue]


class _CommonsPage(BaseModel):
    model_config = ConfigDict(extra="ignore", frozen=True)
    pageid: int = Field(gt=0)
    ns: int
    title: str = Field(min_length=1)
    imageinfo: tuple[_ImageInfo, ...] = ()


class _CommonsQuery(BaseModel):
    model_config = ConfigDict(extra="ignore", frozen=True)
    pages: tuple[_CommonsPage, ...] = ()


class _CommonsResponse(BaseModel):
    model_config = ConfigDict(extra="ignore", frozen=True, populate_by_name=True)
    query: _CommonsQuery
    continuation: dict[str, str] | None = Field(default=None, alias="continue")


@dataclass(frozen=True)
class FetchPaths:
    """Filesystem paths owned by Commons acquisition."""

    fetched_images: Path
    audit: Path
    cache_directory: Path


@dataclass(frozen=True)
class CommonsCachePaths:
    """Raw-response and provenance files for one reviewed category."""

    response: Path
    metadata: Path


@dataclass(frozen=True)
class _Candidate:
    dwarf: DwarfRecord
    page_id: int
    title: str
    source_url: str
    download_url: str
    author: str
    license_name: str
    license_url: str
    commons_sha1: str
    source_revision_at: datetime
    extension: str


@dataclass
class _Counters:
    discovered: int = 0
    eligible: int = 0
    downloaded: int = 0
    reused: int = 0


class _PlainTextParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.parts: list[str] = []

    def handle_data(self, data: str) -> None:
        self.parts.append(data)

    def text(self) -> str:
        return " ".join(" ".join(self.parts).split())


class _NetworkSession:
    def __init__(self, config: WikimediaDataConfig, client: httpx.Client) -> None:
        self.config = config
        self.client = client
        self.user_agent: str | None = None

    def _headers(self, accept: str) -> dict[str, str]:
        if self.user_agent is None:
            self.user_agent = _contact_user_agent()
        return {"Accept": accept, "User-Agent": self.user_agent}

    def get(
        self,
        url: str,
        *,
        params: Mapping[str, str] | None = None,
        accept: str,
        resource: str,
    ) -> httpx.Response:
        for attempt in range(self.config.max_attempts):
            try:
                response = self.client.get(
                    url,
                    params=params,
                    headers=self._headers(accept),
                    timeout=self.config.request_timeout_seconds,
                )
            except (httpx.TimeoutException, httpx.TransportError) as error:
                if attempt + 1 >= self.config.max_attempts:
                    raise CommonsFetchError(
                        f"{resource} failed after {self.config.max_attempts} attempts."
                    ) from error
                time.sleep(self.config.retry_backoff_seconds[attempt])
                continue

            if (
                response.status_code in RETRYABLE_STATUS_CODES
                and attempt + 1 < self.config.max_attempts
            ):
                retry_after = _retry_after_seconds(response, self.config.max_retry_after_seconds)
                delay = (
                    retry_after
                    if retry_after is not None
                    else self.config.retry_backoff_seconds[attempt]
                )
                time.sleep(delay)
                continue

            try:
                response.raise_for_status()
            except httpx.HTTPStatusError as error:
                raise CommonsFetchError(
                    f"{resource} returned HTTP {response.status_code}."
                ) from error
            return response
        raise CommonsFetchError(f"{resource} exhausted its retry policy.")


def fetch_paths(discovery_dir: Path) -> FetchPaths:
    """Resolve generated Commons acquisition artifacts."""
    return FetchPaths(
        fetched_images=discovery_dir / FETCHED_IMAGES_FILENAME,
        audit=discovery_dir / FETCH_AUDIT_FILENAME,
        cache_directory=discovery_dir / COMMONS_CACHE_DIRECTORY,
    )


def commons_cache_paths(discovery_dir: Path, dwarf_id: str) -> CommonsCachePaths:
    """Resolve one dwarf's Commons category cache files."""
    directory = fetch_paths(discovery_dir).cache_directory
    return CommonsCachePaths(
        response=directory / f"{dwarf_id}.json",
        metadata=directory / f"{dwarf_id}.meta.json",
    )


def _qid_sort_key(dwarf_id: str) -> tuple[int, int | str]:
    if re.fullmatch(r"Q[1-9]\d*", dwarf_id):
        return (0, int(dwarf_id[1:]))
    return (1, dwarf_id)


def _atomic_write_json(path: Path, payload: object) -> None:
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
            json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        temporary_path.replace(path)
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


def _read_json(path: Path) -> object:
    with path.open(encoding="utf-8") as handle:
        return cast(object, json.load(handle))


def _load_discovery(discovery_dir: Path) -> tuple[DwarfDiscoveryFile, DiscoveryAuditFile]:
    try:
        discovery = DwarfDiscoveryFile.model_validate(_read_json(discovery_dir / "dwarfs.json"))
        audit = DiscoveryAuditFile.model_validate(_read_json(discovery_dir / "audit.json"))
    except (OSError, json.JSONDecodeError, ValidationError) as error:
        raise CommonsConfigurationError(
            "valid data/discovery/dwarfs.json and audit.json files are required; "
            "run krasnal-id data query first"
        ) from error
    if discovery.query_sha256 != audit.query_sha256:
        raise CommonsConfigurationError("discovery records and audit use different query hashes")
    return discovery, audit


def prepare_category_review(
    discovery_dir: Path,
    review_path: Path,
) -> CategoryReviewFile:
    """Create or update the tracked category-review file without network access."""
    discovery, audit = _load_discovery(discovery_dir)
    warning_ids = {
        record.dwarf_id
        for record in audit.records
        if record.disposition is AuditDisposition.WARNING
    }
    existing: CategoryReviewFile | None = None
    if review_path.exists():
        try:
            existing = CategoryReviewFile.model_validate(_read_json(review_path))
        except (OSError, json.JSONDecodeError, ValidationError) as error:
            raise CommonsConfigurationError(
                f"category review file is malformed: {review_path}"
            ) from error

    previous = {} if existing is None else {record.dwarf_id: record for record in existing.records}
    updated = dict(previous)
    for dwarf in discovery.records:
        prior = previous.get(dwarf.dwarf_id)
        if prior is not None and prior.discovered_category == dwarf.commons_category:
            updated[dwarf.dwarf_id] = prior.model_copy(
                update={
                    "display_name": dwarf.display_name,
                    "discovery_warning": dwarf.dwarf_id in warning_ids,
                }
            )
        else:
            updated[dwarf.dwarf_id] = CategoryReviewRecord(
                dwarf_id=dwarf.dwarf_id,
                display_name=dwarf.display_name,
                discovered_category=dwarf.commons_category,
                status=CategoryReviewStatus.PENDING,
                discovery_warning=dwarf.dwarf_id in warning_ids,
            )

    review = CategoryReviewFile(
        schema_version=SCHEMA_VERSION,
        records=tuple(sorted(updated.values(), key=lambda record: _qid_sort_key(record.dwarf_id))),
    )
    _atomic_write_json(review_path, review.model_dump(mode="json"))
    return review


def _load_review(
    review_path: Path,
    discovery: DwarfDiscoveryFile,
) -> tuple[CategoryReviewFile, tuple[CategoryReviewRecord, ...]]:
    try:
        review = CategoryReviewFile.model_validate(_read_json(review_path))
    except (OSError, json.JSONDecodeError, ValidationError) as error:
        raise CommonsConfigurationError(
            f"valid category review file required at {review_path}; "
            "run krasnal-id data fetch --prepare-review first"
        ) from error

    review_by_id = {record.dwarf_id: record for record in review.records}
    missing = [dwarf.dwarf_id for dwarf in discovery.records if dwarf.dwarf_id not in review_by_id]
    pending = [
        dwarf.dwarf_id
        for dwarf in discovery.records
        if dwarf.dwarf_id in review_by_id
        and review_by_id[dwarf.dwarf_id].status is CategoryReviewStatus.PENDING
    ]
    mismatched = [
        dwarf.dwarf_id
        for dwarf in discovery.records
        if dwarf.dwarf_id in review_by_id
        and review_by_id[dwarf.dwarf_id].discovered_category != dwarf.commons_category
    ]
    problems: list[str] = []
    if missing:
        problems.append(f"missing decisions: {', '.join(missing)}")
    if pending:
        problems.append(f"pending decisions: {', '.join(pending)}")
    if mismatched:
        problems.append(f"changed mappings: {', '.join(mismatched)}")
    if problems:
        raise CommonsConfigurationError(
            "; ".join(problems) + "; rerun --prepare-review and review the file"
        )
    selected = tuple(review_by_id[dwarf.dwarf_id] for dwarf in discovery.records)
    return review, selected


def _contact_user_agent() -> str:
    user_agent = os.environ.get(USER_AGENT_ENV_VAR, "").strip()
    if not user_agent or _CONTACT_PATTERN.search(user_agent) is None:
        raise CommonsConfigurationError(
            f"{USER_AGENT_ENV_VAR} must contain a contact email, mailto URI, or HTTP(S) URL."
        )
    return user_agent


def _retry_after_seconds(response: httpx.Response, maximum: float) -> float | None:
    value = response.headers.get("Retry-After")
    if value is None:
        return None
    try:
        delay = float(value)
    except ValueError:
        try:
            retry_at = parsedate_to_datetime(value)
            if retry_at.tzinfo is None:
                retry_at = retry_at.replace(tzinfo=UTC)
            delay = (retry_at - datetime.now(UTC)).total_seconds()
        except (TypeError, ValueError, OverflowError):
            return None
    return min(max(delay, 0.0), maximum)


def _request_parameters(config: WikimediaDataConfig, category: str) -> dict[str, str]:
    return {
        "action": "query",
        "format": "json",
        "formatversion": "2",
        "generator": "categorymembers",
        "gcmlimit": "max",
        "gcmnamespace": "6",
        "gcmtitle": f"Category:{category}",
        "gcmtype": "file",
        "iiprop": "timestamp|url|size|mime|sha1|extmetadata",
        "iiurlheight": str(config.image_max_long_side),
        "iiurlwidth": str(config.image_max_long_side),
        "prop": "imageinfo",
    }


def _request_hash(config: WikimediaDataConfig, category: str) -> str:
    serialized = json.dumps(
        {
            "endpoint": str(config.commons_api_endpoint),
            "parameters": _request_parameters(config, category),
        },
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(serialized.encode()).hexdigest()


def _parse_response(payload: object) -> _CommonsResponse:
    if isinstance(payload, dict) and "error" in payload:
        raise CommonsFetchError(f"Commons API returned an error: {payload['error']}")
    try:
        return _CommonsResponse.model_validate(payload)
    except ValidationError as error:
        raise CommonsFetchError(f"Commons API returned an invalid payload: {error}") from error


def _load_category_cache(
    paths: CommonsCachePaths,
    config: WikimediaDataConfig,
    category: str,
) -> tuple[_CommonsPage, ...] | None:
    try:
        metadata = CommonsCacheMetadata.model_validate(_read_json(paths.metadata))
        payloads = _read_json(paths.response)
        if not isinstance(payloads, list):
            return None
        responses = tuple(_parse_response(payload) for payload in payloads)
    except (OSError, json.JSONDecodeError, ValidationError, CommonsFetchError):
        return None
    if str(metadata.endpoint) != str(config.commons_api_endpoint):
        return None
    if metadata.category != category or metadata.request_sha256 != _request_hash(config, category):
        return None
    return tuple(page for response in responses for page in response.query.pages)


def _fetch_category(
    session: _NetworkSession,
    config: WikimediaDataConfig,
    category: str,
) -> tuple[tuple[_CommonsPage, ...], tuple[object, ...]]:
    params = _request_parameters(config, category)
    payloads: list[object] = []
    pages: list[_CommonsPage] = []
    continuation: dict[str, str] = {}
    while True:
        response = session.get(
            str(config.commons_api_endpoint),
            params={**params, **continuation},
            accept="application/json",
            resource=f"Commons category {category!r}",
        )
        try:
            payload = cast(object, response.json())
        except ValueError as error:
            raise CommonsFetchError("Commons API returned malformed JSON.") from error
        parsed = _parse_response(payload)
        payloads.append(payload)
        pages.extend(parsed.query.pages)
        if not parsed.continuation:
            break
        continuation = parsed.continuation
    return tuple(pages), tuple(payloads)


def _write_category_cache(
    paths: CommonsCachePaths,
    config: WikimediaDataConfig,
    category: str,
    payloads: tuple[object, ...],
) -> None:
    metadata = CommonsCacheMetadata(
        schema_version=SCHEMA_VERSION,
        endpoint=config.commons_api_endpoint,
        category=category,
        request_sha256=_request_hash(config, category),
        retrieved_at=datetime.now(UTC),
    )
    _atomic_write_json(paths.response, payloads)
    _atomic_write_json(paths.metadata, metadata.model_dump(mode="json"))


def _plain_text(value: str) -> str:
    parser = _PlainTextParser()
    parser.feed(html.unescape(value))
    parser.close()
    return parser.text()


def _metadata_value(info: _ImageInfo, key: str) -> str:
    metadata = info.extmetadata.get(key)
    if metadata is None or not isinstance(metadata.value, str):
        return ""
    return metadata.value.strip()


def _license_family(token: str, short_name: str) -> str | None:
    normalized = f"{token} {short_name}".lower().replace("_", "-")
    if any(marker in normalized for marker in ("-nc", " nc", "-nd", " nd")):
        return None
    if re.search(r"\bcc[- ]?by[- ]?sa\b", normalized):
        return "cc-by-sa"
    if re.search(r"\bcc[- ]?by\b", normalized):
        return "cc-by"
    if "cc0" in normalized or "cc-zero" in normalized:
        return "cc0"
    if token.lower() == "pd" or "public domain" in normalized:
        return "public-domain"
    return None


def _license_url(info: _ImageInfo, family: str) -> str | None:
    raw_url = _metadata_value(info, "LicenseUrl")
    if not raw_url:
        if family == "public-domain":
            return _PUBLIC_DOMAIN_URL
        if family == "cc0":
            return _CC0_URL
        return None
    parsed = urlparse(raw_url)
    if parsed.scheme not in {"http", "https"} or parsed.hostname is None:
        return None
    if not parsed.hostname.lower().endswith("creativecommons.org"):
        return None
    return raw_url.replace("http://", "https://", 1)


def _trusted_wikimedia_url(value: str) -> bool:
    parsed = urlparse(value)
    hostname = (parsed.hostname or "").lower()
    return parsed.scheme == "https" and (
        hostname == "wikimedia.org" or hostname.endswith(".wikimedia.org")
    )


def _audit(
    dwarf_id: str,
    disposition: FetchAuditDisposition,
    reason: FetchAuditReason,
    details: str,
    page: _CommonsPage | None = None,
) -> FetchAuditRecord:
    return FetchAuditRecord(
        dwarf_id=dwarf_id,
        disposition=disposition,
        reason=reason,
        details=details,
        commons_page_id=None if page is None else page.pageid,
        commons_title=None if page is None else page.title,
    )


def _normalize_page(
    dwarf: DwarfRecord,
    page: _CommonsPage,
    config: WikimediaDataConfig,
) -> tuple[_Candidate | None, FetchAuditRecord | None]:
    if page.ns != 6 or len(page.imageinfo) != 1:
        return None, _audit(
            dwarf.dwarf_id,
            FetchAuditDisposition.EXCLUDED,
            FetchAuditReason.INVALID_METADATA,
            "File page must contain exactly one imageinfo record.",
            page,
        )
    info = page.imageinfo[0]
    extension = _MIME_EXTENSIONS.get(info.mime.lower())
    if extension is None:
        return None, _audit(
            dwarf.dwarf_id,
            FetchAuditDisposition.EXCLUDED,
            FetchAuditReason.UNSUPPORTED_MEDIA,
            f"Unsupported MIME type: {info.mime}.",
            page,
        )
    if min(info.width, info.height) < config.image_min_short_side:
        return None, _audit(
            dwarf.dwarf_id,
            FetchAuditDisposition.EXCLUDED,
            FetchAuditReason.IMAGE_TOO_SMALL,
            f"Original dimensions {info.width}x{info.height} are below the short-side minimum.",
            page,
        )
    author = _plain_text(_metadata_value(info, "Artist"))
    license_name = _plain_text(_metadata_value(info, "LicenseShortName"))
    token = _metadata_value(info, "License")
    if not author or not license_name:
        return None, _audit(
            dwarf.dwarf_id,
            FetchAuditDisposition.EXCLUDED,
            FetchAuditReason.MISSING_ATTRIBUTION,
            "Artist and LicenseShortName metadata are required.",
            page,
        )
    family = _license_family(token, license_name)
    license_url = None if family is None else _license_url(info, family)
    if family is None or family not in config.allowed_license_families or license_url is None:
        return None, _audit(
            dwarf.dwarf_id,
            FetchAuditDisposition.EXCLUDED,
            FetchAuditReason.UNSUPPORTED_LICENSE,
            f"Unsupported or incomplete license metadata: {license_name!r}.",
            page,
        )
    source_url = str(info.descriptionurl)
    download_url = str(info.thumburl or info.url)
    if not _trusted_wikimedia_url(source_url) or not _trusted_wikimedia_url(download_url):
        return None, _audit(
            dwarf.dwarf_id,
            FetchAuditDisposition.EXCLUDED,
            FetchAuditReason.INVALID_METADATA,
            "Commons source and download URLs must use trusted HTTPS Wikimedia hosts.",
            page,
        )
    return (
        _Candidate(
            dwarf=dwarf,
            page_id=page.pageid,
            title=page.title,
            source_url=source_url,
            download_url=download_url,
            author=author,
            license_name=license_name,
            license_url=license_url,
            commons_sha1=info.sha1,
            source_revision_at=info.timestamp,
            extension=extension,
        ),
        None,
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _inspect_image(path: Path, config: WikimediaDataConfig) -> tuple[int, int]:
    try:
        with Image.open(path) as image:
            if getattr(image, "n_frames", 1) != 1:
                raise CommonsFetchError("animated images are not supported")
            image.verify()
        with Image.open(path) as image:
            width, height = image.size
    except (OSError, SyntaxError, UnidentifiedImageError) as error:
        raise CommonsFetchError("downloaded content is not a valid static image") from error
    if min(width, height) < config.image_min_short_side:
        raise CommonsFetchError(
            f"stored dimensions {width}x{height} are below the short-side minimum"
        )
    if max(width, height) > config.image_max_long_side:
        raise CommonsFetchError(f"stored dimensions {width}x{height} exceed the long-side maximum")
    return width, height


def _resize_oversized_image(path: Path, config: WikimediaDataConfig) -> tuple[int, int]:
    """Downscale one verified static image to the configured bounding square."""
    try:
        with Image.open(path) as image:
            if getattr(image, "n_frames", 1) != 1:
                raise CommonsFetchError("animated images are not supported")
            image.load()
            width, height = image.size
            if min(width, height) < config.image_min_short_side:
                raise CommonsFetchError(
                    f"stored dimensions {width}x{height} are below the short-side minimum"
                )
            if max(width, height) <= config.image_max_long_side:
                return width, height
            image_format = image.format
            if image_format is None:
                raise CommonsFetchError("downloaded image format could not be determined")
            resized = image.copy()
        try:
            resized.thumbnail(
                (config.image_max_long_side, config.image_max_long_side),
                Image.Resampling.LANCZOS,
                reducing_gap=3.0,
            )
            with path.open("wb") as handle:
                resized.save(handle, format=image_format)
                handle.flush()
                os.fsync(handle.fileno())
        finally:
            resized.close()
    except CommonsFetchError:
        raise
    except (OSError, SyntaxError, UnidentifiedImageError) as error:
        raise CommonsFetchError("downloaded content is not a valid static image") from error
    return _inspect_image(path, config)


def _can_reuse(
    candidate: _Candidate,
    expected_path: Path,
    previous: ImageRecord | None,
    config: WikimediaDataConfig,
) -> bool:
    if previous is None or previous.commons_page_id != candidate.page_id:
        return False
    if previous.commons_sha1 != candidate.commons_sha1:
        return False
    if previous.source_revision_at != candidate.source_revision_at:
        return False
    if previous.local_path != expected_path or not expected_path.is_file():
        return False
    try:
        dimensions = _inspect_image(expected_path, config)
        checksum = _sha256(expected_path)
    except (OSError, CommonsFetchError):
        return False
    return checksum == previous.sha256 and dimensions == (previous.width, previous.height)


def _download_candidate(
    candidate: _Candidate,
    destination: Path,
    session: _NetworkSession,
    config: WikimediaDataConfig,
) -> ImageRecord:
    destination.parent.mkdir(parents=True, exist_ok=True)
    response = session.get(
        candidate.download_url,
        accept="image/*",
        resource=f"Commons file {candidate.title!r}",
    )
    temporary_path: Path | None = None
    try:
        with NamedTemporaryFile(
            mode="wb",
            dir=destination.parent,
            prefix=f".{destination.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temporary_path = Path(handle.name)
            handle.write(response.content)
            handle.flush()
            os.fsync(handle.fileno())
        width, height = _resize_oversized_image(temporary_path, config)
        checksum = _sha256(temporary_path)
        temporary_path.replace(destination)
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)
    return ImageRecord(
        image_id=f"commons-{candidate.page_id}",
        dwarf_id=candidate.dwarf.dwarf_id,
        local_path=destination,
        source_url=cast(HttpUrl, candidate.source_url),
        author=candidate.author,
        license=candidate.license_name,
        license_url=cast(HttpUrl, candidate.license_url),
        sha256=checksum,
        width=width,
        height=height,
        acquired_at=datetime.now(UTC),
        commons_page_id=candidate.page_id,
        commons_sha1=candidate.commons_sha1,
        source_revision_at=candidate.source_revision_at,
    )


def _load_previous_records(path: Path) -> dict[tuple[str, int], ImageRecord]:
    try:
        staged = FetchedImagesFile.model_validate(_read_json(path))
    except (OSError, json.JSONDecodeError, ValidationError):
        return {}
    return {
        (record.dwarf_id, record.commons_page_id): record
        for record in staged.records
        if record.commons_page_id is not None
    }


def _review_hash(review: CategoryReviewFile) -> str:
    return category_review_sha256(review)


def _audit_sort_key(
    record: FetchAuditRecord,
) -> tuple[tuple[int, int | str], int, int, str, str]:
    dispositions = {
        FetchAuditDisposition.EXCLUDED: 0,
        FetchAuditDisposition.WARNING: 1,
        FetchAuditDisposition.ERROR: 2,
    }
    return (
        _qid_sort_key(record.dwarf_id),
        record.commons_page_id or 0,
        dispositions[record.disposition],
        record.reason.value,
        record.details,
    )


def _deduplicate(
    images: list[ImageRecord],
    audit: list[FetchAuditRecord],
) -> tuple[ImageRecord, ...]:
    by_checksum: dict[str, list[ImageRecord]] = defaultdict(list)
    for record in images:
        by_checksum[record.sha256].append(record)
    retained: list[ImageRecord] = []
    for records in by_checksum.values():
        ordered = sorted(
            records,
            key=lambda record: (_qid_sort_key(record.dwarf_id), record.commons_page_id or 0),
        )
        dwarf_ids = {record.dwarf_id for record in ordered}
        if len(dwarf_ids) > 1:
            labels = ", ".join(sorted(dwarf_ids, key=_qid_sort_key))
            for record in ordered:
                audit.append(
                    FetchAuditRecord(
                        dwarf_id=record.dwarf_id,
                        disposition=FetchAuditDisposition.EXCLUDED,
                        reason=FetchAuditReason.CROSS_LABEL_DUPLICATE,
                        details=f"Identical content occurs under multiple labels: {labels}.",
                        commons_page_id=record.commons_page_id,
                    )
                )
            continue
        retained.append(ordered[0])
        for record in ordered[1:]:
            audit.append(
                FetchAuditRecord(
                    dwarf_id=record.dwarf_id,
                    disposition=FetchAuditDisposition.EXCLUDED,
                    reason=FetchAuditReason.SAME_LABEL_DUPLICATE,
                    details=f"Identical content retained as {ordered[0].image_id}.",
                    commons_page_id=record.commons_page_id,
                )
            )
    return tuple(
        sorted(
            retained,
            key=lambda record: (_qid_sort_key(record.dwarf_id), record.commons_page_id or 0),
        )
    )


def fetch_images(
    discovery_dir: Path,
    review_path: Path,
    images_dir: Path,
    config: WikimediaDataConfig,
    *,
    max_images_per_dwarf: int | None = None,
    refresh: bool = False,
    client: httpx.Client | None = None,
) -> FetchResult:
    """Fetch reviewed Commons files into deterministic local staging artifacts."""
    if max_images_per_dwarf is not None and max_images_per_dwarf <= 0:
        raise CommonsConfigurationError("max_images_per_dwarf must be a positive integer")
    discovery, _ = _load_discovery(discovery_dir)
    review, selected_reviews = _load_review(review_path, discovery)
    review_by_id = {record.dwarf_id: record for record in selected_reviews}
    approved = [
        dwarf
        for dwarf in discovery.records
        if review_by_id[dwarf.dwarf_id].status is CategoryReviewStatus.APPROVED
    ]
    rejected = [
        dwarf
        for dwarf in discovery.records
        if review_by_id[dwarf.dwarf_id].status is CategoryReviewStatus.REJECTED
    ]
    paths = fetch_paths(discovery_dir)
    previous = _load_previous_records(paths.fetched_images)
    counters = _Counters()
    audit = [
        _audit(
            dwarf.dwarf_id,
            FetchAuditDisposition.EXCLUDED,
            FetchAuditReason.REJECTED_CATEGORY,
            "Category mapping was rejected during human review.",
        )
        for dwarf in rejected
    ]
    acquired: list[ImageRecord] = []

    owned_client = None if client is not None else httpx.Client()
    context = nullcontext(client) if client is not None else cast(httpx.Client, owned_client)
    try:
        with context as active_client:
            session = _NetworkSession(config, active_client)
            for dwarf in approved:
                review_record = review_by_id[dwarf.dwarf_id]
                category = review_record.selected_category
                cache = commons_cache_paths(discovery_dir, dwarf.dwarf_id)
                cached_pages = None if refresh else _load_category_cache(cache, config, category)
                try:
                    if cached_pages is None:
                        pages, payloads = _fetch_category(session, config, category)
                        _write_category_cache(cache, config, category, payloads)
                    else:
                        pages = cached_pages
                except (CommonsFetchError, OSError) as error:
                    audit.append(
                        _audit(
                            dwarf.dwarf_id,
                            FetchAuditDisposition.ERROR,
                            FetchAuditReason.API_FAILURE,
                            str(error),
                        )
                    )
                    continue

                ordered_pages = sorted(pages, key=lambda page: page.pageid)
                counters.discovered += len(ordered_pages)
                candidates: list[_Candidate] = []
                for page in ordered_pages:
                    candidate, exclusion = _normalize_page(dwarf, page, config)
                    if exclusion is not None:
                        audit.append(exclusion)
                    elif candidate is not None:
                        candidates.append(candidate)
                counters.eligible += len(candidates)
                if max_images_per_dwarf is not None:
                    candidates = candidates[:max_images_per_dwarf]

                for candidate in candidates:
                    destination = (
                        images_dir
                        / dwarf.dwarf_id
                        / f"commons-{candidate.page_id}{candidate.extension}"
                    )
                    old_record = previous.get((dwarf.dwarf_id, candidate.page_id))
                    if _can_reuse(candidate, destination, old_record, config):
                        acquired.append(cast(ImageRecord, old_record))
                        counters.reused += 1
                        continue
                    try:
                        acquired.append(
                            _download_candidate(candidate, destination, session, config)
                        )
                        counters.downloaded += 1
                    except (CommonsFetchError, OSError) as error:
                        audit.append(
                            FetchAuditRecord(
                                dwarf_id=dwarf.dwarf_id,
                                disposition=FetchAuditDisposition.ERROR,
                                reason=FetchAuditReason.DOWNLOAD_FAILURE,
                                details=str(error),
                                commons_page_id=candidate.page_id,
                                commons_title=candidate.title,
                            )
                        )
    finally:
        if owned_client is not None and not owned_client.is_closed:
            owned_client.close()

    images = _deduplicate(acquired, audit)
    ordered_audit = tuple(sorted(audit, key=_audit_sort_key))
    review_sha256 = _review_hash(review)
    staged = FetchedImagesFile(
        schema_version=SCHEMA_VERSION,
        source_query_sha256=discovery.query_sha256,
        review_sha256=review_sha256,
        max_images_per_dwarf=max_images_per_dwarf,
        records=images,
    )
    audit_file = FetchAuditFile(
        schema_version=SCHEMA_VERSION,
        source_query_sha256=discovery.query_sha256,
        review_sha256=review_sha256,
        records=ordered_audit,
    )
    _atomic_write_json(paths.fetched_images, staged.model_dump(mode="json"))
    _atomic_write_json(paths.audit, audit_file.model_dump(mode="json"))
    operational_failures = sum(
        record.disposition is FetchAuditDisposition.ERROR for record in ordered_audit
    )
    result = FetchResult(
        images=images,
        audit=ordered_audit,
        approved_categories=len(approved),
        rejected_categories=len(rejected),
        pending_categories=0,
        discovered_images=counters.discovered,
        eligible_images=counters.eligible,
        downloaded_images=counters.downloaded,
        reused_images=counters.reused,
        operational_failures=operational_failures,
    )
    LOGGER.info(
        "Commons acquisition completed",
        extra={
            "approved_categories": result.approved_categories,
            "downloaded_images": result.downloaded_images,
            "operational_failures": result.operational_failures,
            "reused_images": result.reused_images,
        },
    )
    return result

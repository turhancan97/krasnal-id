"""Validated data contracts shared by acquisition and experiments."""

from datetime import datetime
from enum import StrEnum
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, HttpUrl, model_validator


class Coordinates(BaseModel):
    """Geographic coordinates in decimal degrees."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    latitude: float = Field(ge=-90.0, le=90.0)
    longitude: float = Field(ge=-180.0, le=180.0)


class DwarfRecord(BaseModel):
    """Stable identity and source metadata for one dwarf statue."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    dwarf_id: str = Field(min_length=1)
    display_name: str = Field(min_length=1)
    wikidata_url: HttpUrl
    commons_category: str = Field(min_length=1)
    coordinates: Coordinates | None = None


class ImageRecord(BaseModel):
    """One locally cached image with mandatory attribution metadata."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    image_id: str = Field(min_length=1)
    dwarf_id: str = Field(min_length=1)
    local_path: Path
    source_url: HttpUrl
    author: str = Field(min_length=1)
    license: str = Field(min_length=1)
    license_url: HttpUrl
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    width: int = Field(gt=0)
    height: int = Field(gt=0)
    acquired_at: datetime
    commons_page_id: int | None = Field(default=None, gt=0)
    commons_sha1: str | None = Field(default=None, pattern=r"^[0-9a-z]{31,40}$")
    source_revision_at: datetime | None = None


class DatasetManifest(BaseModel):
    """Versioned collection of dwarf identities and licensed image records."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: str = Field(pattern=r"^\d+\.\d+$")
    generated_at: datetime
    minimum_images_per_dwarf: int = Field(ge=1)
    dwarfs: tuple[DwarfRecord, ...]
    images: tuple[ImageRecord, ...]

    @model_validator(mode="after")
    def validate_references_and_ids(self) -> "DatasetManifest":
        """Require unique IDs and ensure every image points to a known dwarf."""
        dwarf_ids = [record.dwarf_id for record in self.dwarfs]
        if len(dwarf_ids) != len(set(dwarf_ids)):
            raise ValueError("dwarf IDs must be unique")

        image_ids = [record.image_id for record in self.images]
        if len(image_ids) != len(set(image_ids)):
            raise ValueError("image IDs must be unique")

        unknown_ids = {record.dwarf_id for record in self.images}.difference(dwarf_ids)
        if unknown_ids:
            unknown = ", ".join(sorted(unknown_ids))
            raise ValueError(f"images reference unknown dwarf IDs: {unknown}")
        return self


class AuditDisposition(StrEnum):
    """Whether a discovery audit entry blocks inclusion."""

    EXCLUDED = "excluded"
    WARNING = "warning"


class AuditReason(StrEnum):
    """Stable reason codes for Wikidata discovery decisions."""

    MISSING_COMMONS_CATEGORY = "missing_commons_category"
    EXPLICIT_GROUP_ENTITY = "explicit_group_entity"
    CONFLICTING_SOURCE_VALUES = "conflicting_source_values"
    INVALID_RECORD = "invalid_record"
    POSSIBLE_UNLINKED_GROUP = "possible_unlinked_group"


class DiscoveryAuditRecord(BaseModel):
    """One reviewable inclusion warning or exclusion decision."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    dwarf_id: str = Field(min_length=1)
    disposition: AuditDisposition
    reason: AuditReason
    details: str = Field(min_length=1)


class DwarfDiscoveryFile(BaseModel):
    """Deterministic staging artifact consumed by later data stages."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: str = Field(pattern=r"^\d+\.\d+$")
    query_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    selection_limit: int | None = Field(default=None, gt=0)
    eligible_total: int = Field(ge=0)
    records: tuple[DwarfRecord, ...]


class DiscoveryAuditFile(BaseModel):
    """Deterministic exclusions and warnings from one normalization pass."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: str = Field(pattern=r"^\d+\.\d+$")
    query_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    records: tuple[DiscoveryAuditRecord, ...]


class QueryCacheMetadata(BaseModel):
    """Provenance used to validate a cached raw SPARQL response."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: str = Field(pattern=r"^\d+\.\d+$")
    endpoint: HttpUrl
    query_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    retrieved_at: datetime


class DiscoveryResult(BaseModel):
    """In-memory result returned by the Wikidata query stage."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    records: tuple[DwarfRecord, ...]
    audit: tuple[DiscoveryAuditRecord, ...]
    eligible_total: int = Field(ge=0)
    cache_status: Literal["hit", "fetched", "refreshed", "recovered"]


class CategoryReviewStatus(StrEnum):
    """Human decision for one Wikidata-to-Commons category mapping."""

    PENDING = "pending"
    APPROVED = "approved"
    REJECTED = "rejected"


class CategoryReviewRecord(BaseModel):
    """Review state for one discovered dwarf and its Commons category."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    dwarf_id: str = Field(pattern=r"^Q[1-9]\d*$")
    display_name: str = Field(min_length=1)
    discovered_category: str = Field(min_length=1)
    status: CategoryReviewStatus = CategoryReviewStatus.PENDING
    corrected_category: str | None = Field(default=None, min_length=1)
    notes: str | None = Field(default=None, min_length=1)
    discovery_warning: bool = False

    @property
    def selected_category(self) -> str:
        """Return the reviewed category used for Commons requests."""
        return self.corrected_category or self.discovered_category


class CategoryReviewFile(BaseModel):
    """Tracked human review decisions for discovered Commons mappings."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: str = Field(pattern=r"^\d+\.\d+$")
    records: tuple[CategoryReviewRecord, ...]

    @model_validator(mode="after")
    def validate_unique_dwarf_ids(self) -> "CategoryReviewFile":
        """Require at most one decision for every dwarf."""
        dwarf_ids = [record.dwarf_id for record in self.records]
        if len(dwarf_ids) != len(set(dwarf_ids)):
            raise ValueError("category review dwarf IDs must be unique")
        return self


class FetchAuditDisposition(StrEnum):
    """Severity of a Commons acquisition audit entry."""

    EXCLUDED = "excluded"
    WARNING = "warning"
    ERROR = "error"


class FetchAuditReason(StrEnum):
    """Stable reason codes emitted by Commons acquisition."""

    REJECTED_CATEGORY = "rejected_category"
    MISSING_ATTRIBUTION = "missing_attribution"
    UNSUPPORTED_LICENSE = "unsupported_license"
    UNSUPPORTED_MEDIA = "unsupported_media"
    IMAGE_TOO_SMALL = "image_too_small"
    INVALID_IMAGE = "invalid_image"
    SAME_LABEL_DUPLICATE = "same_label_duplicate"
    CROSS_LABEL_DUPLICATE = "cross_label_duplicate"
    INVALID_METADATA = "invalid_metadata"
    API_FAILURE = "api_failure"
    DOWNLOAD_FAILURE = "download_failure"


class FetchAuditRecord(BaseModel):
    """One deterministic Commons exclusion, warning, or operational error."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    dwarf_id: str = Field(min_length=1)
    disposition: FetchAuditDisposition
    reason: FetchAuditReason
    details: str = Field(min_length=1)
    commons_page_id: int | None = Field(default=None, gt=0)
    commons_title: str | None = Field(default=None, min_length=1)


class CommonsCacheMetadata(BaseModel):
    """Provenance used to validate a cached Commons category response."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: str = Field(pattern=r"^\d+\.\d+$")
    endpoint: HttpUrl
    category: str = Field(min_length=1)
    request_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    retrieved_at: datetime


class FetchedImagesFile(BaseModel):
    """Deterministic staging artifact consumed by manifest construction."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: str = Field(pattern=r"^\d+\.\d+$")
    source_query_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    review_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    max_images_per_dwarf: int | None = Field(default=None, gt=0)
    records: tuple[ImageRecord, ...]


class FetchAuditFile(BaseModel):
    """Deterministic audit artifact for one Commons acquisition run."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: str = Field(pattern=r"^\d+\.\d+$")
    source_query_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    review_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    records: tuple[FetchAuditRecord, ...]


class FetchResult(BaseModel):
    """In-memory summary returned by the Commons acquisition stage."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    images: tuple[ImageRecord, ...]
    audit: tuple[FetchAuditRecord, ...]
    approved_categories: int = Field(ge=0)
    rejected_categories: int = Field(ge=0)
    pending_categories: int = Field(ge=0)
    discovered_images: int = Field(ge=0)
    eligible_images: int = Field(ge=0)
    downloaded_images: int = Field(ge=0)
    reused_images: int = Field(ge=0)
    operational_failures: int = Field(ge=0)

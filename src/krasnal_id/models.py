"""Validated data contracts shared by acquisition and experiments."""

from datetime import datetime
from pathlib import Path

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

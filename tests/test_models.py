"""Validation tests for the versioned dataset manifest."""

from collections.abc import Callable
from typing import Any

import pytest
from pydantic import ValidationError

from krasnal_id.models import DatasetManifest


def test_valid_manifest(valid_manifest_data: Callable[[], dict[str, Any]]) -> None:
    manifest = DatasetManifest.model_validate(valid_manifest_data())

    assert manifest.schema_version == "1.0"
    assert manifest.dwarfs[0].coordinates is not None
    assert manifest.images[0].author == "Example Photographer"


@pytest.mark.parametrize("field", ["author", "license"])
def test_rejects_empty_attribution(
    valid_manifest_data: Callable[[], dict[str, Any]], field: str
) -> None:
    data = valid_manifest_data()
    data["images"][0][field] = ""

    with pytest.raises(ValidationError):
        DatasetManifest.model_validate(data)


@pytest.mark.parametrize("field", ["source_url", "license_url"])
def test_rejects_invalid_image_urls(
    valid_manifest_data: Callable[[], dict[str, Any]], field: str
) -> None:
    data = valid_manifest_data()
    data["images"][0][field] = "not-a-url"

    with pytest.raises(ValidationError):
        DatasetManifest.model_validate(data)


def test_rejects_malformed_checksum(valid_manifest_data: Callable[[], dict[str, Any]]) -> None:
    data = valid_manifest_data()
    data["images"][0]["sha256"] = "abc123"

    with pytest.raises(ValidationError):
        DatasetManifest.model_validate(data)


@pytest.mark.parametrize(
    ("coordinate", "value"),
    [("latitude", 90.1), ("latitude", -90.1), ("longitude", 180.1), ("longitude", -180.1)],
)
def test_rejects_invalid_coordinates(
    valid_manifest_data: Callable[[], dict[str, Any]], coordinate: str, value: float
) -> None:
    data = valid_manifest_data()
    data["dwarfs"][0]["coordinates"][coordinate] = value

    with pytest.raises(ValidationError):
        DatasetManifest.model_validate(data)


def test_rejects_duplicate_dwarf_ids(valid_manifest_data: Callable[[], dict[str, Any]]) -> None:
    data = valid_manifest_data()
    data["dwarfs"].append(dict(data["dwarfs"][0]))

    with pytest.raises(ValidationError, match="dwarf IDs must be unique"):
        DatasetManifest.model_validate(data)


def test_rejects_duplicate_image_ids(valid_manifest_data: Callable[[], dict[str, Any]]) -> None:
    data = valid_manifest_data()
    data["images"].append(dict(data["images"][0]))

    with pytest.raises(ValidationError, match="image IDs must be unique"):
        DatasetManifest.model_validate(data)


def test_rejects_unknown_dwarf_reference(
    valid_manifest_data: Callable[[], dict[str, Any]],
) -> None:
    data = valid_manifest_data()
    data["images"][0]["dwarf_id"] = "Q999"

    with pytest.raises(ValidationError, match="unknown dwarf IDs: Q999"):
        DatasetManifest.model_validate(data)

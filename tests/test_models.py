"""Validation tests for the versioned dataset manifest."""

from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest
from pydantic import ValidationError

from krasnal_id.models import CategoryReviewFile, DatasetManifest, ImageReviewFile


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


def test_category_review_uses_display_name_overrides() -> None:
    review = CategoryReviewFile.model_validate_json(Path("data/category-review.json").read_text())
    names = {record.dwarf_id: record.selected_display_name for record in review.records}

    assert names["Q136001294"] == "Abruzjusz"
    assert names["Q136001318"] == "Ossolinek"
    assert names["Q136001344"] == "Demokracja"


def test_image_review_file_has_unique_dwarf_page_keys() -> None:
    data = {
        "schema_version": "1.0",
        "source_query_sha256": "a" * 64,
        "staging_sha256": "b" * 64,
        "records": [
            {
                "dwarf_id": "Q123",
                "commons_page_id": 1,
                "status": "retain",
                "reason": "preferred_duplicate",
                "notes": "Keep canonical copy.",
            }
        ],
    }
    review = ImageReviewFile.model_validate(data)

    assert review.records[0].status.value == "retain"

    data["records"].append(dict(data["records"][0]))
    with pytest.raises(ValidationError, match="dwarf/page pairs must be unique"):
        ImageReviewFile.model_validate(data)


def test_tracked_review_contains_resolved_dataset_decisions() -> None:
    review = ImageReviewFile.model_validate_json(Path("data/image-review.json").read_text())

    assert {record.status.value for record in review.records} == {"retain", "exclude"}
    assert {record.commons_page_id for record in review.records} == {
        166491,
        22381955,
        22398133,
        52890654,
        52890655,
        134103757,
        89462414,
    }

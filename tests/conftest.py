"""Shared valid test records for schema and contract tests."""

from collections.abc import Callable
from typing import Any

import pytest


@pytest.fixture
def valid_manifest_data() -> Callable[[], dict[str, Any]]:
    """Return a factory so tests can mutate independent valid manifests."""

    def factory() -> dict[str, Any]:
        return {
            "schema_version": "1.0",
            "source_query_sha256": "b" * 64,
            "staging_sha256": "c" * 64,
            "image_review_sha256": "d" * 64,
            "generated_at": "2026-08-19T12:00:00Z",
            "minimum_images_per_dwarf": 3,
            "dwarfs": [
                {
                    "dwarf_id": "Q123",
                    "display_name": "Test Dwarf",
                    "wikidata_url": "https://www.wikidata.org/wiki/Q123",
                    "commons_category": "Test dwarf",
                    "coordinates": {"latitude": 51.1079, "longitude": 17.0385},
                    "coordinate_source": "wikidata",
                }
            ],
            "images": [
                {
                    "image_id": "commons-1",
                    "dwarf_id": "Q123",
                    "local_path": "data/images/Q123/commons-1.jpg",
                    "source_url": "https://commons.wikimedia.org/wiki/File:Test.jpg",
                    "author": "Example Photographer",
                    "license": "CC BY-SA 4.0",
                    "license_url": "https://creativecommons.org/licenses/by-sa/4.0/",
                    "sha256": "a" * 64,
                    "width": 1200,
                    "height": 800,
                    "acquired_at": "2026-08-19T12:00:00Z",
                }
            ],
        }

    return factory

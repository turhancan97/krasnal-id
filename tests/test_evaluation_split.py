"""Tests for deterministic evaluation split construction."""

from datetime import UTC, datetime
from pathlib import Path

import pytest

from krasnal_id.data_pipeline.build_split import (
    SplitConfigurationError,
    build_evaluation_split,
    build_split_from_artifact,
    write_evaluation_split,
)
from krasnal_id.models import DatasetManifest, DwarfRecord, EvaluationSplit


def _manifest() -> DatasetManifest:
    dwarf = DwarfRecord(
        dwarf_id="Q1",
        display_name="One",
        wikidata_url="https://www.wikidata.org/wiki/Q1",
        commons_category="One",
    )
    records = []
    for index in range(3):
        records.append(
            {
                "image_id": f"image-{index}",
                "dwarf_id": "Q1",
                "local_path": f"data/images/image-{index}.jpg",
                "source_url": "https://commons.wikimedia.org/wiki/File:One.jpg",
                "author": "Author",
                "license": "CC BY-SA 4.0",
                "license_url": "https://creativecommons.org/licenses/by-sa/4.0/",
                "sha256": f"{index + 1:064x}",
                "width": 4,
                "height": 4,
                "acquired_at": "2026-08-23T12:00:00Z",
            }
        )
    return DatasetManifest(
        schema_version="1.0",
        source_query_sha256="a" * 64,
        staging_sha256="b" * 64,
        image_review_sha256="c" * 64,
        generated_at=datetime(2026, 8, 23, tzinfo=UTC),
        minimum_images_per_dwarf=3,
        dwarfs=(dwarf,),
        images=tuple(records),
    )


def test_leave_one_out_is_deterministic_and_non_leaking() -> None:
    manifest = _manifest()
    generated_at = datetime(2026, 8, 23, tzinfo=UTC)

    first = build_evaluation_split(manifest, generated_at)
    second = build_evaluation_split(manifest, generated_at)

    assert first == second
    assert len(first.folds) == 3
    assert [fold.query_image_id for fold in first.folds] == [
        "image-0",
        "image-1",
        "image-2",
    ]
    for fold in first.folds:
        assert fold.query_image_id not in fold.reference_image_ids
        assert len(fold.reference_image_ids) == 2
        assert fold.query_dwarf_id == "Q1"


def test_split_round_trips_atomically(tmp_path: Path) -> None:
    split = build_evaluation_split(_manifest(), datetime(2026, 8, 23, tzinfo=UTC))
    path = tmp_path / "splits" / "leave-one-out.json"

    write_evaluation_split(path, split)

    loaded = EvaluationSplit.model_validate_json(path.read_text())
    assert loaded == split
    assert not list(path.parent.glob(".*.tmp"))


def test_split_artifact_rejects_missing_or_malformed_manifest(tmp_path: Path) -> None:
    with pytest.raises(SplitConfigurationError):
        build_split_from_artifact(tmp_path / "missing.json")

    path = tmp_path / "manifest.json"
    path.write_text("{}")
    with pytest.raises(SplitConfigurationError):
        build_split_from_artifact(path)

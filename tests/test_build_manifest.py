"""Tests for audited manifest construction and CLI integration."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

import pytest
from pydantic import HttpUrl
from typer.testing import CliRunner

from krasnal_id.cli import app
from krasnal_id.data_pipeline.build_manifest import (
    ManifestConfigurationError,
    build_dataset_manifest,
    build_manifest_from_artifacts,
    canonical_json_sha256,
    category_review_sha256,
)
from krasnal_id.models import (
    CategoryReviewFile,
    CategoryReviewRecord,
    CategoryReviewStatus,
    DatasetManifest,
    DwarfDiscoveryFile,
    DwarfRecord,
    FetchedImagesFile,
    ImageRecord,
    ImageReviewFile,
    ImageReviewReason,
    ImageReviewRecord,
    ImageReviewStatus,
)

QUERY_HASH = "a" * 64
NOW = datetime(2026, 8, 20, tzinfo=UTC)
runner = CliRunner()


def _dump(path: Path, payload: object) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n")


def _dwarf(qid: str, name: str, category: str) -> DwarfRecord:
    return DwarfRecord(
        dwarf_id=qid,
        display_name=name,
        wikidata_url=HttpUrl(f"https://www.wikidata.org/wiki/{qid}"),
        commons_category=category,
    )


def _image(qid: str, page_id: int) -> ImageRecord:
    return ImageRecord(
        image_id=f"commons-{page_id}",
        dwarf_id=qid,
        local_path=Path(f"data/images/{qid}/commons-{page_id}.jpg"),
        source_url=HttpUrl(f"https://commons.wikimedia.org/wiki/File:{page_id}.jpg"),
        author="Example Photographer",
        license="Public domain",
        license_url=HttpUrl("https://creativecommons.org/publicdomain/mark/1.0/"),
        sha256=f"{page_id:064x}",
        width=900,
        height=700,
        acquired_at=NOW,
        commons_page_id=page_id,
    )


def _write_artifacts(root: Path, *, stale_review_hash: bool = False) -> dict[str, Path]:
    discovery_path = root / "dwarfs.json"
    fetched_path = root / "fetched-images.json"
    category_path = root / "category-review.json"
    image_review_path = root / "image-review.json"

    dwarfs = (
        _dwarf("Q1", "Generated One", "Category One"),
        _dwarf("Q2", "Generated Two", "Category Two"),
    )
    discovery = DwarfDiscoveryFile(
        schema_version="1.0",
        query_sha256=QUERY_HASH,
        eligible_total=2,
        records=dwarfs,
    )
    category_review = CategoryReviewFile(
        schema_version="1.0",
        records=(
            CategoryReviewRecord(
                dwarf_id="Q1",
                display_name="Generated One",
                display_name_override="Reviewed One",
                discovered_category="Category One",
                status=CategoryReviewStatus.APPROVED,
            ),
            CategoryReviewRecord(
                dwarf_id="Q2",
                display_name="Generated Two",
                discovered_category="Category Two",
                status=CategoryReviewStatus.APPROVED,
            ),
        ),
    )
    fetched = FetchedImagesFile(
        schema_version="1.0",
        source_query_sha256=QUERY_HASH,
        review_sha256=category_review_sha256(category_review),
        records=tuple(
            [_image("Q1", page_id) for page_id in (1, 2, 3, 4)] + [_image("Q2", 5), _image("Q2", 6)]
        ),
    )
    discovery_raw = discovery.model_dump(mode="json")
    category_raw = category_review.model_dump(mode="json")
    fetched_raw = fetched.model_dump(mode="json")
    image_review = ImageReviewFile(
        schema_version="1.0",
        source_query_sha256=QUERY_HASH,
        staging_sha256=("f" * 64 if stale_review_hash else canonical_json_sha256(fetched_raw)),
        records=(
            ImageReviewRecord(
                dwarf_id="Q1",
                commons_page_id=1,
                status=ImageReviewStatus.EXCLUDE,
                reason=ImageReviewReason.SAME_CONTENT_DUPLICATE,
                notes="Exclude duplicate.",
            ),
            ImageReviewRecord(
                dwarf_id="Q1",
                commons_page_id=2,
                status=ImageReviewStatus.RETAIN,
                reason=ImageReviewReason.PREFERRED_DUPLICATE,
                notes="Retain canonical image.",
            ),
        ),
    )

    _dump(discovery_path, discovery_raw)
    _dump(fetched_path, fetched_raw)
    _dump(category_path, category_raw)
    _dump(image_review_path, image_review.model_dump(mode="json"))
    return {
        "discovery": discovery_path,
        "fetched": fetched_path,
        "category": category_path,
        "image_review": image_review_path,
    }


def _build(paths: dict[str, Path]) -> DatasetManifest:
    return build_manifest_from_artifacts(
        paths["discovery"],
        paths["fetched"],
        paths["category"],
        paths["image_review"],
        3,
        generated_at=NOW,
    )


def test_pure_builder_filters_threshold_and_records_provenance() -> None:
    dwarfs = (_dwarf("Q1", "One", "Category One"), _dwarf("Q2", "Two", "Category Two"))
    images = tuple([_image("Q1", page_id) for page_id in (1, 2, 3)] + [_image("Q2", 4)])

    manifest = build_dataset_manifest(
        dwarfs,
        images,
        NOW,
        3,
        source_query_sha256="a" * 64,
        staging_sha256="b" * 64,
        image_review_sha256="c" * 64,
    )

    assert [dwarf.dwarf_id for dwarf in manifest.dwarfs] == ["Q1"]
    assert len(manifest.images) == 3
    assert manifest.source_query_sha256 == "a" * 64
    assert manifest.staging_sha256 == "b" * 64
    assert manifest.image_review_sha256 == "c" * 64


def test_artifact_builder_applies_image_review_and_name_override(tmp_path: Path) -> None:
    paths = _write_artifacts(tmp_path)
    manifest = _build(paths)

    assert [dwarf.dwarf_id for dwarf in manifest.dwarfs] == ["Q1"]
    assert manifest.dwarfs[0].display_name == "Reviewed One"
    assert [image.commons_page_id for image in manifest.images] == [2, 3, 4]
    assert manifest.minimum_images_per_dwarf == 3


def test_artifact_builder_rejects_stale_image_review(tmp_path: Path) -> None:
    paths = _write_artifacts(tmp_path, stale_review_hash=True)

    with pytest.raises(ManifestConfigurationError, match="staging hash"):
        _build(paths)


def test_artifact_builder_rejects_unknown_review_page(tmp_path: Path) -> None:
    paths = _write_artifacts(tmp_path)
    review = json.loads(paths["image_review"].read_text())
    review["records"][0]["commons_page_id"] = 999
    review["staging_sha256"] = canonical_json_sha256(json.loads(paths["fetched"].read_text()))
    _dump(paths["image_review"], review)

    with pytest.raises(ManifestConfigurationError, match="unstaged images"):
        _build(paths)


def test_pure_builder_rejects_unknown_dwarf_reference() -> None:
    with pytest.raises(ValueError, match="unknown dwarf IDs"):
        build_dataset_manifest(
            (_dwarf("Q1", "One", "Category One"),),
            (_image("Q2", 1),),
            NOW,
            3,
            source_query_sha256="a" * 64,
            staging_sha256="b" * 64,
            image_review_sha256="c" * 64,
        )


def test_cli_builds_manifest_atomically_and_reports_summary(tmp_path: Path) -> None:
    paths = _write_artifacts(tmp_path)
    output_path = tmp_path / "manifest.json"
    result = runner.invoke(
        app,
        [
            "data",
            "build-manifest",
            "--override",
            f"paths.discovery_dir={tmp_path}",
            "--override",
            f"paths.category_review_path={paths['category']}",
            "--override",
            f"paths.image_review_path={paths['image_review']}",
            "--override",
            f"paths.manifest_path={output_path}",
            "--override",
            "logging.json_output=false",
        ],
    )

    assert result.exit_code == 0, result.output
    assert "dwarfs=1 images=3 threshold=3" in result.output
    assert output_path.exists()
    assert not list(tmp_path.glob(".manifest.json.*.tmp"))


def test_cli_fails_when_required_artifact_is_missing(tmp_path: Path) -> None:
    result = runner.invoke(
        app,
        [
            "data",
            "build-manifest",
            "--override",
            f"paths.discovery_dir={tmp_path}",
            "--override",
            f"paths.category_review_path={tmp_path / 'category-review.json'}",
            "--override",
            f"paths.image_review_path={tmp_path / 'image-review.json'}",
            "--override",
            f"paths.manifest_path={tmp_path / 'manifest.json'}",
            "--override",
            "logging.json_output=false",
        ],
    )

    assert result.exit_code == 2
    assert "Manifest configuration error" in result.output


@pytest.mark.skipif(
    not Path("data/discovery/dwarfs.json").exists(),
    reason="ignored local dataset artifacts are unavailable",
)
def test_current_local_artifacts_produce_expected_manifest() -> None:
    manifest = build_manifest_from_artifacts(
        Path("data/discovery/dwarfs.json"),
        Path("data/discovery/fetched-images.json"),
        Path("data/category-review.json"),
        Path("data/image-review.json"),
        3,
    )

    assert len(manifest.dwarfs) == 306
    assert len(manifest.images) == 1691
    names = {dwarf.dwarf_id: dwarf.display_name for dwarf in manifest.dwarfs}
    # Tracked display-name overrides survive a Commons-first rebuild.
    assert names["Q136001318"] == "Ossolinek"
    assert names["Q136001344"] == "Demokracja"
    # Both identity kinds coexist, and only the Wikidata ones carry coordinates.
    wikidata = [d for d in manifest.dwarfs if d.wikidata_url is not None]
    commons_only = [d for d in manifest.dwarfs if d.wikidata_url is None]
    assert len(wikidata) == 23
    assert len(commons_only) == 283
    assert all(d.coordinates is not None for d in wikidata)
    assert all(d.coordinates is None for d in commons_only)

"""Tests for embedding cache and manifest-driven extraction."""

import hashlib
from pathlib import Path

import numpy as np
import pytest
from PIL import Image
from typer.testing import CliRunner

from krasnal_id.cli import app
from krasnal_id.config import load_config
from krasnal_id.embeddings.backbone import EmbeddingBackbone
from krasnal_id.embeddings.cache import EmbeddingCache, EmbeddingCacheKey
from krasnal_id.embeddings.extract import (
    EmbeddingExtractionError,
    extract_manifest_embeddings,
)
from krasnal_id.models import DatasetManifest, DwarfRecord, ImageRecord


class FakeBackbone:
    """Deterministic offline backbone used by CI tests."""

    model_id = "fake/model"
    revision = "fake-revision"
    preprocessing_id = "fake-processor"

    def __init__(self) -> None:
        self.calls = 0

    def get_embeddings(self, images: tuple[Image.Image, ...]) -> np.ndarray:
        self.calls += 1
        vectors = np.asarray(
            [[float(index + 1), 1.0, 0.5] for index in range(len(images))],
            dtype=np.float32,
        )
        return vectors / np.linalg.norm(vectors, axis=1, keepdims=True)

    def get_embedding(self, image: Image.Image) -> np.ndarray:
        return self.get_embeddings((image,))[0]


def _image_record(
    tmp_path: Path, image_id: str, dwarf_id: str, color: tuple[int, int, int]
) -> ImageRecord:
    path = tmp_path / f"{image_id}.png"
    Image.new("RGB", (4, 4), color).save(path)
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    return ImageRecord(
        image_id=image_id,
        dwarf_id=dwarf_id,
        local_path=path,
        source_url="https://commons.wikimedia.org/wiki/File:Example.png",
        author="Author",
        license="CC BY-SA 4.0",
        license_url="https://creativecommons.org/licenses/by-sa/4.0/",
        sha256=digest,
        width=4,
        height=4,
        acquired_at="2026-08-23T12:00:00Z",
    )


def _manifest(tmp_path: Path) -> DatasetManifest:
    dwarfs = tuple(
        DwarfRecord(
            dwarf_id=f"Q{index}",
            display_name=f"Dwarf {index}",
            wikidata_url=f"https://www.wikidata.org/wiki/Q{index}",
            commons_category=f"Dwarf {index}",
        )
        for index in range(2)
    )
    images = tuple(
        _image_record(tmp_path, f"image-{dwarf}-{index}", f"Q{dwarf}", (dwarf * 50, index * 50, 10))
        for dwarf in range(2)
        for index in range(3)
    )
    return DatasetManifest(
        schema_version="1.0",
        source_query_sha256="a" * 64,
        staging_sha256="b" * 64,
        image_review_sha256="c" * 64,
        generated_at="2026-08-23T12:00:00Z",
        minimum_images_per_dwarf=3,
        dwarfs=dwarfs,
        images=images,
    )


def test_cache_round_trip_and_invalid_cache_is_a_miss(tmp_path: Path) -> None:
    cache = EmbeddingCache(tmp_path)
    key = EmbeddingCacheKey("a" * 64, "model", "revision", "processor")
    vector = np.asarray([1.0, 0.0], dtype=np.float32)

    path = cache.store(key, vector)
    assert path == cache.path_for(key)
    np.testing.assert_allclose(cache.load(key), vector)

    np.save(path, np.asarray([0.0, 0.0], dtype=np.float32))
    assert cache.load(key) is None
    with pytest.raises(ValueError):
        cache.store(key, np.asarray([1.0, 0.0], dtype=np.float64))


def test_extraction_is_strict_resumable_and_batched(tmp_path: Path) -> None:
    manifest = _manifest(tmp_path)
    cache = EmbeddingCache(tmp_path / "embeddings")
    backbone = FakeBackbone()

    first = extract_manifest_embeddings(manifest, backbone, cache, batch_size=2)
    assert first.total == 6
    assert first.reused == 0
    assert first.computed == 6
    assert backbone.calls == 3

    second = extract_manifest_embeddings(manifest, backbone, cache, batch_size=2)
    assert second.reused == 6
    assert second.computed == 0
    assert backbone.calls == 3


def test_extraction_rejects_missing_or_changed_images(tmp_path: Path) -> None:
    manifest = _manifest(tmp_path)
    record = manifest.images[0]
    record.local_path.write_bytes(b"changed")

    with pytest.raises(EmbeddingExtractionError, match="checksum mismatch"):
        extract_manifest_embeddings(manifest, FakeBackbone(), EmbeddingCache(tmp_path / "cache"), 2)


def test_cli_build_split_and_invalid_extraction_configuration(tmp_path: Path) -> None:
    manifest = _manifest(tmp_path)
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(manifest.model_dump_json())
    split_path = tmp_path / "split.json"

    runner = CliRunner()
    split_result = runner.invoke(
        app,
        [
            "data",
            "build-split",
            "--override",
            f"paths.manifest_path={manifest_path}",
            "--override",
            f"paths.evaluation_split_path={split_path}",
            "--override",
            "logging.json_output=false",
        ],
    )
    assert split_result.exit_code == 0
    assert "folds=6" in split_result.output
    assert split_path.is_file()

    extract_result = runner.invoke(
        app,
        [
            "embeddings",
            "extract",
            "--override",
            f"paths.manifest_path={tmp_path / 'missing.json'}",
            "--override",
            f"paths.embeddings_dir={tmp_path / 'embeddings'}",
            "--override",
            "logging.json_output=false",
        ],
    )
    assert extract_result.exit_code == 2
    assert "Embedding extraction error" in extract_result.output


def test_backbone_adapters_are_lazy() -> None:
    assert isinstance(FakeBackbone(), EmbeddingBackbone)
    assert load_config().backbone.batch_size == 16

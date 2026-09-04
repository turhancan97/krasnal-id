"""Tests for embedding cache and manifest-driven extraction."""

import hashlib
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from PIL import Image
from pydantic import HttpUrl
from typer.testing import CliRunner

from krasnal_id.cli import app
from krasnal_id.config import load_config
from krasnal_id.embeddings.backbone import EmbeddingBackbone
from krasnal_id.embeddings.cache import EmbeddingCache, EmbeddingCacheKey
from krasnal_id.embeddings.clip import ClipBackbone
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
        return np.asarray(self.get_embeddings((image,))[0])


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
        source_url=HttpUrl("https://commons.wikimedia.org/wiki/File:Example.png"),
        author="Author",
        license="CC BY-SA 4.0",
        license_url=HttpUrl("https://creativecommons.org/licenses/by-sa/4.0/"),
        sha256=digest,
        width=4,
        height=4,
        acquired_at=datetime(2026, 8, 23, 12, 0, tzinfo=UTC),
    )


def _manifest(tmp_path: Path) -> DatasetManifest:
    dwarfs = tuple(
        DwarfRecord(
            dwarf_id=f"Q{index}",
            display_name=f"Dwarf {index}",
            wikidata_url=HttpUrl(f"https://www.wikidata.org/wiki/Q{index}"),
            commons_category=f"Dwarf {index}",
        )
        for index in range(1, 3)
    )
    images = tuple(
        _image_record(tmp_path, f"image-{dwarf}-{index}", f"Q{dwarf}", (dwarf * 50, index * 50, 10))
        for dwarf in range(1, 3)
        for index in range(3)
    )
    return DatasetManifest(
        schema_version="1.0",
        source_query_sha256="a" * 64,
        staging_sha256="b" * 64,
        image_review_sha256="c" * 64,
        generated_at=datetime(2026, 8, 23, 12, 0, tzinfo=UTC),
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
    stored = cache.load(key)
    assert stored is not None
    np.testing.assert_allclose(stored, vector)

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


class _FakeVisionOutput:
    """Stand-in for the transformers output object returned by get_image_features."""

    def __init__(self, pooler_output: object) -> None:
        self.pooler_output = pooler_output


class _FakeClipModel:
    """Stands in for CLIPModel, yielding whatever get_image_features should return."""

    def __init__(self, features: Any) -> None:
        self._features = features

    def get_image_features(self, pixel_values: Any) -> Any:
        return self._features


class _FakeProcessor:
    """Stands in for AutoProcessor, returning a fixed pixel batch."""

    def __init__(self, torch: Any) -> None:
        self._torch = torch

    def __call__(self, images: Any, return_tensors: str) -> dict[str, Any]:
        return {"pixel_values": self._torch.zeros(1, 3, 2, 2)}


def test_clip_accepts_tensor_or_output_object() -> None:
    torch = pytest.importorskip("torch")
    config = load_config(["backbone=clip"]).backbone
    backbone = ClipBackbone(config)
    tensor = torch.tensor([[3.0, 4.0]])

    backbone._processor = _FakeProcessor(torch)
    backbone._torch = torch
    backbone._device = "cpu"

    # transformers 5 returns the vision output object; older shapes returned a
    # bare tensor. Both must normalize to the same unit vector.
    for features in (tensor, _FakeVisionOutput(tensor)):
        backbone._model = _FakeClipModel(features)
        vectors = backbone.get_embeddings((Image.new("RGB", (2, 2)),))
        np.testing.assert_allclose(vectors, np.array([[0.6, 0.8]], dtype=np.float32), atol=1e-6)

    backbone._model = _FakeClipModel(_FakeVisionOutput(None))
    with pytest.raises(ValueError, match="unsupported image features"):
        backbone.get_embeddings((Image.new("RGB", (2, 2)),))

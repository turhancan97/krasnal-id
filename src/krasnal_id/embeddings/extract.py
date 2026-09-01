"""Manifest-driven, resumable embedding extraction."""

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, UnidentifiedImageError
from pydantic import ValidationError

from krasnal_id.config import BackboneConfig
from krasnal_id.embeddings.backbone import (
    EmbeddingBackbone,
    EmbeddingConfigurationError,
)
from krasnal_id.embeddings.cache import EmbeddingCache, EmbeddingCacheKey
from krasnal_id.embeddings.clip import ClipBackbone
from krasnal_id.embeddings.dinov2 import DinoV2Backbone
from krasnal_id.embeddings.store import cache_key_for
from krasnal_id.models import DatasetManifest, ImageRecord


class EmbeddingExtractionError(ValueError):
    """Raised when manifest inputs or extraction outputs are invalid."""


@dataclass(frozen=True, slots=True)
class ExtractionSummary:
    """Counts emitted by one resumable extraction run."""

    total: int
    reused: int
    computed: int


def create_backbone(config: BackboneConfig) -> EmbeddingBackbone:
    """Create the configured adapter without loading optional ML dependencies."""
    if config.name == "dinov2":
        return DinoV2Backbone(config)
    if config.name == "clip":
        return ClipBackbone(config)
    raise EmbeddingConfigurationError(f"unsupported backbone: {config.name}")


def _sha256_file(path: Path) -> str:
    """Hash a local research image without loading the whole file at once."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_and_validate_image(record: ImageRecord) -> Image.Image:
    """Read and validate one manifest image and its recorded provenance."""
    path = record.local_path
    if not path.is_file():
        raise EmbeddingExtractionError(f"image {record.image_id} is missing: {path}")
    actual_sha256 = _sha256_file(path)
    if actual_sha256 != record.sha256:
        raise EmbeddingExtractionError(
            f"image {record.image_id} checksum mismatch: expected {record.sha256}, "
            f"got {actual_sha256}"
        )
    try:
        with Image.open(path) as image:
            image.load()
            if image.size != (record.width, record.height):
                raise EmbeddingExtractionError(
                    f"image {record.image_id} dimensions mismatch: expected "
                    f"{record.width}x{record.height}, got {image.width}x{image.height}"
                )
            return image.convert("RGB")
    except (OSError, UnidentifiedImageError) as error:
        raise EmbeddingExtractionError(
            f"image {record.image_id} cannot be decoded: {path}: {error}"
        ) from error


def extract_manifest_embeddings(
    manifest: DatasetManifest,
    backbone: EmbeddingBackbone,
    cache: EmbeddingCache,
    batch_size: int,
) -> ExtractionSummary:
    """Validate manifest images and compute only missing or invalid vectors."""
    if batch_size <= 0:
        raise EmbeddingExtractionError("batch_size must be positive")

    pending: list[tuple[EmbeddingCacheKey, Image.Image]] = []
    reused = 0
    ordered_records = tuple(sorted(manifest.images, key=lambda image: image.image_id))
    for record in ordered_records:
        image = _load_and_validate_image(record)
        key = cache_key_for(record, backbone)
        if cache.load(key) is not None:
            reused += 1
        else:
            pending.append((key, image))

    computed = 0
    for start in range(0, len(pending), batch_size):
        batch = pending[start : start + batch_size]
        try:
            vectors = backbone.get_embeddings(tuple(image for _, image in batch))
        except Exception as error:
            raise EmbeddingExtractionError(
                f"backbone {backbone.model_id} failed on batch starting at {start}: {error}"
            ) from error
        array = np.asarray(vectors)
        if array.ndim != 2 or array.shape[0] != len(batch):
            raise EmbeddingExtractionError(
                f"backbone returned {array.shape} for {len(batch)} images"
            )
        for (key, _), vector in zip(batch, array, strict=True):
            try:
                cache.store(key, np.asarray(vector, dtype=np.float32))
            except ValueError as error:
                raise EmbeddingExtractionError(
                    f"backbone returned an invalid vector for cache key {key.digest()}: {error}"
                ) from error
            computed += 1

    return ExtractionSummary(
        total=len(ordered_records),
        reused=reused,
        computed=computed,
    )


def _read_manifest(path: Path) -> DatasetManifest:
    """Read and strictly validate a generated manifest."""
    try:
        payload: Any = json.loads(path.read_text(encoding="utf-8"))
        return DatasetManifest.model_validate(payload)
    except (OSError, json.JSONDecodeError, ValidationError, TypeError, ValueError) as error:
        raise EmbeddingExtractionError(f"invalid manifest {path}: {error}") from error


def extract_from_artifact(
    manifest_path: Path,
    config: BackboneConfig,
    cache_root: Path,
) -> ExtractionSummary:
    """Load a manifest, create its configured backbone, and extract embeddings."""
    manifest = _read_manifest(manifest_path)
    backbone = create_backbone(config)
    return extract_manifest_embeddings(
        manifest,
        backbone,
        EmbeddingCache(cache_root),
        config.batch_size,
    )

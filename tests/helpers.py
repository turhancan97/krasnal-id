"""Shared synthetic dataset builders for evaluation tests."""

from pathlib import Path

import numpy as np
import numpy.typing as npt

from krasnal_id.config import BackboneConfig
from krasnal_id.embeddings.cache import EmbeddingCache
from krasnal_id.embeddings.store import cache_key_for
from krasnal_id.models import DatasetManifest, DwarfRecord, ImageRecord

FAKE_BACKBONE = BackboneConfig(
    name="dinov2",
    model_id="fake/model",
    revision="fake-revision",
    preprocessing_id="fake-processor",
    batch_size=4,
)


def image_record(image_id: str, dwarf_id: str, digest_seed: str) -> ImageRecord:
    """Build one valid image record without touching the filesystem."""
    return ImageRecord(
        image_id=image_id,
        dwarf_id=dwarf_id,
        local_path=Path(f"data/images/{dwarf_id}/{image_id}.jpg"),
        source_url="https://commons.wikimedia.org/wiki/File:Example.jpg",
        author="Author",
        license="CC BY-SA 4.0",
        license_url="https://creativecommons.org/licenses/by-sa/4.0/",
        sha256=(digest_seed * 64)[:64],
        width=800,
        height=600,
        acquired_at="2026-08-23T12:00:00Z",
    )


def synthetic_manifest(dwarf_count: int = 3, per_dwarf: int = 3) -> DatasetManifest:
    """Build a manifest with evenly sized classes."""
    dwarfs = tuple(
        DwarfRecord(
            dwarf_id=f"Q{index}",
            display_name=f"Dwarf {index}",
            wikidata_url=f"https://www.wikidata.org/wiki/Q{index}",
            commons_category=f"Dwarf {index}",
        )
        for index in range(dwarf_count)
    )
    images = tuple(
        image_record(f"image-{dwarf}-{position}", f"Q{dwarf}", f"{dwarf}{position}")
        for dwarf in range(dwarf_count)
        for position in range(per_dwarf)
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


def tight_cluster_vector(
    dwarf_index: int, position: int, dwarf_count: int
) -> npt.NDArray[np.float32]:
    """Place each dwarf on its own axis, with a small per-image perturbation."""
    vector = np.zeros(dwarf_count + 1, dtype=np.float32)
    vector[dwarf_index] = 1.0
    vector[dwarf_count] = 0.01 * (position + 1)
    return np.asarray(vector / np.linalg.norm(vector), dtype=np.float32)


def seed_embedding_cache(
    cache_root: Path,
    manifest: DatasetManifest,
    vector_for: object = tight_cluster_vector,
    skip: tuple[str, ...] = (),
    config: BackboneConfig = FAKE_BACKBONE,
) -> None:
    """Populate a cache with one deterministic vector per manifest image."""
    cache = EmbeddingCache(cache_root)
    dwarf_ids = sorted({image.dwarf_id for image in manifest.images})
    for record in manifest.images:
        if record.image_id in skip:
            continue
        position = [
            image.image_id for image in manifest.images if image.dwarf_id == record.dwarf_id
        ].index(record.image_id)
        vector = vector_for(  # type: ignore[operator]
            dwarf_ids.index(record.dwarf_id), position, len(dwarf_ids)
        )
        cache.store(cache_key_for(record, config), np.asarray(vector, dtype=np.float32))

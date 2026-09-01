"""Manifest-ordered access to cached embeddings for evaluation code."""

from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol, runtime_checkable

import numpy as np
import numpy.typing as npt

from krasnal_id.config import BackboneConfig
from krasnal_id.embeddings.cache import EmbeddingCache, EmbeddingCacheKey
from krasnal_id.models import DatasetManifest, ImageRecord


class EmbeddingStoreError(ValueError):
    """Raised when cached vectors are missing or inconsistent with the manifest."""


@runtime_checkable
class BackboneIdentity(Protocol):
    """The pinned identity that a cached vector belongs to.

    Satisfied by both `BackboneConfig` and a loaded backbone adapter, so extraction
    and evaluation cannot drift into building different keys for the same vector.
    """

    @property
    def model_id(self) -> str:
        """Return the upstream model identifier."""
        ...

    @property
    def revision(self) -> str:
        """Return the immutable upstream model revision."""
        ...

    @property
    def preprocessing_id(self) -> str:
        """Return an identifier for the exact preprocessing pipeline."""
        ...


def cache_key_for(record: ImageRecord, identity: BackboneIdentity) -> EmbeddingCacheKey:
    """Build the single cache identity shared by extraction and evaluation."""
    return EmbeddingCacheKey(
        image_sha256=record.sha256,
        model_id=identity.model_id,
        revision=identity.revision,
        preprocessing_id=identity.preprocessing_id,
    )


@dataclass(frozen=True, slots=True)
class EmbeddingMatrix:
    """Cached vectors for one backbone, ordered by image ID."""

    image_ids: tuple[str, ...]
    dwarf_ids: tuple[str, ...]
    vectors: npt.NDArray[np.float32]
    # Built once, because the ablation resolves rows hundreds of thousands of times.
    _row_by_image_id: dict[str, int] = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        """Validate the row alignment and index the image IDs for O(1) lookup."""
        if not (len(self.image_ids) == len(self.dwarf_ids) == self.vectors.shape[0]):
            raise EmbeddingStoreError(
                f"misaligned matrix: {len(self.image_ids)} image IDs, "
                f"{len(self.dwarf_ids)} dwarf IDs, {self.vectors.shape[0]} rows"
            )
        rows = {image_id: row for row, image_id in enumerate(self.image_ids)}
        if len(rows) != len(self.image_ids):
            raise EmbeddingStoreError("image IDs must be unique")
        object.__setattr__(self, "_row_by_image_id", rows)

    def index_of(self, image_id: str) -> int:
        """Return the row index of one image."""
        try:
            return self._row_by_image_id[image_id]
        except KeyError as error:
            raise EmbeddingStoreError(f"image {image_id} has no cached vector") from error

    def rows_for(
        self,
        image_ids: Sequence[str],
    ) -> tuple[npt.NDArray[np.float32], tuple[str, ...]]:
        """Return the vectors and dwarf IDs for a subset, preserving its order."""
        indices = [self.index_of(image_id) for image_id in image_ids]
        return (
            np.asarray(self.vectors[indices], dtype=np.float32),
            tuple(self.dwarf_ids[index] for index in indices),
        )

    def vector_for(self, image_id: str) -> npt.NDArray[np.float32]:
        """Return one image's cached vector."""
        return np.asarray(self.vectors[self.index_of(image_id)], dtype=np.float32)


def load_embedding_matrix(
    manifest: DatasetManifest,
    config: BackboneConfig,
    cache_root: Path,
) -> EmbeddingMatrix:
    """Load every manifest image's cached vector for the configured backbone."""
    records = tuple(sorted(manifest.images, key=lambda image: image.image_id))
    if not records:
        raise EmbeddingStoreError("manifest contains no images")

    cache = EmbeddingCache(cache_root)
    vectors: list[npt.NDArray[np.float32]] = []
    missing: list[str] = []
    for record in records:
        vector = cache.load(cache_key_for(record, config))
        if vector is None:
            missing.append(record.image_id)
            continue
        vectors.append(vector)

    if missing:
        raise EmbeddingStoreError(
            f"{len(missing)} of {len(records)} images have no cached {config.name} vector "
            f"(first: {missing[0]}); run krasnal-id embeddings extract "
            f"--override backbone={config.name}"
        )

    dimensions = {vector.shape[0] for vector in vectors}
    if len(dimensions) != 1:
        raise EmbeddingStoreError(
            f"cached {config.name} vectors have mixed dimensions {dimensions}"
        )

    return EmbeddingMatrix(
        image_ids=tuple(record.image_id for record in records),
        dwarf_ids=tuple(record.dwarf_id for record in records),
        vectors=np.stack(vectors).astype(np.float32, copy=False),
    )

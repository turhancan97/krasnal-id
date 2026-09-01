"""Atomic, validated filesystem caching for normalized embeddings."""

import hashlib
import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from tempfile import NamedTemporaryFile

import numpy as np
import numpy.typing as npt


class EmbeddingCacheError(ValueError):
    """Raised when an embedding cannot be validated or persisted."""


@dataclass(frozen=True, slots=True)
class EmbeddingCacheKey:
    """Inputs that uniquely identify one cached embedding."""

    image_sha256: str
    model_id: str
    revision: str
    preprocessing_id: str

    def digest(self) -> str:
        """Return a stable SHA-256 filename stem for this key."""
        payload = json.dumps(asdict(self), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _validate_embedding(embedding: npt.NDArray[np.float32]) -> npt.NDArray[np.float32]:
    """Validate the persisted vector contract."""
    array = np.asarray(embedding)
    if array.ndim != 1 or array.size == 0:
        raise EmbeddingCacheError("embedding must be a non-empty one-dimensional vector")
    if array.dtype != np.dtype(np.float32):
        raise EmbeddingCacheError("embedding must use float32 storage")
    if not np.isfinite(array).all():
        raise EmbeddingCacheError("embedding contains non-finite values")
    norm = float(np.linalg.norm(array))
    if not np.isfinite(norm) or not np.isclose(norm, 1.0, atol=1e-4):
        raise EmbeddingCacheError("embedding must be L2-normalized")
    return np.asarray(array, dtype=np.float32)


class EmbeddingCache:
    """Filesystem cache keyed by image and immutable backbone identity."""

    def __init__(self, root: Path) -> None:
        self.root = root

    def path_for(self, key: EmbeddingCacheKey) -> Path:
        """Return the deterministic cache path for a key."""
        return self.root / f"{key.digest()}.npy"

    def load(self, key: EmbeddingCacheKey) -> npt.NDArray[np.float32] | None:
        """Load a valid cached embedding, treating invalid files as cache misses."""
        path = self.path_for(key)
        if not path.is_file():
            return None
        try:
            loaded = np.load(path, allow_pickle=False)
            return _validate_embedding(loaded)
        except (OSError, ValueError, EOFError):
            return None

    def store(self, key: EmbeddingCacheKey, embedding: npt.NDArray[np.float32]) -> Path:
        """Validate and persist an embedding atomically."""
        validated = _validate_embedding(embedding)
        self.root.mkdir(parents=True, exist_ok=True)
        temporary_path: Path | None = None
        try:
            with NamedTemporaryFile(
                mode="wb",
                dir=self.root,
                prefix=f".{key.digest()}.",
                suffix=".tmp",
                delete=False,
            ) as temporary:
                temporary_path = Path(temporary.name)
                np.save(temporary, validated, allow_pickle=False)
                temporary.flush()
                os.fsync(temporary.fileno())
            target = self.path_for(key)
            os.replace(temporary_path, target)
            return target
        except OSError as error:
            raise EmbeddingCacheError(f"could not write embedding cache: {error}") from error
        finally:
            if temporary_path is not None:
                temporary_path.unlink(missing_ok=True)

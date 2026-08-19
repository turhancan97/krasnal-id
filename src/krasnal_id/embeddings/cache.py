"""Cache identity contracts for reproducible embedding extraction."""

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import numpy.typing as npt


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


class EmbeddingCache:
    """Filesystem cache interface; persistence is implemented in v0.1."""

    def __init__(self, root: Path) -> None:
        self.root = root

    def load(self, key: EmbeddingCacheKey) -> npt.NDArray[np.float32] | None:
        """Load a cached embedding when present and valid."""
        raise NotImplementedError("Embedding cache reads are scheduled for v0.1")

    def store(self, key: EmbeddingCacheKey, embedding: npt.NDArray[np.float32]) -> Path:
        """Persist an embedding atomically and return its cache path."""
        raise NotImplementedError("Embedding cache writes are scheduled for v0.1")

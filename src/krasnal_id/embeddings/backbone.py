"""Swappable embedding-backbone contracts and shared validation helpers."""

import importlib
from typing import Any, Protocol, cast, runtime_checkable

import numpy as np
import numpy.typing as npt
from PIL import Image

from krasnal_id.config import BackboneConfig


class EmbeddingConfigurationError(RuntimeError):
    """Raised when optional ML dependencies or runtime settings are unavailable."""


@runtime_checkable
class EmbeddingBackbone(Protocol):
    """Stable feature-extraction contract independent of model libraries."""

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

    def get_embedding(self, image: Image.Image) -> npt.NDArray[np.float32]:
        """Return one normalized float32 embedding vector."""
        ...

    def get_embeddings(
        self,
        images: tuple[Image.Image, ...],
    ) -> npt.NDArray[np.float32]:
        """Return one normalized float32 vector for each image."""
        ...


def import_optional_ml(module_name: str) -> Any:
    """Import an optional ML dependency only when extraction is requested."""
    try:
        return importlib.import_module(module_name)
    except ImportError as error:
        raise EmbeddingConfigurationError(
            "ML dependencies are required for embedding extraction; run uv sync --extra ml"
        ) from error


def resolve_device(config: BackboneConfig) -> str:
    """Resolve auto/CPU/CUDA device selection and validate explicit CUDA."""
    if config.device == "cpu":
        return "cpu"
    torch = import_optional_ml("torch")
    cuda_available = bool(torch.cuda.is_available())
    if config.device == "cuda" and not cuda_available:
        raise EmbeddingConfigurationError("CUDA was requested but is not available")
    return "cuda" if cuda_available else "cpu"


def normalize_embedding_batch(values: Any) -> npt.NDArray[np.float32]:
    """Validate and L2-normalize a batch of model outputs."""
    array = np.asarray(values, dtype=np.float32)
    if array.ndim == 1:
        array = array.reshape(1, -1)
    if array.ndim != 2 or array.shape[0] == 0 or array.shape[1] == 0:
        raise ValueError("embedding output must be a non-empty two-dimensional array")
    if not np.isfinite(array).all():
        raise ValueError("embedding output contains non-finite values")
    norms = np.linalg.norm(array, axis=1, keepdims=True)
    if not np.isfinite(norms).all() or np.any(norms <= 0):
        raise ValueError("embedding output contains a zero or invalid norm")
    normalized = (array / norms).astype(np.float32, copy=False)
    return cast(npt.NDArray[np.float32], normalized)


def single_embedding(values: npt.NDArray[np.float32]) -> npt.NDArray[np.float32]:
    """Extract one vector from a normalized batch."""
    if values.shape[0] != 1:
        raise ValueError("single-image extraction returned an unexpected batch size")
    return cast(npt.NDArray[np.float32], values[0])

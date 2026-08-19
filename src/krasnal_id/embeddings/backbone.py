"""Backbone protocol shared by DINOv2, CLIP, and future adapters."""

from typing import Protocol, runtime_checkable

import numpy as np
import numpy.typing as npt
from PIL import Image


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

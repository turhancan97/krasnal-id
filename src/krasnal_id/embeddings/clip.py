"""CLIP vision-backbone adapter placeholder."""

import numpy as np
import numpy.typing as npt
from PIL import Image

from krasnal_id.config import BackboneConfig


class ClipBackbone:
    """Lazy CLIP adapter that will use the optional ML dependencies."""

    def __init__(self, config: BackboneConfig) -> None:
        self._config = config

    @property
    def model_id(self) -> str:
        """Return the configured Hugging Face model identifier."""
        return self._config.model_id

    @property
    def revision(self) -> str:
        """Return the configured model revision."""
        return self._config.revision

    @property
    def preprocessing_id(self) -> str:
        """Return the configured preprocessing identity."""
        return self._config.preprocessing_id

    def get_embedding(self, image: Image.Image) -> npt.NDArray[np.float32]:
        """Extract one normalized CLIP image embedding."""
        raise NotImplementedError("CLIP extraction is scheduled for v0.1")

"""DINOv2 embedding adapter with lazy optional ML loading."""

from typing import Any

import numpy as np
import numpy.typing as npt
from PIL import Image

from krasnal_id.config import BackboneConfig
from krasnal_id.embeddings.backbone import (
    import_optional_ml,
    normalize_embedding_batch,
    resolve_device,
    single_embedding,
)


class DinoV2Backbone:
    """Extract normalized DINOv2 CLS-token image embeddings."""

    def __init__(self, config: BackboneConfig) -> None:
        if config.name != "dinov2":
            raise ValueError("DinoV2Backbone requires a dinov2 configuration")
        self._config = config
        self._processor: Any | None = None
        self._model: Any | None = None
        self._torch: Any | None = None
        self._device: str | None = None

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

    def _ensure_loaded(self) -> None:
        """Load the processor and model once, only when inference begins."""
        if self._model is not None:
            return
        transformers = import_optional_ml("transformers")
        self._torch = import_optional_ml("torch")
        self._device = resolve_device(self._config)
        self._processor = transformers.AutoImageProcessor.from_pretrained(
            self.model_id,
            revision=self.revision,
        )
        self._model = transformers.AutoModel.from_pretrained(
            self.model_id,
            revision=self.revision,
        )
        self._model.to(self._device)
        self._model.eval()

    def get_embeddings(
        self,
        images: tuple[Image.Image, ...],
    ) -> npt.NDArray[np.float32]:
        """Extract one normalized CLS-token vector per image."""
        if not images:
            raise ValueError("at least one image is required")
        self._ensure_loaded()
        assert self._processor is not None
        assert self._model is not None
        assert self._torch is not None
        assert self._device is not None
        rgb_images = [image.convert("RGB") for image in images]
        inputs = self._processor(images=rgb_images, return_tensors="pt")
        moved_inputs = {
            name: value.to(self._device) if hasattr(value, "to") else value
            for name, value in inputs.items()
        }
        with self._torch.inference_mode():
            outputs = self._model(**moved_inputs)
        cls_vectors = outputs.last_hidden_state[:, 0, :].detach().cpu().numpy()
        return normalize_embedding_batch(cls_vectors)

    def get_embedding(self, image: Image.Image) -> npt.NDArray[np.float32]:
        """Extract one normalized DINOv2 CLS-token vector."""
        return single_embedding(self.get_embeddings((image,)))

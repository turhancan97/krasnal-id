"""SigLIP embedding adapter with lazy optional ML loading.

SigLIP 2 is in the project as a *modern* language-aligned backbone (AGENTS.md
§7.15). The paper attributes CLIP's weakness on near-identical statues to its
language alignment; the CLIP checkpoint here is a 2021 ViT-B/32, so without a
current language-aligned model at the same size that attribution could just as
well be about an old checkpoint.

The fixed-resolution SigLIP 2 checkpoints are plain ``siglip`` models: the image
embedding is the vision tower's attention-pooled output, which
``get_image_features`` returns, with no projection layer. Only the image processor
is loaded -- the tokenizer would pull in a text dependency the image path never
uses.
"""

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


class SiglipBackbone:
    """Extract normalized SigLIP image embeddings."""

    def __init__(self, config: BackboneConfig) -> None:
        if config.family != "siglip":
            raise ValueError("SiglipBackbone requires a siglip-family configuration")
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
        """Load the image processor and model once, only when inference begins."""
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
        """Extract one normalized pooled image vector per image."""
        if not images:
            raise ValueError("at least one image is required")
        self._ensure_loaded()
        assert self._processor is not None
        assert self._model is not None
        assert self._torch is not None
        assert self._device is not None
        rgb_images = [image.convert("RGB") for image in images]
        inputs = self._processor(images=rgb_images, return_tensors="pt")
        pixel_values = inputs["pixel_values"].to(self._device)
        with self._torch.inference_mode():
            features = self._model.get_image_features(pixel_values=pixel_values)
        # transformers 5 returns an output object whose pooler_output holds the
        # pooled features; older versions returned the tensor itself.
        vectors = getattr(features, "pooler_output", features)
        if not hasattr(vectors, "detach"):
            raise ValueError(
                f"SigLIP returned unsupported image features of type {type(features).__name__}"
            )
        return normalize_embedding_batch(vectors.detach().cpu().numpy())

    def get_embedding(self, image: Image.Image) -> npt.NDArray[np.float32]:
        """Extract one normalized SigLIP image vector."""
        return single_embedding(self.get_embeddings((image,)))

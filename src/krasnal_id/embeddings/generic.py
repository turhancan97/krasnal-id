"""Generic adapters for the pretraining-family comparison (AGENTS.md §7.17).

The family study adds checkpoints from seven pretraining families. Writing one
adapter per checkpoint would repeat the same load-preprocess-pool loop nine times,
so two adapters cover them, keyed by the library that releases the weights:

* `TimmBackbone` for checkpoints published as timm models (DINO, MAE, EVA-02,
  AIMv2, the supervised ViT). Weights come from `model.safetensors` at a pinned Hub
  revision; timm's own `create_model` would otherwise read the moving `main`.
* `TransformersBackbone` for checkpoints published for `transformers` (I-JEPA,
  BEiT, RADIO), through `AutoModel` and the checkpoint's own image processor.

Pooling is fixed by `BackboneConfig.pooling` and is part of the cache identity.
The rule is the checkpoint's released embedding where it has one, the mean of its
patch tokens where it has none; it is stated once in the paper and never tuned.
Every model keeps the preprocessing it was released with, as DINOv3 does (§7.12).
"""

from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
from PIL import Image

from krasnal_id.config import BackboneConfig
from krasnal_id.embeddings.backbone import (
    EmbeddingConfigurationError,
    import_optional_ml,
    normalize_embedding_batch,
    resolve_device,
    single_embedding,
)


class _GenericBackbone:
    """Identity, lazy loading and batching shared by both generic adapters."""

    family: str = ""

    def __init__(self, config: BackboneConfig) -> None:
        if config.family != self.family or config.pooling is None:
            raise ValueError(f"{type(self).__name__} requires a pooled {self.family} config")
        self._config = config
        self._pooling = config.pooling
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
        """Return the configured preprocessing identity, pooling included."""
        return self._config.preprocessing_id

    def _load(self) -> None:
        """Load the model and its preprocessing once."""
        raise NotImplementedError

    def _forward(self, images: list[Image.Image]) -> Any:
        """Return one pooled feature tensor per image."""
        raise NotImplementedError

    def get_embeddings(self, images: tuple[Image.Image, ...]) -> npt.NDArray[np.float32]:
        """Extract one normalized vector per image."""
        if not images:
            raise ValueError("at least one image is required")
        if self._model is None:
            self._torch = import_optional_ml("torch")
            self._device = resolve_device(self._config)
            self._load()
        assert self._torch is not None
        with self._torch.inference_mode():
            pooled = self._forward([image.convert("RGB") for image in images])
        return normalize_embedding_batch(pooled.float().detach().cpu().numpy())

    def get_embedding(self, image: Image.Image) -> npt.NDArray[np.float32]:
        """Extract one normalized vector."""
        return single_embedding(self.get_embeddings((image,)))


def pool_tokens(tokens: Any, prefix_tokens: int, pooling: str) -> Any:
    """Reduce a (batch, tokens, width) sequence by the configured rule."""
    if pooling == "cls":
        if prefix_tokens < 1:
            raise EmbeddingConfigurationError("cls pooling needs a model with a class token")
        return tokens[:, 0]
    if pooling == "mean":
        return tokens[:, prefix_tokens:].mean(dim=1)
    raise EmbeddingConfigurationError(f"token pooling cannot apply {pooling!r}")


class TimmBackbone(_GenericBackbone):
    """A timm checkpoint, its released data config, pinned to a Hub revision."""

    family = "timm"

    def _load(self) -> None:
        timm = import_optional_ml("timm")
        hub = import_optional_ml("huggingface_hub")
        if not self.model_id.startswith("timm/"):
            raise EmbeddingConfigurationError(f"{self.model_id} is not a timm Hub model")
        weights = Path(
            hub.hf_hub_download(self.model_id, "model.safetensors", revision=self.revision)
        )
        safetensors = import_optional_ml("safetensors.torch")
        # Built empty and loaded strictly from the pinned file: timm's own pretrained
        # path reads the moving main, and for some checkpoints a JAX-format loader.
        # The classifier is dropped only after a strict load, so a missing or
        # unexpected weight fails here rather than silently initialising.
        model = timm.create_model(self.model_id.removeprefix("timm/"), pretrained=False)
        model.load_state_dict(safetensors.load_file(str(weights)), strict=True)
        model.reset_classifier(0)
        config = timm.data.resolve_model_data_config(model)
        self._transform = timm.data.create_transform(**config, is_training=False)
        self._prefix_tokens = int(getattr(model, "num_prefix_tokens", 0))
        model.to(self._device).eval()
        self._model = model

    def _forward(self, images: list[Image.Image]) -> Any:
        assert self._torch is not None and self._model is not None
        batch = self._torch.stack([self._transform(image) for image in images]).to(self._device)
        tokens = self._model.forward_features(batch)
        if self._pooling == "model":
            return self._model.forward_head(tokens, pre_logits=True)
        return pool_tokens(tokens, self._prefix_tokens, self._pooling)


class TransformersBackbone(_GenericBackbone):
    """A transformers checkpoint through `AutoModel` and its own image processor."""

    family = "transformers"

    def _load(self) -> None:
        transformers = import_optional_ml("transformers")
        remote = self._config.trust_remote_code
        self._processor = transformers.AutoImageProcessor.from_pretrained(
            self.model_id, revision=self.revision, trust_remote_code=remote
        )
        model = transformers.AutoModel.from_pretrained(
            self.model_id, revision=self.revision, trust_remote_code=remote
        )
        model.to(self._device).eval()
        self._model = model

    def _forward(self, images: list[Image.Image]) -> Any:
        assert self._model is not None
        inputs = self._processor(
            images=images, return_tensors="pt", **self._config.processor_kwargs
        )
        pixels = inputs["pixel_values"].to(self._device)
        if self._pooling == "summary":
            # RADIO accepts only sides that are multiples of its patch step; its own
            # README snaps the processed tensor to the nearest supported size with
            # bilinear interpolation, which is what is done here. Its forward takes
            # the tensor positionally and returns (summary, spatial features); the
            # summary is its image embedding.
            assert self._torch is not None
            nearest = self._model.get_nearest_supported_resolution(*pixels.shape[-2:])
            pixels = self._torch.nn.functional.interpolate(
                pixels, size=tuple(nearest), mode="bilinear", align_corners=False
            )
            output = self._model(pixels)
            return output[0] if isinstance(output, tuple) else output.summary
        output = self._model(pixel_values=pixels)
        if self._pooling == "model":
            pooled = getattr(output, "pooler_output", None)
            if pooled is None:
                raise EmbeddingConfigurationError(f"{self.model_id} has no pooled output")
            return pooled
        tokens = output.last_hidden_state
        prefix = 1 if self._pooling == "cls" else _prefix_tokens(self._model)
        return pool_tokens(tokens, prefix, self._pooling)


def _prefix_tokens(model: Any) -> int:
    """How many non-patch tokens a transformers vision model prepends: its CLS, if any."""
    return 1 if getattr(getattr(model, "embeddings", None), "cls_token", None) is not None else 0


__all__ = ["TimmBackbone", "TransformersBackbone", "pool_tokens"]

"""The pretraining-family adapters of AGENTS.md section 7.17, tested offline."""

from typing import Any

import numpy as np
import pytest
from PIL import Image
from pydantic import ValidationError

from krasnal_id.config import BackboneConfig, backbone_config, load_config
from krasnal_id.embeddings.backbone import EmbeddingConfigurationError
from krasnal_id.embeddings.clip import ClipBackbone
from krasnal_id.embeddings.extract import create_backbone
from krasnal_id.embeddings.generic import TimmBackbone, TransformersBackbone, pool_tokens

FAMILY_STUDY = {
    "dino": TimmBackbone,
    "mae": TimmBackbone,
    "eva02-mim": TimmBackbone,
    "aimv2": TimmBackbone,
    "vit-supervised": TimmBackbone,
    "beit": TransformersBackbone,
    "ijepa": TransformersBackbone,
    "radio": TransformersBackbone,
    "clip-b16": ClipBackbone,
}


def _config(**changes: Any) -> BackboneConfig:
    base: dict[str, Any] = {
        "name": "dino",
        "family": "timm",
        "model_id": "timm/vit_base_patch16_224.dino",
        "revision": "a" * 40,
        "preprocessing_id": "timm-data-config;pool=model",
        "pooling": "model",
        "batch_size": 4,
    }
    return BackboneConfig.model_validate(base | changes)


def test_every_family_checkpoint_is_pinned_dispatched_and_unpublished() -> None:
    exported = load_config().export.backbones
    for name, adapter in FAMILY_STUDY.items():
        config = backbone_config(name)
        assert len(config.revision) == 40, name  # a commit, never a branch
        assert isinstance(create_backbone(config), adapter), name
        assert name not in exported, name


def test_a_generic_pooling_must_key_the_cache() -> None:
    """Two poolings of one checkpoint are two embeddings, so the key must differ."""
    with pytest.raises(ValidationError, match="must end with ';pool=mean'"):
        _config(pooling="mean")
    with pytest.raises(ValidationError, match="requires a pooling"):
        _config(pooling=None)
    with pytest.raises(ValidationError, match="does not take a pooling"):
        _config(family="dinov2", model_id="facebook/dinov2-base", name="dinov2")


def test_remote_code_and_processor_arguments_are_transformers_only_and_recorded() -> None:
    with pytest.raises(ValidationError, match="apply only to transformers"):
        _config(trust_remote_code=True)
    with pytest.raises(ValidationError, match="must record do_resize=True"):
        _config(
            family="transformers",
            preprocessing_id="processor;pool=model",
            processor_kwargs={"do_resize": True},
        )
    radio = backbone_config("radio")
    assert radio.trust_remote_code and radio.processor_kwargs == {"do_resize": True}
    assert radio.batch_size == 1  # aspect is kept, so images differ in size


def test_token_pooling_skips_prefix_tokens() -> None:
    torch = pytest.importorskip("torch")
    tokens = torch.tensor([[[9.0, 9.0], [1.0, 3.0], [3.0, 5.0]]])
    assert pool_tokens(tokens, 1, "cls").tolist() == [[9.0, 9.0]]
    assert pool_tokens(tokens, 1, "mean").tolist() == [[2.0, 4.0]]
    np.testing.assert_allclose(
        pool_tokens(tokens, 0, "mean").numpy(), [[13 / 3, 17 / 3]], rtol=1e-6
    )
    with pytest.raises(EmbeddingConfigurationError, match="class token"):
        pool_tokens(tokens, 0, "cls")
    with pytest.raises(EmbeddingConfigurationError, match="cannot apply"):
        pool_tokens(tokens, 1, "summary")


def test_adapters_load_nothing_until_used_and_refuse_other_families() -> None:
    backbone = create_backbone(backbone_config("radio"))
    assert backbone.preprocessing_id.endswith(";pool=summary")
    with pytest.raises(ValueError, match="pooled timm config"):
        TimmBackbone(backbone_config("beit"))
    with pytest.raises(ValueError, match="pooled transformers config"):
        TransformersBackbone(backbone_config("dino"))
    with pytest.raises(ValueError, match="at least one image"):
        backbone.get_embeddings(())


class _Output:
    def __init__(self, torch: Any, pooled: bool) -> None:
        self.last_hidden_state = torch.tensor([[[0.0, 1.0], [3.0, 4.0], [5.0, 12.0]]])
        self.pooler_output = torch.tensor([[3.0, 4.0]]) if pooled else None


class _Model:
    def __init__(self, torch: Any, pooled: bool, cls: bool) -> None:
        self._torch, self._pooled = torch, pooled
        self.embeddings = type("E", (), {"cls_token": object() if cls else None})()

    def __call__(self, pixel_values: Any) -> _Output:
        return _Output(self._torch, self._pooled)


def _processor(torch: Any) -> Any:
    return lambda images, return_tensors, **_: {"pixel_values": torch.zeros(len(images), 3, 2, 2)}


@pytest.mark.parametrize(
    ("name", "pooled", "cls", "expected"),
    [("beit", True, True, [0.6, 0.8]), ("ijepa", False, False, [8 / 3, 17 / 3])],
)
def test_transformers_pooling_reads_the_configured_output(
    name: str, pooled: bool, cls: bool, expected: list[float]
) -> None:
    torch = pytest.importorskip("torch")
    backbone = TransformersBackbone(backbone_config(name))
    backbone._torch, backbone._device = torch, "cpu"
    backbone._processor = _processor(torch)
    backbone._model = _Model(torch, pooled, cls)
    vector = backbone.get_embedding(Image.new("RGB", (2, 2)))
    np.testing.assert_allclose(vector, np.asarray(expected) / np.linalg.norm(expected), atol=1e-6)


def test_a_pooled_config_on_a_model_without_a_pooled_output_is_refused() -> None:
    torch = pytest.importorskip("torch")
    backbone = TransformersBackbone(backbone_config("beit"))
    backbone._torch, backbone._device = torch, "cpu"
    backbone._processor = _processor(torch)
    backbone._model = _Model(torch, pooled=False, cls=True)
    with pytest.raises(EmbeddingConfigurationError, match="no pooled output"):
        backbone.get_embedding(Image.new("RGB", (2, 2)))

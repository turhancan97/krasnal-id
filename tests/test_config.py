"""Composition tests for every Hydra configuration branch."""

import pytest
from pydantic import ValidationError

from krasnal_id.config import WikimediaDataConfig, load_config


@pytest.mark.parametrize(
    ("override", "expected_kind"),
    [
        ("experiment=baseline", "baseline"),
        ("experiment=pool_size_ablation", "pool_size_ablation"),
        ("experiment=confusion", "confusion"),
        ("experiment=visualization", "visualization"),
    ],
)
def test_composes_every_experiment(override: str, expected_kind: str) -> None:
    config = load_config([override])

    assert config.experiment.kind == expected_kind
    assert config.thresholds.minimum_images_per_dwarf == 3


def test_composes_clip_backbone_and_runtime_override() -> None:
    config = load_config(["backbone=clip", "logging.json_output=false"])

    assert config.backbone.name == "clip"
    assert config.logging.json_output is False


def test_rejects_invalid_typed_override() -> None:
    with pytest.raises(ValidationError):
        load_config(["thresholds.minimum_images_per_dwarf=2"])


@pytest.mark.parametrize(
    "updates",
    [
        {"max_attempts": 4},
        {"retry_backoff_seconds": [-1.0, 2.0]},
    ],
)
def test_rejects_invalid_wikidata_retry_schedule(updates: dict[str, object]) -> None:
    raw_config = load_config().data.model_dump()
    raw_config.update(updates)

    with pytest.raises(ValidationError):
        WikimediaDataConfig.model_validate(raw_config)

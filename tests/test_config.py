"""Composition tests for every Hydra configuration branch."""

from importlib import metadata

import pytest
from pydantic import ValidationError

import krasnal_id
from krasnal_id.config import WikimediaDataConfig, load_config


def test_the_package_version_matches_the_distribution() -> None:
    """Cutting a release must bump both, and 0.8.0 shipped having bumped only one.

    The module constant sat at 0.7.0 for a whole release while the distribution
    said 0.8.0. Nothing consumed the constant, so nothing complained; this does.
    """
    assert krasnal_id.__version__ == metadata.version("krasnal-id")


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
    assert config.data.image_max_long_side == 2000
    assert config.data.image_min_short_side == 400
    assert config.data.allowed_license_families == ("public-domain", "cc0", "cc-by", "cc-by-sa")
    assert config.paths.category_review_path.as_posix() == "data/category-review.json"
    assert config.paths.image_review_path.as_posix() == "data/image-review.json"

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
        {"image_min_short_side": 2001},
        {"allowed_license_families": ["unknown"]},
        {"allowed_license_families": ["cc-by", "cc-by"]},
    ],
)
def test_rejects_invalid_wikidata_retry_schedule(updates: dict[str, object]) -> None:
    raw_config = load_config().data.model_dump()
    raw_config.update(updates)

    with pytest.raises(ValidationError):
        WikimediaDataConfig.model_validate(raw_config)

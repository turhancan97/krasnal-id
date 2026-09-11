"""Composition tests for every Hydra configuration branch."""

from importlib import metadata
from importlib.resources import files

import pytest
from pydantic import ValidationError

import krasnal_id
from krasnal_id.config import (
    BackboneConfig,
    WikimediaDataConfig,
    backbone_config,
    load_config,
)


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


def _packaged_backbone_names() -> list[str]:
    """Every backbone the config package ships, by Hydra group option name."""
    group = files("krasnal_id.configs").joinpath("backbone")
    return sorted(
        entry.name.removesuffix(".yaml")
        for entry in group.iterdir()
        if entry.name.endswith(".yaml")
    )


def test_every_packaged_backbone_composes_and_names_itself_after_its_file() -> None:
    """A checkpoint added as a file must be reachable as `backbone=<file stem>`.

    Section 8's capacity question turns on comparing four DINOv2 checkpoints, and
    a file whose `name` disagreed with its stem would be composed under one
    identity and written under another.
    """
    for option in _packaged_backbone_names():
        config = backbone_config(option)

        assert config.name == option


def test_backbone_names_are_unique_across_the_package() -> None:
    """Two backbones sharing a name would overwrite each other's result file."""
    names = [backbone_config(option).name for option in _packaged_backbone_names()]

    assert sorted(names) == sorted(set(names))


def test_the_four_dinov2_checkpoints_are_one_family_and_four_identities() -> None:
    """The cross section 8 records: distinct checkpoints, one adapter."""
    cells = ["dinov2", "dinov2-large", "dinov2-registers", "dinov2-registers-large"]
    configs = [backbone_config(name) for name in cells]

    assert {config.family for config in configs} == {"dinov2"}
    assert len({config.model_id for config in configs}) == len(cells)
    assert len({config.revision for config in configs}) == len(cells)


def test_a_backbone_cannot_declare_a_family_with_no_adapter() -> None:
    raw_config = backbone_config("dinov2").model_dump()
    raw_config["family"] = "dinov3"

    with pytest.raises(ValidationError):
        BackboneConfig.model_validate(raw_config)

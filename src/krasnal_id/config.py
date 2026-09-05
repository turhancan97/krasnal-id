"""Hydra composition and Pydantic validation for application configuration."""

from pathlib import Path
from typing import Annotated, Literal

from hydra import compose, initialize_config_module
from omegaconf import OmegaConf
from pydantic import BaseModel, ConfigDict, Field, HttpUrl, model_validator


class PathsConfig(BaseModel):
    """Filesystem locations for local artifacts."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    data_dir: Path
    images_dir: Path
    embeddings_dir: Path
    evaluation_split_path: Path
    discovery_dir: Path
    category_review_path: Path
    image_review_path: Path
    manifest_path: Path
    results_dir: Path


class WikimediaDataConfig(BaseModel):
    """Wikimedia API and dataset filtering settings."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    wikidata_endpoint: HttpUrl
    commons_api_endpoint: HttpUrl
    request_timeout_seconds: float = Field(gt=0)
    max_attempts: int = Field(ge=1)
    retry_backoff_seconds: tuple[float, ...]
    max_retry_after_seconds: float = Field(gt=0)
    commons_root_category: str = Field(min_length=1)
    image_max_long_side: int = Field(gt=0)
    image_min_short_side: int = Field(gt=0)
    allowed_license_families: tuple[str, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def validate_retry_schedule(self) -> "WikimediaDataConfig":
        """Validate retry, image-size, and license-policy invariants."""
        if len(self.retry_backoff_seconds) != self.max_attempts - 1:
            raise ValueError("retry_backoff_seconds must contain max_attempts - 1 values")
        if any(delay < 0 for delay in self.retry_backoff_seconds):
            raise ValueError("retry backoff values cannot be negative")
        if self.image_min_short_side > self.image_max_long_side:
            raise ValueError("image_min_short_side cannot exceed image_max_long_side")
        supported = {"public-domain", "cc0", "cc-by", "cc-by-sa"}
        if not set(self.allowed_license_families).issubset(supported):
            raise ValueError("allowed_license_families contains an unknown family")
        if len(self.allowed_license_families) != len(set(self.allowed_license_families)):
            raise ValueError("allowed_license_families cannot contain duplicates")
        return self


class ThresholdsConfig(BaseModel):
    """Dataset and analysis thresholds kept out of implementation code."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    minimum_images_per_dwarf: int = Field(ge=3)
    confusion_top_pairs: int = Field(gt=0)
    open_set_top_rejections: int = Field(gt=0)
    max_coordinate_drift_km: float = Field(gt=0)


class SeedsConfig(BaseModel):
    """Named deterministic seeds shared by experiment configurations."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    primary: int
    ablation: tuple[int, ...] = Field(min_length=1)


class BackboneConfig(BaseModel):
    """Pinned embedding-backbone identity and preprocessing contract."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    name: Literal["dinov2", "clip"]
    model_id: str = Field(min_length=1)
    revision: str = Field(min_length=1)
    preprocessing_id: str = Field(min_length=1)
    device: Literal["auto", "cpu", "cuda"] = "auto"
    batch_size: int = Field(gt=0)


class BaselineExperimentConfig(BaseModel):
    """Full-pool retrieval evaluation settings."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    kind: Literal["baseline"]
    seed: int
    top_k: tuple[int, ...] = Field(min_length=1)


class PoolSizeAblationConfig(BaseModel):
    """Synthetic candidate-pool ablation settings."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    kind: Literal["pool_size_ablation"]
    pool_sizes: tuple[int, ...] = Field(min_length=1)
    seeds: tuple[int, ...] = Field(min_length=1)


class GeoAblationConfig(BaseModel):
    """Real coordinate-based candidate-pool settings."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    kind: Literal["geo_ablation"]
    pool_sizes: tuple[int, ...] = Field(min_length=1)
    seeds: tuple[int, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def validate_pool_sizes(self) -> "GeoAblationConfig":
        """Require pool sizes a candidate set can actually take."""
        if any(size < 2 for size in self.pool_sizes):
            raise ValueError("pool sizes must be at least two")
        return self


class ProbeExperimentConfig(BaseModel):
    """Trained-classifier comparison settings."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    kind: Literal["probe"]
    seed: int
    methods: tuple[Literal["retrieval", "prototype", "linear_probe"], ...] = Field(min_length=1)
    top_k: tuple[int, ...] = Field(min_length=1)
    max_iterations: int = Field(gt=0)
    regularization: float = Field(gt=0)

    @model_validator(mode="after")
    def validate_methods(self) -> "ProbeExperimentConfig":
        """Require distinct methods and positive cut-offs."""
        if len(self.methods) != len(set(self.methods)):
            raise ValueError("methods cannot contain duplicates")
        if any(k <= 0 for k in self.top_k):
            raise ValueError("top_k values must be positive")
        return self


class OpenSetExperimentConfig(BaseModel):
    """Unknown-query rejection settings."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    kind: Literal["open_set"]
    seed: int
    target_known_acceptance: tuple[float, ...] = Field(min_length=1)
    top_rejections: int = Field(gt=0)

    @model_validator(mode="after")
    def validate_targets(self) -> "OpenSetExperimentConfig":
        """Require distinct acceptance targets a quantile can actually take."""
        if any(not 0.0 < target < 1.0 for target in self.target_known_acceptance):
            raise ValueError("target_known_acceptance values must lie strictly between 0 and 1")
        if len(self.target_known_acceptance) != len(set(self.target_known_acceptance)):
            raise ValueError("target_known_acceptance cannot contain duplicates")
        return self

    @property
    def primary_target(self) -> float:
        """Return the target whose operating point the per-dwarf rows describe."""
        return self.target_known_acceptance[0]


class CameraGapExperimentConfig(BaseModel):
    """Phone-versus-camera query comparison settings."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    kind: Literal["camera_gap"]
    seed: int
    top_k: tuple[int, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def validate_top_k(self) -> "CameraGapExperimentConfig":
        """Require positive cut-offs."""
        if any(k <= 0 for k in self.top_k):
            raise ValueError("top_k values must be positive")
        return self


class ConfusionExperimentConfig(BaseModel):
    """Most-confused-pair analysis settings."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    kind: Literal["confusion"]
    seed: int
    top_pairs: int = Field(gt=0)


class VisualizationExperimentConfig(BaseModel):
    """Embedding projection settings."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    kind: Literal["visualization"]
    method: Literal["umap", "tsne"]
    seed: int


ExperimentConfig = Annotated[
    BaselineExperimentConfig
    | PoolSizeAblationConfig
    | GeoAblationConfig
    | ProbeExperimentConfig
    | OpenSetExperimentConfig
    | CameraGapExperimentConfig
    | ConfusionExperimentConfig
    | VisualizationExperimentConfig,
    Field(discriminator="kind"),
]


class LoggingConfig(BaseModel):
    """Structured application logging settings."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
    json_output: bool


class AppConfig(BaseModel):
    """Fully composed and validated application configuration."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    data: WikimediaDataConfig
    backbone: BackboneConfig
    experiment: ExperimentConfig
    logging: LoggingConfig
    paths: PathsConfig
    thresholds: ThresholdsConfig
    seeds: SeedsConfig


def load_config(overrides: list[str] | None = None) -> AppConfig:
    """Compose the packaged Hydra configuration and validate its complete shape."""
    with initialize_config_module(config_module="krasnal_id.configs", version_base=None):
        composed = compose(config_name="config", overrides=overrides or [])
    raw_config = OmegaConf.to_container(composed, resolve=True)
    return AppConfig.model_validate(raw_config)

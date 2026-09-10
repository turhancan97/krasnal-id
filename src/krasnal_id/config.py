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
    field_queries_dir: Path
    field_route_path: Path
    category_review_path: Path
    image_review_path: Path
    manifest_path: Path
    results_dir: Path
    huggingface_export_dir: Path
    kaggle_export_dir: Path


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


class OpenSetGeometryConfig(BaseModel):
    """Geometric open-set rejection settings."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    kind: Literal["open_set_geometry"]
    seed: int
    # How many candidates from the cosine ranking get their geometry checked. The
    # unknown arm has no correct answer, so a wider k only offers more chances to
    # fluke a homography -- the same effect section 7.7 measured when widening
    # verification from 10 to 50 bought 0.18 points.
    top_k: int = Field(ge=1)
    max_keypoints: int = Field(ge=16)
    # The weight geometry carries in the blended signal. A single value, not a
    # sweep: section 7.6 already swept it for accuracy, and this experiment asks a
    # different question of the same evidence.
    blend_weight: float = Field(ge=0.0)
    target_known_acceptance: float = Field(gt=0.0, lt=1.0)
    # Repeat every signal with the query's own photographer withheld from both
    # arms. Section 7.6 found the inlier separation inflated by same-visit
    # near-duplicates, so a geometric signal that only works when the same person
    # shot the reference has not been shown to work at all.
    photographer_disjoint: bool = True


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


class FieldGapExperimentConfig(BaseModel):
    """Field-photograph query-domain gap settings."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    kind: Literal["field_gap"]
    seed: int
    top_k: tuple[int, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def validate_top_k(self) -> "FieldGapExperimentConfig":
        """Require positive cut-offs."""
        if any(k <= 0 for k in self.top_k):
            raise ValueError("top_k values must be positive")
        return self


class PhotographerGapConfig(BaseModel):
    """Photographer-disjoint evaluation settings."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    kind: Literal["photographer_gap"]
    # Several seeds because the size-matched control samples its reference set,
    # and one draw of it would not show how much the sampling itself moves.
    seeds: tuple[int, ...] = Field(min_length=1)
    top_k: tuple[int, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def validate_settings(self) -> "PhotographerGapConfig":
        """Require positive cut-offs and distinct seeds."""
        if any(k <= 0 for k in self.top_k):
            raise ValueError("top_k values must be positive")
        if len(set(self.seeds)) != len(self.seeds):
            raise ValueError("seeds cannot contain duplicates")
        return self


class RerankAblationConfig(BaseModel):
    """Geometric re-ranking sweep settings."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    kind: Literal["rerank_ablation"]
    seed: int
    # How many distinct statues from the global ranking get their geometry
    # checked. A homography per candidate costs milliseconds against a dot
    # product's microseconds, so it is spent only where the ranking is uncertain.
    top_k: int = Field(ge=2)
    max_keypoints: int = Field(ge=16)
    # The blend weights swept. Zero is the unranked control and is required.
    weights: tuple[float, ...] = Field(min_length=1)
    top_k_metrics: tuple[int, ...] = Field(min_length=1)
    # Run the sweep twice over the answerable queries — once on all references and
    # once with the query's own photographer withheld — so the gain can be read
    # against the regime that produced it. Section 7.6 needs this because its
    # inlier separation is inflated by same-visit near-duplicates.
    photographer_disjoint: bool = False

    @model_validator(mode="after")
    def validate_settings(self) -> "RerankAblationConfig":
        """Require the control weight, non-negative weights and positive cut-offs."""
        if any(weight < 0.0 for weight in self.weights):
            raise ValueError("blend weights cannot be negative")
        if 0.0 not in self.weights:
            raise ValueError("the sweep must include weight 0.0, the unranked control")
        if any(k <= 0 for k in self.top_k_metrics):
            raise ValueError("top_k_metrics values must be positive")
        return self


class RecallCurveConfig(BaseModel):
    """First-stage recall diagnostic settings."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    kind: Literal["recall_curve"]
    seed: int
    top_k: tuple[int, ...] = Field(min_length=1)
    # Backbones whose similarities are summed for the fused variant. Independent
    # of the selected backbone, which supplies the plain and expanded rows, so
    # the fused figure is the same in either artifact.
    fuse_backbones: tuple[Literal["dinov2", "clip"], ...] = ()
    # Neighbour counts for query expansion. Empty skips the variant.
    expansion_neighbours: tuple[int, ...] = ()
    expansion_alpha: float = Field(default=3.0, ge=0.0)

    @model_validator(mode="after")
    def validate_settings(self) -> "RecallCurveConfig":
        """Require positive cut-offs and neighbour counts, and distinct backbones."""
        if any(k <= 0 for k in self.top_k):
            raise ValueError("top_k values must be positive")
        if any(n <= 0 for n in self.expansion_neighbours):
            raise ValueError("expansion_neighbours values must be positive")
        if len(set(self.fuse_backbones)) != len(self.fuse_backbones):
            raise ValueError("fuse_backbones cannot contain duplicates")
        return self


class ConfusionExperimentConfig(BaseModel):
    """Most-confused-pair analysis settings."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    kind: Literal["confusion"]
    seed: int
    top_pairs: int = Field(gt=0)


class VisualizationExperimentConfig(BaseModel):
    """Embedding projection and contact-sheet settings."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    kind: Literal["visualization"]
    method: Literal["umap", "tsne"]
    seed: int
    # Which backbones the retrieval contact sheet draws a row for. The
    # projection commands use one backbone at a time and ignore this.
    backbones: tuple[str, ...] = Field(min_length=1)


ExperimentConfig = Annotated[
    BaselineExperimentConfig
    | PoolSizeAblationConfig
    | GeoAblationConfig
    | ProbeExperimentConfig
    | OpenSetExperimentConfig
    | OpenSetGeometryConfig
    | CameraGapExperimentConfig
    | FieldGapExperimentConfig
    | PhotographerGapConfig
    | RerankAblationConfig
    | RecallCurveConfig
    | ConfusionExperimentConfig
    | VisualizationExperimentConfig,
    Field(discriminator="kind"),
]


class ExportConfig(BaseModel):
    """Dataset export settings, shared by the Hugging Face and Kaggle writers."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    repo_id: str = Field(pattern=r"^[A-Za-z0-9][A-Za-z0-9._-]*/[A-Za-z0-9][A-Za-z0-9._-]*$")
    # Kaggle slugs are lowercase, hyphenated and 3-50 characters; the export
    # checks the length itself so a rejection costs a message, not an upload.
    kaggle_id: str = Field(pattern=r"^[A-Za-z0-9][A-Za-z0-9-]*/[A-Za-z0-9][A-Za-z0-9-]*$")
    backbones: tuple[Literal["dinov2", "clip"], ...] = Field(min_length=1)
    # Under the Hub's ~500 MB shard convention, with headroom for the estimate
    # being wrong: the size is projected from file sizes before any bytes are read.
    shard_target_bytes: int = Field(gt=0)
    # The Hub documents 100 rows per row group for image data, which is what the
    # datasets library itself writes.
    row_group_rows: int = Field(gt=0)
    license_name: str = Field(min_length=1)
    license_link: str = Field(min_length=1)

    @model_validator(mode="after")
    def validate_backbones(self) -> "ExportConfig":
        """Require distinct backbones, so a vector column cannot be written twice."""
        if len(set(self.backbones)) != len(self.backbones):
            raise ValueError("backbones cannot contain duplicates")
        return self


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
    export: ExportConfig
    logging: LoggingConfig
    paths: PathsConfig
    thresholds: ThresholdsConfig
    seeds: SeedsConfig


def backbone_config(name: str, overrides: list[str] | None = None) -> BackboneConfig:
    """Compose one named backbone's pinned configuration.

    `AppConfig.backbone` is whichever backbone the overrides selected, but an
    export writes a vector column per backbone and therefore needs several in one
    run. This keeps `initialize_config_module` from leaking out of this module.
    """
    return load_config([*(overrides or []), f"backbone={name}"]).backbone


def load_config(overrides: list[str] | None = None) -> AppConfig:
    """Compose the packaged Hydra configuration and validate its complete shape."""
    with initialize_config_module(config_module="krasnal_id.configs", version_base=None):
        composed = compose(config_name="config", overrides=overrides or [])
    raw_config = OmegaConf.to_container(composed, resolve=True)
    return AppConfig.model_validate(raw_config)

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
    field_inbox_dir: Path
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
    # GPS sorting of field photographs (`data field-sort`). A fix further than the
    # first from every route statue is off the route; any statue within the second
    # of the nearest makes the proposal ambiguous rather than a single guess.
    field_sort_max_distance_m: float = Field(gt=0)
    field_sort_ambiguity_m: float = Field(ge=0)


class SeedsConfig(BaseModel):
    """Named deterministic seeds shared by experiment configurations."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    primary: int
    ablation: tuple[int, ...] = Field(min_length=1)


# Every backbone identity the config package ships. Declared once, because the
# experiments that name backbones in a list -- fusing them, or comparing against
# them -- must accept exactly the identities that can be selected. Keeping two
# copies of this let `fuse_backbones` reject a checkpoint the pipeline could run.
BackboneName = Literal[
    "dinov2",
    "clip",
    "dinov2-large",
    "dinov2-registers",
    "dinov2-registers-large",
    "dinov3",
    "siglip2",
]


# Every local matcher `create_matcher` can build. Declared once for the same reason
# as `BackboneName`: three experiments name matchers, and a copy per experiment
# lets one of them reject a matcher the others can run.
MatcherName = Literal["sift", "disk-lightglue", "loftr"]


class BackboneConfig(BaseModel):
    """Pinned embedding-backbone identity and preprocessing contract.

    `name` is the artifact identity: it names the result file, the exported vector
    column and the cached-vector owner. `family` is the adapter that knows how to
    run the checkpoint. They were the same field until §8's capacity question
    needed four DINOv2 checkpoints, which are one family and four identities --
    and collapsing them would have made `recall_curve-dinov2.json` mean whichever
    checkpoint ran last.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    name: BackboneName
    family: Literal["dinov2", "dinov3", "clip", "siglip"]
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
    # Percentile bootstrap over statues for every gap's interval. Statues, not
    # photographs, because a walk's photographs of one statue are not independent.
    bootstrap_resamples: int = Field(ge=100)

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

    @property
    def artifact_variant(self) -> str | None:
        """Name the arm this run measures, so the two do not share a filename.

        Withholding the query's own photographer is a different measurement, not a
        different setting of one -- see §6.4.
        """
        return "disjoint" if self.photographer_disjoint else None


class RecallCurveConfig(BaseModel):
    """First-stage recall diagnostic settings."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    kind: Literal["recall_curve"]
    seed: int
    top_k: tuple[int, ...] = Field(min_length=1)
    # Backbones whose similarities are summed for the fused variant. Independent
    # of the selected backbone, which supplies the plain and expanded rows, so
    # the fused figure is the same in either artifact.
    fuse_backbones: tuple[BackboneName, ...] = ()
    # Backbones scored on the same folds as the selected one and compared to it
    # pairwise. Separate from `fuse_backbones`, which combines two backbones into
    # one ranking: these stay separate rankings and are only ever differenced.
    compare_backbones: tuple[BackboneName, ...] = ()
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
        if len(set(self.compare_backbones)) != len(self.compare_backbones):
            raise ValueError("compare_backbones cannot contain duplicates")
        return self


class MatcherRerankConfig(BaseModel):
    """Several local matchers through the re-ranking protocol, side by side."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    kind: Literal["matcher_rerank"]
    seed: int
    # The first entry is the baseline every other matcher is paired against, so
    # it should be the one whose result is already published.
    matchers: tuple[MatcherName, ...] = Field(min_length=1)
    top_k: int = Field(ge=2)
    # SIFT's budget, matched to §7.6 so its column lands against the published
    # 94.0%. A learned detector gets its own, because holding it to SIFT's would
    # handicap the thing being tested.
    max_keypoints: int = Field(ge=16)
    # The learned detector's budget. 1024 rather than the 2048 LightGlue's own
    # evaluations use, and it costs nothing here: `blended_score` caps inliers at
    # `INLIER_CAP`, so every pair scoring above 30 blends identically. Measured on
    # four pairs, 1024 gives 51/115/51 inliers where 2048 gives 96/208/155 -- all
    # above the cap, all the same score -- for 2.4x less compute.
    learned_keypoints: int = Field(default=1024, ge=16)
    weights: tuple[float, ...] = Field(min_length=1)
    top_k_metrics: tuple[int, ...] = Field(min_length=1)
    photographer_disjoint: bool = False
    device: Literal["auto", "cpu", "cuda"] = "auto"

    @model_validator(mode="after")
    def validate_settings(self) -> "MatcherRerankConfig":
        """Require the control weight, distinct matchers and positive cut-offs."""
        if any(weight < 0.0 for weight in self.weights):
            raise ValueError("blend weights cannot be negative")
        if 0.0 not in self.weights:
            raise ValueError("the zero control weight is required")
        if any(k <= 0 for k in self.top_k_metrics):
            raise ValueError("top_k_metrics values must be positive")
        if len(set(self.matchers)) != len(self.matchers):
            raise ValueError("matchers cannot contain duplicates")
        return self

    @property
    def artifact_variant(self) -> str | None:
        """Name the arm this run measures, so the two do not share a filename.

        Withholding the query's own photographer is a different measurement, not a
        different setting of one -- see §6.4.
        """
        return "disjoint" if self.photographer_disjoint else None


class GeometryFirstConfig(BaseModel):
    """Local features promoted from re-ranker to first stage."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    kind: Literal["geometry_first"]
    seed: int
    top_k: tuple[int, ...] = Field(min_length=1)
    # Matched to the re-ranking sweep so the two experiments describe the same
    # keypoints; changing it here would make their inlier counts incomparable.
    max_keypoints: int = Field(ge=16)
    # Stop after this many queries. Zero means every answerable query, which is
    # the published run; a small value is for exercising the path, since a full
    # sweep is 1,955,330 homographies.
    max_queries: int = Field(default=0, ge=0)
    # Which local matcher ranks the references. SIFT is §7.10's published run;
    # `disk-lightglue` asks §7.11's follow-up -- whether the matcher that wins at
    # re-ranking can also retrieve.
    matcher: MatcherName = "sift"
    # The learned detector's own budget, as in `MatcherRerankConfig`; SIFT keeps
    # `max_keypoints` so its rows stay comparable with §7.10's journal.
    learned_keypoints: int = Field(default=1024, ge=16)
    device: Literal["auto", "cpu", "cuda"] = "auto"
    # Score only the queries whose statue appearance leaves outside its top k in
    # the photographer-disjoint arm. Zero scores every answerable query. The
    # rescue rate is defined over exactly those queries, so a learned matcher at
    # 5-25x SIFT's cost per pair can answer it in an hour rather than ten.
    only_appearance_misses_at: int = Field(default=0, ge=0)

    @model_validator(mode="after")
    def validate_settings(self) -> "GeometryFirstConfig":
        """Require positive cut-offs."""
        if any(k <= 0 for k in self.top_k):
            raise ValueError("top_k values must be positive")
        return self

    @property
    def artifact_variant(self) -> str | None:
        """Name a run that is not §7.10's, so it never overwrites that artifact.

        A different matcher measures something else, and a restricted query set
        changes what every `answerable` metric means -- both are §6.4's case for a
        separate filename. The default SIFT sweep keeps the published name.
        """
        parts: list[str] = []
        if self.matcher != "sift":
            parts.append(self.matcher)
        if self.only_appearance_misses_at:
            parts.append(f"misses{self.only_appearance_misses_at}")
        return "-".join(parts) or None


class WorkshopValidationConfig(BaseModel):
    """Re-checks of published results for the CVWW paper, from evidence already on disk.

    Nothing here loads a model or matches an image. The re-ranking part reads the
    two matcher journals; it never constructs a matcher. The identity settings
    below (`rerank_top_k`, `max_keypoints`, `learned_keypoints`) are the ones those
    journals were written under and must equal `matcher_rerank.yaml`'s, or the
    journals will not be found -- a test pins the equality.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    kind: Literal["workshop_validation"]
    seed: int
    # Statue-level folds for choosing the blend weight on queries it does not judge.
    folds: int = Field(ge=2)
    bootstrap_resamples: int = Field(ge=100)
    matchers: tuple[MatcherName, ...] = Field(min_length=1)
    weights: tuple[float, ...] = Field(min_length=1)
    top_k_metrics: tuple[int, ...] = Field(min_length=1)
    rerank_top_k: int = Field(ge=2)
    max_keypoints: int = Field(ge=16)
    learned_keypoints: int = Field(ge=16)
    # Written below `paths.results_dir`, so nothing here can overwrite a released
    # artifact; the journals are still read from `paths.results_dir/journals`.
    output_subdir: str = Field(min_length=1, pattern=r"^[A-Za-z0-9_.-]+$")
    # Which re-checks to run. `rerank` reads journals only; the rest recompute cosine
    # rankings from cached vectors (no model) and refuse to report an interval
    # unless they first reproduce the released artifact exactly.
    parts: tuple[
        Literal[
            "rerank",
            "photographer",
            "baseline",
            "geography",
            "pool",
            "soft_prior",
            "geo_noise",
            "rejection_signals",
            "groups",
            "geo_permutation",
            "families",
            "null_bounds",
        ],
        ...,
    ] = Field(min_length=1)
    backbones: tuple[BackboneName, ...] = Field(min_length=1)
    # Backbones whose released artifacts live in `results_dir/<name>/` rather than
    # `results_dir/`, so no figure glob picks them up (section 7.12).
    separate_results: tuple[BackboneName, ...] = ()
    # Section 7.15. A second, wider shortlist re-ranked by a subset of the matchers;
    # its metrics carry a `cv_k<k>_` prefix so they never collide with the main ones.
    wide_top_k: int | None = Field(default=None, ge=2)
    wide_matchers: tuple[MatcherName, ...] = ()
    # Pairs reported beside the standard ones (each matcher against cosine, each
    # later matcher against the first), as (better, base).
    extra_pairs: tuple[tuple[MatcherName, MatcherName], ...] = ()
    # Soft geographic prior: candidate score = cosine + w * exp(-distance / scale).
    soft_prior_weights: tuple[float, ...] = (0.0,)
    soft_prior_scale_m: float = Field(default=200.0, gt=0.0)
    # Where the simulated user stands, metres from the statue in a seeded direction.
    # Never zero alone: at zero a strong prior just names the statue under the user.
    gps_error_m: tuple[float, ...] = (25.0,)
    # Statue coordinates moved by these distances to test the geographic result.
    noise_metres: tuple[float, ...] = ()
    noise_seeds: tuple[int, ...] = (0,)
    # Rejection signal: similarity to the k-th nearest reference image.
    knn_k: int = Field(default=5, ge=2)
    # Group classes and the member classes they overlap (section 7.3).
    groups: dict[str, tuple[str, ...]] = Field(default_factory=dict)
    # Section 7.16. Shuffled statue layouts for the co-location permutation test.
    permutation_count: int = Field(default=200, ge=19)
    permutation_pool_sizes: tuple[int, ...] = (5, 10, 20)
    # How many of the *other* backbones must call a pair mutual nearest neighbours.
    family_votes: int = Field(default=2, ge=1)
    # Each statue must be within the other's this-many nearest; 1 is mutual nearest
    # neighbours, which can only form pairs (section 7.16).
    family_neighbours: int = Field(default=3, ge=1)
    # Checkpoints whose null result against dinov2 gets an upper bound.
    null_bound_backbones: tuple[BackboneName, ...] = ()
    null_bound_cutoffs: tuple[int, ...] = (1, 10)

    @model_validator(mode="after")
    def validate_settings(self) -> "WorkshopValidationConfig":
        """Require the control weight, distinct matchers and positive cut-offs."""
        if any(weight < 0.0 for weight in self.weights):
            raise ValueError("blend weights cannot be negative")
        if 0.0 not in self.weights:
            raise ValueError("the zero control weight is required")
        if any(k <= 0 for k in self.top_k_metrics):
            raise ValueError("top_k_metrics values must be positive")
        if len(set(self.matchers)) != len(self.matchers):
            raise ValueError("matchers cannot contain duplicates")
        if 0.0 not in self.soft_prior_weights or any(w < 0 for w in self.soft_prior_weights):
            raise ValueError("soft prior weights must be non-negative and include zero")
        if any(error <= 0 for error in self.gps_error_m):
            raise ValueError("gps_error_m must be positive; zero error measures nothing")
        if any(metres <= 0 for metres in self.noise_metres):
            raise ValueError("noise_metres must be positive; the released arm is the zero")
        if not set(self.wide_matchers) <= set(self.matchers):
            raise ValueError("wide_matchers must be a subset of matchers")
        if self.wide_matchers and self.wide_top_k is None:
            raise ValueError("wide_matchers needs wide_top_k")
        for better, base in self.extra_pairs:
            if better == base or not {better, base} <= set(self.matchers):
                raise ValueError(f"extra pair {better} vs {base} must name two configured matchers")
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
    | GeometryFirstConfig
    | MatcherRerankConfig
    | ConfusionExperimentConfig
    | WorkshopValidationConfig
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

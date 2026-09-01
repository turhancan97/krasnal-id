"""Real coordinate-based candidate pools, against randomly sampled ones."""

import math
from dataclasses import dataclass
from datetime import UTC, datetime

import numpy as np

from krasnal_id.config import AppConfig, GeoAblationConfig
from krasnal_id.embeddings.store import EmbeddingMatrix, load_embedding_matrix
from krasnal_id.experiments.baseline_accuracy import (
    BaselineExperimentError,
    evaluate_fold,
    load_evaluation_inputs,
)
from krasnal_id.experiments.contracts import ExperimentResult, MetricSummary
from krasnal_id.experiments.pool_size_ablation import (
    PoolAblationError,
    measure_pool_size,
    resolve_pool_sizes,
)
from krasnal_id.models import DatasetManifest, EvaluationSplit

_EARTH_RADIUS_METRES = 6371008.8


class GeoAblationError(ValueError):
    """Raised when coordinates are missing or the geographic inputs are invalid."""


@dataclass(frozen=True, slots=True)
class GeoMeasurement:
    """Ranking quality at one pool size, with the pools built by proximity."""

    pool_size: int
    top_1: float
    mrr: float
    median_radius_metres: float
    max_radius_metres: float


def haversine_metres(
    first: tuple[float, float],
    second: tuple[float, float],
) -> float:
    """Great-circle distance between two (latitude, longitude) pairs, in metres."""
    lat1, lon1 = math.radians(first[0]), math.radians(first[1])
    lat2, lon2 = math.radians(second[0]), math.radians(second[1])
    delta_lat, delta_lon = lat2 - lat1, lon2 - lon1
    inner = (
        math.sin(delta_lat / 2) ** 2
        + math.cos(lat1) * math.cos(lat2) * math.sin(delta_lon / 2) ** 2
    )
    return 2 * _EARTH_RADIUS_METRES * math.asin(math.sqrt(min(1.0, inner)))


def dwarf_locations(manifest: DatasetManifest) -> dict[str, tuple[float, float]]:
    """Collect every dwarf's coordinates, requiring the whole set to have them."""
    locations: dict[str, tuple[float, float]] = {}
    missing: list[str] = []
    for dwarf in manifest.dwarfs:
        if dwarf.coordinates is None:
            missing.append(dwarf.dwarf_id)
            continue
        locations[dwarf.dwarf_id] = (
            dwarf.coordinates.latitude,
            dwarf.coordinates.longitude,
        )
    if missing:
        raise GeoAblationError(
            f"{len(missing)} of {len(manifest.dwarfs)} dwarves have no coordinates "
            f"(first: {missing[0]}); a geographic pool cannot be built for them"
        )
    if not locations:
        raise GeoAblationError("manifest contains no dwarves")
    return locations


def nearest_pool(
    dwarf_id: str,
    pool_size: int,
    locations: dict[str, tuple[float, float]],
) -> tuple[tuple[str, ...], float]:
    """Return the query's nearest dwarves and the radius that pool spans.

    The pool always contains the query itself, so it holds `pool_size - 1`
    neighbours. Equal distances break by dwarf ID, which keeps the pool stable
    when statues share a location. The radius is the distance to the farthest
    member, which is what a location-narrowed tool would have had to cover.
    """
    if dwarf_id not in locations:
        raise GeoAblationError(f"dwarf {dwarf_id} has no coordinates")
    if pool_size < 2:
        raise GeoAblationError(f"pool size must be at least two, got {pool_size}")
    if pool_size > len(locations):
        raise GeoAblationError(
            f"pool size {pool_size} exceeds the {len(locations)} located dwarves"
        )

    origin = locations[dwarf_id]
    ranked = sorted(
        (
            (haversine_metres(origin, point), other)
            for other, point in locations.items()
            if other != dwarf_id
        ),
        key=lambda pair: (pair[0], pair[1]),
    )
    chosen = ranked[: pool_size - 1]
    radius = chosen[-1][0] if chosen else 0.0
    return (dwarf_id, *(other for _, other in chosen)), radius


def measure_geographic_pool(
    split: EvaluationSplit,
    matrix: EmbeddingMatrix,
    pool_size: int,
    locations: dict[str, tuple[float, float]],
) -> GeoMeasurement:
    """Measure ranking quality with every pool drawn from real proximity.

    Nothing is sampled here, so a geographic measurement is exact rather than a
    mean over seeds, and it carries no seed spread.
    """
    if not split.folds:
        raise GeoAblationError("split contains no folds")

    hits = 0
    reciprocal = 0.0
    radii: list[float] = []
    for fold in split.folds:
        pooled, radius = nearest_pool(fold.query_dwarf_id, pool_size, locations)
        radii.append(radius)
        members = set(pooled)
        references = tuple(
            image_id
            for image_id in fold.reference_image_ids
            if matrix.dwarf_ids[matrix.index_of(image_id)] in members
        )
        outcome = evaluate_fold(fold.query_image_id, fold.query_dwarf_id, references, matrix)
        hits += int(outcome.dwarf_rank == 1)
        reciprocal += 1.0 / outcome.dwarf_rank

    total = len(split.folds)
    return GeoMeasurement(
        pool_size=pool_size,
        top_1=hits / total,
        mrr=reciprocal / total,
        median_radius_metres=float(np.median(radii)),
        max_radius_metres=float(np.max(radii)),
    )


def summarize_geo(
    geo: tuple[GeoMeasurement, ...],
    random_arm: dict[int, tuple[float, float, float]],
    folds: int,
    dwarf_count: int,
) -> tuple[MetricSummary, ...]:
    """Report both arms and the advantage that proximity buys at each pool size."""
    if not geo:
        raise GeoAblationError("no pool sizes were measured")

    metrics: list[MetricSummary] = []
    for measurement in geo:
        metrics.append(
            MetricSummary(name=f"geo_top_1_pool_{measurement.pool_size}", value=measurement.top_1)
        )
    for measurement in geo:
        mean, low, high = random_arm[measurement.pool_size]
        metrics.append(
            MetricSummary(
                name=f"random_top_1_pool_{measurement.pool_size}",
                value=mean,
                lower_bound=low,
                upper_bound=high,
            )
        )
    for measurement in geo:
        mean, _, _ = random_arm[measurement.pool_size]
        metrics.append(
            MetricSummary(
                name=f"geo_advantage_pool_{measurement.pool_size}",
                value=measurement.top_1 - mean,
            )
        )
    for measurement in geo:
        metrics.append(
            MetricSummary(
                name=f"geo_radius_metres_pool_{measurement.pool_size}",
                value=measurement.median_radius_metres,
                upper_bound=measurement.max_radius_metres,
            )
        )
    for measurement in geo:
        metrics.append(
            MetricSummary(name=f"geo_mrr_pool_{measurement.pool_size}", value=measurement.mrr)
        )
    metrics.append(MetricSummary(name="evaluated_folds", value=float(folds)))
    metrics.append(MetricSummary(name="candidate_dwarfs", value=float(dwarf_count)))
    return tuple(metrics)


def run_geo_ablation(config: AppConfig) -> ExperimentResult:
    """Compare real proximity-based candidate pools against randomly sampled ones."""
    if not isinstance(config.experiment, GeoAblationConfig):
        raise GeoAblationError(
            f"the geographic ablation requires experiment=geo_ablation, "
            f"got {config.experiment.kind}"
        )

    try:
        manifest, split = load_evaluation_inputs(
            config.paths.manifest_path,
            config.paths.evaluation_split_path,
        )
    except BaselineExperimentError as error:
        raise GeoAblationError(str(error)) from error

    matrix = load_embedding_matrix(manifest, config.backbone, config.paths.embeddings_dir)
    locations = dwarf_locations(manifest)
    dwarf_ids = tuple(sorted(set(matrix.dwarf_ids)))
    unlocated = set(dwarf_ids) - set(locations)
    if unlocated:
        raise GeoAblationError(
            f"{len(unlocated)} dwarves in the embedding set have no coordinates: "
            f"{', '.join(sorted(unlocated))}"
        )

    try:
        pool_sizes = resolve_pool_sizes(config.experiment.pool_sizes, len(dwarf_ids))
    except PoolAblationError as error:
        raise GeoAblationError(str(error)) from error

    geo = tuple(
        measure_geographic_pool(split, matrix, pool_size, locations) for pool_size in pool_sizes
    )
    random_arm: dict[int, tuple[float, float, float]] = {}
    for pool_size in pool_sizes:
        sampled = measure_pool_size(split, matrix, pool_size, config.experiment.seeds, dwarf_ids)
        random_arm[pool_size] = (
            sampled.top_1,
            min(sampled.top_1_per_seed),
            max(sampled.top_1_per_seed),
        )

    return ExperimentResult(
        experiment="geo_ablation",
        backbone=config.backbone.name,
        created_at=datetime.now(UTC),
        # Only the random comparison arm samples; the geographic pools are exact.
        seed=config.experiment.seeds[0],
        metrics=summarize_geo(geo, random_arm, len(split.folds), len(dwarf_ids)),
    )

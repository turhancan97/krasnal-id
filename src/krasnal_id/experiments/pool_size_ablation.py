"""Accuracy as a function of candidate-pool size, the headline experiment."""

import logging
from dataclasses import dataclass
from datetime import UTC, datetime

import numpy as np

from krasnal_id.config import AppConfig, PoolSizeAblationConfig
from krasnal_id.embeddings.store import EmbeddingMatrix, load_embedding_matrix
from krasnal_id.experiments.baseline_accuracy import (
    BaselineExperimentError,
    evaluate_fold,
    load_evaluation_inputs,
)
from krasnal_id.experiments.contracts import ExperimentResult, MetricSummary
from krasnal_id.models import EvaluationSplit

_LOGGER = logging.getLogger(__name__)


class PoolAblationError(ValueError):
    """Raised when ablation inputs are missing, invalid, or inconsistent."""


@dataclass(frozen=True, slots=True)
class PoolMeasurement:
    """Aggregated ranking quality at one candidate-pool size."""

    pool_size: int
    top_1_per_seed: tuple[float, ...]
    mrr_per_seed: tuple[float, ...]

    @property
    def top_1(self) -> float:
        """Return mean top-1 accuracy across seeds."""
        return float(np.mean(self.top_1_per_seed))

    @property
    def mrr(self) -> float:
        """Return mean reciprocal rank across seeds."""
        return float(np.mean(self.mrr_per_seed))


def resolve_pool_sizes(configured: tuple[int, ...], dwarf_count: int) -> tuple[int, ...]:
    """Keep the configured sizes a dataset can support, always including its full pool.

    A pool larger than the dataset cannot be sampled, so configured sizes above the
    class count are dropped rather than silently clamped onto each other. The full
    pool is always measured so every run has a comparable right-hand anchor.
    """
    if dwarf_count < 2:
        raise PoolAblationError(f"a pool ablation needs at least two dwarves, got {dwarf_count}")
    if any(size < 2 for size in configured):
        raise PoolAblationError(f"pool sizes must be at least two: {configured}")

    usable = {size for size in configured if size <= dwarf_count}
    skipped = sorted(set(configured) - usable)
    if skipped:
        _LOGGER.warning(
            "skipping pool sizes larger than the %d available dwarves: %s", dwarf_count, skipped
        )
    usable.add(dwarf_count)
    return tuple(sorted(usable))


def measure_pool_size(
    split: EvaluationSplit,
    matrix: EmbeddingMatrix,
    pool_size: int,
    seeds: tuple[int, ...],
    dwarf_ids: tuple[str, ...],
) -> PoolMeasurement:
    """Measure ranking quality at one pool size across every seed.

    Each query is scored against its own dwarf plus `pool_size - 1` other dwarves
    drawn without replacement, which simulates the candidate narrowing a
    location-aware tool would perform.
    """
    # `dwarf_ids` is the universe the distractors are drawn from, and it need not
    # cover the whole matrix: the geographic arm passes only the located subset.
    top_1_per_seed: list[float] = []
    mrr_per_seed: list[float] = []
    for seed in seeds:
        # One generator per (pool size, seed) keeps a run reproducible regardless of
        # how many pool sizes were requested or in which order.
        rng = np.random.default_rng(seed)
        hits = 0
        reciprocal_rank = 0.0
        for fold in split.folds:
            available = [dwarf_id for dwarf_id in dwarf_ids if dwarf_id != fold.query_dwarf_id]
            distractors = rng.choice(
                np.asarray(available), size=pool_size - 1, replace=False
            ).tolist()
            pooled = {fold.query_dwarf_id, *distractors}
            references = tuple(
                image_id
                for image_id in fold.reference_image_ids
                if matrix.dwarf_ids[matrix.index_of(image_id)] in pooled
            )
            outcome = evaluate_fold(fold.query_image_id, fold.query_dwarf_id, references, matrix)
            hits += int(outcome.dwarf_rank == 1)
            reciprocal_rank += 1.0 / outcome.dwarf_rank
        total = len(split.folds)
        top_1_per_seed.append(hits / total)
        mrr_per_seed.append(reciprocal_rank / total)

    return PoolMeasurement(
        pool_size=pool_size,
        top_1_per_seed=tuple(top_1_per_seed),
        mrr_per_seed=tuple(mrr_per_seed),
    )


def points_per_doubling(measurements: tuple[PoolMeasurement, ...]) -> float:
    """Fit top-1 accuracy against log2 pool size over every measured pool size.

    Reported in accuracy points, so a value of -1.5 means one and a half points of
    top-1 accuracy lost each time the candidate pool doubles. The fit spans every
    measured size, including small pools where accuracy sits near its ceiling, so
    it is a conservative estimate of degradation in the larger-pool regime.
    """
    if len(measurements) < 2:
        raise PoolAblationError("a slope needs at least two measured pool sizes")
    sizes = np.log2([measurement.pool_size for measurement in measurements])
    accuracies = np.asarray([measurement.top_1 for measurement in measurements])
    slope = float(np.polyfit(sizes, accuracies, 1)[0])
    return slope * 100.0


def summarize_measurements(measurements: tuple[PoolMeasurement, ...]) -> tuple[MetricSummary, ...]:
    """Turn per-pool measurements into the serializable metric curve."""
    metrics: list[MetricSummary] = []
    for measurement in measurements:
        metrics.append(
            MetricSummary(
                name=f"top_1_pool_{measurement.pool_size}",
                value=measurement.top_1,
                lower_bound=min(measurement.top_1_per_seed),
                upper_bound=max(measurement.top_1_per_seed),
            )
        )
    for measurement in measurements:
        metrics.append(
            MetricSummary(
                name=f"mrr_pool_{measurement.pool_size}",
                value=measurement.mrr,
                lower_bound=min(measurement.mrr_per_seed),
                upper_bound=max(measurement.mrr_per_seed),
            )
        )
    metrics.append(
        MetricSummary(
            name="top_1_points_per_doubling",
            value=points_per_doubling(measurements),
        )
    )
    metrics.append(MetricSummary(name="candidate_dwarfs", value=float(measurements[-1].pool_size)))
    metrics.append(MetricSummary(name="evaluated_pool_sizes", value=float(len(measurements))))
    return tuple(metrics)


def run_pool_size_ablation(config: AppConfig) -> ExperimentResult:
    """Run the configured accuracy-versus-pool-size ablation."""
    if not isinstance(config.experiment, PoolSizeAblationConfig):
        raise PoolAblationError(
            f"the ablation requires experiment=pool_size_ablation, got {config.experiment.kind}"
        )

    try:
        manifest, split = load_evaluation_inputs(
            config.paths.manifest_path,
            config.paths.evaluation_split_path,
        )
    except BaselineExperimentError as error:
        raise PoolAblationError(str(error)) from error

    matrix = load_embedding_matrix(manifest, config.backbone, config.paths.embeddings_dir)
    if not split.folds:
        raise PoolAblationError("split contains no folds")

    dwarf_ids = tuple(sorted(set(matrix.dwarf_ids)))
    pool_sizes = resolve_pool_sizes(config.experiment.pool_sizes, len(dwarf_ids))
    measurements = tuple(
        measure_pool_size(split, matrix, pool_size, config.experiment.seeds, dwarf_ids)
        for pool_size in pool_sizes
    )

    return ExperimentResult(
        experiment="pool_size_ablation",
        backbone=config.backbone.name,
        created_at=datetime.now(UTC),
        # The ablation samples, so record the first seed as the run's identity; the
        # complete seed list is what makes the per-pool error bars reproducible.
        seed=config.experiment.seeds[0],
        metrics=summarize_measurements(measurements),
    )

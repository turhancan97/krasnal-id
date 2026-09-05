"""Most-confused dwarf-pair analysis."""

from dataclasses import dataclass
from datetime import UTC, datetime

import numpy as np

from krasnal_id.config import AppConfig, ConfusionExperimentConfig
from krasnal_id.embeddings.store import EmbeddingMatrix, load_embedding_matrix
from krasnal_id.experiments.baseline_accuracy import (
    BaselineExperimentError,
    accuracy_metric,
    load_evaluation_inputs,
)
from krasnal_id.experiments.contracts import (
    ConfusionAnalysisResult,
    ConfusionPair,
    MetricSummary,
)
from krasnal_id.geometry import haversine_metres
from krasnal_id.models import DatasetManifest, EvaluationSplit
from krasnal_id.retrieval.knn import cosine_knn
from krasnal_id.statistics import separability_auroc

# One installation spans tens of metres; a neighbourhood spans hundreds. Reporting
# both says whether co-location acts at the scale of a shared plinth or a shared
# street, which the aggregate rank statistic cannot distinguish.
SEPARATION_BANDS_METRES = (100.0, 300.0)


class ConfusionAnalysisError(ValueError):
    """Raised when confusion inputs are missing, invalid, or inconsistent."""


@dataclass(frozen=True, slots=True)
class Competition:
    """The strongest wrong candidate for one query."""

    query_image_id: str
    true_dwarf_id: str
    competitor_dwarf_id: str
    margin: float

    @property
    def misidentified(self) -> bool:
        """Return whether the competitor outranked the correct dwarf."""
        return self.margin < 0.0


def find_strongest_competitor(
    query_image_id: str,
    query_dwarf_id: str,
    reference_image_ids: tuple[str, ...],
    matrix: EmbeddingMatrix,
) -> Competition:
    """Find the best wrong dwarf for one query and how close it came.

    The margin is the correct dwarf's best similarity minus the competitor's best
    similarity, so it is negative exactly when the query was misidentified. Every
    query yields a competitor, not only the failures, which is what makes the
    analysis informative on a dataset where outright errors are rare.
    """
    vectors, dwarf_ids = matrix.rows_for(reference_image_ids)
    ranked = cosine_knn(
        query_image_id,
        matrix.vector_for(query_image_id),
        vectors,
        reference_image_ids,
        dwarf_ids,
        top_k=len(reference_image_ids),
    )

    best: dict[str, float] = {}
    for match in ranked.matches:
        best.setdefault(match.dwarf_id, match.cosine_similarity)

    if query_dwarf_id not in best:
        raise ConfusionAnalysisError(
            f"dwarf {query_dwarf_id} of query {query_image_id} has no reference image"
        )
    competitors = [dwarf_id for dwarf_id in best if dwarf_id != query_dwarf_id]
    if not competitors:
        raise ConfusionAnalysisError(
            f"query {query_image_id} has no competing dwarf; a confusion analysis "
            "needs at least two dwarves in the candidate pool"
        )

    competitor = competitors[0]
    return Competition(
        query_image_id=query_image_id,
        true_dwarf_id=query_dwarf_id,
        competitor_dwarf_id=competitor,
        margin=best[query_dwarf_id] - best[competitor],
    )


def aggregate_confusion_pairs(
    competitions: tuple[Competition, ...],
    display_names: dict[str, str],
    locations: dict[str, tuple[float, float]] | None = None,
) -> tuple[ConfusionPair, ...]:
    """Aggregate every competition into directed pairs, most-confused first.

    Ordered by outright misidentifications first, then by how often the pair
    competed, then by the tightest mean margin, so that pairs which are repeatedly
    near-misses surface even when nothing was actually misidentified.

    Returns *all* pairs rather than a top slice, because the separation statistics
    are a property of the whole population and would be badly biased by computing
    them over the pairs that were already selected for being confused.
    """
    placed = locations or {}
    grouped: dict[tuple[str, str], list[Competition]] = {}
    for competition in competitions:
        key = (competition.true_dwarf_id, competition.competitor_dwarf_id)
        grouped.setdefault(key, []).append(competition)

    pairs = [
        ConfusionPair(
            true_dwarf_id=true_id,
            true_display_name=display_names.get(true_id, true_id),
            confused_dwarf_id=confused_id,
            confused_display_name=display_names.get(confused_id, confused_id),
            queries=len(members),
            misidentifications=sum(1 for member in members if member.misidentified),
            mean_margin=float(np.mean([member.margin for member in members])),
            separation_metres=(
                haversine_metres(placed[true_id], placed[confused_id])
                if true_id in placed and confused_id in placed
                else None
            ),
        )
        for (true_id, confused_id), members in grouped.items()
    ]
    pairs.sort(
        key=lambda pair: (
            -pair.misidentifications,
            -pair.queries,
            pair.mean_margin,
            pair.true_dwarf_id,
            pair.confused_dwarf_id,
        )
    )
    return tuple(pairs)


def rank_confusion_pairs(
    competitions: tuple[Competition, ...],
    display_names: dict[str, str],
    top_pairs: int,
    locations: dict[str, tuple[float, float]] | None = None,
) -> tuple[ConfusionPair, ...]:
    """Return the most-confused directed pairs, capped for reporting."""
    if top_pairs <= 0:
        raise ConfusionAnalysisError(f"top_pairs must be positive, got {top_pairs}")
    return aggregate_confusion_pairs(competitions, display_names, locations)[:top_pairs]


def summarize_separation(pairs: tuple[ConfusionPair, ...]) -> tuple[MetricSummary, ...]:
    """Measure whether the statues that get confused are the ones standing together.

    The geographic ablation shows the *shape* of a proximity penalty; this tests the
    explanation for it directly. Every competing pair that has both statues placed
    is split by whether it produced an outright misidentification, and the two
    distance populations are compared.

    The headline is a rank statistic: the probability that a randomly chosen
    confused pair stands closer together than a randomly chosen merely-competing
    one. 0.5 means distance says nothing about confusion.
    """
    measured = [pair for pair in pairs if pair.separation_metres is not None]
    confused = [p.separation_metres for p in measured if p.misidentifications]
    rest = [p.separation_metres for p in measured if not p.misidentifications]
    if not confused or not rest:
        return (MetricSummary(name="separation_pairs_measured", value=float(len(measured))),)

    confused_array = np.asarray(confused, dtype=np.float64)
    rest_array = np.asarray(rest, dtype=np.float64)
    return (
        MetricSummary(name="separation_pairs_measured", value=float(len(measured))),
        MetricSummary(name="confused_pairs_measured", value=float(len(confused))),
        MetricSummary(
            name="confused_pair_separation_median_metres",
            value=float(np.median(confused_array)),
        ),
        MetricSummary(
            name="competing_pair_separation_median_metres",
            value=float(np.median(rest_array)),
        ),
        # Oriented so that above 0.5 means confusion goes with proximity.
        MetricSummary(
            name="proximity_predicts_confusion_auroc",
            value=separability_auroc(rest_array, confused_array),
        ),
        *(
            metric
            for band in SEPARATION_BANDS_METRES
            for metric in (
                MetricSummary(
                    name=f"confused_pairs_within_{band:.0f}m",
                    value=float((confused_array <= band).mean()),
                ),
                MetricSummary(
                    name=f"competing_pairs_within_{band:.0f}m",
                    value=float((rest_array <= band).mean()),
                ),
            )
        ),
    )


def analyze_confusion(
    manifest: DatasetManifest,
    split: EvaluationSplit,
    matrix: EmbeddingMatrix,
    top_pairs: int,
) -> tuple[tuple[MetricSummary, ...], tuple[ConfusionPair, ...]]:
    """Measure systematic cross-class competition over every fold."""
    if not split.folds:
        raise ConfusionAnalysisError("split contains no folds")

    competitions = tuple(
        find_strongest_competitor(
            fold.query_image_id, fold.query_dwarf_id, fold.reference_image_ids, matrix
        )
        for fold in split.folds
    )
    display_names = {dwarf.dwarf_id: dwarf.display_name for dwarf in manifest.dwarfs}
    locations = {
        dwarf.dwarf_id: (dwarf.coordinates.latitude, dwarf.coordinates.longitude)
        for dwarf in manifest.dwarfs
        if dwarf.coordinates is not None
    }
    every_pair = aggregate_confusion_pairs(competitions, display_names, locations)
    pairs = every_pair[:top_pairs]

    total = len(competitions)
    errors = sum(1 for competition in competitions if competition.misidentified)
    margins = np.asarray([competition.margin for competition in competitions])
    distinct = len(
        {
            (competition.true_dwarf_id, competition.competitor_dwarf_id)
            for competition in competitions
        }
    )

    metrics = (
        accuracy_metric("top_1_error_rate", errors, total),
        MetricSummary(name="top_1_errors", value=float(errors)),
        MetricSummary(name="competing_pairs", value=float(distinct)),
        MetricSummary(name="reported_pairs", value=float(len(pairs))),
        MetricSummary(
            name="mean_margin",
            value=float(np.mean(margins)),
            lower_bound=float(np.min(margins)),
            upper_bound=float(np.max(margins)),
        ),
        MetricSummary(name="median_margin", value=float(np.median(margins))),
        *summarize_separation(every_pair),
    )
    return metrics, pairs


def run_confusion_analysis(config: AppConfig) -> ConfusionAnalysisResult:
    """Identify and summarize systematic cross-class retrieval errors."""
    if not isinstance(config.experiment, ConfusionExperimentConfig):
        raise ConfusionAnalysisError(
            f"confusion analysis requires experiment=confusion, got {config.experiment.kind}"
        )

    try:
        manifest, split = load_evaluation_inputs(
            config.paths.manifest_path,
            config.paths.evaluation_split_path,
        )
    except BaselineExperimentError as error:
        raise ConfusionAnalysisError(str(error)) from error

    matrix = load_embedding_matrix(manifest, config.backbone, config.paths.embeddings_dir)
    metrics, pairs = analyze_confusion(manifest, split, matrix, config.experiment.top_pairs)

    return ConfusionAnalysisResult(
        experiment="confusion",
        backbone=config.backbone.name,
        created_at=datetime.now(UTC),
        # Nothing here samples; the seed is recorded for provenance.
        seed=config.experiment.seed,
        metrics=metrics,
        pairs=pairs,
    )

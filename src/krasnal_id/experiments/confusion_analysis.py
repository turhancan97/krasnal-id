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
from krasnal_id.models import DatasetManifest, EvaluationSplit
from krasnal_id.retrieval.knn import cosine_knn


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


def rank_confusion_pairs(
    competitions: tuple[Competition, ...],
    display_names: dict[str, str],
    top_pairs: int,
) -> tuple[ConfusionPair, ...]:
    """Aggregate competitions into the most-confused directed pairs.

    Ordered by outright misidentifications first, then by how often the pair
    competed, then by the tightest mean margin, so that pairs which are repeatedly
    near-misses surface even when nothing was actually misidentified.
    """
    if top_pairs <= 0:
        raise ConfusionAnalysisError(f"top_pairs must be positive, got {top_pairs}")

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
    return tuple(pairs[:top_pairs])


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
    pairs = rank_confusion_pairs(competitions, display_names, top_pairs)

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

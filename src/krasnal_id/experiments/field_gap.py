"""How far does accuracy fall when the query is a phone photograph taken in the street?

`RESULTS.md` names the gap between a Commons upload and a casual snapshot as the
largest untested thing in the project. Section 8 puts a lower bound on it from the
51 references that were themselves shot on phones; this measures it directly, with
photographs taken in Wroclaw under the protocol in `data/field-guide.md`.

Three things make the number mean something, and all three come from `AGENTS.md`
section 5.8:

* **Field photographs are queries, never references.** The reference set is exactly
  the manifest, unchanged, so the ranking a field query faces is the one the
  published demo would give it.
* **The finding is a difference, not an absolute.** The comparison is the
  leave-one-out folds of the *same statues*, so pool size and class difficulty are
  held fixed and only the origin of the query changes.
* **Cohorts are fixed before the walk.** The route files each statue as a member of
  a cluster both backbones already confuse or as a control, so a drop concentrated
  on the hard cases is distinguishable from a uniform one.

One asymmetry is unavoidable and is reported rather than hidden: a leave-one-out
query is withheld from its own class, so it sees one fewer reference of the right
statue than a field query does. That favours the field queries, so it understates
the gap rather than inventing one.
"""

from dataclasses import dataclass
from datetime import UTC, datetime

import numpy as np
import numpy.typing as npt

from krasnal_id.config import AppConfig, FieldGapExperimentConfig
from krasnal_id.data_pipeline.field_queries import (
    FieldCohort,
    FieldQueryManifest,
    FieldQueryRecord,
)
from krasnal_id.embeddings.cache import EmbeddingCache
from krasnal_id.embeddings.store import EmbeddingMatrix, cache_key_for, load_embedding_matrix
from krasnal_id.experiments.baseline_accuracy import (
    BaselineExperimentError,
    accuracy_metric,
    evaluate_fold,
    load_evaluation_inputs,
)
from krasnal_id.experiments.contracts import FieldClassOutcome, FieldGapResult, MetricSummary
from krasnal_id.models import DatasetManifest, EvaluationSplit
from krasnal_id.retrieval.knn import cosine_knn

# The whole-route figures, reported under names without a cohort in them.
OVERALL = "all"


class FieldGapError(ValueError):
    """Raised when field-gap inputs are missing, invalid, or inconsistent."""


@dataclass(frozen=True, slots=True)
class OriginOutcome:
    """Where the correct statue ranked for one origin and cohort of queries."""

    origin: str
    cohort: str
    ranks: tuple[int, ...]

    def top_k_hits(self, k: int) -> int:
        """Count queries whose correct dwarf ranked within the first k candidates."""
        return sum(1 for rank in self.ranks if rank <= k)

    @property
    def label(self) -> str:
        """Return the metric-name prefix for this group."""
        return self.origin if self.cohort == OVERALL else f"{self.origin}_{self.cohort}"


def rank_field_query(
    vector: npt.NDArray[np.float32],
    query_id: str,
    true_dwarf_id: str,
    matrix: EmbeddingMatrix,
) -> int:
    """Rank the whole reference set for one field photograph.

    Nothing is withheld. A field photograph is not in the manifest, so there is no
    copy of it to leak, and the staging step already refuses one that is
    byte-identical to a reference.
    """
    result = cosine_knn(
        query_id,
        vector,
        matrix.vectors,
        matrix.image_ids,
        matrix.dwarf_ids,
        top_k=len(matrix.image_ids),
    )
    ranked: list[str] = []
    for match in result.matches:
        if match.dwarf_id not in ranked:
            ranked.append(match.dwarf_id)
    if true_dwarf_id not in ranked:
        raise FieldGapError(
            f"dwarf {true_dwarf_id} of field query {query_id} has no reference image"
        )
    return ranked.index(true_dwarf_id) + 1


def load_field_vectors(
    staged: FieldQueryManifest,
    config: AppConfig,
) -> dict[str, npt.NDArray[np.float32]]:
    """Read the cached vector for every staged photograph, refusing to compute one.

    Extraction stays the job of `embeddings extract --field-queries`, so a run that
    is missing vectors reports which command to run rather than quietly loading a
    model mid-experiment.
    """
    cache = EmbeddingCache(config.paths.embeddings_dir)
    vectors: dict[str, npt.NDArray[np.float32]] = {}
    missing: list[str] = []
    for query in staged.queries:
        vector = cache.load(cache_key_for(query, config.backbone))
        if vector is None:
            missing.append(query.image_id)
            continue
        vectors[query.image_id] = vector
    if missing:
        raise FieldGapError(
            f"{len(missing)} of {len(staged.queries)} field photographs have no cached "
            f"{config.backbone.name} vector (first: {missing[0]}); run krasnal-id "
            f"embeddings extract --field-queries --override backbone={config.backbone.name}"
        )
    return vectors


def score_field_queries(
    staged: FieldQueryManifest,
    matrix: EmbeddingMatrix,
    vectors: dict[str, npt.NDArray[np.float32]],
) -> dict[str, tuple[FieldQueryRecord, int]]:
    """Rank every staged photograph against the unchanged reference set."""
    return {
        query.image_id: (
            query,
            rank_field_query(vectors[query.image_id], query.image_id, query.dwarf_id, matrix),
        )
        for query in staged.queries
    }


def score_commons_queries(
    split: EvaluationSplit,
    matrix: EmbeddingMatrix,
    photographed: frozenset[str],
) -> dict[str, list[int]]:
    """Rank the leave-one-out folds of the statues that were photographed.

    Restricting to those statues is what makes the comparison a measurement of the
    query's origin: the reference pool, the cut-offs and the classes are identical
    on both sides, and only where the photograph came from differs.
    """
    ranks: dict[str, list[int]] = {}
    for fold in split.folds:
        if fold.query_dwarf_id not in photographed:
            continue
        outcome = evaluate_fold(
            fold.query_image_id, fold.query_dwarf_id, fold.reference_image_ids, matrix
        )
        ranks.setdefault(fold.query_dwarf_id, []).append(outcome.dwarf_rank)
    if not ranks:
        raise FieldGapError(
            "no leave-one-out fold covers a photographed statue, so there is nothing "
            "to compare the field queries against"
        )
    return ranks


def group_outcomes(
    field_scores: dict[str, tuple[FieldQueryRecord, int]],
    commons_ranks: dict[str, list[int]],
    cohort_by_dwarf: dict[str, FieldCohort],
) -> tuple[OriginOutcome, ...]:
    """Collect ranks by query origin, whole-route first and then by cohort."""
    field_by_cohort: dict[str, list[int]] = {}
    commons_by_cohort: dict[str, list[int]] = {}
    for query, rank in field_scores.values():
        field_by_cohort.setdefault(str(query.cohort), []).append(rank)
    for dwarf_id, ranks in commons_ranks.items():
        commons_by_cohort.setdefault(str(cohort_by_dwarf[dwarf_id]), []).extend(ranks)

    outcomes = [
        OriginOutcome(
            origin="field",
            cohort=OVERALL,
            ranks=tuple(rank for _, rank in field_scores.values()),
        ),
        OriginOutcome(
            origin="commons",
            cohort=OVERALL,
            ranks=tuple(rank for ranks in commons_ranks.values() for rank in ranks),
        ),
    ]
    for cohort in FieldCohort:
        for origin, grouped in (("field", field_by_cohort), ("commons", commons_by_cohort)):
            cohort_ranks = grouped.get(str(cohort))
            if cohort_ranks:
                outcomes.append(
                    OriginOutcome(origin=origin, cohort=str(cohort), ranks=tuple(cohort_ranks))
                )
    return tuple(outcomes)


def summarize_field_gap(
    outcomes: tuple[OriginOutcome, ...],
    top_k: tuple[int, ...],
    classes: int,
) -> tuple[MetricSummary, ...]:
    """Report each group, then the gap between the two origins within each cohort."""
    if not outcomes:
        raise FieldGapError("no queries were grouped")
    if any(k <= 0 for k in top_k):
        raise FieldGapError(f"top_k values must be positive: {top_k}")

    cut_offs = sorted(set(top_k))
    metrics: list[MetricSummary] = []
    for outcome in outcomes:
        total = len(outcome.ranks)
        for k in cut_offs:
            metrics.append(
                accuracy_metric(f"{outcome.label}_top_{k}", outcome.top_k_hits(k), total)
            )
        metrics.append(MetricSummary(name=f"{outcome.label}_queries", value=float(total)))
        metrics.append(
            MetricSummary(
                name=f"{outcome.label}_mrr",
                value=sum(1.0 / rank for rank in outcome.ranks) / total,
            )
        )

    by_group = {(outcome.origin, outcome.cohort): outcome for outcome in outcomes}
    for cohort in (OVERALL, *(str(cohort) for cohort in FieldCohort)):
        field, commons = by_group.get(("field", cohort)), by_group.get(("commons", cohort))
        if field is None or commons is None:
            continue
        prefix = "" if cohort == OVERALL else f"{cohort}_"
        for k in cut_offs:
            # Positive means the field photographs did worse, which is the
            # expected direction and the one the finding is stated in.
            gap = commons.top_k_hits(k) / len(commons.ranks) - field.top_k_hits(k) / len(
                field.ranks
            )
            metrics.append(MetricSummary(name=f"{prefix}top_{k}_gap", value=gap))

    metrics.append(MetricSummary(name="field_classes", value=float(classes)))
    return tuple(metrics)


def summarize_classes(
    field_scores: dict[str, tuple[FieldQueryRecord, int]],
    commons_ranks: dict[str, list[int]],
    manifest: DatasetManifest,
) -> tuple[FieldClassOutcome, ...]:
    """Report every photographed statue, worst field accuracy first."""
    display_names = {dwarf.dwarf_id: dwarf.display_name for dwarf in manifest.dwarfs}
    by_dwarf: dict[str, list[tuple[FieldQueryRecord, int]]] = {}
    for query, rank in field_scores.values():
        by_dwarf.setdefault(query.dwarf_id, []).append((query, rank))

    rows = tuple(
        FieldClassOutcome(
            dwarf_id=dwarf_id,
            display_name=display_names.get(dwarf_id, dwarf_id),
            cohort=str(scored[0][0].cohort),
            field_queries=len(scored),
            field_top_1_hits=sum(1 for _, rank in scored if rank == 1),
            field_mean_rank=sum(rank for _, rank in scored) / len(scored),
            commons_queries=len(commons_ranks.get(dwarf_id, [])),
            commons_top_1_hits=sum(1 for rank in commons_ranks.get(dwarf_id, []) if rank == 1),
        )
        for dwarf_id, scored in by_dwarf.items()
    )
    return tuple(
        sorted(
            rows,
            key=lambda row: (row.field_top_1_hits / row.field_queries, row.dwarf_id),
        )
    )


def run_field_gap(config: AppConfig, staged: FieldQueryManifest) -> FieldGapResult:
    """Score the field photographs and compare them with the same statues' folds."""
    if not isinstance(config.experiment, FieldGapExperimentConfig):
        raise FieldGapError(
            f"the field gap requires experiment=field_gap, got {config.experiment.kind}"
        )

    try:
        manifest, split = load_evaluation_inputs(
            config.paths.manifest_path, config.paths.evaluation_split_path
        )
    except BaselineExperimentError as error:
        raise FieldGapError(str(error)) from error

    matrix = load_embedding_matrix(manifest, config.backbone, config.paths.embeddings_dir)
    field_scores = score_field_queries(staged, matrix, load_field_vectors(staged, config))
    cohort_by_dwarf = {query.dwarf_id: query.cohort for query in staged.queries}
    commons_ranks = score_commons_queries(split, matrix, frozenset(cohort_by_dwarf))
    outcomes = group_outcomes(field_scores, commons_ranks, cohort_by_dwarf)

    return FieldGapResult(
        experiment="field_gap",
        backbone=config.backbone.name,
        created_at=datetime.now(UTC),
        seed=config.experiment.seed,
        metrics=summarize_field_gap(outcomes, config.experiment.top_k, len(cohort_by_dwarf)),
        classes=summarize_classes(field_scores, commons_ranks, manifest),
    )

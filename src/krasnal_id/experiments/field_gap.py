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

Two comparators, because a field photograph differs from a Commons query in two
ways at once. It comes from a phone in the street, and it comes from a photographer
who is not in the reference set. Section 7.5 measured the second on its own, so the
experiment scores the same statues' folds twice: as published, and with every
reference by the query's own photographer withheld (`AGENTS.md` section 5.14).

* ``top_k_gap`` against the standard folds is the whole gap a reader would see.
* ``disjoint_top_k_gap`` against the photographer-disjoint folds is the part a new
  photographer does not explain, which is the phone and the street.

Both comparators see *fewer* correct references than a field query: one fewer in
the standard folds, every one by the query's photographer in the disjoint folds.
Section 7.5 found that losing references costs accuracy, so both favour the field
side and the disjoint gap is a lower bound on the phone-and-street component. The
standard gap is not bounded either way, because the photographer leakage pushes it
up while the withheld reference pushes it down.

Uncertainty is resampled by statue, not by photograph. A walk yields three to five
photographs of each of a few dozen statues, and photographs of one statue are not
independent draws, so a per-photograph Wilson interval on the gap would be far too
narrow. Each gap's interval is a percentile bootstrap over statues, pairing a
statue's field and Commons queries in every resample. The per-arm accuracies keep
their Wilson intervals by section 6.2's convention, and those *are* per photograph:
quote the gap's interval, not theirs, for anything about the field arm.
"""

import zlib
from collections.abc import Sequence
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
from krasnal_id.photographers import (
    authors_by_image,
    disjoint_references,
    is_answerable,
    photographers_by_class,
)
from krasnal_id.retrieval.knn import cosine_knn

# The whole-route figures, reported under names without a cohort in them.
OVERALL = "all"
# The two Commons comparators, by the name each arm is reported under.
STANDARD = "commons"
DISJOINT = "commons_disjoint"


class FieldGapError(ValueError):
    """Raised when field-gap inputs are missing, invalid, or inconsistent."""


@dataclass(frozen=True, slots=True)
class StatueRanks:
    """Where the correct statue ranked, per origin, for one photographed statue.

    The statue is the unit the bootstrap resamples, so everything known about it
    travels together. `commons_disjoint` is empty for a statue documented by a
    single photographer: withholding that photographer leaves no correct
    reference, so the statue cannot be asked cross-photographer at all.
    """

    dwarf_id: str
    cohort: FieldCohort
    field: tuple[int, ...]
    commons: tuple[int, ...]
    commons_disjoint: tuple[int, ...]

    def ranks(self, origin: str) -> tuple[int, ...]:
        """Return this statue's ranks for one origin."""
        if origin == "field":
            return self.field
        if origin == STANDARD:
            return self.commons
        if origin == DISJOINT:
            return self.commons_disjoint
        raise FieldGapError(f"unknown query origin {origin}")


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


def score_commons_disjoint_queries(
    split: EvaluationSplit,
    matrix: EmbeddingMatrix,
    manifest: DatasetManifest,
    photographed: frozenset[str],
) -> dict[str, list[int]]:
    """Rank the same folds with every reference by the query's photographer withheld.

    A field photograph is by definition by someone the reference set has never
    seen, so this is the comparator that holds the photographer fixed as well as
    the statue. Statues with a single photographer are left out rather than scored
    as failures; section 7.5 gives the reason, and the result counts them.
    """
    authors = authors_by_image(manifest)
    by_class = photographers_by_class(manifest)
    ranks: dict[str, list[int]] = {}
    for fold in split.folds:
        dwarf_id = fold.query_dwarf_id
        if dwarf_id not in photographed or not is_answerable(dwarf_id, by_class):
            continue
        kept = disjoint_references(fold.reference_image_ids, authors, authors[fold.query_image_id])
        outcome = evaluate_fold(fold.query_image_id, dwarf_id, kept, matrix)
        ranks.setdefault(dwarf_id, []).append(outcome.dwarf_rank)
    return ranks


def collect_statue_ranks(
    field_scores: dict[str, tuple[FieldQueryRecord, int]],
    commons_ranks: dict[str, list[int]],
    disjoint_ranks: dict[str, list[int]],
) -> tuple[StatueRanks, ...]:
    """Gather every origin's ranks by statue, in a stable order."""
    by_dwarf: dict[str, list[tuple[FieldQueryRecord, int]]] = {}
    for query, rank in field_scores.values():
        by_dwarf.setdefault(query.dwarf_id, []).append((query, rank))
    statues: list[StatueRanks] = []
    for dwarf_id in sorted(by_dwarf):
        scored = by_dwarf[dwarf_id]
        commons = commons_ranks.get(dwarf_id)
        if not commons:
            raise FieldGapError(
                f"statue {dwarf_id} was photographed but has no leave-one-out fold to "
                "compare against"
            )
        statues.append(
            StatueRanks(
                dwarf_id=dwarf_id,
                cohort=scored[0][0].cohort,
                field=tuple(rank for _, rank in scored),
                commons=tuple(commons),
                commons_disjoint=tuple(disjoint_ranks.get(dwarf_id, ())),
            )
        )
    return tuple(statues)


def _hits(ranks: Sequence[int], k: int) -> int:
    return sum(1 for rank in ranks if rank <= k)


def _pooled(statues: Sequence[StatueRanks], origin: str) -> tuple[int, ...]:
    return tuple(rank for statue in statues for rank in statue.ranks(origin))


def gap(statues: Sequence[StatueRanks], comparator: str, k: int) -> float:
    """Comparator accuracy minus field accuracy at k, pooled over the statues.

    Positive means the field photographs did worse, which is the expected
    direction and the one the finding is stated in.
    """
    commons, field = _pooled(statues, comparator), _pooled(statues, "field")
    return _hits(commons, k) / len(commons) - _hits(field, k) / len(field)


def bootstrap_gaps(
    statues: Sequence[StatueRanks],
    comparator: str,
    k: int,
    resamples: int,
    generator: np.random.Generator,
) -> npt.NDArray[np.float64]:
    """Resample whole statues with replacement and recompute the gap each time.

    A statue's field and Commons queries are drawn together, which pairs the two
    sides exactly as the point estimate does.
    """
    field_hits = np.array([_hits(statue.field, k) for statue in statues], dtype=np.float64)
    field_n = np.array([len(statue.field) for statue in statues], dtype=np.float64)
    comparator_ranks = [statue.ranks(comparator) for statue in statues]
    comparator_hits = np.array([_hits(ranks, k) for ranks in comparator_ranks], dtype=np.float64)
    comparator_n = np.array([len(ranks) for ranks in comparator_ranks], dtype=np.float64)
    draws = generator.integers(0, len(statues), size=(resamples, len(statues)))
    comparator_accuracy = comparator_hits[draws].sum(axis=1) / comparator_n[draws].sum(axis=1)
    field_accuracy = field_hits[draws].sum(axis=1) / field_n[draws].sum(axis=1)
    return np.asarray(comparator_accuracy - field_accuracy, dtype=np.float64)


def _generator(seed: int, label: str) -> np.random.Generator:
    """One generator per resampled group, independent of which other groups exist.

    Keyed by the group's name rather than its position, so adding a cohort or a
    cut-off never changes the draws behind an existing interval.
    """
    return np.random.default_rng([seed, zlib.crc32(label.encode("utf-8"))])


def _interval(name: str, value: float, samples: npt.NDArray[np.float64]) -> MetricSummary:
    lower, upper = np.percentile(samples, [2.5, 97.5])
    return MetricSummary(name=name, value=value, lower_bound=float(lower), upper_bound=float(upper))


def _arm_metrics(label: str, ranks: tuple[int, ...], cut_offs: list[int]) -> list[MetricSummary]:
    metrics = [accuracy_metric(f"{label}_top_{k}", _hits(ranks, k), len(ranks)) for k in cut_offs]
    metrics.append(MetricSummary(name=f"{label}_queries", value=float(len(ranks))))
    metrics.append(
        MetricSummary(name=f"{label}_mrr", value=sum(1.0 / rank for rank in ranks) / len(ranks))
    )
    return metrics


def summarize_field_gap(
    statues: Sequence[StatueRanks],
    top_k: tuple[int, ...],
    resamples: int,
    seed: int,
) -> tuple[MetricSummary, ...]:
    """Report each arm, each gap with its by-statue interval, and the cohort contrast."""
    if not statues:
        raise FieldGapError("no photographed statues to summarize")
    if any(k <= 0 for k in top_k):
        raise FieldGapError(f"top_k values must be positive: {top_k}")
    if resamples <= 0:
        raise FieldGapError(f"bootstrap resamples must be positive: {resamples}")

    cut_offs = sorted(set(top_k))
    scopes: dict[str, tuple[StatueRanks, ...]] = {OVERALL: tuple(statues)}
    for cohort in FieldCohort:
        members = tuple(statue for statue in statues if statue.cohort == cohort)
        if members:
            scopes[str(cohort)] = members

    metrics: list[MetricSummary] = []
    # The comparator's statues per scope: every one for the standard folds, and
    # only those answerable cross-photographer for the disjoint ones, with the
    # field side restricted to the same statues so both sides cover one set.
    covered: dict[tuple[str, str], tuple[StatueRanks, ...]] = {}
    for scope, members in scopes.items():
        covered[(scope, STANDARD)] = members
        answerable = tuple(statue for statue in members if statue.commons_disjoint)
        if answerable:
            covered[(scope, DISJOINT)] = answerable

    for scope, members in scopes.items():
        suffix = "" if scope == OVERALL else f"_{scope}"
        metrics.extend(_arm_metrics(f"field{suffix}", _pooled(members, "field"), cut_offs))
        for comparator in (STANDARD, DISJOINT):
            subset = covered.get((scope, comparator))
            if subset:
                metrics.extend(
                    _arm_metrics(f"{comparator}{suffix}", _pooled(subset, comparator), cut_offs)
                )

    samples: dict[tuple[str, str, int], npt.NDArray[np.float64]] = {}
    for (scope, comparator), subset in covered.items():
        prefix = "" if scope == OVERALL else f"{scope}_"
        kind = "" if comparator == STANDARD else "disjoint_"
        for k in cut_offs:
            draws = bootstrap_gaps(
                subset, comparator, k, resamples, _generator(seed, f"{scope}/{comparator}")
            )
            samples[(scope, comparator, k)] = draws
            metrics.append(
                _interval(f"{prefix}{kind}top_{k}_gap", gap(subset, comparator, k), draws)
            )

    # The finding section 5.8 is built for: is the drop concentrated on the
    # families both backbones already confuse? Confusable gap minus control gap,
    # each cohort resampled on its own, so the contrast carries its own interval.
    confusable, control = str(FieldCohort.CONFUSABLE), str(FieldCohort.CONTROL)
    for comparator in (STANDARD, DISJOINT):
        if (confusable, comparator) not in covered or (control, comparator) not in covered:
            continue
        kind = "" if comparator == STANDARD else "disjoint_"
        for k in cut_offs:
            contrast = gap(covered[(confusable, comparator)], comparator, k) - gap(
                covered[(control, comparator)], comparator, k
            )
            difference = samples[(confusable, comparator, k)] - samples[(control, comparator, k)]
            metrics.append(_interval(f"cohort_{kind}top_{k}_gap_difference", contrast, difference))

    metrics.append(MetricSummary(name="field_classes", value=float(len(statues))))
    metrics.append(
        MetricSummary(
            name="disjoint_classes", value=float(len(covered.get((OVERALL, DISJOINT), ())))
        )
    )
    metrics.append(MetricSummary(name="bootstrap_resamples", value=float(resamples)))
    return tuple(metrics)


def summarize_classes(
    statues: Sequence[StatueRanks],
    manifest: DatasetManifest,
) -> tuple[FieldClassOutcome, ...]:
    """Report every photographed statue, worst field accuracy first."""
    display_names = {dwarf.dwarf_id: dwarf.display_name for dwarf in manifest.dwarfs}
    rows = tuple(
        FieldClassOutcome(
            dwarf_id=statue.dwarf_id,
            display_name=display_names.get(statue.dwarf_id, statue.dwarf_id),
            cohort=str(statue.cohort),
            field_queries=len(statue.field),
            field_top_1_hits=_hits(statue.field, 1),
            field_mean_rank=sum(statue.field) / len(statue.field),
            commons_queries=len(statue.commons),
            commons_top_1_hits=_hits(statue.commons, 1),
            commons_disjoint_queries=len(statue.commons_disjoint),
            commons_disjoint_top_1_hits=_hits(statue.commons_disjoint, 1),
        )
        for statue in statues
    )
    return tuple(
        sorted(rows, key=lambda row: (row.field_top_1_hits / row.field_queries, row.dwarf_id))
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
    photographed = frozenset(query.dwarf_id for query in staged.queries)
    statues = collect_statue_ranks(
        field_scores,
        score_commons_queries(split, matrix, photographed),
        score_commons_disjoint_queries(split, matrix, manifest, photographed),
    )

    return FieldGapResult(
        experiment="field_gap",
        backbone=config.backbone.name,
        created_at=datetime.now(UTC),
        seed=config.experiment.seed,
        configuration=config.experiment.model_dump(mode="json"),
        metrics=summarize_field_gap(
            statues,
            config.experiment.top_k,
            config.experiment.bootstrap_resamples,
            config.experiment.seed,
        ),
        classes=summarize_classes(statues, manifest),
    )

"""Does the model recognise the statue, or the photographer who shot it?

122 people contributed the 1,691 reference photographs, but two of them took
67.4% of them, and near-duplicates from a single visit to a statue were never
removed — only byte-identical cross-label duplicates were. So a leave-one-out
query is frequently only near-duplicate-distant from one of its own references,
and a method can score by recognising a camera, a distance and a processing style
rather than a sculpture. `RESULTS.md` says the headline 93.1% is inflated by an
unmeasured amount. This measures it.

Three conditions over the same queries:

* **standard** — the ordinary leave-one-out protocol.
* **disjoint** — every image by the query's own photographer is withheld, so the
  correct statue can only be matched through somebody else's photograph.
* **control** — the reference set is cut to the *same size* with the *same number
  of correct references*, chosen at random. This is the arm that makes the result
  mean anything: withholding a photographer also removes distractors and whole
  candidate classes, and this project's headline finding is that accuracy *rises*
  as the pool shrinks. Without a size-matched control, that inflation would hide
  the very penalty being measured. The geographic ablation of section 5.2 answers
  its question the same way.

The gap against `standard` is therefore decomposed rather than reported whole:
the part the control also shows is the cost of having fewer references, and only
what remains is attributable to the photographer's identity.

**41% of the classes here have a single photographer.** Their queries are not
hard under this protocol, they are unanswerable: no correct reference exists at
all. They are excluded from every rate and counted, because scoring them as
failures would measure the dataset's coverage and call it a model's weakness.
"""

from dataclasses import dataclass
from datetime import UTC, datetime

import numpy as np

from krasnal_id.config import AppConfig, PhotographerGapConfig
from krasnal_id.embeddings.store import EmbeddingMatrix, load_embedding_matrix
from krasnal_id.experiments.baseline_accuracy import (
    BaselineExperimentError,
    accuracy_metric,
    evaluate_fold,
    load_evaluation_inputs,
)
from krasnal_id.experiments.contracts import ExperimentResult, MetricSummary
from krasnal_id.models import DatasetManifest, EvaluationSplit
from krasnal_id.photographers import (
    authors_by_image,
    photographers_by_class,
)

STANDARD = "standard"
DISJOINT = "disjoint"
CONTROL = "control"


class PhotographerGapError(ValueError):
    """Raised when the photographer-gap inputs are missing or inconsistent."""


@dataclass(frozen=True, slots=True)
class Coverage:
    """How much of the dataset this protocol can speak about at all."""

    answerable_queries: int
    unanswerable_queries: int
    single_photographer_classes: int
    classes: int
    photographers: int


@dataclass(frozen=True, slots=True)
class ConditionOutcome:
    """Where the correct statue ranked, under one condition."""

    condition: str
    ranks: tuple[int, ...]
    candidate_classes: tuple[int, ...]

    def top_k_hits(self, k: int) -> int:
        """Count queries whose correct statue ranked within the first k."""
        return sum(1 for rank in self.ranks if rank <= k)

    @property
    def mrr(self) -> float:
        """Mean reciprocal rank over this condition's queries."""
        return sum(1.0 / rank for rank in self.ranks) / len(self.ranks)


def measure_coverage(manifest: DatasetManifest, split: EvaluationSplit) -> Coverage:
    """Count what this protocol can and cannot ask about."""
    authors = authors_by_image(manifest)
    by_class = photographers_by_class(manifest)

    unanswerable = sum(
        1 for fold in split.folds if len(by_class.get(fold.query_dwarf_id, set())) < 2
    )
    return Coverage(
        answerable_queries=len(split.folds) - unanswerable,
        unanswerable_queries=unanswerable,
        single_photographer_classes=sum(1 for names in by_class.values() if len(names) == 1),
        classes=len(by_class),
        photographers=len(set(authors.values())),
    )


def _candidate_classes(references: tuple[str, ...], matrix: EmbeddingMatrix) -> int:
    """Count the distinct statues a reference set can still propose."""
    return len({matrix.dwarf_ids[matrix.index_of(image_id)] for image_id in references})


def evaluate_conditions(
    split: EvaluationSplit,
    manifest: DatasetManifest,
    matrix: EmbeddingMatrix,
    seeds: tuple[int, ...],
) -> tuple[dict[str, ConditionOutcome], Coverage]:
    """Score every answerable query under all three conditions."""
    if not split.folds:
        raise PhotographerGapError("split contains no folds")
    if not seeds:
        raise PhotographerGapError("the control arm needs at least one seed")

    authors = authors_by_image(manifest)
    classes = {image.image_id: image.dwarf_id for image in manifest.images}
    coverage = measure_coverage(manifest, split)
    if coverage.answerable_queries == 0:
        raise PhotographerGapError(
            "every class has a single photographer, so no query can be answered "
            "without one of its own photographer's references"
        )

    standard: list[tuple[int, int]] = []
    disjoint: list[tuple[int, int]] = []
    # One list of ranks per seed, so the control's spread across seeds is visible
    # rather than averaged away before anyone can see it.
    control: list[list[tuple[int, int]]] = [[] for _ in seeds]

    for fold in split.folds:
        query, truth = fold.query_image_id, fold.query_dwarf_id
        mine = authors.get(query)
        if mine is None:
            raise PhotographerGapError(f"query {query} is not in the manifest")

        kept = tuple(image_id for image_id in fold.reference_image_ids if authors[image_id] != mine)
        own_kept = tuple(image_id for image_id in kept if classes[image_id] == truth)
        if not own_kept:
            # Unanswerable, not hard: nobody else photographed this statue.
            continue

        standard.append(
            (
                evaluate_fold(query, truth, fold.reference_image_ids, matrix).dwarf_rank,
                _candidate_classes(fold.reference_image_ids, matrix),
            )
        )
        disjoint.append(
            (evaluate_fold(query, truth, kept, matrix).dwarf_rank, _candidate_classes(kept, matrix))
        )

        own_available = np.asarray(
            [i for i in fold.reference_image_ids if classes[i] == truth], dtype=object
        )
        other_available = np.asarray(
            [i for i in fold.reference_image_ids if classes[i] != truth], dtype=object
        )
        for position, seed in enumerate(seeds):
            # Seeded per query so the control is reproducible regardless of the
            # order folds happen to be visited in.
            rng = np.random.default_rng([seed, len(standard)])
            sampled = (
                *rng.choice(own_available, size=len(own_kept), replace=False).tolist(),
                *rng.choice(
                    other_available,
                    size=min(len(kept) - len(own_kept), len(other_available)),
                    replace=False,
                ).tolist(),
            )
            control[position].append(
                (
                    evaluate_fold(query, truth, sampled, matrix).dwarf_rank,
                    _candidate_classes(sampled, matrix),
                )
            )

    outcomes = {
        STANDARD: ConditionOutcome(
            STANDARD, tuple(r for r, _ in standard), tuple(c for _, c in standard)
        ),
        DISJOINT: ConditionOutcome(
            DISJOINT, tuple(r for r, _ in disjoint), tuple(c for _, c in disjoint)
        ),
    }
    # The control's per-seed runs are pooled: every seed scores the same queries,
    # so the pooled rate is the mean rate and its spread is reported separately.
    outcomes[CONTROL] = ConditionOutcome(
        CONTROL,
        tuple(rank for run in control for rank, _ in run),
        tuple(count for run in control for _, count in run),
    )
    return outcomes, coverage


def summarize(
    outcomes: dict[str, ConditionOutcome],
    coverage: Coverage,
    top_k: tuple[int, ...],
) -> tuple[MetricSummary, ...]:
    """Report each condition, then decompose the gap between them."""
    if any(k <= 0 for k in top_k):
        raise PhotographerGapError(f"top_k values must be positive: {top_k}")
    cut_offs = sorted(set(top_k))

    metrics: list[MetricSummary] = []
    for name in (STANDARD, DISJOINT, CONTROL):
        outcome = outcomes[name]
        total = len(outcome.ranks)
        for k in cut_offs:
            metrics.append(accuracy_metric(f"{name}_top_{k}", outcome.top_k_hits(k), total))
        metrics.append(MetricSummary(name=f"{name}_mrr", value=outcome.mrr))
        metrics.append(
            MetricSummary(
                name=f"{name}_median_candidate_classes",
                value=float(np.median(np.asarray(outcome.candidate_classes))),
            )
        )

    for k in cut_offs:
        standard_rate = outcomes[STANDARD].top_k_hits(k) / len(outcomes[STANDARD].ranks)
        disjoint_rate = outcomes[DISJOINT].top_k_hits(k) / len(outcomes[DISJOINT].ranks)
        control_rate = outcomes[CONTROL].top_k_hits(k) / len(outcomes[CONTROL].ranks)
        # Positive means the disjoint condition did worse, which is the expected
        # direction and the one the finding is stated in.
        metrics.append(MetricSummary(name=f"top_{k}_gap", value=standard_rate - disjoint_rate))
        # The part the size-matched control does not explain: what is left once
        # "fewer references" is accounted for is the photographer's identity.
        metrics.append(
            MetricSummary(name=f"attributable_top_{k}_gap", value=control_rate - disjoint_rate)
        )

    metrics.extend(
        (
            MetricSummary(name="answerable_queries", value=float(coverage.answerable_queries)),
            MetricSummary(name="unanswerable_queries", value=float(coverage.unanswerable_queries)),
            MetricSummary(
                name="single_photographer_classes",
                value=float(coverage.single_photographer_classes),
            ),
            MetricSummary(name="classes", value=float(coverage.classes)),
            MetricSummary(name="photographers", value=float(coverage.photographers)),
        )
    )
    return tuple(metrics)


def run_photographer_gap(config: AppConfig) -> ExperimentResult:
    """Measure how much of the headline accuracy survives a photographer change."""
    if not isinstance(config.experiment, PhotographerGapConfig):
        raise PhotographerGapError(
            f"the photographer gap requires experiment=photographer_gap, "
            f"got {config.experiment.kind}"
        )

    try:
        manifest, split = load_evaluation_inputs(
            config.paths.manifest_path, config.paths.evaluation_split_path
        )
    except BaselineExperimentError as error:
        raise PhotographerGapError(str(error)) from error

    matrix = load_embedding_matrix(manifest, config.backbone, config.paths.embeddings_dir)
    outcomes, coverage = evaluate_conditions(split, manifest, matrix, config.experiment.seeds)

    return ExperimentResult(
        experiment="photographer_gap",
        backbone=config.backbone.name,
        created_at=datetime.now(UTC),
        seed=config.experiment.seeds[0],
        configuration=config.experiment.model_dump(mode="json"),
        metrics=summarize(outcomes, coverage, config.experiment.top_k),
    )

"""Does verifying geometry on the top candidates fix the near-identical statues?

Section 5 finds the errors concentrated on families of near-identical
sculptures, and section 9 finds that part of the global ranking's signal is the
photographer's style rather than the statue's shape. Both point the same way: a
whole-image embedding compares *appearance*, and appearance is what these
families share. Matching keypoints and fitting a homography compares *geometry*,
which they do not.

So this re-ranks the global top-k by blending each candidate's RANSAC inlier
count into its cosine similarity, and sweeps the blend weight. The sweep is the
experiment rather than a tuning exercise:

* **Weight zero must reproduce the baseline exactly.** It is the control, and it
  is checked rather than assumed — if it drifts, every other column is
  meaningless.
* **Geometry is blended, not substituted.** Sampled pairs give a median of 10
  inliers for the same statue against 4 for a different one, but the
  distributions overlap and some correct pairs verify at zero. Sorting by inliers
  alone would demote correct answers that happen to photograph badly.
* **Promotions and demotions are counted, not just the net.** A weight that gains
  two points by fixing five queries and breaking three is a different result from
  one that fixes two and breaks none, and the net accuracy hides the difference.
"""

from collections.abc import Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

from krasnal_id.config import AppConfig, RerankAblationConfig
from krasnal_id.embeddings.store import EmbeddingMatrix, load_embedding_matrix
from krasnal_id.experiments.baseline_accuracy import (
    BaselineExperimentError,
    accuracy_metric,
    load_evaluation_inputs,
)
from krasnal_id.experiments.contracts import ExperimentResult, MetricSummary
from krasnal_id.models import DatasetManifest, EvaluationSplit
from krasnal_id.photographers import (
    authors_by_image,
    disjoint_references,
    is_answerable,
    photographers_by_class,
)
from krasnal_id.retrieval.knn import cosine_knn
from krasnal_id.retrieval.rerank import FeatureCache, blended_score, count_inliers


class RerankAblationError(ValueError):
    """Raised when the re-ranking inputs are missing or inconsistent."""


@dataclass(frozen=True, slots=True)
class Candidate:
    """One distinct statue proposed for a query, with both kinds of evidence."""

    dwarf_id: str
    cosine: float
    inliers: int
    correct: bool


@dataclass(frozen=True, slots=True)
class QueryEvidence:
    """Every candidate considered for one query, in global-ranking order."""

    query_image_id: str
    candidates: tuple[Candidate, ...]
    # Where the correct statue sat before re-ranking. Zero means it was not among
    # the top-k at all, so no blend weight can rescue it.
    baseline_rank: int

    def ranked(self, weight: float) -> tuple[str, ...]:
        """Order the candidates under one blend weight.

        Ties are broken by the global order, which is what makes weight zero
        reproduce the baseline instead of shuffling equal scores.
        """
        scored = [
            (-blended_score(candidate.cosine, candidate.inliers, weight), position)
            for position, candidate in enumerate(self.candidates)
        ]
        return tuple(self.candidates[position].dwarf_id for _, position in sorted(scored))

    def rank_of_truth(self, weight: float, truth: str) -> int:
        """Return the correct statue's one-based rank, or 0 if it is absent."""
        order = self.ranked(weight)
        return order.index(truth) + 1 if truth in order else 0


def collect_evidence(
    split: EvaluationSplit,
    manifest: DatasetManifest,
    matrix: EmbeddingMatrix,
    cache: FeatureCache,
    top_k: int,
    *,
    photographer_disjoint: bool = False,
    only_answerable: bool = False,
) -> tuple[QueryEvidence, ...]:
    """Rank globally, then verify the geometry of the top candidates.

    Only the top-k are verified, which is the whole point of re-ranking: a
    homography per candidate costs milliseconds where a dot product costs
    microseconds, so it is spent where the global ranking is already uncertain.
    """
    if top_k < 2:
        raise RerankAblationError(f"re-ranking needs at least two candidates, got {top_k}")

    paths = {image.image_id: image.local_path for image in manifest.images}
    authors = authors_by_image(manifest)
    by_class = photographers_by_class(manifest)
    evidence: list[QueryEvidence] = []

    for fold in split.folds:
        query, truth = fold.query_image_id, fold.query_dwarf_id
        # Both arms score the same queries, so the two columns are comparable:
        # the answerable subset is fixed by the dataset, not by the arm.
        if (only_answerable or photographer_disjoint) and not is_answerable(truth, by_class):
            continue

        references = fold.reference_image_ids
        if photographer_disjoint:
            references = disjoint_references(references, authors, authors[query])
        if not references:
            continue

        vectors, dwarf_ids = matrix.rows_for(references)
        ranked = cosine_knn(
            query,
            matrix.vector_for(query),
            vectors,
            references,
            dwarf_ids,
            top_k=len(references),
        )

        # Collapse to distinct statues, each represented by its best image, which
        # is the candidate list an identification tool would show.
        best: list[tuple[str, str, float]] = []
        seen: set[str] = set()
        for match in ranked.matches:
            if match.dwarf_id in seen:
                continue
            seen.add(match.dwarf_id)
            best.append((match.dwarf_id, match.image_id, match.cosine_similarity))
            if len(best) == top_k:
                break

        baseline_rank = next(
            (position for position, (dwarf, _, _) in enumerate(best, start=1) if dwarf == truth),
            0,
        )
        query_features = cache.get(query, paths[query])
        evidence.append(
            QueryEvidence(
                query_image_id=query,
                baseline_rank=baseline_rank,
                candidates=tuple(
                    Candidate(
                        dwarf_id=dwarf,
                        cosine=cosine,
                        inliers=count_inliers(query_features, cache.get(image_id, paths[image_id])),
                        correct=dwarf == truth,
                    )
                    for dwarf, image_id, cosine in best
                ),
            )
        )
    return tuple(evidence)


def summarize_arms(
    # A Mapping rather than a dict because it is only read, and dict's invariance
    # would reject a caller's narrower value type for no benefit.
    arms: Mapping[str, tuple[QueryEvidence, ...]],
    truths: dict[str, str],
    weights: tuple[float, ...],
    cut_offs: tuple[int, ...],
) -> tuple[MetricSummary, ...]:
    """Report several reference regimes side by side, each with its own control.

    Each arm keeps its own weight-zero control, so a gain is always read against
    the same regime that produced it — the point of running the disjoint arm is
    that its baseline is *not* the ordinary one.
    """
    metrics: list[MetricSummary] = []
    for arm, evidence in arms.items():
        prefix = f"{arm}_" if arm else ""
        metrics.extend(
            summary.model_copy(update={"name": f"{prefix}{summary.name}"})
            for summary in summarize(evidence, truths, weights, cut_offs)
        )
    return tuple(metrics)


def separation(evidence: tuple[QueryEvidence, ...]) -> tuple[float, float]:
    """Return the median inlier count for correct and for wrong candidates.

    This is the experiment's own check on whether geometry carries any signal
    here at all. If the two medians coincide, no blend weight can help and the
    accuracy columns will say so.
    """
    correct = [c.inliers for e in evidence for c in e.candidates if c.correct]
    wrong = [c.inliers for e in evidence for c in e.candidates if not c.correct]
    return (
        float(np.median(correct)) if correct else 0.0,
        float(np.median(wrong)) if wrong else 0.0,
    )


def summarize(
    evidence: tuple[QueryEvidence, ...],
    truths: dict[str, str],
    weights: tuple[float, ...],
    cut_offs: tuple[int, ...],
) -> tuple[MetricSummary, ...]:
    """Report accuracy at every blend weight, with what each weight moved."""
    if not evidence:
        raise RerankAblationError("no queries were scored")
    if any(weight < 0.0 for weight in weights):
        raise RerankAblationError(f"blend weights cannot be negative: {weights}")
    if 0.0 not in weights:
        raise RerankAblationError(
            "the sweep must include weight 0.0, which is the unranked control"
        )

    total = len(evidence)
    baseline_top_1 = sum(1 for e in evidence if e.baseline_rank == 1)
    metrics: list[MetricSummary] = []

    for weight in sorted(set(weights)):
        ranks = [e.rank_of_truth(weight, truths[e.query_image_id]) for e in evidence]
        for k in sorted(set(cut_offs)):
            hits = sum(1 for rank in ranks if 0 < rank <= k)
            metrics.append(accuracy_metric(f"weight_{weight:g}_top_{k}", hits, total))
        # A query whose correct statue never entered the top-k contributes zero
        # reciprocal rank rather than being dropped, so the column stays
        # comparable with the baseline over the same denominator.
        metrics.append(
            MetricSummary(
                name=f"weight_{weight:g}_mrr",
                value=sum(1.0 / rank for rank in ranks if rank) / total,
            )
        )
        promoted = sum(
            1 for e, rank in zip(evidence, ranks, strict=True) if rank == 1 != e.baseline_rank
        )
        demoted = sum(
            1 for e, rank in zip(evidence, ranks, strict=True) if e.baseline_rank == 1 != rank
        )
        metrics.append(MetricSummary(name=f"weight_{weight:g}_promoted", value=float(promoted)))
        metrics.append(MetricSummary(name=f"weight_{weight:g}_demoted", value=float(demoted)))

    correct_median, wrong_median = separation(evidence)
    metrics.extend(
        (
            MetricSummary(name="median_inliers_correct", value=correct_median),
            MetricSummary(name="median_inliers_wrong", value=wrong_median),
            MetricSummary(name="baseline_top_1_hits", value=float(baseline_top_1)),
            MetricSummary(name="queries", value=float(total)),
            MetricSummary(
                name="candidates_per_query",
                value=float(np.median([len(e.candidates) for e in evidence])),
            ),
            MetricSummary(
                name="truth_outside_top_k",
                value=float(sum(1 for e in evidence if e.baseline_rank == 0)),
            ),
        )
    )
    return tuple(metrics)


def run_rerank_ablation(config: AppConfig) -> ExperimentResult:
    """Sweep how much geometric verification is worth on top of the ranking."""
    if not isinstance(config.experiment, RerankAblationConfig):
        raise RerankAblationError(
            f"re-ranking requires experiment=rerank_ablation, got {config.experiment.kind}"
        )

    try:
        manifest, split = load_evaluation_inputs(
            config.paths.manifest_path, config.paths.evaluation_split_path
        )
    except BaselineExperimentError as error:
        raise RerankAblationError(str(error)) from error

    matrix = load_embedding_matrix(manifest, config.backbone, config.paths.embeddings_dir)
    for image in manifest.images:
        if not Path(image.local_path).is_file():
            raise RerankAblationError(
                f"image {image.image_id} is missing: {image.local_path}; geometric "
                "verification reads the photographs themselves, not just their vectors"
            )

    cache = FeatureCache(config.experiment.max_keypoints)
    truths = {fold.query_image_id: fold.query_dwarf_id for fold in split.folds}

    if config.experiment.photographer_disjoint:
        # Two arms over one query set. The features are described once and reused,
        # so the second arm costs matching rather than detection.
        arms = {
            "all": collect_evidence(
                split, manifest, matrix, cache, config.experiment.top_k, only_answerable=True
            ),
            "disjoint": collect_evidence(
                split,
                manifest,
                matrix,
                cache,
                config.experiment.top_k,
                photographer_disjoint=True,
            ),
        }
    else:
        arms = {"": collect_evidence(split, manifest, matrix, cache, config.experiment.top_k)}

    return ExperimentResult(
        experiment="rerank_ablation",
        backbone=config.backbone.name,
        created_at=datetime.now(UTC),
        seed=config.experiment.seed,
        configuration=config.experiment.model_dump(mode="json"),
        metrics=summarize_arms(
            arms, truths, config.experiment.weights, config.experiment.top_k_metrics
        ),
    )

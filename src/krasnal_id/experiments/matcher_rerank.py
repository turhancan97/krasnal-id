"""Does a learned matcher re-rank better than SIFT?

Section 10 blended SIFT inlier counts into the cosine ranking and reached 94.0%
for DINOv2. Section 14 then asked SIFT to *retrieve* rather than reorder and it
failed badly -- but that conclusion was written from one hand-designed detector
from 1999, on bronze, which is close to its worst case: specular, low-texture,
few stable corners. The regime where it collapsed, wide baseline with changed
illumination and sensor, is precisely what learned matchers exist for.

This runs several matchers through section 10's protocol unchanged. Every
matcher sees the same queries, the same candidate statues in the same order and
the same blend weights, so the columns differ only in where the correspondences
came from -- and a difference between them is paired query by query rather than
read off two separate confidence intervals.

The cheap arm on purpose. Verifying the top ten of 1,157 queries is 11,570 pairs
against section 14's 1,955,330, so this says whether a learned matcher is worth
the expensive question before the expensive question is asked.
"""

from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from krasnal_id.config import AppConfig, MatcherRerankConfig
from krasnal_id.embeddings.store import load_embedding_matrix
from krasnal_id.experiments.baseline_accuracy import (
    BaselineExperimentError,
    accuracy_metric,
    load_evaluation_inputs,
)
from krasnal_id.experiments.contracts import ExperimentResult, MetricSummary
from krasnal_id.experiments.rerank_ablation import QueryEvidence, collect_evidence
from krasnal_id.retrieval.matchers import create_matcher
from krasnal_id.statistics import exact_mcnemar_p_value


class MatcherRerankError(ValueError):
    """Raised when the matcher comparison's inputs are missing or inconsistent."""


@dataclass(frozen=True, slots=True)
class MatcherRun:
    """One matcher's evidence over the shared query set."""

    name: str
    evidence: tuple[QueryEvidence, ...]


def best_weight(
    evidence: tuple[QueryEvidence, ...],
    truths: dict[str, str],
    weights: tuple[float, ...],
    cut_off: int,
) -> tuple[float, float]:
    """Return the weight with the highest accuracy at one cut-off, and that accuracy.

    Ties resolve to the smaller weight, so a matcher that gains nothing is
    reported at the control rather than at whichever large weight happened to tie.
    """
    scored = []
    for weight in sorted(weights):
        hits = sum(
            1
            for item in evidence
            if 0 < item.rank_of_truth(weight, truths[item.query_image_id]) <= cut_off
        )
        scored.append((hits / len(evidence), weight))
    accuracy, weight = max(scored, key=lambda pair: (pair[0], -pair[1]))
    return weight, accuracy


def summarize(
    runs: tuple[MatcherRun, ...],
    truths: dict[str, str],
    weights: tuple[float, ...],
    cut_offs: tuple[int, ...],
) -> tuple[MetricSummary, ...]:
    """Report every matcher at every weight, and pair them against the first one."""
    if not runs:
        raise MatcherRerankError("at least one matcher is required")
    baseline = runs[0]
    total = len(baseline.evidence)
    for run in runs:
        if len(run.evidence) != total:
            raise MatcherRerankError(
                f"{run.name} scored {len(run.evidence)} queries against "
                f"{baseline.name}'s {total}, so the comparison would not be paired"
            )

    metrics: list[MetricSummary] = []
    for run in runs:
        for weight in sorted(weights):
            for cut_off in sorted(set(cut_offs)):
                hits = sum(
                    1
                    for item in run.evidence
                    if 0 < item.rank_of_truth(weight, truths[item.query_image_id]) <= cut_off
                )
                metrics.append(
                    accuracy_metric(f"{run.name}_w{weight:g}_top_{cut_off}", hits, total)
                )
        for cut_off in sorted(set(cut_offs)):
            weight, accuracy = best_weight(run.evidence, truths, weights, cut_off)
            metrics.append(
                MetricSummary(name=f"{run.name}_best_weight_top_{cut_off}", value=weight)
            )
            metrics.append(MetricSummary(name=f"{run.name}_best_top_{cut_off}", value=accuracy))

    # Paired against the first matcher, each at its own best weight, because the
    # question is which matcher wins rather than which weight does.
    for run in runs[1:]:
        for cut_off in sorted(set(cut_offs)):
            mine, _ = best_weight(run.evidence, truths, weights, cut_off)
            theirs, _ = best_weight(baseline.evidence, truths, weights, cut_off)
            wins = losses = 0
            for new, old in zip(run.evidence, baseline.evidence, strict=True):
                truth = truths[new.query_image_id]
                got = 0 < new.rank_of_truth(mine, truth) <= cut_off
                had = 0 < old.rank_of_truth(theirs, truth) <= cut_off
                wins += got and not had
                losses += had and not got
            label = f"{run.name}_vs_{baseline.name}_top_{cut_off}"
            metrics.append(MetricSummary(name=f"{label}_wins", value=float(wins)))
            metrics.append(MetricSummary(name=f"{label}_losses", value=float(losses)))
            metrics.append(
                MetricSummary(name=f"{label}_p_value", value=exact_mcnemar_p_value(wins, losses))
            )
    metrics.append(MetricSummary(name="queries", value=float(total)))
    return tuple(metrics)


def run_matcher_rerank(config: AppConfig) -> ExperimentResult:
    """Run section 10's protocol once per matcher and compare them pairwise."""
    if not isinstance(config.experiment, MatcherRerankConfig):
        raise MatcherRerankError(
            f"this requires experiment=matcher_rerank, got {config.experiment.kind}"
        )
    settings = config.experiment

    try:
        manifest, split = load_evaluation_inputs(
            config.paths.manifest_path, config.paths.evaluation_split_path
        )
    except BaselineExperimentError as error:
        raise MatcherRerankError(str(error)) from error

    matrix = load_embedding_matrix(manifest, config.backbone, Path(config.paths.embeddings_dir))
    truths = {fold.query_image_id: fold.query_dwarf_id for fold in split.folds}

    runs = tuple(
        MatcherRun(
            name=name,
            evidence=collect_evidence(
                split,
                manifest,
                matrix,
                create_matcher(name, settings.max_keypoints, settings.device),
                settings.top_k,
                photographer_disjoint=settings.photographer_disjoint,
                only_answerable=True,
            ),
        )
        for name in settings.matchers
    )

    return ExperimentResult(
        experiment="matcher_rerank",
        backbone=config.backbone.name,
        created_at=datetime.now(UTC),
        seed=settings.seed,
        configuration=settings.model_dump(mode="json"),
        metrics=summarize(runs, truths, settings.weights, settings.top_k_metrics),
    )

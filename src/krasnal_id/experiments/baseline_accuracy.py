"""Full-candidate-pool top-1, top-5, and MRR baseline."""

import json
import math
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from pydantic import ValidationError

from krasnal_id.config import AppConfig, BaselineExperimentConfig
from krasnal_id.data_pipeline.build_manifest import canonical_json_sha256
from krasnal_id.embeddings.store import EmbeddingMatrix, load_embedding_matrix
from krasnal_id.experiments.contracts import ExperimentResult, MetricSummary
from krasnal_id.models import DatasetManifest, EvaluationSplit
from krasnal_id.retrieval.knn import cosine_knn

# Two-sided 95% normal quantile, used for the Wilson score interval.
_WILSON_Z = 1.959963984540054


class BaselineExperimentError(ValueError):
    """Raised when baseline inputs are missing, invalid, or mutually inconsistent."""


@dataclass(frozen=True, slots=True)
class FoldOutcome:
    """Where the correct dwarf ranked for one leave-one-out query."""

    query_image_id: str
    query_dwarf_id: str
    dwarf_rank: int
    image_rank: int
    candidate_dwarfs: int


def wilson_interval(successes: int, total: int) -> tuple[float, float]:
    """Return the 95% Wilson score interval for an observed proportion."""
    if total <= 0:
        raise BaselineExperimentError("cannot summarize an accuracy over zero folds")
    proportion = successes / total
    z_squared = _WILSON_Z**2
    denominator = 1.0 + z_squared / total
    center = (proportion + z_squared / (2 * total)) / denominator
    margin = (
        _WILSON_Z
        * math.sqrt(proportion * (1.0 - proportion) / total + z_squared / (4 * total**2))
        / denominator
    )
    return (max(0.0, center - margin), min(1.0, center + margin))


def accuracy_metric(name: str, successes: int, total: int) -> MetricSummary:
    """Summarize a proportion with its Wilson interval as error bars."""
    lower, upper = wilson_interval(successes, total)
    return MetricSummary(
        name=name,
        value=successes / total,
        lower_bound=lower,
        upper_bound=upper,
    )


def evaluate_fold(
    query_image_id: str,
    query_dwarf_id: str,
    reference_image_ids: tuple[str, ...],
    matrix: EmbeddingMatrix,
) -> FoldOutcome:
    """Rank one query against its references and locate the correct dwarf.

    Two ranks are reported. `image_rank` is the position of the first reference
    image of the correct dwarf. `dwarf_rank` is that dwarf's position once the
    ranking is collapsed to distinct dwarves by their best-matching image, which
    is the candidate list an identification tool would actually present.
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

    ranked_dwarfs: list[str] = []
    image_rank = 0
    for position, match in enumerate(ranked.matches, start=1):
        if match.dwarf_id not in ranked_dwarfs:
            ranked_dwarfs.append(match.dwarf_id)
        if image_rank == 0 and match.dwarf_id == query_dwarf_id:
            image_rank = position

    if image_rank == 0:
        raise BaselineExperimentError(
            f"dwarf {query_dwarf_id} of query {query_image_id} has no reference image; "
            "the manifest threshold should guarantee at least two per dwarf"
        )

    return FoldOutcome(
        query_image_id=query_image_id,
        query_dwarf_id=query_dwarf_id,
        dwarf_rank=ranked_dwarfs.index(query_dwarf_id) + 1,
        image_rank=image_rank,
        candidate_dwarfs=len(ranked_dwarfs),
    )


def evaluate_baseline(
    split: EvaluationSplit,
    matrix: EmbeddingMatrix,
    top_k: tuple[int, ...],
) -> tuple[MetricSummary, ...]:
    """Measure full-pool ranking quality over every fold in the split."""
    if not split.folds:
        raise BaselineExperimentError("split contains no folds")
    if any(k <= 0 for k in top_k):
        raise BaselineExperimentError(f"top_k values must be positive: {top_k}")

    outcomes = tuple(
        evaluate_fold(fold.query_image_id, fold.query_dwarf_id, fold.reference_image_ids, matrix)
        for fold in split.folds
    )
    total = len(outcomes)

    metrics: list[MetricSummary] = []
    for k in sorted(set(top_k)):
        hits = sum(1 for outcome in outcomes if outcome.dwarf_rank <= k)
        metrics.append(accuracy_metric(f"top_{k}", hits, total))
    metrics.append(
        MetricSummary(
            name="mrr",
            value=sum(1.0 / outcome.dwarf_rank for outcome in outcomes) / total,
        )
    )
    for k in sorted(set(top_k)):
        hits = sum(1 for outcome in outcomes if outcome.image_rank <= k)
        metrics.append(accuracy_metric(f"image_top_{k}", hits, total))
    metrics.append(
        MetricSummary(
            name="image_mrr",
            value=sum(1.0 / outcome.image_rank for outcome in outcomes) / total,
        )
    )
    metrics.append(MetricSummary(name="evaluated_folds", value=float(total)))
    metrics.append(
        MetricSummary(
            name="candidate_dwarfs",
            value=float(max(outcome.candidate_dwarfs for outcome in outcomes)),
        )
    )
    return tuple(metrics)


def _read_json(path: Path, label: str) -> object:
    """Read one required JSON artifact."""
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise BaselineExperimentError(f"invalid {label} {path}: {error}") from error


def load_evaluation_inputs(
    manifest_path: Path,
    split_path: Path,
) -> tuple[DatasetManifest, EvaluationSplit]:
    """Load the manifest and split, requiring the split to describe that manifest."""
    try:
        manifest = DatasetManifest.model_validate(_read_json(manifest_path, "manifest"))
        split = EvaluationSplit.model_validate(_read_json(split_path, "split"))
    except (ValidationError, TypeError, ValueError) as error:
        if isinstance(error, BaselineExperimentError):
            raise
        raise BaselineExperimentError(f"invalid evaluation inputs: {error}") from error

    manifest_sha256 = canonical_json_sha256(manifest.model_dump(mode="json"))
    if split.manifest_sha256 != manifest_sha256:
        raise BaselineExperimentError(
            f"split {split_path} was built for manifest {split.manifest_sha256[:12]} but "
            f"{manifest_path} hashes to {manifest_sha256[:12]}; rebuild it with "
            "krasnal-id data build-split"
        )
    return manifest, split


def run_baseline(config: AppConfig) -> ExperimentResult:
    """Run the configured full-pool retrieval baseline.

    The baseline is fully deterministic: every image is queried against every other
    image exactly once, so the configured seed is recorded for provenance rather
    than used to sample anything.
    """
    if not isinstance(config.experiment, BaselineExperimentConfig):
        raise BaselineExperimentError(
            f"baseline requires experiment=baseline, got {config.experiment.kind}"
        )

    manifest, split = load_evaluation_inputs(
        config.paths.manifest_path,
        config.paths.evaluation_split_path,
    )
    matrix = load_embedding_matrix(manifest, config.backbone, config.paths.embeddings_dir)
    metrics = evaluate_baseline(split, matrix, config.experiment.top_k)

    return ExperimentResult(
        experiment="baseline",
        backbone=config.backbone.name,
        created_at=datetime.now(UTC),
        seed=config.experiment.seed,
        metrics=metrics,
    )

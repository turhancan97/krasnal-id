"""Trained-classifier comparison against raw cosine retrieval."""

import contextlib
import importlib
from collections import defaultdict
from collections.abc import Iterator
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

import numpy as np
import numpy.typing as npt

from krasnal_id.config import AppConfig, ProbeExperimentConfig
from krasnal_id.embeddings.store import EmbeddingMatrix, load_embedding_matrix
from krasnal_id.experiments.baseline_accuracy import (
    BaselineExperimentError,
    accuracy_metric,
    evaluate_fold,
    load_evaluation_inputs,
)
from krasnal_id.experiments.contracts import ExperimentResult, MetricSummary
from krasnal_id.models import EvaluationSplit


class ProbeExperimentError(ValueError):
    """Raised when probe inputs are missing, invalid, or inconsistent."""


def import_optional_analysis(module_name: str) -> Any:
    """Import an optional analysis dependency only when a probe needs it."""
    try:
        return importlib.import_module(module_name)
    except ImportError as error:
        raise ProbeExperimentError(
            "analysis dependencies are required for the linear probe; run uv sync --extra analysis"
        ) from error


@contextlib.contextmanager
def single_threaded_math() -> Iterator[None]:
    """Hold BLAS to one thread while many small classifiers are fitted.

    Each fold fits on about 145 vectors, so the threads a BLAS library spawns per
    call cost far more in oversubscription than they recover: limiting them takes
    the full leave-one-out sweep from minutes to seconds. Purely a performance
    measure, so a missing threadpoolctl is not an error.
    """
    try:
        threadpoolctl = importlib.import_module("threadpoolctl")
    except ImportError:
        yield
        return
    with threadpoolctl.threadpool_limits(limits=1):
        yield


@dataclass(frozen=True, slots=True)
class MethodOutcome:
    """Where the correct dwarf ranked for one method across every fold."""

    method: str
    ranks: tuple[int, ...]

    def top_k_hits(self, k: int) -> int:
        """Count folds whose correct dwarf ranked within the first k candidates."""
        return sum(1 for rank in self.ranks if rank <= k)

    @property
    def mrr(self) -> float:
        """Return the mean reciprocal rank of the correct dwarf."""
        return float(np.mean([1.0 / rank for rank in self.ranks]))


def _rank_of(labels: tuple[str, ...], scores: npt.NDArray[np.float32], target: str) -> int:
    """Return the one-based rank of a target label under descending scores.

    Ties break by ascending label, matching the retrieval tie-break so no method
    gains an advantage from how its scores happen to collide.
    """
    if target not in labels:
        raise ProbeExperimentError(f"dwarf {target} is absent from the classifier's classes")
    order = np.lexsort((np.asarray(labels), -np.asarray(scores, dtype=np.float64)))
    ranked = [labels[index] for index in order]
    return ranked.index(target) + 1


def class_prototypes(
    vectors: npt.NDArray[np.float32],
    labels: tuple[str, ...],
) -> tuple[tuple[str, ...], npt.NDArray[np.float32]]:
    """Build one L2-normalized mean embedding per class."""
    grouped: dict[str, list[int]] = defaultdict(list)
    for index, label in enumerate(labels):
        grouped[label].append(index)

    names = tuple(sorted(grouped))
    means = np.stack([np.asarray(vectors[grouped[name]]).mean(axis=0) for name in names])
    norms = np.linalg.norm(means, axis=1, keepdims=True)
    if not np.isfinite(norms).all() or np.any(norms <= 0.0):
        raise ProbeExperimentError("a class prototype has a zero or invalid norm")
    return names, np.asarray(means / norms, dtype=np.float32)


def prototype_rank(
    query: npt.NDArray[np.float32],
    vectors: npt.NDArray[np.float32],
    labels: tuple[str, ...],
    target: str,
) -> int:
    """Rank classes by cosine similarity to their mean embedding."""
    names, prototypes = class_prototypes(vectors, labels)
    similarities = np.asarray(prototypes @ query, dtype=np.float32)
    return _rank_of(names, similarities, target)


def linear_probe_rank(
    query: npt.NDArray[np.float32],
    vectors: npt.NDArray[np.float32],
    labels: tuple[str, ...],
    target: str,
    seed: int,
    max_iterations: int,
    regularization: float,
) -> int:
    """Fit a multinomial logistic regression on the fold and rank its classes.

    One classifier is fitted per fold, which is what keeps the comparison honest:
    the query is never part of the data its own classifier was trained on.
    """
    linear_model = import_optional_analysis("sklearn.linear_model")
    classifier = linear_model.LogisticRegression(
        C=regularization,
        max_iter=max_iterations,
        random_state=seed,
    )
    classifier.fit(np.asarray(vectors, dtype=np.float64), np.asarray(labels))
    probabilities = np.asarray(
        classifier.predict_proba(np.asarray(query, dtype=np.float64).reshape(1, -1))[0]
    )
    return _rank_of(tuple(str(name) for name in classifier.classes_), probabilities, target)


def _folds_with_limited_threads(split: EvaluationSplit) -> Iterator[Any]:
    """Iterate folds with BLAS held to one thread for the whole sweep."""
    with single_threaded_math():
        yield from split.folds


def evaluate_methods(
    split: EvaluationSplit,
    matrix: EmbeddingMatrix,
    config: ProbeExperimentConfig,
) -> tuple[MethodOutcome, ...]:
    """Rank the correct dwarf under every configured method, fold by fold."""
    if not split.folds:
        raise ProbeExperimentError("split contains no folds")

    ranks: dict[str, list[int]] = {method: [] for method in config.methods}
    for fold in _folds_with_limited_threads(split):
        vectors, dwarf_ids = matrix.rows_for(fold.reference_image_ids)
        query = matrix.vector_for(fold.query_image_id)
        for method in config.methods:
            if method == "retrieval":
                outcome = evaluate_fold(
                    fold.query_image_id,
                    fold.query_dwarf_id,
                    fold.reference_image_ids,
                    matrix,
                )
                ranks[method].append(outcome.dwarf_rank)
            elif method == "prototype":
                ranks[method].append(prototype_rank(query, vectors, dwarf_ids, fold.query_dwarf_id))
            elif method == "linear_probe":
                ranks[method].append(
                    linear_probe_rank(
                        query,
                        vectors,
                        dwarf_ids,
                        fold.query_dwarf_id,
                        config.seed,
                        config.max_iterations,
                        config.regularization,
                    )
                )
            else:  # pragma: no cover - the config contract restricts the values
                raise ProbeExperimentError(f"unsupported method: {method}")

    return tuple(
        MethodOutcome(method=method, ranks=tuple(ranks[method])) for method in config.methods
    )


def summarize_methods(
    outcomes: tuple[MethodOutcome, ...],
    top_k: tuple[int, ...],
) -> tuple[MetricSummary, ...]:
    """Turn per-method ranks into comparable accuracy and MRR metrics."""
    if not outcomes:
        raise ProbeExperimentError("no methods were evaluated")

    metrics: list[MetricSummary] = []
    for outcome in outcomes:
        total = len(outcome.ranks)
        for k in sorted(set(top_k)):
            metrics.append(
                accuracy_metric(f"{outcome.method}_top_{k}", outcome.top_k_hits(k), total)
            )
        metrics.append(MetricSummary(name=f"{outcome.method}_mrr", value=outcome.mrr))

    # The headline of this experiment is the gap, so it is recorded rather than
    # left for a reader to subtract.
    baseline = next((outcome for outcome in outcomes if outcome.method == "retrieval"), None)
    if baseline is not None:
        total = len(baseline.ranks)
        for outcome in outcomes:
            if outcome.method == "retrieval":
                continue
            gain = (outcome.top_k_hits(1) - baseline.top_k_hits(1)) / total
            metrics.append(
                MetricSummary(name=f"{outcome.method}_top_1_gain_over_retrieval", value=gain)
            )
    metrics.append(MetricSummary(name="evaluated_folds", value=float(len(outcomes[0].ranks))))
    return tuple(metrics)


def run_probe_comparison(config: AppConfig) -> ExperimentResult:
    """Compare trained classifiers against raw cosine retrieval on one split."""
    if not isinstance(config.experiment, ProbeExperimentConfig):
        raise ProbeExperimentError(
            f"the probe comparison requires experiment=probe, got {config.experiment.kind}"
        )

    try:
        manifest, split = load_evaluation_inputs(
            config.paths.manifest_path,
            config.paths.evaluation_split_path,
        )
    except BaselineExperimentError as error:
        raise ProbeExperimentError(str(error)) from error

    matrix = load_embedding_matrix(manifest, config.backbone, config.paths.embeddings_dir)
    outcomes = evaluate_methods(split, matrix, config.experiment)

    return ExperimentResult(
        experiment="probe",
        backbone=config.backbone.name,
        created_at=datetime.now(UTC),
        seed=config.experiment.seed,
        metrics=summarize_methods(outcomes, config.experiment.top_k),
    )

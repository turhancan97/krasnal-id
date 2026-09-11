"""How much of the answer is in the candidate list at all?

Re-ranking can only reorder what the first stage proposed, so its ceiling is the
first stage's recall. Section 10 hit that ceiling: withholding a photographer
pushes the correct statue outside CLIP's top 10 for a quarter of the queries, and
no amount of geometric verification can rescue them.

This measures the ceiling, and the two standard ways of raising it — both of
which fail here, which is the point of measuring rather than assuming:

* **Fusing the backbones.** Averaging two models' similarities is the obvious move
  when one is stronger. It loses to the stronger model alone, because CLIP
  contributes noise rather than an independent view.
* **Query expansion.** Re-querying with the query averaged into its own top
  results is the classical recall fix, and here it *costs* several points. It
  assumes the top results are mostly correct; at 54% precision the wrong ones
  are lookalikes, so expansion pulls the query onto its own confuser and cements
  the error.

Everything here reads cached vectors and no photographs, so it runs in seconds
and can be consulted before committing to an expensive verification sweep.
"""

from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import numpy.typing as npt

from krasnal_id.config import AppConfig, RecallCurveConfig, backbone_config
from krasnal_id.embeddings.store import EmbeddingMatrix, load_embedding_matrix
from krasnal_id.experiments.baseline_accuracy import (
    BaselineExperimentError,
    accuracy_metric,
    load_evaluation_inputs,
)
from krasnal_id.experiments.contracts import ExperimentResult, MetricSummary
from krasnal_id.models import DatasetManifest, EvaluationSplit
from krasnal_id.photographers import authors_by_image, photographers_by_class
from krasnal_id.statistics import exact_mcnemar_p_value

# Every query whose correct statue never appears. Kept out of band so a rank is
# always a positive integer and "absent" cannot be mistaken for "last".
ABSENT = 0

FULL = "full"
ANSWERABLE = "answerable"
DISJOINT = "disjoint"


class RecallCurveError(ValueError):
    """Raised when the recall inputs are missing or inconsistent."""


@dataclass(frozen=True, slots=True)
class Arm:
    """One reference regime: which queries it asks, and what it withholds."""

    name: str
    # Restrict to statues documented by more than one photographer. Section 9's
    # subset, so these columns are comparable with sections 9 and 10.
    only_answerable: bool
    photographer_disjoint: bool


ARMS = (
    Arm(FULL, only_answerable=False, photographer_disjoint=False),
    Arm(ANSWERABLE, only_answerable=True, photographer_disjoint=False),
    Arm(DISJOINT, only_answerable=True, photographer_disjoint=True),
)


def class_rank(
    scores: npt.NDArray[np.float32],
    dwarf_ids: npt.NDArray[np.str_],
    truth: str,
) -> int:
    """Return the correct statue's rank once candidates collapse to statues.

    A statue is represented by its best-scoring photograph, which is the
    candidate list an identification tool would present, and the same collapse
    the baseline and the re-ranking use.
    """
    ranked = dwarf_ids[np.argsort(-scores, kind="stable")]
    _, first = np.unique(ranked, return_index=True)
    classes = ranked[np.sort(first)]
    where = np.flatnonzero(classes == truth)
    return int(where[0]) + 1 if where.size else ABSENT


def expanded_query(
    vector: npt.NDArray[np.float32],
    references: npt.NDArray[np.float32],
    scores: npt.NDArray[np.float32],
    neighbours: int,
    alpha: float,
) -> npt.NDArray[np.float32]:
    """Return the query averaged into its own top results, similarity-weighted."""
    if neighbours <= 0:
        return vector
    top = np.argsort(-scores)[:neighbours]
    weights = np.maximum(scores[top], 0.0) ** alpha
    combined = vector + (weights[:, None] * references[top]).sum(axis=0)
    norm = float(np.linalg.norm(combined))
    return vector if norm == 0.0 else np.asarray(combined / norm, dtype=np.float32)


def measure_arm(
    arm: Arm,
    split: EvaluationSplit,
    manifest: DatasetManifest,
    matrices: dict[str, EmbeddingMatrix],
    selected: str,
    config: RecallCurveConfig,
) -> tuple[dict[str, list[int]], int]:
    """Rank every query in one regime, under the plain and the rejected variants."""
    primary = matrices[selected]
    authors = authors_by_image(manifest)
    by_class = photographers_by_class(manifest)
    rows = {image_id: index for index, image_id in enumerate(primary.image_ids)}
    dwarf_ids = np.asarray(primary.dwarf_ids)
    author_of = np.asarray([authors[image_id] for image_id in primary.image_ids])

    fused = [name for name in config.fuse_backbones if name in matrices]
    compared = [name for name in config.compare_backbones if name in matrices and name != selected]
    ranks: dict[str, list[int]] = {"plain": []}
    if len(fused) >= 2:
        ranks["fused"] = []
    for name in compared:
        ranks[f"against_{name}"] = []
    for neighbours in config.expansion_neighbours:
        ranks[f"expanded_{neighbours}"] = []

    total = 0
    for fold in split.folds:
        truth = fold.query_dwarf_id
        if arm.only_answerable and len(by_class.get(truth, set())) < 2:
            continue
        query = rows[fold.query_image_id]

        keep = np.ones(len(dwarf_ids), dtype=bool)
        keep[query] = False
        if arm.photographer_disjoint:
            keep &= author_of != author_of[query]
        indices = np.flatnonzero(keep)
        if indices.size == 0:
            continue
        total += 1

        vectors = primary.vectors[indices]
        scores = vectors @ primary.vectors[query]
        candidates = dwarf_ids[indices]
        ranks["plain"].append(class_rank(scores, candidates, truth))

        if "fused" in ranks:
            combined = np.zeros_like(scores)
            for name in fused:
                other = matrices[name]
                combined = combined + other.vectors[indices] @ other.vectors[query]
            ranks["fused"].append(class_rank(combined, candidates, truth))

        for name in compared:
            other = matrices[name]
            ranks[f"against_{name}"].append(
                class_rank(other.vectors[indices] @ other.vectors[query], candidates, truth)
            )

        for neighbours in config.expansion_neighbours:
            vector = expanded_query(
                primary.vectors[query], vectors, scores, neighbours, config.expansion_alpha
            )
            ranks[f"expanded_{neighbours}"].append(class_rank(vectors @ vector, candidates, truth))

    if total == 0:
        raise RecallCurveError(f"the {arm.name} arm scored no queries")
    return ranks, total


def summarize(
    measurements: dict[str, tuple[dict[str, list[int]], int]],
    cut_offs: tuple[int, ...],
) -> tuple[MetricSummary, ...]:
    """Report recall at each cut-off, per regime and per variant."""
    if any(k <= 0 for k in cut_offs):
        raise RecallCurveError(f"cut-offs must be positive: {cut_offs}")

    metrics: list[MetricSummary] = []
    for arm_name, (variants, total) in measurements.items():
        for variant, ranks in variants.items():
            label = arm_name if variant == "plain" else f"{arm_name}_{variant}"
            for k in sorted(set(cut_offs)):
                hits = sum(1 for rank in ranks if rank != ABSENT and rank <= k)
                metrics.append(accuracy_metric(f"{label}_r_at_{k}", hits, total))
        metrics.extend(paired_metrics(arm_name, variants, cut_offs))
        metrics.append(MetricSummary(name=f"{arm_name}_queries", value=float(total)))
    return tuple(metrics)


def paired_metrics(
    arm_name: str,
    variants: dict[str, list[int]],
    cut_offs: tuple[int, ...],
) -> list[MetricSummary]:
    """Compare each `against_*` backbone to the selected one, query by query.

    Two backbones' separate intervals are the wrong comparison here: every query is
    answered by both, so reading two overlapping intervals throws away the pairing
    and calls a real difference undecided. What carries the evidence is the queries
    where exactly one of them succeeds, which is what these counts report and what
    the exact McNemar p-value is computed from.
    """
    baseline = variants.get("plain")
    if baseline is None:
        return []

    metrics: list[MetricSummary] = []
    for variant, ranks in sorted(variants.items()):
        if not variant.startswith("against_"):
            continue
        other = variant.removeprefix("against_")
        if len(ranks) != len(baseline):
            raise RecallCurveError(
                f"{other} scored {len(ranks)} queries against {len(baseline)}, "
                "so the comparison would not be paired"
            )
        for k in sorted(set(cut_offs)):
            label = f"{arm_name}_{other}_vs_selected_at_{k}"
            hit = [rank != ABSENT and rank <= k for rank in ranks]
            was = [rank != ABSENT and rank <= k for rank in baseline]
            wins = sum(1 for new, old in zip(hit, was, strict=True) if new and not old)
            losses = sum(1 for new, old in zip(hit, was, strict=True) if old and not new)
            metrics.append(
                MetricSummary(name=f"{label}_delta", value=(wins - losses) / len(baseline))
            )
            metrics.append(MetricSummary(name=f"{label}_wins", value=float(wins)))
            metrics.append(MetricSummary(name=f"{label}_losses", value=float(losses)))
            metrics.append(
                MetricSummary(name=f"{label}_p_value", value=exact_mcnemar_p_value(wins, losses))
            )
    return metrics


def run_recall_curve(config: AppConfig) -> ExperimentResult:
    """Measure the first stage's recall, and whether two standard fixes help."""
    if not isinstance(config.experiment, RecallCurveConfig):
        raise RecallCurveError(
            f"the recall curve requires experiment=recall_curve, got {config.experiment.kind}"
        )

    try:
        manifest, split = load_evaluation_inputs(
            config.paths.manifest_path, config.paths.evaluation_split_path
        )
    except BaselineExperimentError as error:
        raise RecallCurveError(str(error)) from error

    wanted = {
        config.backbone.name,
        *config.experiment.fuse_backbones,
        *config.experiment.compare_backbones,
    }
    matrices: dict[str, EmbeddingMatrix] = {}
    for name in sorted(wanted):
        backbone = config.backbone if name == config.backbone.name else backbone_config(name)
        matrices[name] = load_embedding_matrix(
            manifest, backbone, Path(config.paths.embeddings_dir)
        )

    primary = matrices[config.backbone.name]
    for loaded, matrix in matrices.items():
        if matrix.image_ids != primary.image_ids:
            raise RecallCurveError(
                f"the {loaded} vectors are not in the same row order as "
                f"{config.backbone.name}, so a fused score would mix images"
            )

    measurements = {
        arm.name: measure_arm(
            arm, split, manifest, matrices, config.backbone.name, config.experiment
        )
        for arm in ARMS
    }

    return ExperimentResult(
        experiment="recall_curve",
        backbone=config.backbone.name,
        created_at=datetime.now(UTC),
        seed=config.experiment.seed,
        configuration=config.experiment.model_dump(mode="json"),
        metrics=summarize(measurements, config.experiment.top_k),
    )

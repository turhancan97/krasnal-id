"""Open-set rejection: can a similarity threshold say "I don't know this one"?

Retrieval always returns a nearest neighbour, so a photograph of a statue that is
absent from the reference set still produces a confident-looking ranking. This
experiment measures whether thresholding the top-1 cosine similarity separates
queries whose dwarf is present from queries whose dwarf is not.

The protocol is recorded in `AGENTS.md` section 7.2. Two populations of equal size
are built from the manifest without sampling: the known arm is the existing
leave-one-out split, and the unknown arm removes every image of a query's own dwarf
so that the dwarf is genuinely missing. The unknown arm therefore searches a gallery
holding one fewer class, which is inherent to making a class absent rather than a
flaw; top-1 similarity to the nearest *other* statue is what both arms measure.
"""

from collections import Counter
from dataclasses import dataclass
from datetime import UTC, datetime

import numpy as np
import numpy.typing as npt

from krasnal_id.config import AppConfig, OpenSetExperimentConfig
from krasnal_id.embeddings.store import EmbeddingMatrix, load_embedding_matrix
from krasnal_id.experiments.baseline_accuracy import (
    BaselineExperimentError,
    accuracy_metric,
    load_evaluation_inputs,
)
from krasnal_id.experiments.contracts import (
    DwarfRejection,
    MetricSummary,
    OpenSetRejectionResult,
    RejectionOperatingPoint,
)
from krasnal_id.models import DatasetManifest, EvaluationSplit
from krasnal_id.retrieval.knn import cosine_knn


class OpenSetExperimentError(ValueError):
    """Raised when open-set inputs are missing, invalid, or inconsistent."""


@dataclass(frozen=True, slots=True)
class ScoredQuery:
    """The top-1 match for one query, and whether its own dwarf was reachable.

    `present` distinguishes the two arms: a known query's dwarf is in the gallery
    it searched, an unknown query's dwarf is not.
    """

    query_image_id: str
    query_dwarf_id: str
    top_similarity: float
    top_dwarf_id: str
    present: bool

    @property
    def identified(self) -> bool:
        """Whether the top-1 match names the query's own dwarf."""
        return self.top_dwarf_id == self.query_dwarf_id


def _top_match(
    query_image_id: str,
    reference_image_ids: tuple[str, ...],
    matrix: EmbeddingMatrix,
    present: bool,
    query_dwarf_id: str,
) -> ScoredQuery:
    """Rank one query against a gallery and keep only its best match."""
    if not reference_image_ids:
        raise OpenSetExperimentError(
            f"query {query_image_id} has an empty gallery; open-set scoring needs "
            "at least two dwarves in the manifest"
        )
    vectors, dwarf_ids = matrix.rows_for(reference_image_ids)
    ranked = cosine_knn(
        query_image_id,
        matrix.vector_for(query_image_id),
        vectors,
        reference_image_ids,
        dwarf_ids,
        top_k=1,
    )
    best = ranked.matches[0]
    return ScoredQuery(
        query_image_id=query_image_id,
        query_dwarf_id=query_dwarf_id,
        top_similarity=best.cosine_similarity,
        top_dwarf_id=best.dwarf_id,
        present=present,
    )


def score_known_queries(
    split: EvaluationSplit,
    matrix: EmbeddingMatrix,
) -> tuple[ScoredQuery, ...]:
    """Score every leave-one-out fold, whose correct dwarf is in the gallery."""
    if not split.folds:
        raise OpenSetExperimentError("split contains no folds")
    return tuple(
        _top_match(
            fold.query_image_id,
            fold.reference_image_ids,
            matrix,
            present=True,
            query_dwarf_id=fold.query_dwarf_id,
        )
        for fold in split.folds
    )


def score_unknown_queries(matrix: EmbeddingMatrix) -> tuple[ScoredQuery, ...]:
    """Score every image against a gallery holding none of its own dwarf.

    Removing the whole class, rather than one image, is what makes the query
    genuinely unknown: with a sibling image left in place the correct answer would
    still be reachable and the query would not be open-set at all.
    """
    if len(set(matrix.dwarf_ids)) < 2:
        raise OpenSetExperimentError(
            "open-set scoring needs at least two dwarves; removing the only class "
            "leaves no gallery to search"
        )

    by_dwarf: dict[str, list[str]] = {}
    for image_id, dwarf_id in zip(matrix.image_ids, matrix.dwarf_ids, strict=True):
        by_dwarf.setdefault(dwarf_id, []).append(image_id)

    scored: list[ScoredQuery] = []
    for dwarf_id, own_image_ids in sorted(by_dwarf.items()):
        gallery = tuple(
            image_id for image_id in matrix.image_ids if image_id not in set(own_image_ids)
        )
        for image_id in own_image_ids:
            scored.append(
                _top_match(image_id, gallery, matrix, present=False, query_dwarf_id=dwarf_id)
            )
    return tuple(scored)


def _average_ranks(values: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    """Return one-based ranks of ascending values, averaging over ties.

    Ties matter here: identical similarities must not be broken arbitrarily, or the
    AUROC would depend on input order rather than on the scores themselves.
    """
    order = np.argsort(values, kind="stable")
    ranks = np.empty(values.shape[0], dtype=np.float64)
    sorted_values = values[order]
    start = 0
    while start < sorted_values.shape[0]:
        stop = start
        while stop + 1 < sorted_values.shape[0] and sorted_values[stop + 1] == sorted_values[start]:
            stop += 1
        ranks[order[start : stop + 1]] = (start + stop) / 2.0 + 1.0
        start = stop + 1
    return ranks


def separability_auroc(known: npt.NDArray[np.float64], unknown: npt.NDArray[np.float64]) -> float:
    """Return the probability that a known query outscores an unknown one.

    Ties count as half. This is the threshold-free headline: 1.0 means some
    threshold separates the two populations perfectly, 0.5 means top-1 similarity
    carries no information about whether the dwarf is present at all.
    """
    if known.shape[0] == 0 or unknown.shape[0] == 0:
        raise OpenSetExperimentError("AUROC needs at least one query in each population")
    ranks = _average_ranks(np.concatenate([known, unknown]))
    known_rank_sum = float(ranks[: known.shape[0]].sum())
    count_known = float(known.shape[0])
    count_unknown = float(unknown.shape[0])
    return (known_rank_sum - count_known * (count_known + 1.0) / 2.0) / (
        count_known * count_unknown
    )


def threshold_for_target(known: npt.NDArray[np.float64], target: float) -> float:
    """Return the highest observed score that still accepts `target` of the known arm.

    The quantile is taken with `lower` interpolation, so the threshold is always an
    observed score and the achieved acceptance rate is at least the target. A
    quantile of finitely many discrete scores rarely lands exactly on a requested
    rate, which is why the achieved rate is reported alongside the target.
    """
    if known.shape[0] == 0:
        raise OpenSetExperimentError("cannot calibrate a threshold on zero known queries")
    if not 0.0 < target < 1.0:
        raise OpenSetExperimentError(f"target acceptance must lie in (0, 1): {target}")
    return float(np.quantile(known, 1.0 - target, method="lower"))


def calibrate_leave_one_class_out(
    known: tuple[ScoredQuery, ...],
    target: float,
) -> dict[str, float]:
    """Calibrate one threshold per dwarf from the *other* dwarves' known scores.

    A threshold fitted on all the data has already seen the queries it judges, so it
    reports an optimistic false-acceptance rate. Holding each dwarf out of its own
    calibration removes that leak at the cost of a slightly smaller calibration set.
    """
    dwarf_ids = sorted({query.query_dwarf_id for query in known})
    if len(dwarf_ids) < 2:
        raise OpenSetExperimentError(
            "leave-one-class-out calibration needs at least two dwarves in the known arm"
        )
    thresholds: dict[str, float] = {}
    for dwarf_id in dwarf_ids:
        others = np.asarray(
            [query.top_similarity for query in known if query.query_dwarf_id != dwarf_id],
            dtype=np.float64,
        )
        thresholds[dwarf_id] = threshold_for_target(others, target)
    return thresholds


def best_balanced_accuracy(
    known: npt.NDArray[np.float64],
    unknown: npt.NDArray[np.float64],
) -> tuple[float, float]:
    """Return the best in-sample balanced accuracy and the threshold reaching it.

    Swept over every observed score, so it is the most favourable operating point
    this data admits. It is an upper bound and is reported as one: the threshold was
    chosen with the answers in view.
    """
    candidates = np.unique(np.concatenate([known, unknown]))
    best_score = -1.0
    best_threshold = float(candidates[0])
    for threshold in candidates:
        accept_rate = float((known >= threshold).mean())
        reject_rate = float((unknown < threshold).mean())
        balanced = (accept_rate + reject_rate) / 2.0
        if balanced > best_score:
            best_score = balanced
            best_threshold = float(threshold)
    return best_score, best_threshold


def rejection_curve(
    known: npt.NDArray[np.float64],
    unknown: npt.NDArray[np.float64],
) -> tuple[RejectionOperatingPoint, ...]:
    """Sweep every observed score to record the whole acceptance tradeoff.

    This is the descriptive object behind the AUROC: what any threshold would
    accept on both populations. It is deliberately not calibrated, so it must not
    be read as a set of achievable operating points — those are in the metrics,
    fitted leave-one-class-out. The endpoints accept everything and reject
    everything, so the curve spans the full range rather than only the observed one.
    """
    if known.shape[0] == 0 or unknown.shape[0] == 0:
        raise OpenSetExperimentError("a rejection curve needs both populations")

    observed = np.unique(np.concatenate([known, unknown]))
    # Finite endpoints on both ends: an infinity would not survive a JSON round
    # trip, and one step past the extreme observed score accepts or rejects
    # everything just as well.
    thresholds = np.concatenate(
        [
            [np.nextafter(observed[0], -np.inf)],
            observed,
            [np.nextafter(observed[-1], np.inf)],
        ]
    )
    return tuple(
        RejectionOperatingPoint(
            threshold=float(threshold),
            known_acceptance=float((known >= threshold).mean()),
            false_acceptance=float((unknown >= threshold).mean()),
        )
        for threshold in thresholds
    )


def _operating_point_metrics(
    known: tuple[ScoredQuery, ...],
    unknown: tuple[ScoredQuery, ...],
    thresholds: dict[str, float],
    label: str,
) -> list[MetricSummary]:
    """Summarize one thresholded operating point over both populations."""
    accepted_known = [
        query for query in known if query.top_similarity >= thresholds[query.query_dwarf_id]
    ]
    identified_known = [query for query in accepted_known if query.identified]
    accepted_unknown = [
        query
        for query in unknown
        if query.top_similarity >= thresholds.get(query.query_dwarf_id, float("inf"))
    ]
    total = len(known) + len(unknown)
    correct = len(identified_known) + (len(unknown) - len(accepted_unknown))
    values = sorted(thresholds.values())

    return [
        accuracy_metric(f"{label}_open_set_accuracy", correct, total),
        accuracy_metric(f"{label}_known_acceptance", len(accepted_known), len(known)),
        accuracy_metric(f"{label}_known_correct_acceptance", len(identified_known), len(known)),
        accuracy_metric(f"{label}_false_acceptance", len(accepted_unknown), len(unknown)),
        MetricSummary(
            name=f"{label}_threshold",
            value=float(np.mean(values)),
            lower_bound=values[0],
            upper_bound=values[-1],
        ),
    ]


def summarize_open_set(
    known: tuple[ScoredQuery, ...],
    unknown: tuple[ScoredQuery, ...],
    config: OpenSetExperimentConfig,
) -> tuple[MetricSummary, ...]:
    """Turn both scored populations into threshold-free and operating-point metrics."""
    if not known or not unknown:
        raise OpenSetExperimentError("both the known and unknown populations must be non-empty")

    known_scores = np.asarray([query.top_similarity for query in known], dtype=np.float64)
    unknown_scores = np.asarray([query.top_similarity for query in unknown], dtype=np.float64)

    metrics: list[MetricSummary] = [
        MetricSummary(name="auroc", value=separability_auroc(known_scores, unknown_scores)),
        MetricSummary(
            name="mean_similarity_gap",
            value=float(known_scores.mean() - unknown_scores.mean()),
            lower_bound=float(unknown_scores.mean()),
            upper_bound=float(known_scores.mean()),
        ),
    ]

    # Closed-set top-1 over the same folds, so the cost of adding rejection is
    # readable from this artifact alone rather than by opening the baseline's.
    metrics.append(
        accuracy_metric(
            "closed_set_top_1", sum(1 for query in known if query.identified), len(known)
        )
    )

    for target in config.target_known_acceptance:
        label = f"target_{target * 100:g}"
        thresholds = calibrate_leave_one_class_out(known, target)
        metrics.extend(_operating_point_metrics(known, unknown, thresholds, label))

    in_sample_threshold = threshold_for_target(known_scores, config.primary_target)
    metrics.extend(
        _operating_point_metrics(
            known,
            unknown,
            dict.fromkeys(
                {query.query_dwarf_id for query in [*known, *unknown]}, in_sample_threshold
            ),
            f"in_sample_target_{config.primary_target * 100:g}",
        )
    )
    balanced, balanced_threshold = best_balanced_accuracy(known_scores, unknown_scores)
    metrics.append(MetricSummary(name="in_sample_best_balanced_accuracy", value=balanced))
    metrics.append(MetricSummary(name="in_sample_best_threshold", value=balanced_threshold))

    metrics.append(MetricSummary(name="known_queries", value=float(len(known))))
    metrics.append(MetricSummary(name="unknown_queries", value=float(len(unknown))))
    return tuple(metrics)


def rank_rejections(
    known: tuple[ScoredQuery, ...],
    unknown: tuple[ScoredQuery, ...],
    manifest: DatasetManifest,
    config: OpenSetExperimentConfig,
) -> tuple[DwarfRejection, ...]:
    """Report, per removed dwarf, how often it slipped through and what covered for it."""
    thresholds = calibrate_leave_one_class_out(known, config.primary_target)
    names = {dwarf.dwarf_id: dwarf.display_name for dwarf in manifest.dwarfs}

    grouped: dict[str, list[ScoredQuery]] = {}
    for query in unknown:
        grouped.setdefault(query.query_dwarf_id, []).append(query)

    rejections: list[DwarfRejection] = []
    for dwarf_id, queries in grouped.items():
        if dwarf_id not in names:
            raise OpenSetExperimentError(f"dwarf {dwarf_id} is absent from the manifest")
        threshold = thresholds.get(dwarf_id, float("inf"))
        nearest_dwarf_id, _ = Counter(query.top_dwarf_id for query in queries).most_common(1)[0]
        if nearest_dwarf_id not in names:
            raise OpenSetExperimentError(f"dwarf {nearest_dwarf_id} is absent from the manifest")
        rejections.append(
            DwarfRejection(
                dwarf_id=dwarf_id,
                display_name=names[dwarf_id],
                unknown_queries=len(queries),
                false_accepts=sum(1 for query in queries if query.top_similarity >= threshold),
                mean_top_similarity=float(np.mean([query.top_similarity for query in queries])),
                nearest_dwarf_id=nearest_dwarf_id,
                nearest_display_name=names[nearest_dwarf_id],
            )
        )

    # Worst first: the dwarves a threshold cannot protect are the reportable ones.
    rejections.sort(key=lambda row: (-row.false_accepts, -row.mean_top_similarity, row.dwarf_id))
    return tuple(rejections[: config.top_rejections])


def run_open_set_rejection(config: AppConfig) -> OpenSetRejectionResult:
    """Measure whether a similarity threshold can reject unknown dwarves."""
    if not isinstance(config.experiment, OpenSetExperimentConfig):
        raise OpenSetExperimentError(
            f"open-set rejection requires experiment=open_set, got {config.experiment.kind}"
        )

    try:
        manifest, split = load_evaluation_inputs(
            config.paths.manifest_path,
            config.paths.evaluation_split_path,
        )
    except BaselineExperimentError as error:
        raise OpenSetExperimentError(str(error)) from error

    matrix = load_embedding_matrix(manifest, config.backbone, config.paths.embeddings_dir)
    known = score_known_queries(split, matrix)
    unknown = score_unknown_queries(matrix)

    return OpenSetRejectionResult(
        experiment="open_set",
        backbone=config.backbone.name,
        created_at=datetime.now(UTC),
        seed=config.experiment.seed,
        metrics=summarize_open_set(known, unknown, config.experiment),
        rejections=rank_rejections(known, unknown, manifest, config.experiment),
        curve=rejection_curve(
            np.asarray([query.top_similarity for query in known], dtype=np.float64),
            np.asarray([query.top_similarity for query in unknown], dtype=np.float64),
        ),
    )

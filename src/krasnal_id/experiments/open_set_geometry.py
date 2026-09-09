"""Open-set rejection from geometry: can inliers say "I don't know" where cosine cannot?

Section 7.3 measured a similarity threshold and found no operating point worth
shipping at this scale: DINOv2's false acceptance rises from 4% at 23 classes to
38% at 306 at the same target. The reason is that cosine similarity is a poor
confidence score here — every statue is the same semantic category, so the nearest
*wrong* statue scores nearly as high as the right one, and a missing statue's
nearest neighbour looks exactly like a present one's.

Section 7.6 then showed that geometry discriminates where similarity does not:
two photographs of one physical object admit a consistent homography and two
photographs of similar-but-different objects do not. That is a different kind of
evidence, and nobody has asked whether it rejects. This experiment asks.

**The signals are compared on one query population, not across runs.** Four scores
are computed for every query, and cosine similarity is among them as a control: it
must reproduce the AUROC section 7.3 reports, which is what makes any difference in
the geometric rows readable as the signal rather than as the harness.

**The photographer-disjoint condition is not optional here.** Section 7.6 found the
inlier separation inflated by same-visit near-duplicates, so a known query may be
matching its own photographer's other frame from the same angle rather than the
statue. A geometric signal that works only when the same person shot the reference
has not been shown to work at all, so both arms are re-scored with the query's own
photographer withheld and both numbers are reported.
"""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import numpy.typing as npt

from krasnal_id.config import AppConfig, OpenSetGeometryConfig
from krasnal_id.embeddings.store import EmbeddingMatrix, load_embedding_matrix
from krasnal_id.experiments.baseline_accuracy import (
    BaselineExperimentError,
    load_evaluation_inputs,
)
from krasnal_id.experiments.contracts import (
    MetricSummary,
    OpenSetGeometryResult,
    RejectionSignalOutcome,
)
from krasnal_id.models import DatasetManifest, EvaluationSplit
from krasnal_id.photographers import (
    authors_by_image,
    disjoint_references,
    is_answerable,
    photographers_by_class,
)
from krasnal_id.retrieval.knn import cosine_knn
from krasnal_id.retrieval.rerank import FeatureCache, RerankError, blended_score, count_inliers
from krasnal_id.statistics import separability_auroc

# The signals compared, in the order they are reported. `cosine` is the control.
SIGNALS = ("cosine", "inliers_top_1", "inliers_best", "blended")
STANDARD = "standard"
DISJOINT = "photographer_disjoint"


class OpenSetGeometryError(ValueError):
    """Raised when geometric rejection inputs are missing, invalid, or inconsistent."""


@dataclass(frozen=True, slots=True)
class QueryEvidence:
    """One query's appearance and geometric evidence against its best candidates.

    `present` distinguishes the arms: a known query's own statue is in the gallery
    it searched, an unknown query's is not. Every field is evidence about the same
    ranking, so the signals derived from it differ only in what they read.
    """

    query_image_id: str
    query_dwarf_id: str
    present: bool
    top_cosine: float
    top_inliers: int
    best_inliers: int
    best_blended: float

    def score(self, signal: str) -> float:
        """Return this query's value for one rejection signal."""
        if signal == "cosine":
            return self.top_cosine
        if signal == "inliers_top_1":
            return float(self.top_inliers)
        if signal == "inliers_best":
            return float(self.best_inliers)
        if signal == "blended":
            return self.best_blended
        raise OpenSetGeometryError(f"unknown rejection signal: {signal}")


def _evidence_for_query(
    query_image_id: str,
    query_dwarf_id: str,
    gallery: tuple[str, ...],
    matrix: EmbeddingMatrix,
    cache: FeatureCache,
    paths: Mapping[str, Path],
    *,
    present: bool,
    top_k: int,
    weight: float,
) -> QueryEvidence:
    """Rank one query, verify its top candidates, and record both kinds of evidence."""
    if not gallery:
        raise OpenSetGeometryError(
            f"query {query_image_id} has an empty gallery; rejection scoring needs "
            "at least one reachable reference"
        )
    vectors, dwarf_ids = matrix.rows_for(gallery)
    ranked = cosine_knn(
        query_image_id,
        matrix.vector_for(query_image_id),
        vectors,
        gallery,
        dwarf_ids,
        top_k=min(top_k, len(gallery)),
    )
    matches = ranked.matches
    query_features = cache.get(query_image_id, paths[query_image_id])

    inliers = [
        count_inliers(query_features, cache.get(match.image_id, paths[match.image_id]))
        for match in matches
    ]
    blended = [
        blended_score(match.cosine_similarity, count, weight)
        for match, count in zip(matches, inliers, strict=True)
    ]
    return QueryEvidence(
        query_image_id=query_image_id,
        query_dwarf_id=query_dwarf_id,
        present=present,
        top_cosine=matches[0].cosine_similarity,
        top_inliers=inliers[0],
        # The maximum over the checked candidates, not the top-1's count. A present
        # statue that the ranking put second still has geometry available, and a
        # rejector is allowed to look at every candidate it was given.
        best_inliers=max(inliers),
        best_blended=max(blended),
    )


def collect_known_arm(
    split: EvaluationSplit,
    matrix: EmbeddingMatrix,
    cache: FeatureCache,
    paths: Mapping[str, Path],
    config: OpenSetGeometryConfig,
    *,
    authors: Mapping[str, str] | None = None,
    answerable: Mapping[str, set[str]] | None = None,
) -> tuple[QueryEvidence, ...]:
    """Score the leave-one-out folds, whose own statue is reachable.

    Passing `authors` withholds each query's own photographer, which also drops the
    queries whose statue only one person documented: with that person removed the
    correct answer is gone, so the query is unanswerable rather than hard and
    scoring it would measure the dataset's coverage instead of the signal.
    """
    if not split.folds:
        raise OpenSetGeometryError("split contains no folds")

    collected: list[QueryEvidence] = []
    for fold in split.folds:
        gallery = fold.reference_image_ids
        if authors is not None:
            if answerable is None:
                raise OpenSetGeometryError(
                    "withholding a photographer needs the per-class photographer sets"
                )
            if not is_answerable(fold.query_dwarf_id, dict(answerable)):
                continue
            gallery = disjoint_references(gallery, dict(authors), authors[fold.query_image_id])
            if not gallery:
                continue
        collected.append(
            _evidence_for_query(
                fold.query_image_id,
                fold.query_dwarf_id,
                gallery,
                matrix,
                cache,
                paths,
                present=True,
                top_k=config.top_k,
                weight=config.blend_weight,
            )
        )
    if not collected:
        raise OpenSetGeometryError("no known query survived the requested condition")
    return tuple(collected)


def collect_unknown_arm(
    matrix: EmbeddingMatrix,
    cache: FeatureCache,
    paths: Mapping[str, Path],
    config: OpenSetGeometryConfig,
    *,
    authors: Mapping[str, str] | None = None,
    answerable: Mapping[str, set[str]] | None = None,
) -> tuple[QueryEvidence, ...]:
    """Score every image against a gallery holding none of its own statue.

    Removing the whole class is what makes the query genuinely unknown; with one
    sibling left in place the correct answer would still be reachable. When a
    photographer is withheld it is withheld here too, even though the unknown arm
    has no correct answer to leak: the arms would otherwise search galleries of
    different sizes, and a difference in gallery size is itself a difference in how
    hard the nearest wrong statue is to find.
    """
    by_dwarf: dict[str, list[str]] = {}
    for image_id, dwarf_id in zip(matrix.image_ids, matrix.dwarf_ids, strict=True):
        by_dwarf.setdefault(dwarf_id, []).append(image_id)
    if len(by_dwarf) < 2:
        raise OpenSetGeometryError(
            "rejection scoring needs at least two statues; removing the only class "
            "leaves no gallery to search"
        )

    collected: list[QueryEvidence] = []
    for dwarf_id, own_image_ids in sorted(by_dwarf.items()):
        if authors is not None:
            if answerable is None:
                raise OpenSetGeometryError(
                    "withholding a photographer needs the per-class photographer sets"
                )
            if not is_answerable(dwarf_id, dict(answerable)):
                continue
        own = set(own_image_ids)
        outside = tuple(image_id for image_id in matrix.image_ids if image_id not in own)
        for image_id in own_image_ids:
            gallery = outside
            if authors is not None:
                gallery = disjoint_references(gallery, dict(authors), authors[image_id])
                if not gallery:
                    continue
            collected.append(
                _evidence_for_query(
                    image_id,
                    dwarf_id,
                    gallery,
                    matrix,
                    cache,
                    paths,
                    present=False,
                    top_k=config.top_k,
                    weight=config.blend_weight,
                )
            )
    if not collected:
        raise OpenSetGeometryError("no unknown query survived the requested condition")
    return tuple(collected)


def calibrate_leave_one_class_out(
    known: Sequence[QueryEvidence],
    signal: str,
    target: float,
) -> dict[str, float]:
    """Fit one threshold per statue from the *other* statues' known scores.

    A threshold fitted on all the data has already seen the queries it judges and
    reports an optimistic false-acceptance rate. This mirrors the calibration the
    similarity experiment uses; it is written against this module's evidence rather
    than imported because the score being calibrated is a parameter here.
    """
    dwarf_ids = sorted({query.query_dwarf_id for query in known})
    if len(dwarf_ids) < 2:
        raise OpenSetGeometryError(
            "leave-one-class-out calibration needs at least two statues in the known arm"
        )
    if not 0.0 < target < 1.0:
        raise OpenSetGeometryError(f"target acceptance must lie in (0, 1): {target}")

    thresholds: dict[str, float] = {}
    for dwarf_id in dwarf_ids:
        others = np.asarray(
            [query.score(signal) for query in known if query.query_dwarf_id != dwarf_id],
            dtype=np.float64,
        )
        # `lower` interpolation keeps the threshold an observed score, so the
        # achieved acceptance is at least the target and is reported alongside it.
        thresholds[dwarf_id] = float(np.quantile(others, 1.0 - target, method="lower"))
    return thresholds


def best_balanced_accuracy(
    known: npt.NDArray[np.float64],
    unknown: npt.NDArray[np.float64],
) -> float:
    """Return the best balanced accuracy any threshold reaches on this data.

    Swept over every observed score with the answers in view, so it is an upper
    bound and is reported as one. It is the fairest possible comparison between
    signals whose units differ: no calibration can beat it.
    """
    candidates = np.unique(np.concatenate([known, unknown]))
    best = -1.0
    for threshold in candidates:
        accept = float((known >= threshold).mean())
        reject = float((unknown < threshold).mean())
        best = max(best, (accept + reject) / 2.0)
    return best


def measure_signal(
    known: Sequence[QueryEvidence],
    unknown: Sequence[QueryEvidence],
    signal: str,
    condition: str,
    target: float,
) -> RejectionSignalOutcome:
    """Summarize how well one signal separates the two arms."""
    known_scores = np.asarray([query.score(signal) for query in known], dtype=np.float64)
    unknown_scores = np.asarray([query.score(signal) for query in unknown], dtype=np.float64)

    thresholds = calibrate_leave_one_class_out(known, signal, target)
    accepted_known = sum(
        1 for query in known if query.score(signal) >= thresholds[query.query_dwarf_id]
    )
    # A statue absent from the known arm has no calibrated threshold of its own, so
    # nothing about it can be accepted. Infinity records that as a rejection rather
    # than silently skipping the query and shrinking the denominator.
    accepted_unknown = sum(
        1
        for query in unknown
        if query.score(signal) >= thresholds.get(query.query_dwarf_id, float("inf"))
    )

    return RejectionSignalOutcome(
        signal=signal,
        condition=condition,
        auroc=separability_auroc(known_scores, unknown_scores),
        known_queries=len(known),
        unknown_queries=len(unknown),
        target_known_acceptance=target,
        achieved_known_acceptance=accepted_known / len(known),
        false_acceptance=accepted_unknown / len(unknown),
        in_sample_balanced_accuracy=best_balanced_accuracy(known_scores, unknown_scores),
        known_mean_score=float(known_scores.mean()),
        unknown_mean_score=float(unknown_scores.mean()),
    )


def summarize(signals: Sequence[RejectionSignalOutcome]) -> tuple[MetricSummary, ...]:
    """Promote the comparison's headline numbers into metrics.

    The signal rows carry the full comparison; these exist so the artifact answers
    "did geometry beat similarity" without a reader joining rows themselves.
    """
    if not signals:
        raise OpenSetGeometryError("cannot summarize an empty signal comparison")

    metrics: list[MetricSummary] = []
    for row in signals:
        metrics.append(MetricSummary(name=f"{row.condition}_{row.signal}_auroc", value=row.auroc))
        metrics.append(
            MetricSummary(
                name=f"{row.condition}_{row.signal}_false_acceptance",
                value=row.false_acceptance,
            )
        )

    for condition in dict.fromkeys(row.condition for row in signals):
        rows = [row for row in signals if row.condition == condition]
        control = next((row for row in rows if row.signal == "cosine"), None)
        if control is None:
            continue
        geometric = [row for row in rows if row.signal != "cosine"]
        if not geometric:
            continue
        best = max(geometric, key=lambda row: row.auroc)
        metrics.append(
            MetricSummary(
                name=f"{condition}_best_geometric_auroc_gain",
                value=best.auroc - control.auroc,
                lower_bound=control.auroc,
                upper_bound=best.auroc,
            )
        )
        metrics.append(
            MetricSummary(
                name=f"{condition}_best_geometric_false_acceptance_change",
                value=best.false_acceptance - control.false_acceptance,
                lower_bound=control.false_acceptance,
                upper_bound=best.false_acceptance,
            )
        )
    return tuple(metrics)


def _require_images(manifest: DatasetManifest) -> dict[str, Path]:
    """Return every image's path, failing before any feature is computed."""
    paths = {image.image_id: Path(image.local_path) for image in manifest.images}
    missing = [image_id for image_id, local_path in paths.items() if not local_path.is_file()]
    if missing:
        raise OpenSetGeometryError(
            f"{len(missing)} image(s) are missing, first {missing[0]} at "
            f"{paths[missing[0]]}; geometric rejection reads pixels, so run "
            "data fetch-images before it"
        )
    return paths


def run_open_set_geometry(config: AppConfig) -> OpenSetGeometryResult:
    """Compare appearance and geometry as open-set rejection signals."""
    if not isinstance(config.experiment, OpenSetGeometryConfig):
        raise OpenSetGeometryError(
            "geometric open-set rejection requires experiment=open_set_geometry, "
            f"got {config.experiment.kind}"
        )
    settings = config.experiment

    try:
        manifest, split = load_evaluation_inputs(
            config.paths.manifest_path,
            config.paths.evaluation_split_path,
        )
    except BaselineExperimentError as error:
        raise OpenSetGeometryError(str(error)) from error

    paths = _require_images(manifest)
    matrix = load_embedding_matrix(manifest, config.backbone, config.paths.embeddings_dir)
    cache = FeatureCache(settings.max_keypoints)

    conditions: list[tuple[str, dict[str, str] | None]] = [(STANDARD, None)]
    if settings.photographer_disjoint:
        conditions.append((DISJOINT, authors_by_image(manifest)))
    by_class = photographers_by_class(manifest)

    rows: list[RejectionSignalOutcome] = []
    try:
        for condition, authors in conditions:
            known = collect_known_arm(
                split,
                matrix,
                cache,
                paths,
                settings,
                authors=authors,
                answerable=by_class if authors is not None else None,
            )
            unknown = collect_unknown_arm(
                matrix,
                cache,
                paths,
                settings,
                authors=authors,
                answerable=by_class if authors is not None else None,
            )
            rows.extend(
                measure_signal(known, unknown, signal, condition, settings.target_known_acceptance)
                for signal in SIGNALS
            )
    except RerankError as error:
        raise OpenSetGeometryError(str(error)) from error

    return OpenSetGeometryResult(
        experiment="open_set_geometry",
        backbone=config.backbone.name,
        created_at=datetime.now(UTC),
        seed=settings.seed,
        configuration=settings.model_dump(mode="json"),
        metrics=summarize(rows),
        signals=tuple(rows),
    )

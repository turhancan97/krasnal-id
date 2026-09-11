"""Can local features retrieve what appearance loses?

Section 10 verified geometry on the top candidates a cosine ranking had already
chosen, so geometry has only ever been shown a shortlist. Section 11 then found
that the shortlist is the limit: the correct statue is outside the top 10 for a
tenth of the disjoint queries and no re-ranking can rescue what was never
proposed. Section 13 ruled out buying the missing recall with capacity.

That leaves the question this asks. Rank *every* reference by its inlier count
against the query, with no appearance involved at any point, and see where the
correct statue lands -- and in particular whether it lands inside the top k for
the queries appearance loses. That number, the rescue rate, is what decides
whether a real local-feature index is worth building.

**This is a feasibility probe and not a retrieval system.** It fits 1,690
homographies per query, which is seven seconds a photograph against a cosine
ranking's microseconds. Nothing here is deployable, and a positive result would
be an argument for building an index (ASMK, VLAD, a learned detector), not for
shipping this loop.
"""

import hashlib
import json
import logging
import os
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import numpy.typing as npt

from krasnal_id.config import AppConfig, GeometryFirstConfig
from krasnal_id.embeddings.store import (
    BackboneIdentity,
    EmbeddingMatrix,
    load_embedding_matrix,
)
from krasnal_id.experiments.baseline_accuracy import (
    BaselineExperimentError,
    accuracy_metric,
    load_evaluation_inputs,
)
from krasnal_id.experiments.contracts import ExperimentResult, MetricSummary
from krasnal_id.experiments.recall_curve import ABSENT, class_rank
from krasnal_id.models import DatasetManifest, EvaluationSplit
from krasnal_id.photographers import authors_by_image, photographers_by_class
from krasnal_id.retrieval.rerank import FeatureCache
from krasnal_id.statistics import exact_mcnemar_p_value

# The two regimes, named as in the recall curve so the columns line up. `full` is
# deliberately absent: its extra queries are the single-photographer classes,
# which cannot be asked cross-photographer at all, so it would cost a third of
# the runtime for a column section 11 already reports appearance-only.
ANSWERABLE = "answerable"
DISJOINT = "disjoint"

APPEARANCE = "appearance"
GEOMETRY = "geometry"

# How often a long sweep says where it has got to. Two hours in silence is
# indistinguishable from two hours hung.
PROGRESS_EVERY = 50

logger = logging.getLogger(__name__)


class GeometryFirstError(ValueError):
    """Raised when the geometry-first inputs are missing or inconsistent."""


@dataclass(frozen=True, slots=True)
class QueryRanks:
    """One query's outcome under both signals, in both regimes.

    Both rankings come from one pass over one candidate set, so the paired
    comparison is paired by construction rather than by assumption.
    """

    answerable: dict[str, int]
    disjoint: dict[str, int]


def _inliers_against(
    cache: FeatureCache,
    query_image: str,
    query_path: Path,
    candidates: tuple[tuple[str, Path], ...],
) -> npt.NDArray[np.float32]:
    """Count RANSAC inliers between one query and every candidate photograph."""
    from krasnal_id.retrieval.rerank import count_inliers

    query = cache.get(query_image, query_path)
    return np.asarray(
        [count_inliers(query, cache.get(image_id, path)) for image_id, path in candidates],
        dtype=np.float32,
    )


def rank_one_query(
    inliers: npt.NDArray[np.float32],
    cosine: npt.NDArray[np.float32],
    dwarf_ids: npt.NDArray[np.str_],
    truth: str,
    disjoint_mask: npt.NDArray[np.bool_],
) -> QueryRanks:
    """Rank the correct statue under both signals, over both candidate sets.

    The disjoint regime is a mask over the same inlier vector rather than a second
    matching pass, which is what makes two arms cost the same as one.
    """
    ranks = {}
    for name, keep in ((ANSWERABLE, np.ones_like(disjoint_mask)), (DISJOINT, disjoint_mask)):
        indices = np.flatnonzero(keep)
        if indices.size == 0:
            ranks[name] = {APPEARANCE: ABSENT, GEOMETRY: ABSENT}
            continue
        candidates = dwarf_ids[indices]
        ranks[name] = {
            APPEARANCE: class_rank(cosine[indices], candidates, truth),
            GEOMETRY: class_rank(inliers[indices], candidates, truth),
        }
    return QueryRanks(answerable=ranks[ANSWERABLE], disjoint=ranks[DISJOINT])


def journal_identity(
    split: EvaluationSplit,
    backbone: BackboneIdentity,
    config: GeometryFirstConfig,
) -> str:
    """Digest everything that decides a query's row, so rows can never be mixed.

    The split's `manifest_sha256` covers the candidate set and the truth; the
    backbone covers the appearance column; `max_keypoints` covers the geometry
    one. Change any of them and the digest changes, which starts a fresh journal
    rather than resuming onto rows that mean something else.
    """
    payload = json.dumps(
        {
            "manifest_sha256": split.manifest_sha256,
            "model_id": backbone.model_id,
            "revision": backbone.revision,
            "preprocessing_id": backbone.preprocessing_id,
            "max_keypoints": config.max_keypoints,
        },
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def read_journal(path: Path) -> dict[str, QueryRanks]:
    """Load the rows a previous run finished, tolerating a truncated tail.

    A sweep killed mid-write leaves a partial final line. That line is a miss,
    not a corruption: the query is simply recomputed.
    """
    if not path.is_file():
        return {}
    done: dict[str, QueryRanks] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            row = json.loads(line)
            done[row["query_image_id"]] = QueryRanks(
                answerable={k: int(v) for k, v in row[ANSWERABLE].items()},
                disjoint={k: int(v) for k, v in row[DISJOINT].items()},
            )
        except (ValueError, KeyError, TypeError):
            continue
    return done


def append_journal(path: Path, query_image_id: str, ranks: QueryRanks) -> None:
    """Record one finished query, durably, before the next one starts."""
    path.parent.mkdir(parents=True, exist_ok=True)
    row = {
        "query_image_id": query_image_id,
        ANSWERABLE: ranks.answerable,
        DISJOINT: ranks.disjoint,
    }
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def measure(
    split: EvaluationSplit,
    manifest: DatasetManifest,
    matrix: EmbeddingMatrix,
    config: GeometryFirstConfig,
    journal: Path | None = None,
) -> tuple[QueryRanks, ...]:
    """Rank every answerable query by geometry alone, and by appearance.

    Resumable, because the full sweep is hours and anything can stop it. Each
    query's four ranks are appended to `journal` as soon as they exist, and a
    restart recomputes only what is not already there — the same contract
    `embeddings extract` offers, for the same reason.
    """
    authors = authors_by_image(manifest)
    by_class = photographers_by_class(manifest)
    paths = {image.image_id: Path(image.local_path) for image in manifest.images}
    rows = {image_id: index for index, image_id in enumerate(matrix.image_ids)}
    dwarf_ids = np.asarray(matrix.dwarf_ids)
    author_of = np.asarray([authors[image_id] for image_id in matrix.image_ids])
    cache = FeatureCache(config.max_keypoints)
    done = read_journal(journal) if journal is not None else {}
    if done:
        logger.info("geometry-first resuming: %d queries already scored", len(done))

    measured: list[QueryRanks] = []
    for fold in split.folds:
        truth = fold.query_dwarf_id
        if len(by_class.get(truth, set())) < 2:
            continue
        if config.max_queries and len(measured) >= config.max_queries:
            break

        finished = done.get(fold.query_image_id)
        if finished is not None:
            measured.append(finished)
            continue

        query = rows[fold.query_image_id]
        keep = np.ones(len(dwarf_ids), dtype=bool)
        keep[query] = False
        indices = np.flatnonzero(keep)
        if indices.size == 0:
            continue

        candidates = tuple(
            (matrix.image_ids[index], paths[matrix.image_ids[index]]) for index in indices
        )
        inliers = _inliers_against(
            cache, fold.query_image_id, paths[fold.query_image_id], candidates
        )
        cosine = np.asarray(matrix.vectors[indices] @ matrix.vectors[query], dtype=np.float32)
        ranks = rank_one_query(
            inliers,
            cosine,
            dwarf_ids[indices],
            truth,
            author_of[indices] != author_of[query],
        )
        measured.append(ranks)
        if journal is not None:
            append_journal(journal, fold.query_image_id, ranks)
        if len(measured) % PROGRESS_EVERY == 0:
            logger.info("geometry-first progress: %d queries scored", len(measured))

    if not measured:
        raise GeometryFirstError("no answerable query could be scored")
    return tuple(measured)


def summarize(
    measured: tuple[QueryRanks, ...],
    cut_offs: tuple[int, ...],
) -> tuple[MetricSummary, ...]:
    """Report each signal's recall, and what geometry rescues from appearance."""
    if any(k <= 0 for k in cut_offs):
        raise GeometryFirstError(f"cut-offs must be positive: {cut_offs}")

    metrics: list[MetricSummary] = []
    for arm in (ANSWERABLE, DISJOINT):
        ranks = [getattr(query, arm) for query in measured]
        total = len(ranks)
        for k in sorted(set(cut_offs)):
            hit = {
                signal: [r[signal] != ABSENT and r[signal] <= k for r in ranks]
                for signal in (APPEARANCE, GEOMETRY)
            }
            for signal, hits in hit.items():
                metrics.append(accuracy_metric(f"{arm}_{signal}_r_at_{k}", sum(hits), total))

            # The decisive number: a first stage does not have to beat appearance
            # everywhere, only to find what appearance loses.
            missed = [i for i, found in enumerate(hit[APPEARANCE]) if not found]
            rescued = sum(1 for i in missed if hit[GEOMETRY][i])
            metrics.append(
                MetricSummary(name=f"{arm}_appearance_misses_at_{k}", value=float(len(missed)))
            )
            metrics.append(MetricSummary(name=f"{arm}_rescued_at_{k}", value=float(rescued)))
            if missed:
                metrics.append(accuracy_metric(f"{arm}_rescue_rate_at_{k}", rescued, len(missed)))

            wins = sum(
                1 for g, a in zip(hit[GEOMETRY], hit[APPEARANCE], strict=True) if g and not a
            )
            losses = sum(
                1 for g, a in zip(hit[GEOMETRY], hit[APPEARANCE], strict=True) if a and not g
            )
            label = f"{arm}_geometry_vs_appearance_at_{k}"
            metrics.append(MetricSummary(name=f"{label}_wins", value=float(wins)))
            metrics.append(MetricSummary(name=f"{label}_losses", value=float(losses)))
            metrics.append(
                MetricSummary(name=f"{label}_p_value", value=exact_mcnemar_p_value(wins, losses))
            )
        metrics.append(MetricSummary(name=f"{arm}_queries", value=float(total)))
    return tuple(metrics)


def run_geometry_first(config: AppConfig) -> ExperimentResult:
    """Rank every reference by geometry alone, and see what it finds."""
    if not isinstance(config.experiment, GeometryFirstConfig):
        raise GeometryFirstError(
            f"geometry-first requires experiment=geometry_first, got {config.experiment.kind}"
        )

    try:
        manifest, split = load_evaluation_inputs(
            config.paths.manifest_path, config.paths.evaluation_split_path
        )
    except BaselineExperimentError as error:
        raise GeometryFirstError(str(error)) from error

    matrix = load_embedding_matrix(manifest, config.backbone, Path(config.paths.embeddings_dir))
    digest = journal_identity(split, config.backbone, config.experiment)
    journal = (
        Path(config.paths.results_dir)
        / "journals"
        / f"geometry_first-{config.backbone.name}-{digest[:16]}.jsonl"
    )
    measured = measure(split, manifest, matrix, config.experiment, journal)

    return ExperimentResult(
        experiment="geometry_first",
        backbone=config.backbone.name,
        created_at=datetime.now(UTC),
        seed=config.experiment.seed,
        configuration=config.experiment.model_dump(mode="json"),
        metrics=summarize(measured, config.experiment.top_k),
    )

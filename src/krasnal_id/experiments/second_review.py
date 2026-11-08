"""The second internal review's checks (AGENTS.md section 7.18), run inside workshop validation.

* **Multi-region baseline.** A compact multi-vector representation scored by
  late interaction, against the CLS cosine on the same queries. The cosine arm
  reproduces the released baseline and photographer-disjoint top-1 first.
* **Reference strata.** Unseen-photographer top-1 split by how many references of
  the correct statue remain once the query's photographer is withheld, which
  separates scarcity from contributor overlap. Its total reproduces the released
  disjoint arm.
* **Margin-gated verification.** Verify only the queries whose cosine top-two
  margin is smallest, using the held-out weights the re-ranking check chose, and
  read accuracy against the fraction verified. Verifying none is cosine and all is
  the published held-out result, which bound the curve.
"""

from collections.abc import Mapping, Sequence

import numpy as np
import numpy.typing as npt

from krasnal_id.config import WorkshopValidationConfig
from krasnal_id.embeddings.store import EmbeddingMatrix
from krasnal_id.experiments.contracts import MetricSummary
from krasnal_id.experiments.rerank_ablation import QueryEvidence
from krasnal_id.experiments.reviewer_checks import statue_ranking
from krasnal_id.experiments.workshop_validation import (
    COSINE,
    HeldOutQuery,
    WorkshopValidationError,
    check_reproduced,
    paired_gap,
    rate_interval,
)
from krasnal_id.models import DatasetManifest, EvaluationSplit
from krasnal_id.photographers import authors_by_image

# Region sets of the multi-vector baseline, as index ranges into the stored stack:
# the CLS token and the four 2x2 cells, then all fourteen with the 3x3 cells.
REGION_SETS: dict[str, slice] = {"grid5": slice(0, 5), "grid14": slice(0, 14)}
# Remaining references of the correct statue, as (label, lowest, highest).
STRATA: tuple[tuple[str, int, int], ...] = (
    ("1", 1, 1),
    ("2", 2, 2),
    ("3to5", 3, 5),
    ("6plus", 6, 10**9),
)


def image_scores(regions: npt.NDArray[np.float32]) -> npt.NDArray[np.float32]:
    """Late-interaction score between every pair of images: (n, n), query by reference.

    For query regions q_i and reference regions r_j the score is the mean over i of
    the largest q_i . r_j, as `regions.late_interaction` computes for one query.
    """
    n, count, dim = regions.shape
    flat = regions.reshape(n * count, dim)
    out = np.empty((n, n), dtype=np.float32)
    for query in range(n):
        similarity = (flat @ regions[query].T).reshape(n, count, count)
        out[query] = similarity.max(axis=1).mean(axis=1)
    return out


def _statue_rank(
    scores: npt.NDArray[np.float32], references: Sequence[int], dwarf_of: Sequence[str], truth: str
) -> int:
    """The correct statue's rank when each statue scores its best reference."""
    best: dict[str, float] = {}
    for index in references:
        statue = dwarf_of[index]
        value = float(scores[index])
        if value > best.get(statue, -np.inf):
            best[statue] = value
    if truth not in best:
        raise WorkshopValidationError(f"no reference of {truth} in the gallery")
    return 1 + sum(1 for value in best.values() if value > best[truth])


def multi_region_part(
    backbone: str,
    split: EvaluationSplit,
    manifest: DatasetManifest,
    matrix: EmbeddingMatrix,
    regions: npt.NDArray[np.float32],
    released_baseline: Mapping[str, float],
    released_photographer: Mapping[str, float],
    settings: WorkshopValidationConfig,
) -> list[MetricSummary]:
    """Top-1 and recall at ten for each region set, against cosine, in both arms."""
    if regions.shape[0] != len(matrix.image_ids):
        raise WorkshopValidationError("the region stack does not match the embedding matrix")
    position = {image_id: index for index, image_id in enumerate(matrix.image_ids)}
    dwarf_of = list(matrix.dwarf_ids)
    author = authors_by_image(manifest)
    statue_of = {fold.query_image_id: fold.query_dwarf_id for fold in split.folds}
    resamples, seed = settings.bootstrap_resamples, settings.seed

    arms: dict[str, list[tuple[str, str, tuple[str, ...]]]] = {"full": [], "disjoint": []}
    for fold in split.folds:
        query, truth = fold.query_image_id, fold.query_dwarf_id
        arms["full"].append((query, truth, fold.reference_image_ids))
        kept = tuple(r for r in fold.reference_image_ids if author[r] != author[query])
        if any(dwarf_of[position[r]] == truth for r in kept):
            arms["disjoint"].append((query, truth, kept))

    ranks: dict[str, dict[str, dict[str, int]]] = {arm: {COSINE: {}} for arm in arms}
    for arm, rows in arms.items():
        for query, truth, references in rows:
            ranking = statue_ranking(query, truth, references, matrix)
            ranks[arm][COSINE][query] = ranking.truth + 1
    check_reproduced(
        f"{backbone} multi-region cosine",
        sum(r == 1 for r in ranks["full"][COSINE].values()) / len(ranks["full"][COSINE]),
        released_baseline,
        "top_1",
    )
    check_reproduced(
        f"{backbone} multi-region disjoint cosine",
        sum(r == 1 for r in ranks["disjoint"][COSINE].values()) / len(ranks["disjoint"][COSINE]),
        released_photographer,
        "disjoint_top_1",
    )

    for method, cells in REGION_SETS.items():
        scores = image_scores(np.ascontiguousarray(regions[:, cells, :]))
        for arm, rows in arms.items():
            ranks[arm][method] = {
                query: _statue_rank(
                    scores[position[query]], [position[r] for r in references], dwarf_of, truth
                )
                for query, truth, references in rows
            }

    metrics: list[MetricSummary] = []
    for arm in arms:
        for method, by_query in ranks[arm].items():
            for k in (1, 10):
                hits = {q: float(rank <= k) for q, rank in by_query.items()}
                metrics.append(
                    rate_interval(
                        f"{backbone}_mr_{arm}_{method}_top_{k}", hits, statue_of, resamples, seed
                    )
                )
                if method == COSINE:
                    continue
                base = {q: float(rank <= k) for q, rank in ranks[arm][COSINE].items()}
                label = f"{backbone}_mr_{arm}_{method}_vs_cosine_top_{k}"
                metrics.append(paired_gap(f"{label}_gain", hits, base, statue_of, resamples, seed))
                wins = sum(1 for q in hits if hits[q] and not base[q])
                losses = sum(1 for q in hits if base[q] and not hits[q])
                metrics.append(MetricSummary(name=f"{label}_wins", value=float(wins)))
                metrics.append(MetricSummary(name=f"{label}_losses", value=float(losses)))
        metrics.append(
            MetricSummary(name=f"{backbone}_mr_{arm}_queries", value=float(len(arms[arm])))
        )
    return metrics


def strata_part(
    backbone: str,
    split: EvaluationSplit,
    manifest: DatasetManifest,
    matrix: EmbeddingMatrix,
    released_photographer: Mapping[str, float],
    settings: WorkshopValidationConfig,
) -> list[MetricSummary]:
    """Unseen-photographer top-1 by the number of correct references left."""
    author = authors_by_image(manifest)
    dwarf_of = dict(zip(matrix.image_ids, matrix.dwarf_ids, strict=True))
    statue_of = {fold.query_image_id: fold.query_dwarf_id for fold in split.folds}
    hits: dict[str, float] = {}
    remaining: dict[str, int] = {}
    for fold in split.folds:
        query, truth = fold.query_image_id, fold.query_dwarf_id
        kept = tuple(r for r in fold.reference_image_ids if author[r] != author[query])
        own = sum(1 for r in kept if dwarf_of[r] == truth)
        if own == 0:
            continue
        hits[query] = float(statue_ranking(query, truth, kept, matrix).truth == 0)
        remaining[query] = own
    check_reproduced(
        f"{backbone} strata",
        sum(hits.values()) / len(hits),
        released_photographer,
        "disjoint_top_1",
    )
    metrics: list[MetricSummary] = []
    for label, lowest, highest in STRATA:
        members = {q: h for q, h in hits.items() if lowest <= remaining[q] <= highest}
        if not members:
            continue
        name = f"{backbone}_strata_{label}"
        metrics.append(
            rate_interval(
                f"{name}_top_1", members, statue_of, settings.bootstrap_resamples, settings.seed
            )
        )
        metrics.append(MetricSummary(name=f"{name}_queries", value=float(len(members))))
        metrics.append(
            MetricSummary(name=f"{name}_statues", value=float(len({statue_of[q] for q in members})))
        )
    return metrics


def cosine_margin(item: QueryEvidence) -> float:
    """How far the first-stage winner leads the runner-up, by cosine."""
    if len(item.candidates) < 2:
        return float("inf")
    return item.candidates[0].cosine - item.candidates[1].cosine


def gated_part(
    held_out: Sequence[HeldOutQuery],
    evidence: Mapping[str, Mapping[str, QueryEvidence]],
    settings: WorkshopValidationConfig,
) -> list[MetricSummary]:
    """Top-1 when only the least certain fraction of queries is verified.

    Queries are ordered by cosine margin, smallest first, ties by image ID, and the
    first `fraction` of them take the matcher's held-out rank; the rest keep
    cosine's. The order uses cosine alone, so no matcher output selects the queries.
    """
    if not held_out:
        raise WorkshopValidationError("no held-out queries to gate")
    first = next(iter(evidence))
    margin = {
        item.query_image_id: cosine_margin(evidence[first][item.query_image_id])
        for item in held_out
    }
    order = sorted(margin, key=lambda q: (margin[q], q))
    statue_of = {item.query_image_id: item.dwarf_id for item in held_out}
    by_query = {item.query_image_id: item for item in held_out}
    metrics: list[MetricSummary] = []
    for matcher in evidence:
        for fraction in settings.gate_fractions:
            verified = set(order[: round(fraction * len(order))])
            hits = {
                q: float(by_query[q].ranks[matcher if q in verified else COSINE] == 1)
                for q in order
            }
            pct = round(100 * fraction)
            metrics.append(
                rate_interval(
                    f"cv_gate_{matcher}_{pct}_top_1",
                    hits,
                    statue_of,
                    settings.bootstrap_resamples,
                    settings.seed,
                )
            )
    return metrics

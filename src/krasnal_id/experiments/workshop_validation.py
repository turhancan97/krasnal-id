"""Re-checks of published results for the CVWW paper, from evidence already on disk.

The re-ranking numbers this project published -- 82.28% for SIFT and 84.62% for
DISK+LightGlue, photographer-disjoint -- are each the best point of a weight sweep
scored on the same queries that chose it. That is an in-sample selection, and a
reviewer is right to discount it. This module asks the out-of-sample version
without matching a single image: both matchers' per-query evidence is journalled,
so a weight can be chosen on some statues and scored on others for free.

* **Folds are statues, not photographs.** All of a statue's queries stay in one
  fold, so the weight that judges a statue was never chosen on any photograph of
  it. The reference gallery is untouched: holding a statue out of *selection* does
  not remove it from the set every query is matched against.
* **Every eligible query is scored exactly once**, by the weight chosen on the
  other folds, and the held-out predictions are pooled.
* **This is a validation check on a benchmark already explored, not an untouched
  confirmatory test.** The weight grid, the matchers and the protocol were all
  fixed while looking at these queries. What it removes is the selection of the
  weight on the queries it is scored on; it cannot remove everything else.

Nothing here loads a model or constructs a matcher. A journal that is missing,
partial, duplicated, or written for a different split is refused with the command
that rebuilds it, never recomputed.
"""

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

from krasnal_id.bootstrap import (
    generator,
    interval_metric,
    per_statue,
    resampled_rate,
    statue_draws,
)
from krasnal_id.config import (
    AppConfig,
    GeoAblationConfig,
    PhotographerGapConfig,
    PoolSizeAblationConfig,
    WorkshopValidationConfig,
    backbone_config,
    load_config,
)
from krasnal_id.embeddings.store import EmbeddingMatrix, load_embedding_matrix
from krasnal_id.experiments.baseline_accuracy import (
    BaselineExperimentError,
    accuracy_metric,
    evaluate_fold,
    load_evaluation_inputs,
)
from krasnal_id.experiments.contracts import FoldWeight, MetricSummary, WorkshopValidationResult
from krasnal_id.experiments.geo_ablation import located_inputs, measure_geographic_pool
from krasnal_id.experiments.matcher_rerank import best_weight, evidence_identity
from krasnal_id.experiments.photographer_gap import evaluate_conditions
from krasnal_id.experiments.pool_size_ablation import measure_pool_size, resolve_pool_sizes
from krasnal_id.experiments.rerank_ablation import QueryEvidence, evidence_from_row
from krasnal_id.models import DatasetManifest, EvaluationSplit
from krasnal_id.photographers import is_answerable, photographers_by_class
from krasnal_id.retrieval.rerank import INLIER_CAP
from krasnal_id.statistics import exact_mcnemar_p_value

# The method every matcher is compared against: the blend at weight zero, which is
# the cosine ranking by construction.
COSINE = "cosine"


class WorkshopValidationError(ValueError):
    """Raised when the evidence on disk cannot support the validation."""


@dataclass(frozen=True, slots=True)
class HeldOutQuery:
    """One query's rank under every method, each weight chosen without its statue."""

    query_image_id: str
    dwarf_id: str
    fold: int
    # Method name to the correct statue's rank; 0 means outside the top-k shortlist.
    ranks: Mapping[str, int]


def eligible_queries(split: EvaluationSplit, manifest: DatasetManifest) -> tuple[str, ...]:
    """The photographer-disjoint queries the matcher journals must cover, in split order."""
    by_class = photographers_by_class(manifest)
    return tuple(
        fold.query_image_id for fold in split.folds if is_answerable(fold.query_dwarf_id, by_class)
    )


def journal_path(
    journals_dir: Path,
    split: EvaluationSplit,
    backbone: str,
    matcher: str,
    settings: WorkshopValidationConfig,
    top_k: int | None = None,
) -> Path:
    """Where `experiment matcher-rerank` wrote this matcher's disjoint evidence.

    `top_k` defaults to the main shortlist; section 7.15's wider one passes its own.
    """
    keypoints = settings.max_keypoints if matcher == "sift" else settings.learned_keypoints
    shortlist = settings.rerank_top_k if top_k is None else top_k
    digest = evidence_identity(
        split, backbone, matcher, keypoints, shortlist, photographer_disjoint=True
    )
    return journals_dir / f"matcher_rerank-{backbone}-{matcher}-{digest[:16]}.jsonl"


def load_journal_strictly(path: Path, expected: Sequence[str]) -> dict[str, QueryEvidence]:
    """Read a complete journal, refusing anything that would change the population.

    `read_evidence_journal` tolerates a truncated tail because a resumed sweep
    recomputes what is missing. Here nothing is recomputed, so a missing, malformed
    or duplicated row would silently alter what is being validated; each is an error.
    The file name already encodes the split's manifest hash, so a journal written
    for a different dataset is not found rather than misread.
    """
    rebuild = "run krasnal-id experiment matcher-rerank to write it"
    if not path.is_file():
        raise WorkshopValidationError(f"matcher journal is missing: {path}; {rebuild}")
    evidence: dict[str, QueryEvidence] = {}
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        try:
            item = evidence_from_row(json.loads(line))
        except (ValueError, KeyError, TypeError) as error:
            raise WorkshopValidationError(
                f"{path}:{number} is not a valid evidence row ({error}); {rebuild}"
            ) from error
        if item.query_image_id in evidence:
            raise WorkshopValidationError(
                f"{path}:{number} repeats query {item.query_image_id}; {rebuild}"
            )
        evidence[item.query_image_id] = item
    wanted = set(expected)
    if set(evidence) != wanted:
        missing, extra = len(wanted - set(evidence)), len(set(evidence) - wanted)
        raise WorkshopValidationError(
            f"{path} covers {len(evidence)} queries, expected {len(wanted)} "
            f"({missing} missing, {extra} not eligible); {rebuild}"
        )
    return evidence


def statue_folds(statues: Sequence[str], folds: int, seed: int) -> dict[str, int]:
    """Assign statues to `folds` near-equal groups, deterministically.

    Sorted first, so the assignment depends on the set of statues and the seed and
    on nothing about the order they were discovered in.
    """
    ordered = sorted(set(statues))
    if len(ordered) < folds:
        raise WorkshopValidationError(f"{len(ordered)} statues cannot fill {folds} folds")
    permuted = np.random.default_rng(seed).permutation(len(ordered))
    groups = np.array_split(permuted, folds)
    return {ordered[index]: fold for fold, group in enumerate(groups) for index in group}


def cross_validate(
    evidence: Mapping[str, Mapping[str, QueryEvidence]],
    truths: Mapping[str, str],
    fold_of_statue: Mapping[str, int],
    weights: tuple[float, ...],
    cut_off: int = 1,
    cap: int = INLIER_CAP,
) -> tuple[tuple[HeldOutQuery, ...], tuple[FoldWeight, ...]]:
    """Choose each matcher's weight on four folds and score it on the fifth.

    The cosine ranking is read from every matcher's evidence at weight zero and
    must agree query by query: both journals were built over the same shortlist,
    and a disagreement means they are not paired.
    """
    matchers = tuple(evidence)
    queries = sorted(next(iter(evidence.values())))
    folds = sorted(set(fold_of_statue.values()))
    ranks: dict[str, dict[str, int]] = {query: {} for query in queries}
    for query in queries:
        cosine = {evidence[m][query].rank_of_truth(0.0, truths[query]) for m in matchers}
        if len(cosine) != 1:
            raise WorkshopValidationError(
                f"the matcher journals disagree on the cosine ranking of {query}; "
                "they were not built over the same shortlist"
            )
        ranks[query][COSINE] = cosine.pop()

    chosen: list[FoldWeight] = []
    for matcher in matchers:
        for fold in folds:
            held = [q for q in queries if fold_of_statue[truths[q]] == fold]
            train = tuple(
                evidence[matcher][q] for q in queries if fold_of_statue[truths[q]] != fold
            )
            weight, accuracy = best_weight(train, dict(truths), weights, cut_off, cap)
            for query in held:
                ranks[query][matcher] = evidence[matcher][query].rank_of_truth(
                    weight, truths[query], cap
                )
            chosen.append(
                FoldWeight(
                    matcher=matcher,
                    fold=fold,
                    weight=weight,
                    selection_top_1=accuracy,
                    held_out_queries=len(held),
                    held_out_statues=len({truths[q] for q in held}),
                )
            )

    held_out = tuple(
        HeldOutQuery(
            query_image_id=query,
            dwarf_id=truths[query],
            fold=fold_of_statue[truths[query]],
            ranks=ranks[query],
        )
        for query in queries
    )
    return held_out, tuple(chosen)


def _hit(rank: int, k: int) -> bool:
    return 0 < rank <= k


def summarize_held_out(
    held_out: Sequence[HeldOutQuery],
    methods: Sequence[str],
    cut_offs: Sequence[int],
    resamples: int,
    seed: int,
    prefix: str = "cv",
    extra_pairs: Sequence[tuple[str, str]] = (),
) -> tuple[MetricSummary, ...]:
    """Per-method accuracy, then every pairwise gain with McNemar and a statue interval.

    Pairs are each matcher against cosine, and each later matcher against the first,
    which is the comparison the published table makes; `extra_pairs` adds more
    without renaming any of those. `prefix` keeps a second shortlist's metrics apart.
    """
    if not held_out:
        raise WorkshopValidationError("no held-out queries to summarize")
    statue_of = {item.query_image_id: item.dwarf_id for item in held_out}
    statues = sorted(set(statue_of.values()))
    metrics: list[MetricSummary] = []
    for method in methods:
        for k in cut_offs:
            hits = sum(_hit(item.ranks[method], k) for item in held_out)
            metrics.append(accuracy_metric(f"{prefix}_{method}_top_{k}", hits, len(held_out)))
    metrics.append(MetricSummary(name=f"{prefix}_queries", value=float(len(held_out))))
    metrics.append(MetricSummary(name=f"{prefix}_statues", value=float(len(statues))))

    matchers = [method for method in methods if method != COSINE]
    pairs = [(matcher, COSINE) for matcher in matchers]
    pairs += [(later, matchers[0]) for later in matchers[1:]]
    pairs += [pair for pair in extra_pairs if pair not in pairs]
    for better, base in pairs:
        label = f"{prefix}_{better}_vs_{base}"
        draws = statue_draws(len(statues), resamples, generator(seed, label))
        for k in cut_offs:
            a = {item.query_image_id: float(_hit(item.ranks[better], k)) for item in held_out}
            b = {item.query_image_id: float(_hit(item.ranks[base], k)) for item in held_out}
            wins = sum(1 for q in a if a[q] and not b[q])
            losses = sum(1 for q in a if b[q] and not a[q])
            a_sum, counts = per_statue(a, statue_of, statues)
            b_sum, _ = per_statue(b, statue_of, statues)
            samples = resampled_rate(a_sum, counts, draws) - resampled_rate(b_sum, counts, draws)
            gain = (sum(a.values()) - sum(b.values())) / len(held_out)
            metrics.append(interval_metric(f"{label}_top_{k}_gain", gain, samples))
            metrics.append(MetricSummary(name=f"{label}_top_{k}_wins", value=float(wins)))
            metrics.append(MetricSummary(name=f"{label}_top_{k}_losses", value=float(losses)))
            metrics.append(
                MetricSummary(
                    name=f"{label}_top_{k}_p_value", value=exact_mcnemar_p_value(wins, losses)
                )
            )
    return tuple(metrics)


# ---------------------------------------------------------------------------
# Recomputed parts. Each recomputes per-query cosine outcomes from cached vectors,
# proves it reproduces the released artifact, then resamples statues.
# ---------------------------------------------------------------------------

# A recomputation must match the released number to this tolerance or nothing is
# reported: an interval around a number that is not the published one would be an
# interval around nothing anybody has read.
REPRODUCTION_TOLERANCE = 1e-9


def released_metrics(results_dir: Path, stem: str) -> dict[str, float]:
    """Read a released artifact's metrics by name."""
    path = results_dir / f"{stem}.json"
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        return {str(m["name"]): float(m["value"]) for m in payload["metrics"]}
    except (OSError, ValueError, KeyError, TypeError) as error:
        raise WorkshopValidationError(
            f"released artifact {path} is unreadable ({error}); the recomputation has "
            "nothing to be checked against"
        ) from error


def check_reproduced(
    label: str, recomputed: float, released: Mapping[str, float], key: str
) -> None:
    """Refuse to go on if a recomputed number is not the released one."""
    if key not in released:
        raise WorkshopValidationError(f"{label}: released artifact has no metric {key}")
    if abs(recomputed - released[key]) > REPRODUCTION_TOLERANCE:
        raise WorkshopValidationError(
            f"{label}: recomputed {key} = {recomputed!r} but the released artifact holds "
            f"{released[key]!r}; the inputs have changed since it was written"
        )


def mean_by_query(query_ids: Sequence[str], values: Sequence[float]) -> dict[str, float]:
    """Average repeated outcomes per query, e.g. a control arm stacked seed after seed."""
    sums: dict[str, float] = {}
    counts: dict[str, int] = {}
    for query, value in zip(query_ids, values, strict=True):
        sums[query] = sums.get(query, 0.0) + float(value)
        counts[query] = counts.get(query, 0) + 1
    return {query: sums[query] / counts[query] for query in sums}


def paired_gap(
    name: str,
    a: Mapping[str, float],
    b: Mapping[str, float],
    statue_of: Mapping[str, str],
    resamples: int,
    seed: int,
) -> MetricSummary:
    """Mean of `a` minus mean of `b` over the same queries, with a statue interval."""
    if set(a) != set(b) or not a:
        raise WorkshopValidationError(f"{name}: the two arms do not cover the same queries")
    statues = sorted({statue_of[q] for q in a})
    a_sum, counts = per_statue(a, statue_of, statues)
    b_sum, _ = per_statue(b, statue_of, statues)
    draws = statue_draws(len(statues), resamples, generator(seed, name))
    samples = resampled_rate(a_sum, counts, draws) - resampled_rate(b_sum, counts, draws)
    return interval_metric(name, (sum(a.values()) - sum(b.values())) / len(a), samples)


def rate_interval(
    name: str,
    outcomes: Mapping[str, float],
    statue_of: Mapping[str, str],
    resamples: int,
    seed: int,
) -> MetricSummary:
    """A per-query rate with a statue interval in place of the Wilson one."""
    statues = sorted({statue_of[q] for q in outcomes})
    sums, counts = per_statue(outcomes, statue_of, statues)
    draws = statue_draws(len(statues), resamples, generator(seed, name))
    return interval_metric(
        name, sum(outcomes.values()) / len(outcomes), resampled_rate(sums, counts, draws)
    )


def slope_interval(
    name: str,
    pool_sizes: Sequence[int],
    curves: Sequence[Mapping[str, float]],
    statue_of: Mapping[str, str],
    resamples: int,
    seed: int,
) -> MetricSummary:
    """Points per doubling, refitted on every statue resample of the whole curve.

    One draw is shared by every pool size, so a resample moves the whole curve at
    once, which is what makes the slope's interval mean something.
    """
    statues = sorted({statue_of[q] for q in curves[0]})
    draws = statue_draws(len(statues), resamples, generator(seed, name))
    x = np.log2(np.asarray(pool_sizes, dtype=np.float64))
    point = np.asarray([sum(c.values()) / len(c) for c in curves])
    rates = np.stack(
        [resampled_rate(*per_statue(c, statue_of, statues), draws) for c in curves], axis=1
    )
    centred = x - x.mean()
    slopes = (rates - rates.mean(axis=1, keepdims=True)) @ centred / (centred @ centred)
    value = float(np.polyfit(x, point, 1)[0])
    return interval_metric(name, value * 100.0, np.asarray(slopes * 100.0, dtype=np.float64))


def photographer_part(
    backbone: str,
    split: EvaluationSplit,
    manifest: DatasetManifest,
    matrix: EmbeddingMatrix,
    released: Mapping[str, float],
    resamples: int,
    seed: int,
) -> list[MetricSummary]:
    """The three photographer arms, the two effects between them, with statue intervals."""
    seeds = packaged(PhotographerGapConfig, "photographer_gap").seeds
    outcomes, _ = evaluate_conditions(split, manifest, matrix, seeds)
    statue_of = {fold.query_image_id: fold.query_dwarf_id for fold in split.folds}
    arms = {
        name: mean_by_query(outcome.query_image_ids, [float(rank == 1) for rank in outcome.ranks])
        for name, outcome in outcomes.items()
    }
    for name, arm in arms.items():
        check_reproduced(
            f"{backbone} photographer", sum(arm.values()) / len(arm), released, f"{name}_top_1"
        )
    label = f"{backbone}_photographer"
    return [
        paired_gap(
            f"{label}_raw_gap", arms["standard"], arms["disjoint"], statue_of, resamples, seed
        ),
        paired_gap(
            f"{label}_reference_cost", arms["standard"], arms["control"], statue_of, resamples, seed
        ),
        paired_gap(
            f"{label}_attributable", arms["control"], arms["disjoint"], statue_of, resamples, seed
        ),
    ]


def baseline_part(
    split: EvaluationSplit,
    matrices: Mapping[str, EmbeddingMatrix],
    released: Mapping[str, Mapping[str, float]],
    resamples: int,
    seed: int,
) -> list[MetricSummary]:
    """Each backbone's top-1 and the paired differences between backbones."""
    statue_of = {fold.query_image_id: fold.query_dwarf_id for fold in split.folds}
    top_1: dict[str, dict[str, float]] = {}
    for name, matrix in matrices.items():
        top_1[name] = {
            fold.query_image_id: float(
                evaluate_fold(
                    fold.query_image_id, fold.query_dwarf_id, fold.reference_image_ids, matrix
                ).dwarf_rank
                == 1
            )
            for fold in split.folds
        }
        check_reproduced(
            f"{name} baseline",
            sum(top_1[name].values()) / len(top_1[name]),
            released[name],
            "top_1",
        )
    metrics = [
        rate_interval(f"{name}_top_1", outcomes, statue_of, resamples, seed)
        for name, outcomes in top_1.items()
    ]
    names = list(matrices)
    for position, first in enumerate(names):
        for second in names[position + 1 :]:
            metrics.append(
                paired_gap(
                    f"{first}_minus_{second}_top_1",
                    top_1[first],
                    top_1[second],
                    statue_of,
                    resamples,
                    seed,
                )
            )
    return metrics


def geography_part(
    backbone: str,
    split: EvaluationSplit,
    manifest: DatasetManifest,
    matrix: EmbeddingMatrix,
    released: Mapping[str, float],
    resamples: int,
    seed: int,
) -> list[MetricSummary]:
    """Geographic minus equal-sized random pools, per pool size, with statue intervals.

    The geographic arm is exact; the random arm is the mean over its seeds, as
    published. The interval therefore covers which statues were asked about, not
    which random pools were drawn -- the seed spread stays a separate quantity.
    """
    settings = packaged(GeoAblationConfig, "geo_ablation")
    located, dwarf_ids, _, locations = located_inputs(manifest, split, matrix)
    statue_of = {fold.query_image_id: fold.query_dwarf_id for fold in located.folds}
    metrics: list[MetricSummary] = []
    for pool_size in resolve_pool_sizes(settings.pool_sizes, len(dwarf_ids)):
        geo = measure_geographic_pool(located, matrix, pool_size, locations)
        rand = measure_pool_size(located, matrix, pool_size, settings.seeds, dwarf_ids)
        check_reproduced(
            f"{backbone} geography", geo.top_1, released, f"geo_top_1_pool_{pool_size}"
        )
        check_reproduced(
            f"{backbone} geography", rand.top_1, released, f"random_top_1_pool_{pool_size}"
        )
        metrics.append(
            paired_gap(
                f"{backbone}_geo_minus_random_pool_{pool_size}",
                dict(zip(geo.query_image_ids, geo.query_top_1, strict=True)),
                dict(zip(rand.query_image_ids, rand.query_top_1, strict=True)),
                statue_of,
                resamples,
                seed,
            )
        )
    return metrics


def pool_part(
    backbone: str,
    split: EvaluationSplit,
    matrix: EmbeddingMatrix,
    released: Mapping[str, float],
    resamples: int,
    seed: int,
) -> list[MetricSummary]:
    """The decay rate per doubling of the candidate pool, with a statue interval."""
    settings = packaged(PoolSizeAblationConfig, "pool_size_ablation")
    dwarf_ids = tuple(sorted(set(matrix.dwarf_ids)))
    sizes = resolve_pool_sizes(settings.pool_sizes, len(dwarf_ids))
    measurements = [
        measure_pool_size(split, matrix, size, settings.seeds, dwarf_ids) for size in sizes
    ]
    for measurement in measurements:
        check_reproduced(
            f"{backbone} pool", measurement.top_1, released, f"top_1_pool_{measurement.pool_size}"
        )
    statue_of = {fold.query_image_id: fold.query_dwarf_id for fold in split.folds}
    curves = [dict(zip(m.query_image_ids, m.query_top_1, strict=True)) for m in measurements]
    slope = slope_interval(
        f"{backbone}_points_per_doubling", sizes, curves, statue_of, resamples, seed
    )
    check_reproduced(f"{backbone} pool", slope.value, released, "top_1_points_per_doubling")
    return [slope]


def packaged[ConfigT](kind: type[ConfigT], name: str) -> ConfigT:
    """The packaged settings another experiment was published with."""
    experiment = load_config([f"experiment={name}"]).experiment
    if not isinstance(experiment, kind):
        raise WorkshopValidationError(f"packaged experiment={name} is not a {kind.__name__}")
    return experiment


def run_workshop_validation(config: AppConfig, journals_dir: Path) -> WorkshopValidationResult:
    """Run every configured re-check, from evidence and cached vectors alone.

    `journals_dir` sits in the released results directory, whose artifacts the
    recomputed parts are checked against.
    """
    if not isinstance(config.experiment, WorkshopValidationConfig):
        raise WorkshopValidationError(
            f"this requires experiment=workshop_validation, got {config.experiment.kind}"
        )
    settings = config.experiment
    try:
        manifest, split = load_evaluation_inputs(
            config.paths.manifest_path, config.paths.evaluation_split_path
        )
    except BaselineExperimentError as error:
        raise WorkshopValidationError(str(error)) from error

    released_root = journals_dir.parent
    resamples, seed = settings.bootstrap_resamples, settings.seed
    metrics: list[MetricSummary] = []
    fold_weights: tuple[FoldWeight, ...] = ()

    if "rerank" in settings.parts or "gated" in settings.parts:
        expected = eligible_queries(split, manifest)
        truths = {fold.query_image_id: fold.query_dwarf_id for fold in split.folds}
        evidence: dict[str, dict[str, QueryEvidence]] = {
            matcher: load_journal_strictly(
                journal_path(journals_dir, split, config.backbone.name, matcher, settings),
                expected,
            )
            for matcher in settings.matchers
        }
        fold_of_statue = statue_folds([truths[q] for q in expected], settings.folds, seed)
        held_out, fold_weights = cross_validate(
            evidence, truths, fold_of_statue, settings.weights, cap=settings.rerank_inlier_cap
        )
        if "gated" in settings.parts:
            from krasnal_id.experiments.second_review import gated_part

            metrics += gated_part(held_out, evidence, settings)
    if "rerank" in settings.parts:
        metrics += summarize_held_out(
            held_out,
            (COSINE, *settings.matchers),
            settings.top_k_metrics,
            resamples,
            seed,
            extra_pairs=settings.extra_pairs,
        )
        if settings.wide_top_k is not None and settings.wide_matchers:
            # Section 7.15: the same held-out selection over a wider shortlist.
            wide = settings.wide_top_k
            wide_evidence: dict[str, dict[str, QueryEvidence]] = {
                matcher: load_journal_strictly(
                    journal_path(
                        journals_dir, split, config.backbone.name, matcher, settings, wide
                    ),
                    expected,
                )
                for matcher in settings.wide_matchers
            }
            wide_held, wide_weights = cross_validate(
                wide_evidence,
                truths,
                fold_of_statue,
                settings.weights,
                cap=settings.rerank_inlier_cap,
            )
            metrics += summarize_held_out(
                wide_held,
                (COSINE, *settings.wide_matchers),
                settings.top_k_metrics,
                resamples,
                seed,
                prefix=f"cv_k{wide}",
            )
            metrics += [
                MetricSummary(
                    name=f"cv_k{wide}_{row.matcher}_fold_{row.fold}_weight", value=row.weight
                )
                for row in wide_weights
            ]

    recomputed = [part for part in settings.parts if part not in ("rerank", "gated")]
    if recomputed:
        # Imported here: the reviewer checks build on this module's helpers.
        from krasnal_id.experiments.reviewer_checks import (
            families_part,
            geo_noise_part,
            geo_permutation_part,
            groups_part,
            null_bounds_part,
            rejection_signals_part,
            soft_prior_part,
        )

        def released_dir(backbone: str) -> Path:
            return (
                released_root / backbone if backbone in settings.separate_results else released_root
            )

        matrices: dict[str, EmbeddingMatrix] = {
            name: load_embedding_matrix(
                manifest, backbone_config(name), Path(config.paths.embeddings_dir)
            )
            for name in settings.backbones
        }
        if "baseline" in recomputed:
            released: dict[str, Mapping[str, float]] = {
                name: released_metrics(released_dir(name), f"baseline-{name}")
                for name in settings.backbones
            }
            metrics += baseline_part(split, matrices, released, resamples, seed)
        for name, matrix in matrices.items():
            if "photographer" in recomputed:
                metrics += photographer_part(
                    name,
                    split,
                    manifest,
                    matrix,
                    released_metrics(released_dir(name), f"photographer_gap-{name}"),
                    resamples,
                    seed,
                )
            if "geography" in recomputed:
                metrics += geography_part(
                    name,
                    split,
                    manifest,
                    matrix,
                    released_metrics(released_dir(name), f"geo_ablation-{name}"),
                    resamples,
                    seed,
                )
            if "geo_permutation" in recomputed:
                metrics += geo_permutation_part(
                    name,
                    split,
                    manifest,
                    matrix,
                    released_metrics(released_dir(name), f"geo_ablation-{name}"),
                    settings,
                )
            if "soft_prior" in recomputed:
                metrics += soft_prior_part(
                    name,
                    split,
                    manifest,
                    matrix,
                    released_metrics(released_dir(name), f"geo_ablation-{name}"),
                    settings,
                )
            if "geo_noise" in recomputed:
                metrics += geo_noise_part(
                    name,
                    split,
                    manifest,
                    matrix,
                    released_metrics(released_dir(name), f"geo_ablation-{name}"),
                    settings,
                )
            if "rejection_signals" in recomputed:
                metrics += rejection_signals_part(
                    name,
                    split,
                    matrix,
                    released_metrics(released_dir(name), f"open_set-{name}"),
                    settings,
                )
            if "groups" in recomputed:
                metrics += groups_part(
                    name,
                    split,
                    matrix,
                    released_metrics(released_dir(name), f"baseline-{name}"),
                    settings,
                )
            if "strata" in recomputed or "multi_region" in recomputed:
                from krasnal_id.embeddings.regions import load_regions
                from krasnal_id.experiments.second_review import multi_region_part, strata_part

                photographer = released_metrics(released_dir(name), f"photographer_gap-{name}")
                if "strata" in recomputed:
                    metrics += strata_part(name, split, manifest, matrix, photographer, settings)
                if "multi_region" in recomputed:
                    regions = load_regions(
                        Path(config.paths.embeddings_dir) / f"regions-{name}.npz",
                        matrix.image_ids,
                        backbone_config(name),
                    )
                    metrics += multi_region_part(
                        name,
                        split,
                        manifest,
                        matrix,
                        regions,
                        released_metrics(released_dir(name), f"baseline-{name}"),
                        photographer,
                        settings,
                    )
            if "pool" in recomputed:
                metrics += pool_part(
                    name,
                    split,
                    matrix,
                    released_metrics(released_dir(name), f"pool_size_ablation-{name}"),
                    resamples,
                    seed,
                )

        if "families" in recomputed:
            metrics += families_part(split, matrices, settings)
        if "null_bounds" in recomputed:
            capacity: dict[str, EmbeddingMatrix] = {
                name: load_embedding_matrix(
                    manifest, backbone_config(name), Path(config.paths.embeddings_dir)
                )
                for name in ("dinov2", *settings.null_bound_backbones)
            }
            metrics += null_bounds_part(
                split,
                manifest,
                capacity,
                released_metrics(released_root, "recall_curve-dinov2"),
                settings,
            )

    return WorkshopValidationResult(
        experiment="workshop_validation",
        backbone=config.backbone.name,
        created_at=datetime.now(UTC),
        seed=seed,
        configuration=settings.model_dump(mode="json"),
        metrics=tuple(metrics),
        fold_weights=fold_weights,
    )

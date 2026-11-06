"""The four vector-only reviewer checks of section 7.15, run inside workshop validation.

Each check recomputes a released number before it measures anything new, and
refuses to report if the recomputation differs by more than
`REPRODUCTION_TOLERANCE`. The control is what licenses reading the rest:

* **Soft geographic prior.** Weight zero is the released full-pool geographic
  top-1. The prior's weight is chosen on held-out statue folds, exactly as the
  re-ranking weights are, because a weight picked on the queries it is scored on
  is the defect the first revision corrected.
* **Coordinate noise.** The released geographic arm is the zero-metre control.
* **Rejection signals.** The cosine signal reproduces the released open-set AUROC
  and false-acceptance rate through the very calibration code that wrote them.
* **Group and member classes.** Unmerged top-1 is the released baseline.

Nothing here loads a model; everything is cosine over cached vectors.
"""

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from krasnal_id.bootstrap import generator, interval_metric, per_statue, statue_draws
from krasnal_id.config import GeoAblationConfig, OpenSetExperimentConfig, WorkshopValidationConfig
from krasnal_id.embeddings.store import EmbeddingMatrix
from krasnal_id.experiments.contracts import MetricSummary
from krasnal_id.experiments.geo_ablation import located_inputs, measure_geographic_pool
from krasnal_id.experiments.open_set import (
    ScoredQuery,
    _operating_point_metrics,
    calibrate_leave_one_class_out,
    summarize_open_set,
)
from krasnal_id.experiments.pool_size_ablation import measure_pool_size, resolve_pool_sizes
from krasnal_id.experiments.workshop_validation import (
    WorkshopValidationError,
    check_reproduced,
    packaged,
    paired_gap,
    rate_interval,
    statue_folds,
)
from krasnal_id.geometry import offset_metres
from krasnal_id.models import DatasetManifest, EvaluationSplit
from krasnal_id.retrieval.knn import cosine_knn

_EARTH_RADIUS_METRES = 6371008.8


def _haversine_many(
    origin: tuple[float, float],
    latitudes: npt.NDArray[np.float64],
    longitudes: npt.NDArray[np.float64],
) -> npt.NDArray[np.float64]:
    """Great-circle distance from one point to many, in metres."""
    lat1, lon1 = math.radians(origin[0]), math.radians(origin[1])
    lat2, lon2 = np.radians(latitudes), np.radians(longitudes)
    inner = (
        np.sin((lat2 - lat1) / 2) ** 2
        + math.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2
    )
    return np.asarray(
        2 * _EARTH_RADIUS_METRES * np.arcsin(np.sqrt(np.minimum(1.0, inner))), dtype=np.float64
    )


@dataclass(frozen=True, slots=True)
class StatueRanking:
    """One query's statues in cosine order, each scored by its best-matching image."""

    query_image_id: str
    dwarf_id: str
    statues: tuple[str, ...]
    best: npt.NDArray[np.float64]

    @property
    def truth(self) -> int:
        """The correct statue's position in cosine order."""
        return self.statues.index(self.dwarf_id)


def statue_ranking(
    query_image_id: str,
    dwarf_id: str,
    references: tuple[str, ...],
    matrix: EmbeddingMatrix,
) -> StatueRanking:
    """Collapse a full cosine ranking to statues, as `evaluate_fold` does.

    The same `cosine_knn` call and the same first-appearance rule, so position
    order here is the baseline's dwarf order to the tie.
    """
    vectors, dwarf_ids = matrix.rows_for(references)
    ranked = cosine_knn(
        query_image_id,
        matrix.vector_for(query_image_id),
        vectors,
        references,
        dwarf_ids,
        top_k=len(references),
    )
    statues: list[str] = []
    best: list[float] = []
    seen: set[str] = set()
    for match in ranked.matches:
        if match.dwarf_id not in seen:
            seen.add(match.dwarf_id)
            statues.append(match.dwarf_id)
            best.append(match.cosine_similarity)
    return StatueRanking(query_image_id, dwarf_id, tuple(statues), np.asarray(best))


def rank_with_prior(ranking: StatueRanking, bonus: npt.NDArray[np.float64]) -> int:
    """The correct statue's rank once each statue's score gains its prior bonus.

    Ties fall back to cosine order, so a zero bonus returns the cosine rank exactly.
    """
    scores = ranking.best + bonus
    truth = ranking.truth
    positions = np.arange(scores.shape[0])
    ahead = (scores > scores[truth]) | ((scores == scores[truth]) & (positions < truth))
    return int(ahead.sum()) + 1


def soft_prior_part(
    backbone: str,
    split: EvaluationSplit,
    manifest: DatasetManifest,
    matrix: EmbeddingMatrix,
    released: Mapping[str, float],
    settings: WorkshopValidationConfig,
) -> list[MetricSummary]:
    """Location as a weight rather than a filter, with the weight chosen held-out.

    The simulated user stands `gps_error_m` from the statue in a seeded direction:
    at zero error a strong prior simply names the statue under the user, which
    measures perfect localisation rather than the prior. Each candidate statue's
    best cosine gains `w * exp(-distance / scale)`.
    """
    located, dwarf_ids, _, locations = located_inputs(manifest, split, matrix)
    geo_sizes = resolve_pool_sizes(
        packaged(GeoAblationConfig, "geo_ablation").pool_sizes, len(dwarf_ids)
    )
    if geo_sizes[-1] != len(dwarf_ids):
        raise WorkshopValidationError(
            f"{backbone} soft prior: the released full pool is {geo_sizes[-1]}, "
            f"not the {len(dwarf_ids)} located statues"
        )
    located_set = set(dwarf_ids)
    dwarf_of = dict(zip(matrix.image_ids, matrix.dwarf_ids, strict=True))
    rankings = [
        statue_ranking(
            fold.query_image_id,
            fold.query_dwarf_id,
            tuple(r for r in fold.reference_image_ids if dwarf_of[r] in located_set),
            matrix,
        )
        for fold in located.folds
    ]
    statue_of = {r.query_image_id: r.dwarf_id for r in rankings}
    queries = [r.query_image_id for r in rankings]
    control = {r.query_image_id: float(r.truth == 0) for r in rankings}
    check_reproduced(
        f"{backbone} soft prior",
        sum(control.values()) / len(control),
        released,
        f"geo_top_1_pool_{len(dwarf_ids)}",
    )

    weights = tuple(sorted(settings.soft_prior_weights))
    fold_of_statue = statue_folds(list(statue_of.values()), settings.folds, settings.seed)
    resamples, seed = settings.bootstrap_resamples, settings.seed
    metrics = [
        rate_interval(f"{backbone}_soft_prior_none_top_1", control, statue_of, resamples, seed)
    ]

    for error in settings.gps_error_m:
        label = f"{backbone}_soft_prior_gps{error:g}"
        hits = {w: dict.fromkeys(queries, 0.0) for w in weights}
        location_only = dict.fromkeys(queries, 0.0)
        for draw in settings.noise_seeds:
            rng = generator(draw, f"gps-{error:g}")
            for ranking in rankings:
                user = offset_metres(
                    locations[ranking.dwarf_id], error, float(rng.uniform(0.0, 2.0 * math.pi))
                )
                distance = _haversine_many(
                    user,
                    np.asarray([locations[s][0] for s in ranking.statues]),
                    np.asarray([locations[s][1] for s in ranking.statues]),
                )
                closeness = np.exp(-distance / settings.soft_prior_scale_m)
                for w in weights:
                    if rank_with_prior(ranking, w * closeness) == 1:
                        hits[w][ranking.query_image_id] += 1.0 / len(settings.noise_seeds)
                # Location alone: the statue nearest the user, ties to cosine order.
                nearest = int(np.flatnonzero(distance == distance.min())[0])
                if nearest == ranking.truth:
                    location_only[ranking.query_image_id] += 1.0 / len(settings.noise_seeds)

        held_out: dict[str, float] = {}
        for fold in sorted(set(fold_of_statue.values())):
            train = [q for q in queries if fold_of_statue[statue_of[q]] != fold]
            # Strictly better only, iterating upward: ties go to the smaller weight.
            chosen = weights[0]
            for w in weights[1:]:
                if sum(hits[w][q] for q in train) > sum(hits[chosen][q] for q in train) + 1e-12:
                    chosen = w
            for q in queries:
                if fold_of_statue[statue_of[q]] == fold:
                    held_out[q] = hits[chosen][q]
            metrics.append(MetricSummary(name=f"{label}_fold_{fold}_weight", value=chosen))
        metrics += [
            rate_interval(f"{label}_top_1", held_out, statue_of, resamples, seed),
            paired_gap(f"{label}_gain", held_out, control, statue_of, resamples, seed),
            rate_interval(
                f"{label}_location_only_top_1", location_only, statue_of, resamples, seed
            ),
        ]
        # The in-sample sweep, descriptive only: the held-out number is the result.
        metrics += [
            MetricSummary(name=f"{label}_w{w:g}_top_1", value=sum(hits[w].values()) / len(queries))
            for w in weights
        ]
    return metrics


def geo_noise_part(
    backbone: str,
    split: EvaluationSplit,
    manifest: DatasetManifest,
    matrix: EmbeddingMatrix,
    released: Mapping[str, float],
    settings: WorkshopValidationConfig,
) -> list[MetricSummary]:
    """Re-run geographic minus random with every coordinate moved a fixed distance.

    Every statue moves, the query's own included, in a direction drawn per seed.
    The random arm ignores coordinates, so it is the released one throughout.
    """
    geo_settings = packaged(GeoAblationConfig, "geo_ablation")
    located, dwarf_ids, _, locations = located_inputs(manifest, split, matrix)
    statue_of = {fold.query_image_id: fold.query_dwarf_id for fold in located.folds}
    sizes = resolve_pool_sizes(geo_settings.pool_sizes, len(dwarf_ids))
    resamples, seed = settings.bootstrap_resamples, settings.seed
    random_arm = {}
    for size in sizes:
        rand = measure_pool_size(located, matrix, size, geo_settings.seeds, dwarf_ids)
        check_reproduced(f"{backbone} noise", rand.top_1, released, f"random_top_1_pool_{size}")
        random_arm[size] = dict(zip(rand.query_image_ids, rand.query_top_1, strict=True))

    metrics: list[MetricSummary] = []
    for metres in settings.noise_metres:
        moved_sets = []
        for draw in settings.noise_seeds:
            rng = generator(draw, f"coordinates-{metres:g}")
            moved_sets.append(
                {
                    dwarf: offset_metres(
                        locations[dwarf], metres, float(rng.uniform(0.0, 2.0 * math.pi))
                    )
                    for dwarf in sorted(locations)
                }
            )
        for size in sizes:
            totals: dict[str, float] = {}
            for moved in moved_sets:
                geo = measure_geographic_pool(located, matrix, size, moved)
                for query, hit in zip(geo.query_image_ids, geo.query_top_1, strict=True):
                    totals[query] = totals.get(query, 0.0) + float(hit) / len(moved_sets)
            metrics.append(
                paired_gap(
                    f"{backbone}_geo_minus_random_pool_{size}_noise_{metres:g}",
                    totals,
                    random_arm[size],
                    statue_of,
                    resamples,
                    seed,
                )
            )
    return metrics


@dataclass(frozen=True, slots=True)
class SignalScores:
    """One query's rejection signals and the statue it would be answered with."""

    query_image_id: str
    dwarf_id: str
    top_dwarf_id: str
    present: bool
    cosine: float
    margin: float
    knn: float


SIGNALS = ("cosine", "margin", "knn")


def signal_scores(
    query_image_id: str,
    dwarf_id: str,
    references: tuple[str, ...],
    matrix: EmbeddingMatrix,
    present: bool,
    knn_k: int,
) -> SignalScores:
    """Score one query: top-1 cosine, its margin over the next statue, and k-NN.

    `margin` is the top-1 similarity minus the best similarity of any *other*
    statue; `knn` is the similarity of the k-th nearest reference image.
    """
    ranking = statue_ranking(query_image_id, dwarf_id, references, matrix)
    vectors, _ = matrix.rows_for(references)
    similarities = np.sort(vectors @ matrix.vector_for(query_image_id))[::-1]
    k = min(knn_k, similarities.shape[0])
    margin = float(ranking.best[0] - ranking.best[1]) if ranking.best.shape[0] > 1 else 1.0
    return SignalScores(
        query_image_id=query_image_id,
        dwarf_id=dwarf_id,
        top_dwarf_id=ranking.statues[0],
        present=present,
        cosine=float(ranking.best[0]),
        margin=margin,
        knn=float(similarities[k - 1]),
    )


def weighted_auroc(
    known: npt.NDArray[np.float64],
    unknown: npt.NDArray[np.float64],
    known_weights: npt.NDArray[np.float64],
    unknown_weights: npt.NDArray[np.float64],
) -> float:
    """Probability a known query outscores an unknown one, ties counted as half.

    With unit weights this is `separability_auroc`; with a statue resample's
    multiplicities it is that resample's AUROC, computed without copying scores.
    """
    values, groups = np.unique(np.concatenate([known, unknown]), return_inverse=True)
    known_group, unknown_group = groups[: known.shape[0]], groups[known.shape[0] :]
    unknown_mass = np.bincount(unknown_group, weights=unknown_weights, minlength=values.shape[0])
    below = np.cumsum(unknown_mass) - unknown_mass
    favourable = below[known_group] + 0.5 * unknown_mass[known_group]
    total = known_weights.sum() * unknown_weights.sum()
    return float((known_weights * favourable).sum() / total)


def rejection_signals_part(
    backbone: str,
    split: EvaluationSplit,
    matrix: EmbeddingMatrix,
    released: Mapping[str, float],
    settings: WorkshopValidationConfig,
) -> list[MetricSummary]:
    """AUROC and calibrated false acceptance for margin and k-NN beside cosine."""
    open_set = packaged(OpenSetExperimentConfig, "open_set")
    known = [
        signal_scores(
            fold.query_image_id,
            fold.query_dwarf_id,
            fold.reference_image_ids,
            matrix,
            True,
            settings.knn_k,
        )
        for fold in split.folds
    ]
    by_dwarf: dict[str, list[str]] = {}
    for image_id, dwarf_id in zip(matrix.image_ids, matrix.dwarf_ids, strict=True):
        by_dwarf.setdefault(dwarf_id, []).append(image_id)
    unknown = []
    for dwarf_id, own in sorted(by_dwarf.items()):
        gallery = tuple(image_id for image_id in matrix.image_ids if image_id not in set(own))
        unknown += [
            signal_scores(image_id, dwarf_id, gallery, matrix, False, settings.knn_k)
            for image_id in own
        ]

    def scored(rows: Sequence[SignalScores], signal: str) -> tuple[ScoredQuery, ...]:
        # The signal rides in `top_similarity`, so open_set's own calibration code
        # judges it -- the cosine control then runs exactly the released path.
        return tuple(
            ScoredQuery(r.query_image_id, r.dwarf_id, getattr(r, signal), r.top_dwarf_id, r.present)
            for r in rows
        )

    control = {
        m.name: m.value
        for m in summarize_open_set(scored(known, "cosine"), scored(unknown, "cosine"), open_set)
    }
    for key in ("auroc", "target_90_false_acceptance", "target_90_known_acceptance"):
        check_reproduced(f"{backbone} rejection", control[key], released, key)

    statues = sorted(by_dwarf)
    index = {statue: position for position, statue in enumerate(statues)}
    known_statue = np.asarray([index[r.dwarf_id] for r in known])
    unknown_statue = np.asarray([index[r.dwarf_id] for r in unknown])
    label = f"{backbone}_reject"
    draws = statue_draws(
        len(statues), settings.bootstrap_resamples, generator(settings.seed, label)
    )
    samples: dict[str, npt.NDArray[np.float64]] = {}
    points: dict[str, float] = {}
    for signal in SIGNALS:
        k_scores = np.asarray([getattr(r, signal) for r in known], dtype=np.float64)
        u_scores = np.asarray([getattr(r, signal) for r in unknown], dtype=np.float64)
        points[signal] = weighted_auroc(
            k_scores, u_scores, np.ones(len(known)), np.ones(len(unknown))
        )
        resampled = np.empty(draws.shape[0])
        for row, draw in enumerate(draws):
            multiplicity = np.bincount(draw, minlength=len(statues)).astype(np.float64)
            resampled[row] = weighted_auroc(
                k_scores, u_scores, multiplicity[known_statue], multiplicity[unknown_statue]
            )
        samples[signal] = resampled
    if abs(points["cosine"] - released["auroc"]) > 1e-9:
        raise WorkshopValidationError(
            f"{label}: weighted AUROC does not reproduce the released one"
        )

    metrics: list[MetricSummary] = []
    for signal in SIGNALS:
        metrics.append(interval_metric(f"{label}_{signal}_auroc", points[signal], samples[signal]))
        if signal != "cosine":
            metrics.append(
                interval_metric(
                    f"{label}_{signal}_minus_cosine_auroc",
                    points[signal] - points["cosine"],
                    samples[signal] - samples["cosine"],
                )
            )
        thresholds = calibrate_leave_one_class_out(scored(known, signal), 0.9)
        operating = {
            m.name: m.value
            for m in _operating_point_metrics(
                scored(known, signal), scored(unknown, signal), thresholds, "t90"
            )
        }
        metrics.append(
            MetricSummary(
                name=f"{label}_{signal}_false_acceptance_90",
                value=operating["t90_false_acceptance"],
            )
        )
        metrics.append(
            MetricSummary(
                name=f"{label}_{signal}_known_acceptance_90",
                value=operating["t90_known_acceptance"],
            )
        )
    return metrics


def groups_part(
    backbone: str,
    split: EvaluationSplit,
    matrix: EmbeddingMatrix,
    released: Mapping[str, float],
    settings: WorkshopValidationConfig,
) -> list[MetricSummary]:
    """Top-1 with each group class merged into its members, against unmerged."""
    family = {dwarf: dwarf for dwarf in set(matrix.dwarf_ids)}
    for group, members in settings.groups.items():
        for dwarf in (group, *members):
            if dwarf not in family:
                raise WorkshopValidationError(f"group setting names an unknown statue: {dwarf}")
            family[dwarf] = group
    unmerged: dict[str, float] = {}
    merged: dict[str, float] = {}
    for fold in split.folds:
        vectors, dwarf_ids = matrix.rows_for(fold.reference_image_ids)
        top = (
            cosine_knn(
                fold.query_image_id,
                matrix.vector_for(fold.query_image_id),
                vectors,
                fold.reference_image_ids,
                dwarf_ids,
                top_k=1,
            )
            .matches[0]
            .dwarf_id
        )
        unmerged[fold.query_image_id] = float(top == fold.query_dwarf_id)
        merged[fold.query_image_id] = float(family[top] == family[fold.query_dwarf_id])
    check_reproduced(
        f"{backbone} groups", sum(unmerged.values()) / len(unmerged), released, "top_1"
    )
    statue_of = {fold.query_image_id: fold.query_dwarf_id for fold in split.folds}
    resamples, seed = settings.bootstrap_resamples, settings.seed
    return [
        rate_interval(f"{backbone}_groups_merged_top_1", merged, statue_of, resamples, seed),
        paired_gap(f"{backbone}_groups_merged_gain", merged, unmerged, statue_of, resamples, seed),
        MetricSummary(
            name=f"{backbone}_groups_changed_queries",
            value=float(sum(merged[q] != unmerged[q] for q in merged)),
        ),
    ]


# ---------------------------------------------------------------------------
# Section 7.16: tests of the explanations, not only of the measurements.
# ---------------------------------------------------------------------------


def geo_permutation_part(
    backbone: str,
    split: EvaluationSplit,
    manifest: DatasetManifest,
    matrix: EmbeddingMatrix,
    released: Mapping[str, float],
    settings: WorkshopValidationConfig,
) -> list[MetricSummary]:
    """Does the geographic penalty need lookalikes to stand together?

    Statue positions are permuted among the located statues: the set of positions
    and the random arm are untouched, and only who stands next to whom is broken.
    If co-located lookalikes cause the penalty, shuffled layouts should not
    reproduce it. The p-value is the share of layouts at least as harmful as the
    real one, counting the real one, so it is never zero.
    """
    located, dwarf_ids, _, locations = located_inputs(manifest, split, matrix)
    statues = sorted(locations)
    positions = [locations[statue] for statue in statues]
    sizes = tuple(size for size in settings.permutation_pool_sizes if size < len(dwarf_ids))
    observed = {}
    for size in sizes:
        geo = measure_geographic_pool(located, matrix, size, locations)
        check_reproduced(f"{backbone} permutation", geo.top_1, released, f"geo_top_1_pool_{size}")
        observed[size] = geo.top_1 - released[f"random_top_1_pool_{size}"]

    rng = generator(settings.seed, f"{backbone}-permutation")
    shuffled = {size: np.empty(settings.permutation_count) for size in sizes}
    for draw in range(settings.permutation_count):
        order = rng.permutation(len(statues))
        layout = {statue: positions[index] for statue, index in zip(statues, order, strict=True)}
        for size in sizes:
            geo = measure_geographic_pool(located, matrix, size, layout)
            shuffled[size][draw] = geo.top_1 - released[f"random_top_1_pool_{size}"]

    label = f"{backbone}_geo_permutation"
    metrics: list[MetricSummary] = []
    for size in sizes:
        null = shuffled[size]
        metrics += [
            MetricSummary(
                name=f"{label}_pool_{size}_null_mean",
                value=float(null.mean()),
                lower_bound=float(np.percentile(null, 2.5)),
                upper_bound=float(np.percentile(null, 97.5)),
            ),
            MetricSummary(
                name=f"{label}_pool_{size}_p_value",
                value=float((1 + np.sum(null <= observed[size])) / (1 + null.shape[0])),
            ),
        ]
    # One test over the sizes together, so the reader is not left to combine them.
    pooled_null = np.mean([shuffled[size] for size in sizes], axis=0)
    pooled = float(np.mean([observed[size] for size in sizes]))
    metrics += [
        MetricSummary(name=f"{label}_mean_observed", value=pooled),
        MetricSummary(
            name=f"{label}_mean_null",
            value=float(pooled_null.mean()),
            lower_bound=float(np.percentile(pooled_null, 2.5)),
            upper_bound=float(np.percentile(pooled_null, 97.5)),
        ),
        MetricSummary(
            name=f"{label}_mean_p_value",
            value=float((1 + np.sum(pooled_null <= pooled)) / (1 + pooled_null.shape[0])),
        ),
        MetricSummary(name=f"{label}_permutations", value=float(settings.permutation_count)),
    ]
    return metrics


def statue_centroids(matrix: EmbeddingMatrix) -> tuple[tuple[str, ...], npt.NDArray[np.float64]]:
    """Each statue's mean image vector, renormalised, in sorted statue order."""
    statues = tuple(sorted(set(matrix.dwarf_ids)))
    index = {statue: position for position, statue in enumerate(statues)}
    sums = np.zeros((len(statues), matrix.vectors.shape[1]), dtype=np.float64)
    for dwarf, vector in zip(matrix.dwarf_ids, matrix.vectors, strict=True):
        sums[index[dwarf]] += vector
    return statues, sums / np.linalg.norm(sums, axis=1, keepdims=True)


def mutual_pairs(matrix: EmbeddingMatrix) -> set[tuple[str, str]]:
    """Statue pairs that are each other's nearest neighbour by centroid."""
    statues, centroids = statue_centroids(matrix)
    similarity = centroids @ centroids.T
    np.fill_diagonal(similarity, -np.inf)
    nearest = similarity.argmax(axis=1)
    return {
        (statues[a], statues[b])
        for a, b in enumerate(nearest)
        if nearest[b] == a and statues[a] < statues[b]
    }


def lookalike_families(
    matrices: Mapping[str, EmbeddingMatrix], minimum_votes: int
) -> dict[str, int]:
    """Connected components of pairs that at least `minimum_votes` backbones agree on."""
    votes: dict[tuple[str, str], int] = {}
    for matrix in matrices.values():
        for pair in mutual_pairs(matrix):
            votes[pair] = votes.get(pair, 0) + 1
    parent: dict[str, str] = {}

    def root(statue: str) -> str:
        while parent.setdefault(statue, statue) != statue:
            statue = parent[statue]
        return statue

    for (a, b), count in sorted(votes.items()):
        if count >= minimum_votes:
            parent[root(a)] = root(b)
    groups: dict[str, int] = {}
    family_of: dict[str, int] = {}
    for statue in sorted(parent):
        family_of[statue] = groups.setdefault(root(statue), len(groups))
    return family_of


def families_part(
    split: EvaluationSplit,
    matrices: Mapping[str, EmbeddingMatrix],
    settings: WorkshopValidationConfig,
) -> list[MetricSummary]:
    """How much of each backbone's error lies inside a lookalike family.

    The families a backbone is judged against are drawn by the *other* backbones,
    so no backbone is scored against groups its own embedding defined.
    """
    resamples, seed = settings.bootstrap_resamples, settings.seed
    statue_of = {fold.query_image_id: fold.query_dwarf_id for fold in split.folds}
    statues = sorted(set(statue_of.values()))
    metrics: list[MetricSummary] = []
    for name, matrix in matrices.items():
        others = {other: m for other, m in matrices.items() if other != name}
        family_of = lookalike_families(others, settings.family_votes)
        in_family = {s for s in statues if s in family_of}
        wrong: dict[str, float] = {}
        within: dict[str, float] = {}
        for fold in split.folds:
            vectors, dwarf_ids = matrix.rows_for(fold.reference_image_ids)
            top = (
                cosine_knn(
                    fold.query_image_id,
                    matrix.vector_for(fold.query_image_id),
                    vectors,
                    fold.reference_image_ids,
                    dwarf_ids,
                    top_k=1,
                )
                .matches[0]
                .dwarf_id
            )
            truth = fold.query_dwarf_id
            wrong[fold.query_image_id] = float(top != truth)
            within[fold.query_image_id] = float(
                top != truth and truth in family_of and family_of.get(top) == family_of[truth]
            )
        label = f"{name}_families"
        errors, _ = per_statue(wrong, statue_of, statues)
        inside, _ = per_statue(within, statue_of, statues)
        draws = statue_draws(len(statues), resamples, generator(seed, f"{label}_error_share"))
        samples = inside[draws].sum(axis=1) / np.maximum(errors[draws].sum(axis=1), 1.0)
        metrics += [
            MetricSummary(name=f"{label}_count", value=float(len(set(family_of.values())))),
            MetricSummary(name=f"{label}_statues", value=float(len(in_family))),
            MetricSummary(name=f"{label}_statue_share", value=len(in_family) / len(statues)),
            interval_metric(
                f"{label}_error_share",
                sum(within.values()) / max(sum(wrong.values()), 1.0),
                samples,
            ),
            MetricSummary(name=f"{label}_errors", value=float(sum(wrong.values()))),
        ]
        for subset, tag in (
            ({q for q in wrong if statue_of[q] in in_family}, "inside"),
            ({q for q in wrong if statue_of[q] not in in_family}, "outside"),
        ):
            metrics.append(
                rate_interval(
                    f"{label}_{tag}_error_rate",
                    {q: wrong[q] for q in subset},
                    statue_of,
                    resamples,
                    seed,
                )
            )
    return metrics


def null_bounds_part(
    split: EvaluationSplit,
    manifest: DatasetManifest,
    matrices: Mapping[str, EmbeddingMatrix],
    released: Mapping[str, float],
    settings: WorkshopValidationConfig,
) -> list[MetricSummary]:
    """The largest gain each null comparison could be hiding, photographer withheld.

    Each checkpoint against `dinov2` on the same photographer-disjoint queries, with
    a paired statue interval. The upper bound is what "no improvement" rules out.
    """
    from krasnal_id.config import RecallCurveConfig
    from krasnal_id.experiments.recall_curve import ARMS, DISJOINT, measure_arm
    from krasnal_id.photographers import photographers_by_class

    recall = packaged(RecallCurveConfig, "recall_curve").model_copy(
        update={
            "fuse_backbones": (),
            "expansion_neighbours": (),
            "compare_backbones": tuple(settings.null_bound_backbones),
        }
    )
    arm = next(a for a in ARMS if a.name == DISJOINT)
    ranks, total = measure_arm(arm, split, manifest, dict(matrices), "dinov2", recall)
    for k in settings.null_bound_cutoffs:
        hits = sum(0 < rank <= k for rank in ranks["plain"]) / total
        check_reproduced("null bounds", hits, released, f"disjoint_r_at_{k}")

    by_class = photographers_by_class(manifest)
    queries = [
        fold.query_image_id
        for fold in split.folds
        if len(by_class.get(fold.query_dwarf_id, set())) >= 2
    ]
    if len(queries) != total:
        raise WorkshopValidationError(
            f"null bounds: {total} disjoint queries ranked but {len(queries)} answerable"
        )
    statue_of = {fold.query_image_id: fold.query_dwarf_id for fold in split.folds}
    metrics: list[MetricSummary] = []
    for other in settings.null_bound_backbones:
        for k in settings.null_bound_cutoffs:
            base = {q: float(0 < r <= k) for q, r in zip(queries, ranks["plain"], strict=True)}
            variant = {
                q: float(0 < r <= k)
                for q, r in zip(queries, ranks[f"against_{other}"], strict=True)
            }
            metrics.append(
                paired_gap(
                    f"null_{other}_minus_dinov2_disjoint_r_at_{k}",
                    variant,
                    base,
                    statue_of,
                    settings.bootstrap_resamples,
                    settings.seed,
                )
            )
    return metrics

"""Rank statistics shared by the experiments."""

import math

import numpy as np
import numpy.typing as npt


class StatisticsError(ValueError):
    """Raised when a rank statistic is asked for on an empty population."""


def average_ranks(values: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
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
        raise StatisticsError("AUROC needs at least one query in each population")
    ranks = average_ranks(np.concatenate([known, unknown]))
    known_rank_sum = float(ranks[: known.shape[0]].sum())
    count_known = float(known.shape[0])
    count_unknown = float(unknown.shape[0])
    return (known_rank_sum - count_known * (count_known + 1.0) / 2.0) / (
        count_known * count_unknown
    )


def exact_mcnemar_p_value(wins: int, losses: int) -> float:
    """Two-sided exact McNemar p-value for two methods scored on the same queries.

    Comparing two backbones through their separate confidence intervals wastes the
    pairing and is the wrong test: every query is answered by both, so the only
    evidence about which is better lives in the queries where they disagree. Under
    the null those discordant queries split like a fair coin, which makes the exact
    binomial test on `wins` of `wins + losses` the whole comparison.

    Exact rather than the chi-square approximation because the discordant counts
    here are tens, not thousands, and dependency-free because a p-value is four
    lines of `math.comb` and the alternative is a new runtime dependency for it.
    """
    if wins < 0 or losses < 0:
        raise StatisticsError("discordant counts cannot be negative")
    total = wins + losses
    if total == 0:
        # No query distinguishes the two methods, so nothing is evidence against
        # the null. Returning 1.0 keeps a caller from reading "no data" as "equal".
        return 1.0
    smaller = min(wins, losses)
    tail = sum(math.comb(total, index) for index in range(smaller + 1))
    return min(1.0, 2.0 * tail / (2.0**total))

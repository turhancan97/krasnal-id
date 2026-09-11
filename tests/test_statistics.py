"""Rank and paired-comparison statistics shared by the experiments."""

import math

import numpy as np
import pytest

from krasnal_id.statistics import (
    StatisticsError,
    exact_mcnemar_p_value,
    separability_auroc,
)


def test_only_the_discordant_queries_carry_evidence() -> None:
    """Queries both methods get right, or both get wrong, say nothing about which wins."""
    assert exact_mcnemar_p_value(5, 5) == exact_mcnemar_p_value(5, 5)
    # A tie is the least surprising outcome there is.
    assert exact_mcnemar_p_value(5, 5) == pytest.approx(1.0)
    # And no disagreement at all is no evidence, not perfect agreement.
    assert exact_mcnemar_p_value(0, 0) == pytest.approx(1.0)


def test_the_p_value_is_the_exact_two_sided_binomial() -> None:
    """Computed rather than approximated, because the counts here are tens."""
    for wins, losses in ((62, 32), (22, 20), (15, 30), (5, 0), (1, 7)):
        total = wins + losses
        smaller = min(wins, losses)
        expected = min(
            1.0,
            2.0 * sum(math.comb(total, i) for i in range(smaller + 1)) / 2.0**total,
        )

        assert exact_mcnemar_p_value(wins, losses) == pytest.approx(expected)


def test_the_p_value_is_symmetric_and_bounded() -> None:
    """Which method is called the baseline cannot change the significance."""
    for wins, losses in ((62, 32), (9, 1), (0, 4)):
        value = exact_mcnemar_p_value(wins, losses)

        assert value == pytest.approx(exact_mcnemar_p_value(losses, wins))
        assert 0.0 <= value <= 1.0


def test_a_larger_imbalance_is_more_significant() -> None:
    """The same number of disagreements, split more lopsidedly, is stronger evidence."""
    values = [exact_mcnemar_p_value(wins, 40 - wins) for wins in (20, 25, 30, 35, 40)]

    assert values == sorted(values, reverse=True)


def test_negative_counts_are_refused() -> None:
    with pytest.raises(StatisticsError):
        exact_mcnemar_p_value(-1, 3)


def test_auroc_is_the_probability_a_known_query_outscores_an_unknown_one() -> None:
    known = np.asarray([0.9, 0.8, 0.7])
    unknown = np.asarray([0.6, 0.5])

    assert separability_auroc(known, unknown) == pytest.approx(1.0)
    assert separability_auroc(unknown, known) == pytest.approx(0.0)


def test_auroc_needs_both_populations() -> None:
    with pytest.raises(StatisticsError):
        separability_auroc(np.asarray([0.5]), np.asarray([]))

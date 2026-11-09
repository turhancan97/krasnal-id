"""Paired bootstrap over whole statues, shared by every experiment that needs one.

A statue is the independent unit in this corpus. Its photographs share a sculpture,
usually a photographer and often a visit, so treating each photograph as an
independent draw overstates precision -- by how much depends on how unevenly the
photographs are spread, which here is very (a median of four per statue, a maximum
of 31). Every interval built here therefore resamples whole statues with
replacement and keeps each statue's queries together.

Two rules make the intervals mean what a reader assumes:

* **Paired draws.** Methods compared on the same queries are resampled with the
  same statue indices, so an interval on a difference reflects the difference and
  not two independent sampling errors.
* **Image-weighted estimates.** Within a resample, accuracy is hits over queries
  pooled across the drawn statues, so the point estimate the interval surrounds is
  the ordinary per-photograph accuracy every other number in the project reports.

`krasnal_id.experiments.field_gap` introduced this for the field photographs; it
lives here so `workshop_validation` and the field gap cannot drift apart.
"""

import zlib
from collections.abc import Mapping, Sequence

import numpy as np
import numpy.typing as npt

from krasnal_id.experiments.contracts import MetricSummary


class BootstrapError(ValueError):
    """Raised when a bootstrap is asked for over nothing, or with mismatched inputs."""


def generator(seed: int, label: str) -> np.random.Generator:
    """One generator per resampled group, independent of which other groups exist.

    Keyed by the group's name rather than its position, so adding a group never
    changes the draws behind an existing interval.
    """
    return np.random.default_rng([seed, zlib.crc32(label.encode("utf-8"))])


def statue_draws(statues: int, resamples: int, rng: np.random.Generator) -> npt.NDArray[np.int64]:
    """Draw `resamples` resamples of `statues` statue indices, with replacement."""
    if statues <= 0 or resamples <= 0:
        raise BootstrapError("a bootstrap needs at least one statue and one resample")
    return np.asarray(rng.integers(0, statues, size=(resamples, statues)), dtype=np.int64)


def resampled_rate(
    successes: npt.NDArray[np.float64],
    counts: npt.NDArray[np.float64],
    draws: npt.NDArray[np.int64],
) -> npt.NDArray[np.float64]:
    """Pooled rate per resample: drawn statues' successes over their queries."""
    if successes.shape != counts.shape:
        raise BootstrapError("one success count is needed per statue count")
    return np.asarray(successes[draws].sum(axis=1) / counts[draws].sum(axis=1), dtype=np.float64)


def per_statue(
    outcomes: Mapping[str, float],
    statue_of: Mapping[str, str],
    statues: Sequence[str],
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """Sum one per-query outcome by statue, in the given statue order.

    Returns the summed outcome and the query count per statue. Every query must
    belong to one of `statues`; an unknown one is an error rather than a silent
    drop, because a dropped query would change the point estimate.
    """
    index = {statue: position for position, statue in enumerate(statues)}
    sums = np.zeros(len(statues), dtype=np.float64)
    counts = np.zeros(len(statues), dtype=np.float64)
    for query, value in outcomes.items():
        statue = statue_of.get(query)
        if statue is None or statue not in index:
            raise BootstrapError(f"query {query} belongs to no statue being resampled")
        sums[index[statue]] += float(value)
        counts[index[statue]] += 1.0
    return sums, counts


def percentile_interval(
    samples: npt.NDArray[np.float64], level: float = 0.95
) -> tuple[float, float]:
    """Return the central percentile interval of the bootstrap samples."""
    if samples.size == 0:
        raise BootstrapError("no bootstrap samples to summarize")
    tail = (1.0 - level) / 2.0 * 100.0
    lower, upper = np.percentile(samples, [tail, 100.0 - tail])
    return float(lower), float(upper)


def interval_metric(name: str, value: float, samples: npt.NDArray[np.float64]) -> MetricSummary:
    """A point estimate with its 95% statue-bootstrap interval."""
    lower, upper = percentile_interval(samples)
    return MetricSummary(name=name, value=value, lower_bound=lower, upper_bound=upper)

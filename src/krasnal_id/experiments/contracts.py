"""Serializable result contracts shared by experiment implementations."""

from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field


class MetricSummary(BaseModel):
    """Aggregate metric value with optional uncertainty bounds."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    name: str = Field(min_length=1)
    value: float
    lower_bound: float | None = None
    upper_bound: float | None = None


class ExperimentResult(BaseModel):
    """Versioned structured result for one reproducible experiment run."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    experiment: str = Field(min_length=1)
    backbone: str = Field(min_length=1)
    created_at: datetime
    seed: int
    metrics: tuple[MetricSummary, ...]

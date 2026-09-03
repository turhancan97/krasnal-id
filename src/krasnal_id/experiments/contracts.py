"""Serializable result contracts shared by experiment implementations."""

from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field, model_validator


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


class ConfusionPair(BaseModel):
    """One directed dwarf pair that competes for the same queries.

    Directed: `true_dwarf_id` is the dwarf being queried and `confused_dwarf_id` is
    the strongest wrong candidate for it. A genuinely symmetric confusion appears as
    two entries, which keeps the asymmetric cases visible instead of averaging them.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    true_dwarf_id: str = Field(min_length=1)
    true_display_name: str = Field(min_length=1)
    confused_dwarf_id: str = Field(min_length=1)
    confused_display_name: str = Field(min_length=1)
    queries: int = Field(gt=0)
    misidentifications: int = Field(ge=0)
    mean_margin: float

    @model_validator(mode="after")
    def validate_counts(self) -> "ConfusionPair":
        """Require the pair to describe a subset of its own queries."""
        if self.misidentifications > self.queries:
            raise ValueError("misidentifications cannot exceed the queries they came from")
        if self.true_dwarf_id == self.confused_dwarf_id:
            raise ValueError("a dwarf cannot be confused with itself")
        return self


class ConfusionAnalysisResult(ExperimentResult):
    """Experiment result carrying the ranked confusion pairs behind its metrics."""

    pairs: tuple[ConfusionPair, ...]


class DwarfRejection(BaseModel):
    """How one dwarf's queries behave once that dwarf is removed from the gallery.

    Every image of `dwarf_id` is queried against a gallery holding none of them, so
    the correct answer for all `unknown_queries` is rejection. `false_accepts` counts
    the ones a threshold let through anyway, and `nearest_dwarf_id` names the dwarf
    they fell through to most often — the statue that can pass for a missing one.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    dwarf_id: str = Field(min_length=1)
    display_name: str = Field(min_length=1)
    unknown_queries: int = Field(gt=0)
    false_accepts: int = Field(ge=0)
    mean_top_similarity: float = Field(ge=-1.0, le=1.0)
    nearest_dwarf_id: str = Field(min_length=1)
    nearest_display_name: str = Field(min_length=1)

    @model_validator(mode="after")
    def validate_counts(self) -> "DwarfRejection":
        """Require the accepted queries to be a subset of the dwarf's own queries."""
        if self.false_accepts > self.unknown_queries:
            raise ValueError("false accepts cannot exceed the queries they came from")
        if self.nearest_dwarf_id == self.dwarf_id:
            raise ValueError("a removed dwarf cannot be its own nearest neighbour")
        return self


class OpenSetRejectionResult(ExperimentResult):
    """Experiment result carrying the per-dwarf rejection rows behind its metrics."""

    rejections: tuple[DwarfRejection, ...]

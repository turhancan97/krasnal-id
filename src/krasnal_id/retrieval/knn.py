"""Typed contracts for cosine-similarity k-nearest-neighbor retrieval."""

import numpy as np
import numpy.typing as npt
from pydantic import BaseModel, ConfigDict, Field


class RetrievalMatch(BaseModel):
    """One ranked reference-image match."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    rank: int = Field(gt=0)
    image_id: str = Field(min_length=1)
    dwarf_id: str = Field(min_length=1)
    cosine_similarity: float = Field(ge=-1.0, le=1.0)


class RetrievalResult(BaseModel):
    """Ranked matches for one query image."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    query_image_id: str = Field(min_length=1)
    matches: tuple[RetrievalMatch, ...]


def cosine_knn(
    query: npt.NDArray[np.float32],
    references: npt.NDArray[np.float32],
    reference_image_ids: tuple[str, ...],
    reference_dwarf_ids: tuple[str, ...],
    top_k: int,
) -> RetrievalResult:
    """Rank reference embeddings by cosine similarity to a query."""
    raise NotImplementedError("Cosine k-NN retrieval is scheduled for v0.1")

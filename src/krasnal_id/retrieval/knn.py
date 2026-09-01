"""Typed contracts for cosine-similarity k-nearest-neighbor retrieval."""

import numpy as np
import numpy.typing as npt
from pydantic import BaseModel, ConfigDict, Field


class RetrievalError(ValueError):
    """Raised when query or reference inputs violate the retrieval contract."""


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


def _unit_vector(values: npt.NDArray[np.float32], label: str) -> npt.NDArray[np.float32]:
    """Validate one vector and rescale it to unit length."""
    array = np.asarray(values, dtype=np.float32)
    if array.ndim != 1 or array.size == 0:
        raise RetrievalError(f"{label} must be a non-empty one-dimensional vector")
    if not np.isfinite(array).all():
        raise RetrievalError(f"{label} contains non-finite values")
    norm = float(np.linalg.norm(array))
    if not np.isfinite(norm) or norm <= 0.0:
        raise RetrievalError(f"{label} must have a positive finite norm")
    return np.asarray(array / norm, dtype=np.float32)


def _unit_matrix(values: npt.NDArray[np.float32], dimension: int) -> npt.NDArray[np.float32]:
    """Validate the reference matrix and rescale every row to unit length."""
    array = np.asarray(values, dtype=np.float32)
    if array.ndim != 2 or array.shape[0] == 0 or array.shape[1] == 0:
        raise RetrievalError("references must be a non-empty two-dimensional array")
    if array.shape[1] != dimension:
        raise RetrievalError(
            f"reference dimension {array.shape[1]} does not match query dimension {dimension}"
        )
    if not np.isfinite(array).all():
        raise RetrievalError("references contain non-finite values")
    norms = np.linalg.norm(array, axis=1, keepdims=True)
    if not np.isfinite(norms).all() or np.any(norms <= 0.0):
        raise RetrievalError("every reference vector must have a positive finite norm")
    return np.asarray(array / norms, dtype=np.float32)


def cosine_knn(
    query_image_id: str,
    query: npt.NDArray[np.float32],
    references: npt.NDArray[np.float32],
    reference_image_ids: tuple[str, ...],
    reference_dwarf_ids: tuple[str, ...],
    top_k: int,
) -> RetrievalResult:
    """Rank reference embeddings by cosine similarity to a query.

    Inputs are rescaled to unit length, so the result is a true cosine ranking even
    if a caller supplies unnormalized vectors. Equal similarities are broken by
    ascending image ID, which keeps every ranking reproducible across runs and
    platforms. Fewer than `top_k` matches are returned when the reference set is
    smaller, which is the normal case for small candidate pools.
    """
    if not query_image_id:
        raise RetrievalError("query_image_id must not be empty")
    if top_k <= 0:
        raise RetrievalError("top_k must be positive")

    count = len(reference_image_ids)
    if len(reference_dwarf_ids) != count:
        raise RetrievalError(
            f"received {count} reference image IDs and {len(reference_dwarf_ids)} dwarf IDs"
        )
    if count == 0:
        raise RetrievalError("at least one reference image is required")
    if len(set(reference_image_ids)) != count:
        raise RetrievalError("reference image IDs must be unique")
    if query_image_id in reference_image_ids:
        raise RetrievalError(f"query image {query_image_id} cannot appear in its reference set")
    if any(not image_id for image_id in reference_image_ids):
        raise RetrievalError("reference image IDs must not be empty")
    if any(not dwarf_id for dwarf_id in reference_dwarf_ids):
        raise RetrievalError("reference dwarf IDs must not be empty")

    unit_query = _unit_vector(query, "query")
    unit_references = _unit_matrix(references, unit_query.shape[0])
    if unit_references.shape[0] != count:
        raise RetrievalError(
            f"references hold {unit_references.shape[0]} rows for {count} reference IDs"
        )

    # Unit-length inputs bound the dot product to [-1, 1] up to float error, which
    # the clip absorbs so that RetrievalMatch validation cannot fail on rounding.
    similarities = np.clip(unit_references @ unit_query, -1.0, 1.0)
    # lexsort applies the last key first: descending similarity, then ascending ID.
    order = np.lexsort((np.asarray(reference_image_ids), -similarities))[:top_k]

    matches = tuple(
        RetrievalMatch(
            rank=rank,
            image_id=reference_image_ids[index],
            dwarf_id=reference_dwarf_ids[index],
            cosine_similarity=float(similarities[index]),
        )
        for rank, index in enumerate(order, start=1)
    )
    return RetrievalResult(query_image_id=query_image_id, matches=matches)

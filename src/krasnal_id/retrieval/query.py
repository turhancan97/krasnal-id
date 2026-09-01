"""Single-image retrieval against the cached reference embeddings."""

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import numpy.typing as npt
from PIL import Image, UnidentifiedImageError
from pydantic import ValidationError

from krasnal_id.config import AppConfig, BackboneConfig
from krasnal_id.embeddings.cache import EmbeddingCache
from krasnal_id.embeddings.store import (
    EmbeddingMatrix,
    cache_key_for_digest,
    load_embedding_matrix,
)
from krasnal_id.models import DatasetManifest
from krasnal_id.retrieval.knn import RetrievalResult, cosine_knn


class QueryError(ValueError):
    """Raised when a query image or the reference set cannot be used."""


@dataclass(frozen=True, slots=True)
class DwarfCandidate:
    """One ranked candidate dwarf for a query image."""

    rank: int
    dwarf_id: str
    display_name: str
    cosine_similarity: float
    matched_image_id: str


@dataclass(frozen=True, slots=True)
class QueryOutcome:
    """Ranked candidates for one query, with how the query was embedded."""

    query_sha256: str
    candidates: tuple[DwarfCandidate, ...]
    reused_cached_vector: bool
    excluded_reference_image_ids: tuple[str, ...]


def sha256_file(path: Path) -> str:
    """Hash a file without loading all of it at once."""
    digest = hashlib.sha256()
    try:
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
    except OSError as error:
        raise QueryError(f"could not read query image {path}: {error}") from error
    return digest.hexdigest()


def read_manifest(path: Path) -> DatasetManifest:
    """Read and strictly validate the generated manifest."""
    try:
        return DatasetManifest.model_validate(json.loads(path.read_text(encoding="utf-8")))
    except (OSError, json.JSONDecodeError, ValidationError, TypeError, ValueError) as error:
        raise QueryError(f"invalid manifest {path}: {error}") from error


def load_query_vector(
    image_path: Path,
    config: BackboneConfig,
    cache_root: Path,
) -> tuple[npt.NDArray[np.float32], str, bool]:
    """Embed one query image, reusing a cached vector for identical file content.

    Reuse is by content hash, so querying an image that is already in the dataset
    costs no model load at all. The cache is only read here: populating it stays
    the job of `embeddings extract`, which keeps ad-hoc queries from growing it.
    """
    if not image_path.is_file():
        raise QueryError(f"query image does not exist: {image_path}")

    digest = sha256_file(image_path)
    cached = EmbeddingCache(cache_root).load(cache_key_for_digest(digest, config))
    if cached is not None:
        return cached, digest, True

    try:
        with Image.open(image_path) as image:
            image.load()
            rgb = image.convert("RGB")
    except (OSError, UnidentifiedImageError) as error:
        raise QueryError(f"query image cannot be decoded: {image_path}: {error}") from error

    # Imported here so the command works from cache without the ml extra installed.
    from krasnal_id.embeddings.backbone import EmbeddingConfigurationError
    from krasnal_id.embeddings.extract import create_backbone

    try:
        backbone = create_backbone(config)
        vector = backbone.get_embedding(rgb)
    except EmbeddingConfigurationError as error:
        raise QueryError(str(error)) from error
    except Exception as error:
        raise QueryError(f"backbone {config.model_id} failed on {image_path}: {error}") from error
    return np.asarray(vector, dtype=np.float32), digest, False


def collapse_to_dwarfs(
    result: RetrievalResult,
    display_names: dict[str, str],
    top_k: int,
) -> tuple[DwarfCandidate, ...]:
    """Rank distinct dwarves by their best-matching reference image."""
    candidates: list[DwarfCandidate] = []
    seen: set[str] = set()
    for match in result.matches:
        if match.dwarf_id in seen:
            continue
        seen.add(match.dwarf_id)
        candidates.append(
            DwarfCandidate(
                rank=len(candidates) + 1,
                dwarf_id=match.dwarf_id,
                display_name=display_names.get(match.dwarf_id, match.dwarf_id),
                cosine_similarity=match.cosine_similarity,
                matched_image_id=match.image_id,
            )
        )
        if len(candidates) == top_k:
            break
    return tuple(candidates)


def rank_candidates(
    query_vector: npt.NDArray[np.float32],
    query_sha256: str,
    manifest: DatasetManifest,
    matrix: EmbeddingMatrix,
    top_k: int,
) -> QueryOutcome:
    """Rank the reference set for one query, excluding the query's own copies.

    A query that is itself a dataset image would otherwise match its own reference
    at similarity 1.0 and tell you nothing, so every reference sharing the query's
    content hash is withheld, which is the same leave-one-out rule the split uses.
    """
    if top_k <= 0:
        raise QueryError(f"top_k must be positive, got {top_k}")

    excluded = tuple(record.image_id for record in manifest.images if record.sha256 == query_sha256)
    references = tuple(image_id for image_id in matrix.image_ids if image_id not in set(excluded))
    if not references:
        raise QueryError("no reference images remain after excluding the query's own copies")

    vectors, dwarf_ids = matrix.rows_for(references)
    result = cosine_knn(
        f"query:{query_sha256[:12]}",
        query_vector,
        vectors,
        references,
        dwarf_ids,
        top_k=len(references),
    )
    display_names = {dwarf.dwarf_id: dwarf.display_name for dwarf in manifest.dwarfs}
    return QueryOutcome(
        query_sha256=query_sha256,
        candidates=collapse_to_dwarfs(result, display_names, top_k),
        reused_cached_vector=False,
        excluded_reference_image_ids=excluded,
    )


def retrieve_image(image_path: Path, config: AppConfig, top_k: int) -> QueryOutcome:
    """Identify the most likely dwarves for one query image."""
    manifest = read_manifest(config.paths.manifest_path)
    matrix = load_embedding_matrix(manifest, config.backbone, config.paths.embeddings_dir)
    vector, digest, reused = load_query_vector(
        image_path, config.backbone, config.paths.embeddings_dir
    )
    outcome = rank_candidates(vector, digest, manifest, matrix, top_k)
    return QueryOutcome(
        query_sha256=outcome.query_sha256,
        candidates=outcome.candidates,
        reused_cached_vector=reused,
        excluded_reference_image_ids=outcome.excluded_reference_image_ids,
    )

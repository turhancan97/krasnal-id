"""Single-image retrieval: query embedding, self-exclusion, and candidate ranking."""

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from PIL import Image
from typer.testing import CliRunner

from helpers import FAKE_BACKBONE, image_record, seed_embedding_cache, synthetic_manifest
from krasnal_id.cli import app
from krasnal_id.config import load_config
from krasnal_id.embeddings import extract as extract_module
from krasnal_id.embeddings.cache import EmbeddingCache
from krasnal_id.embeddings.store import cache_key_for_digest, load_embedding_matrix
from krasnal_id.models import DatasetManifest
from krasnal_id.retrieval.knn import RetrievalMatch, RetrievalResult
from krasnal_id.retrieval.query import (
    QueryError,
    collapse_to_dwarfs,
    load_query_vector,
    rank_candidates,
    read_manifest,
    retrieve_image,
    sha256_file,
)


def _write_image(path: Path, color: tuple[int, int, int] = (10, 20, 30)) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (8, 8), color).save(path)
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _unit(*values: float) -> np.ndarray:
    vector = np.asarray(values, dtype=np.float32)
    return np.asarray(vector / np.linalg.norm(vector), dtype=np.float32)


def test_file_hash_matches_the_content(tmp_path: Path) -> None:
    path = tmp_path / "query.png"
    digest = _write_image(path)

    assert sha256_file(path) == digest
    with pytest.raises(QueryError, match="could not read query image"):
        sha256_file(tmp_path / "absent.png")


def test_manifest_errors_are_reported(tmp_path: Path) -> None:
    with pytest.raises(QueryError, match="invalid manifest"):
        read_manifest(tmp_path / "absent.json")

    broken = tmp_path / "manifest.json"
    broken.write_text("{}", encoding="utf-8")
    with pytest.raises(QueryError, match="invalid manifest"):
        read_manifest(broken)

    valid = tmp_path / "valid.json"
    manifest = synthetic_manifest(dwarf_count=3)
    valid.write_text(json.dumps(manifest.model_dump(mode="json")), encoding="utf-8")
    assert read_manifest(valid) == manifest


def test_identical_content_reuses_its_cached_vector(tmp_path: Path) -> None:
    path = tmp_path / "query.png"
    digest = _write_image(path)
    stored = _unit(1.0, 2.0, 3.0)
    EmbeddingCache(tmp_path / "cache").store(cache_key_for_digest(digest, FAKE_BACKBONE), stored)

    vector, returned_digest, reused = load_query_vector(path, FAKE_BACKBONE, tmp_path / "cache")

    assert reused
    assert returned_digest == digest
    np.testing.assert_allclose(vector, stored)


def test_an_uncached_image_is_embedded_by_the_backbone(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "query.png"
    _write_image(path)
    expected = _unit(0.0, 1.0)

    class FakeBackbone:
        def __init__(self) -> None:
            self.images: list[Any] = []

        def get_embedding(self, image: Image.Image) -> np.ndarray:
            # The adapter must receive a decoded RGB image, not a path.
            assert image.mode == "RGB"
            self.images.append(image)
            return expected

    fake = FakeBackbone()
    monkeypatch.setattr(extract_module, "create_backbone", lambda config: fake)

    vector, _, reused = load_query_vector(path, FAKE_BACKBONE, tmp_path / "empty-cache")

    assert not reused
    assert len(fake.images) == 1
    np.testing.assert_allclose(vector, expected)


def test_unusable_query_images_are_rejected(tmp_path: Path) -> None:
    with pytest.raises(QueryError, match="does not exist"):
        load_query_vector(tmp_path / "absent.png", FAKE_BACKBONE, tmp_path)

    corrupt = tmp_path / "corrupt.png"
    corrupt.write_bytes(b"not an image")
    with pytest.raises(QueryError, match="cannot be decoded"):
        load_query_vector(corrupt, FAKE_BACKBONE, tmp_path)


def test_a_backbone_failure_is_surfaced(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    path = tmp_path / "query.png"
    _write_image(path)

    class Broken:
        def get_embedding(self, image: Image.Image) -> np.ndarray:
            raise RuntimeError("no weights")

    monkeypatch.setattr(extract_module, "create_backbone", lambda config: Broken())

    with pytest.raises(QueryError, match="failed on"):
        load_query_vector(path, FAKE_BACKBONE, tmp_path / "empty-cache")


def test_candidates_collapse_to_distinct_dwarves_keeping_the_best_match() -> None:
    result = RetrievalResult(
        query_image_id="query",
        matches=(
            RetrievalMatch(rank=1, image_id="a1", dwarf_id="Q1", cosine_similarity=0.9),
            RetrievalMatch(rank=2, image_id="a2", dwarf_id="Q1", cosine_similarity=0.8),
            RetrievalMatch(rank=3, image_id="b1", dwarf_id="Q2", cosine_similarity=0.7),
            RetrievalMatch(rank=4, image_id="c1", dwarf_id="Q3", cosine_similarity=0.6),
        ),
    )

    candidates = collapse_to_dwarfs(result, {"Q1": "First", "Q2": "Second"}, top_k=2)

    assert [candidate.dwarf_id for candidate in candidates] == ["Q1", "Q2"]
    assert [candidate.rank for candidate in candidates] == [1, 2]
    # The reported similarity and image are the dwarf's best, not its last.
    assert candidates[0].cosine_similarity == pytest.approx(0.9)
    assert candidates[0].matched_image_id == "a1"
    assert candidates[0].display_name == "First"
    # An unnamed dwarf falls back to its QID.
    assert collapse_to_dwarfs(result, {}, top_k=3)[2].display_name == "Q3"


def test_a_query_that_is_a_dataset_image_withholds_its_own_copies(tmp_path: Path) -> None:
    manifest = synthetic_manifest(dwarf_count=3)
    seed_embedding_cache(tmp_path, manifest)
    matrix = load_embedding_matrix(manifest, FAKE_BACKBONE, tmp_path)
    own = manifest.images[0]

    outcome = rank_candidates(
        matrix.vector_for(own.image_id), own.sha256, manifest, matrix, top_k=3
    )

    assert outcome.excluded_reference_image_ids == (own.image_id,)
    assert own.image_id not in {candidate.matched_image_id for candidate in outcome.candidates}
    # Its own dwarf still wins, on a different reference image.
    assert outcome.candidates[0].dwarf_id == own.dwarf_id
    assert outcome.candidates[0].cosine_similarity < 1.0


def test_every_byte_identical_copy_is_withheld(tmp_path: Path) -> None:
    # Two records sharing one content hash: both must be withheld, not just the first.
    duplicated = "d" * 64
    manifest = synthetic_manifest(dwarf_count=3)
    images = (
        image_record("copy-a", "Q0", "d"),
        image_record("copy-b", "Q1", "d"),
        *manifest.images,
    )
    manifest = manifest.model_copy(update={"images": images})
    seed_embedding_cache(tmp_path, manifest)
    matrix = load_embedding_matrix(manifest, FAKE_BACKBONE, tmp_path)

    outcome = rank_candidates(_unit(1.0, 0.0, 0.0, 0.0), duplicated, manifest, matrix, top_k=3)

    assert set(outcome.excluded_reference_image_ids) == {"copy-a", "copy-b"}
    assert not {"copy-a", "copy-b"} & {
        candidate.matched_image_id for candidate in outcome.candidates
    }


def test_ranking_rejects_bad_top_k_and_an_empty_reference_set(tmp_path: Path) -> None:
    manifest = synthetic_manifest(dwarf_count=3)
    seed_embedding_cache(tmp_path, manifest)
    matrix = load_embedding_matrix(manifest, FAKE_BACKBONE, tmp_path)

    with pytest.raises(QueryError, match="top_k must be positive"):
        rank_candidates(_unit(1.0, 0.0), "a" * 64, manifest, matrix, top_k=0)

    # A manifest whose every image shares the query's hash leaves nothing to rank.
    identical = manifest.model_copy(
        update={
            "images": tuple(
                record.model_copy(update={"sha256": "e" * 64}) for record in manifest.images
            )
        }
    )
    with pytest.raises(QueryError, match="no reference images remain"):
        rank_candidates(matrix.vectors[0], "e" * 64, identical, matrix, top_k=3)


def test_retrieve_image_reports_how_the_query_was_embedded(tmp_path: Path) -> None:
    manifest = synthetic_manifest(dwarf_count=3)
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(manifest.model_dump(mode="json")), encoding="utf-8")
    cache_root = tmp_path / "embeddings"
    seed_embedding_cache(cache_root, manifest)

    query_path = tmp_path / "query.png"
    digest = _write_image(query_path)
    # Give the query file's own content a cached vector so no backbone is needed.
    EmbeddingCache(cache_root).store(
        cache_key_for_digest(digest, FAKE_BACKBONE), _unit(1.0, 0.0, 0.0, 0.0)
    )
    config = load_config(
        [
            f"paths.manifest_path={manifest_path}",
            f"paths.embeddings_dir={cache_root}",
            f"backbone.model_id={FAKE_BACKBONE.model_id}",
            f"backbone.revision={FAKE_BACKBONE.revision}",
            f"backbone.preprocessing_id={FAKE_BACKBONE.preprocessing_id}",
        ]
    )

    outcome = retrieve_image(query_path, config, top_k=2)

    assert outcome.reused_cached_vector
    assert outcome.query_sha256 == digest
    assert len(outcome.candidates) == 2
    assert outcome.excluded_reference_image_ids == ()


def test_cli_retrieve_ranks_candidates_and_reports_failures(tmp_path: Path) -> None:
    manifest = synthetic_manifest(dwarf_count=3)
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(manifest.model_dump(mode="json")), encoding="utf-8")
    cache_root = tmp_path / "embeddings"
    query_path = tmp_path / "query.png"
    digest = _write_image(query_path)
    overrides = [
        f"paths.manifest_path={manifest_path}",
        f"paths.embeddings_dir={cache_root}",
        f"backbone.model_id={FAKE_BACKBONE.model_id}",
        f"backbone.revision={FAKE_BACKBONE.revision}",
        f"backbone.preprocessing_id={FAKE_BACKBONE.preprocessing_id}",
        "logging.json_output=false",
    ]
    flags = [f"-o{value}" for value in overrides]
    runner = CliRunner()

    missing_embeddings = runner.invoke(app, ["retrieve", str(query_path), *flags])
    assert missing_embeddings.exit_code == 2
    assert "Retrieval error" in missing_embeddings.output

    seed_embedding_cache(cache_root, manifest)
    EmbeddingCache(cache_root).store(
        cache_key_for_digest(digest, FAKE_BACKBONE), _unit(1.0, 0.0, 0.0, 0.0)
    )

    missing_image = runner.invoke(app, ["retrieve", str(tmp_path / "absent.png"), *flags])
    assert missing_image.exit_code == 2
    assert "does not exist" in missing_image.output

    result = runner.invoke(app, ["retrieve", str(query_path), "--top-k", "2", *flags])
    assert result.exit_code == 0, result.output
    assert "Top 2 candidates" in result.output
    assert "cached vector" in result.output
    assert "similarity" in result.output
    assert "1. Dwarf" in result.output


def test_cli_retrieve_notes_a_withheld_self_match(tmp_path: Path) -> None:
    manifest = synthetic_manifest(dwarf_count=3)
    query_path = tmp_path / "query.png"
    digest = _write_image(query_path)
    # Point the first manifest record at the query's own content hash.
    images = (
        manifest.images[0].model_copy(update={"sha256": digest}),
        *manifest.images[1:],
    )
    manifest = manifest.model_copy(update={"images": images})
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(manifest.model_dump(mode="json")), encoding="utf-8")
    cache_root = tmp_path / "embeddings"
    seed_embedding_cache(cache_root, manifest)
    runner = CliRunner()

    result = runner.invoke(
        app,
        [
            "retrieve",
            str(query_path),
            f"-opaths.manifest_path={manifest_path}",
            f"-opaths.embeddings_dir={cache_root}",
            f"-obackbone.model_id={FAKE_BACKBONE.model_id}",
            f"-obackbone.revision={FAKE_BACKBONE.revision}",
            f"-obackbone.preprocessing_id={FAKE_BACKBONE.preprocessing_id}",
            "-ologging.json_output=false",
        ],
    )

    assert result.exit_code == 0, result.output
    assert "withheld the query's own reference copies" in result.output
    assert manifest.images[0].image_id in result.output


def test_query_and_manifest_stay_consistent_for_a_real_manifest() -> None:
    manifest_path = Path("data/manifest.json")
    if not manifest_path.is_file():
        pytest.skip("no generated manifest in this checkout")

    manifest = read_manifest(manifest_path)
    assert isinstance(manifest, DatasetManifest)
    assert manifest.images

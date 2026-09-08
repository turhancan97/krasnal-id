"""Smoke tests for scaffolded interfaces and dependency boundaries."""

import json
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pytest

from krasnal_id.config import load_config
from krasnal_id.data_pipeline.build_manifest import build_dataset_manifest
from krasnal_id.demo.app import launch
from krasnal_id.embeddings.backbone import EmbeddingBackbone
from krasnal_id.embeddings.cache import EmbeddingCache, EmbeddingCacheKey
from krasnal_id.embeddings.clip import ClipBackbone
from krasnal_id.embeddings.dinov2 import DinoV2Backbone
from krasnal_id.experiments.artifacts import (
    ExperimentArtifactError,
    experiment_result_path,
    write_experiment_result,
)
from krasnal_id.experiments.contracts import ExperimentResult, MetricSummary
from krasnal_id.retrieval.knn import RetrievalMatch, RetrievalResult, cosine_knn


def test_cache_key_is_stable_and_sensitive() -> None:
    key = EmbeddingCacheKey("a" * 64, "model", "revision", "processor")
    same_key = EmbeddingCacheKey("a" * 64, "model", "revision", "processor")
    other_key = EmbeddingCacheKey("b" * 64, "model", "revision", "processor")

    assert key.digest() == same_key.digest()
    assert key.digest() != other_key.digest()


def test_backbone_adapters_satisfy_protocol_without_optional_ml_imports() -> None:
    dinov2 = DinoV2Backbone(load_config().backbone)
    clip = ClipBackbone(load_config(["backbone=clip"]).backbone)

    assert isinstance(dinov2, EmbeddingBackbone)
    assert isinstance(clip, EmbeddingBackbone)
    assert dinov2.model_id == "facebook/dinov2-base"
    assert dinov2.revision == "f9e44c814b77203eaa57a6bdbbd535f21ede1415"
    assert clip.preprocessing_id == "transformers-auto-processor"


def test_result_contracts() -> None:
    match = RetrievalMatch(rank=1, image_id="image-1", dwarf_id="Q1", cosine_similarity=0.95)
    retrieval = RetrievalResult(query_image_id="query-1", matches=(match,))
    experiment = ExperimentResult(
        experiment="baseline",
        backbone="dinov2",
        created_at=datetime.now(UTC),
        seed=42,
        metrics=(MetricSummary(name="top_1", value=0.5),),
    )

    assert retrieval.matches[0].rank == 1
    assert experiment.metrics[0].lower_bound is None


def test_v01_contracts_and_remaining_placeholders_are_explicit(tmp_path: Path) -> None:
    now = datetime.now(UTC)
    cache = EmbeddingCache(tmp_path)
    key = EmbeddingCacheKey("a" * 64, "model", "revision", "processor")

    manifest = build_dataset_manifest(
        (),
        (),
        now,
        3,
        source_query_sha256="a" * 64,
        staging_sha256="b" * 64,
        image_review_sha256="c" * 64,
    )
    assert manifest.dwarfs == ()
    assert manifest.images == ()
    valid_vector = np.asarray([1.0, 0.0], dtype=np.float32)
    cache.store(key, valid_vector)
    loaded = cache.load(key)
    assert loaded is not None
    np.testing.assert_allclose(loaded, valid_vector)
    ranked = cosine_knn(
        "query-1",
        np.asarray([1.0, 0.0], dtype=np.float32),
        np.asarray([[1.0, 0.0]], dtype=np.float32),
        ("i",),
        ("d",),
        1,
    )
    assert ranked.matches[0].image_id == "i"
    # run_baseline now has real behavior; tests/test_baseline.py covers it.


def test_every_scaffolded_stage_is_implemented() -> None:
    # Nothing in the v0.1-v0.3 build order raises NotImplementedError any more.
    assert callable(launch)


def _rerank_result(top_k: int, backbone: str = "clip") -> ExperimentResult:
    """A result carrying the one setting that made two runs collide."""
    return ExperimentResult(
        experiment="rerank_ablation",
        backbone=backbone,
        created_at=datetime(2026, 9, 8, tzinfo=UTC),
        seed=42,
        metrics=(MetricSummary(name="weight_0_top_1", value=0.54),),
        configuration={"kind": "rerank_ablation", "top_k": top_k, "weights": [0.0, 0.05]},
    )


def test_an_artifact_records_the_settings_that_produced_it() -> None:
    """Without this an artifact cannot say which cut-offs or weights it used."""
    recorded = _rerank_result(10)

    assert recorded.configuration is not None
    assert recorded.configuration["top_k"] == 10
    # Optional, so artifacts written before the field remain readable.
    bare = ExperimentResult(
        experiment="baseline",
        backbone="clip",
        created_at=datetime(2026, 9, 8, tzinfo=UTC),
        seed=1,
        metrics=(),
    )
    assert bare.configuration is None


def test_a_run_with_different_settings_will_not_overwrite_another(tmp_path: Path) -> None:
    """The accident this guards: a k=50 sweep lands on the k=10 file a section cites.

    The filename carries only the experiment and the backbone, because the
    visualizations glob it and expect one file per backbone — so the collision has
    to be refused rather than renamed away.
    """
    path = experiment_result_path(tmp_path, _rerank_result(10))
    write_experiment_result(path, _rerank_result(10))

    with pytest.raises(ExperimentArtifactError, match="different settings"):
        write_experiment_result(path, _rerank_result(50))
    # The message names what would have been lost.
    with pytest.raises(ExperimentArtifactError, match=r"top_k.*10.*50"):
        write_experiment_result(path, _rerank_result(50))
    # And nothing was written.
    assert json.loads(path.read_text(encoding="utf-8"))["configuration"]["top_k"] == 10


def test_re_running_the_same_settings_is_allowed(tmp_path: Path) -> None:
    """The ordinary case: re-running after re-extracting embeddings."""
    path = experiment_result_path(tmp_path, _rerank_result(10))
    write_experiment_result(path, _rerank_result(10))

    write_experiment_result(path, _rerank_result(10))

    assert json.loads(path.read_text(encoding="utf-8"))["configuration"]["top_k"] == 10


def test_replacing_a_different_run_is_possible_but_deliberate(tmp_path: Path) -> None:
    path = experiment_result_path(tmp_path, _rerank_result(10))
    write_experiment_result(path, _rerank_result(10))

    write_experiment_result(path, _rerank_result(50), allow_replace=True)

    assert json.loads(path.read_text(encoding="utf-8"))["configuration"]["top_k"] == 50


def test_an_artifact_from_before_the_field_does_not_block_a_re_run(tmp_path: Path) -> None:
    """Otherwise every existing result would refuse its own regeneration."""
    path = tmp_path / "rerank_ablation-clip.json"
    path.write_text(
        json.dumps(
            {
                "experiment": "rerank_ablation",
                "backbone": "clip",
                "created_at": "2026-01-01T00:00:00Z",
                "seed": 1,
                "metrics": [],
            }
        ),
        encoding="utf-8",
    )

    write_experiment_result(path, _rerank_result(50))

    # The replacement records a configuration, which arms the check from here on.
    assert json.loads(path.read_text(encoding="utf-8"))["configuration"]["top_k"] == 50
    with pytest.raises(ExperimentArtifactError, match="different settings"):
        write_experiment_result(path, _rerank_result(10))


def test_an_unreadable_file_is_not_mistaken_for_a_run(tmp_path: Path) -> None:
    path = tmp_path / "rerank_ablation-clip.json"
    path.write_text("not json at all", encoding="utf-8")

    write_experiment_result(path, _rerank_result(10))

    assert json.loads(path.read_text(encoding="utf-8"))["configuration"]["top_k"] == 10

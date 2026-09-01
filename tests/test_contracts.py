"""Smoke tests for scaffolded interfaces and dependency boundaries."""

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
from krasnal_id.experiments.confusion_analysis import run_confusion_analysis
from krasnal_id.experiments.contracts import ExperimentResult, MetricSummary
from krasnal_id.retrieval.knn import RetrievalMatch, RetrievalResult, cosine_knn
from krasnal_id.viz.embedding_plot import create_embedding_plot


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


def test_later_version_placeholders_are_explicit() -> None:
    config = load_config()

    with pytest.raises(NotImplementedError):
        run_confusion_analysis(config)
    with pytest.raises(NotImplementedError):
        create_embedding_plot(config)
    with pytest.raises(NotImplementedError):
        launch()

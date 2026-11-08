"""The second internal review's checks (AGENTS.md section 7.18)."""

from pathlib import Path

import numpy as np
import pytest

from krasnal_id.config import backbone_config
from krasnal_id.embeddings.regions import (
    RegionExtractionError,
    late_interaction,
    load_regions,
    pool_regions,
    region_count,
)
from krasnal_id.experiments.photographer_gap import evaluate_conditions
from krasnal_id.experiments.rerank_ablation import Candidate, QueryEvidence
from krasnal_id.experiments.second_review import (
    cosine_margin,
    gated_part,
    image_scores,
    multi_region_part,
    strata_part,
)
from krasnal_id.experiments.workshop_validation import (
    COSINE,
    HeldOutQuery,
    WorkshopValidationError,
)
from krasnal_id.models import DatasetManifest
from test_reviewer_checks import _located_world, _settings


def test_regions_are_the_cls_then_every_grid_cell_normalized() -> None:
    rng = np.random.default_rng(0)
    cls = rng.normal(size=(2, 8))
    patches = rng.normal(size=(2, 6, 6, 8))
    regions = pool_regions(cls, patches)
    assert regions.shape == (2, region_count(), 8) == (2, 14, 8)
    assert np.allclose(np.linalg.norm(regions, axis=2), 1.0, atol=1e-6)
    assert np.allclose(regions[:, 0], cls / np.linalg.norm(cls, axis=1, keepdims=True), atol=1e-6)
    # The first 2x2 cell is the mean of the top-left 3x3 block of patches.
    cell = patches[:, :3, :3].mean(axis=(1, 2))
    assert np.allclose(regions[:, 1], cell / np.linalg.norm(cell, axis=1, keepdims=True), atol=1e-6)
    with pytest.raises(RegionExtractionError, match="needs at least"):
        pool_regions(cls, rng.normal(size=(2, 2, 2, 8)))


def test_image_scores_are_late_interaction_for_every_pair() -> None:
    rng = np.random.default_rng(1)
    raw = rng.normal(size=(5, 4, 6))
    regions = (raw / np.linalg.norm(raw, axis=2, keepdims=True)).astype(np.float32)
    scores = image_scores(regions)
    for query in range(5):
        assert np.allclose(scores[query], late_interaction(regions[query], regions), atol=1e-5)
    # An image matches itself perfectly in every region.
    assert np.allclose(np.diag(scores), 1.0, atol=1e-5)


def test_stale_region_files_are_refused(tmp_path: Path) -> None:
    path = tmp_path / "regions.npz"
    config = backbone_config("dinov2")
    np.savez(
        path,
        image_ids=np.asarray(["a", "b"]),
        regions=np.zeros((2, 14, 4), dtype=np.float32),
        model_id=np.asarray("someone/else"),
        revision=np.asarray(config.revision),
    )
    with pytest.raises(RegionExtractionError, match="was written by"):
        load_regions(path, ["a"], config)
    np.savez(
        path,
        image_ids=np.asarray(["a"]),
        regions=np.ones((1, 14, 4), dtype=np.float32),
        model_id=np.asarray(config.model_id),
        revision=np.asarray(config.revision),
    )
    assert load_regions(path, ["a"], config).shape == (1, 14, 4)
    with pytest.raises(RegionExtractionError, match="lacks 1"):
        load_regions(path, ["a", "b"], config)


def _two_photographers(manifest: DatasetManifest) -> DatasetManifest:
    return manifest.model_copy(
        update={
            "images": tuple(
                image.model_copy(
                    update={"author": "Bo" if image.image_id.endswith(("-1", "-3")) else "Al"}
                )
                for image in manifest.images
            )
        }
    )


def _released(manifest: DatasetManifest, split, matrix) -> dict[str, float]:  # type: ignore[no-untyped-def]
    outcomes, _ = evaluate_conditions(split, manifest, matrix, (0,))
    return {
        f"{name}_top_1": sum(rank == 1 for rank in outcome.ranks) / len(outcome.ranks)
        for name, outcome in outcomes.items()
    }


def test_strata_cover_the_disjoint_arm_and_reproduce_it(tmp_path: Path) -> None:
    manifest, split, matrix = _located_world(tmp_path)
    manifest = _two_photographers(manifest)
    released = _released(manifest, split, matrix)
    metrics = {
        m.name: m for m in strata_part("dinov2", split, manifest, matrix, released, _settings())
    }
    counted = sum(m.value for n, m in metrics.items() if n.endswith("_queries"))
    assert counted == len(split.folds)
    with pytest.raises(WorkshopValidationError, match="inputs have changed"):
        strata_part("dinov2", split, manifest, matrix, {"disjoint_top_1": 0.0}, _settings())


def test_a_cls_only_region_set_reproduces_cosine(tmp_path: Path) -> None:
    manifest, split, matrix = _located_world(tmp_path)
    manifest = _two_photographers(manifest)
    released = _released(manifest, split, matrix)
    vectors = np.stack([matrix.vector_for(i) for i in matrix.image_ids])
    # Every region is the CLS vector, so late interaction is cosine exactly.
    regions = np.repeat(vectors[:, None, :], 14, axis=1).astype(np.float32)
    baseline = {"top_1": released["standard_top_1"]}
    metrics = {
        m.name: m
        for m in multi_region_part(
            "dinov2", split, manifest, matrix, regions, baseline, released, _settings()
        )
    }
    for arm in ("full", "disjoint"):
        for method in ("grid5", "grid14"):
            gain = metrics[f"dinov2_mr_{arm}_{method}_vs_cosine_top_1_gain"]
            assert gain.value == pytest.approx(0.0, abs=1e-12)


def _evidence(query: str, cosines: tuple[float, ...]) -> QueryEvidence:
    return QueryEvidence(
        query_image_id=query,
        candidates=tuple(
            Candidate(dwarf_id=f"S{i}", cosine=c, inliers=0, correct=i == 0)
            for i, c in enumerate(cosines)
        ),
        baseline_rank=1,
    )


def test_gating_runs_from_cosine_to_full_verification() -> None:
    evidence = {
        "disk-lightglue": {
            "q1": _evidence("q1", (0.90, 0.10)),
            "q2": _evidence("q2", (0.50, 0.49)),
        }
    }
    held_out = [
        HeldOutQuery("q1", "A", 0, {COSINE: 2, "disk-lightglue": 1}),
        HeldOutQuery("q2", "B", 1, {COSINE: 2, "disk-lightglue": 1}),
    ]
    settings = _settings(gate_fractions=(0.0, 0.5, 1.0))
    metrics = {m.name: m.value for m in gated_part(held_out, evidence, settings)}
    assert metrics["cv_gate_disk-lightglue_0_top_1"] == 0.0
    # Half verified means the least certain query, q2, takes the matcher's rank.
    assert metrics["cv_gate_disk-lightglue_50_top_1"] == 0.5
    assert metrics["cv_gate_disk-lightglue_100_top_1"] == 1.0
    assert cosine_margin(_evidence("q", (0.5,))) == float("inf")

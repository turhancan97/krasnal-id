"""Geometric verification of the top candidates, and the sweep that measures it."""

import json
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("cv2")

from typer.testing import CliRunner

from helpers import FAKE_BACKBONE, seed_embedding_cache, synthetic_manifest
from krasnal_id.cli import app
from krasnal_id.config import RerankAblationConfig, load_config
from krasnal_id.data_pipeline.build_split import (
    build_evaluation_split,
    write_evaluation_split,
)
from krasnal_id.embeddings.store import load_embedding_matrix
from krasnal_id.experiments.rerank_ablation import (
    Candidate,
    QueryEvidence,
    RerankAblationError,
    collect_evidence,
    run_rerank_ablation,
    separation,
    summarize,
    summarize_arms,
)
from krasnal_id.models import DatasetManifest
from krasnal_id.retrieval.rerank import (
    INLIER_CAP,
    FeatureCache,
    RerankError,
    blended_score,
    count_inliers,
    extract_features,
)

runner = CliRunner()


def _textured(path: Path, seed: int, size: int = 160) -> None:
    """Write an image with enough texture for a keypoint detector to work on.

    Flat colour yields no keypoints, so the synthetic images here are noise:
    it is the cheapest thing SIFT can actually describe.
    """
    from PIL import Image

    rng = np.random.default_rng(seed)
    pixels = rng.integers(0, 256, size=(size, size, 3), dtype=np.uint8)
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(pixels).save(path)


def _manifest_with_images(tmp_path: Path, dwarfs: int = 3) -> DatasetManifest:
    """A manifest whose images exist on disk, one distinct texture per class."""
    manifest = synthetic_manifest(dwarf_count=dwarfs, per_dwarf=3)
    records = []
    for index, image in enumerate(manifest.images):
        path = tmp_path / "images" / image.dwarf_id / f"{image.image_id}.png"
        # Same seed within a class, so a class's images verify against each other.
        _textured(path, seed=int(image.dwarf_id.removeprefix("Q")))
        records.append(image.model_copy(update={"local_path": path, "width": 160, "height": 160}))
        del index
    return manifest.model_copy(update={"images": tuple(records)})


def test_a_blend_weight_of_zero_changes_nothing() -> None:
    """The control property the whole sweep is read against."""
    assert blended_score(0.9, 0, 0.0) == 0.9
    assert blended_score(0.9, 500, 0.0) == 0.9
    # Above the cap, more inliers add nothing, so one spectacular match cannot
    # dominate a blended score.
    assert blended_score(0.5, INLIER_CAP, 1.0) == blended_score(0.5, INLIER_CAP * 10, 1.0)
    assert blended_score(0.5, INLIER_CAP, 1.0) == pytest.approx(1.5)
    with pytest.raises(RerankError, match="cannot be negative"):
        blended_score(0.9, 5, -0.1)


def test_an_image_verifies_against_itself_and_not_against_noise(tmp_path: Path) -> None:
    first, second = tmp_path / "a.png", tmp_path / "b.png"
    _textured(first, seed=1)
    _textured(second, seed=2)

    features = extract_features(first, 400)
    other = extract_features(second, 400)

    assert len(features) > 20
    assert count_inliers(features, features) > count_inliers(features, other)
    # An image with nothing to describe is absence of evidence, not a mismatch.
    empty = extract_features(first, 400).__class__(
        points=np.zeros((0, 2), dtype=np.float32),
        descriptors=np.zeros((0, 128), dtype=np.uint8),
    )
    assert count_inliers(features, empty) == 0
    assert count_inliers(empty, features) == 0
    with pytest.raises(RerankError, match="could not read"):
        extract_features(tmp_path / "absent.png", 400)


def test_features_are_described_once_per_image(tmp_path: Path) -> None:
    """A candidate is proposed for many queries; describing it each time would
    dominate the cost."""
    path = tmp_path / "a.png"
    _textured(path, seed=3)
    cache = FeatureCache(200)

    first = cache.get("a", path)
    again = cache.get("a", path)

    assert first is again
    assert len(cache) == 1


def test_the_sweep_reports_what_each_weight_moved() -> None:
    truths = {"q1": "Q1", "q2": "Q2", "q3": "Q3"}
    evidence = (
        # Correct already first, and geometry agrees: nothing to move.
        QueryEvidence(
            "q1",
            (Candidate("Q1", 0.9, 30, True), Candidate("Q2", 0.8, 0, False)),
            baseline_rank=1,
        ),
        # Correct second on cosine but verifies strongly: a weight can promote it.
        QueryEvidence(
            "q2",
            (Candidate("Q3", 0.9, 0, False), Candidate("Q2", 0.85, 30, True)),
            baseline_rank=2,
        ),
        # Correct first on cosine but verifies at zero while a rival verifies
        # well: too large a weight demotes it.
        QueryEvidence(
            "q3",
            (Candidate("Q3", 0.9, 0, True), Candidate("Q1", 0.85, 30, False)),
            baseline_rank=1,
        ),
    )

    metrics = {m.name: m.value for m in summarize(evidence, truths, (0.0, 0.2), (1,))}

    assert metrics["weight_0_top_1"] == pytest.approx(2 / 3)
    assert metrics["weight_0_promoted"] == 0
    assert metrics["weight_0_demoted"] == 0
    # At 0.2 the second query is promoted and the third demoted, so the net is nil
    # while two queries changed — which the net alone would hide.
    assert metrics["weight_0.2_promoted"] == 1
    assert metrics["weight_0.2_demoted"] == 1
    assert metrics["weight_0.2_top_1"] == pytest.approx(2 / 3)
    assert metrics["median_inliers_correct"] == pytest.approx(30.0)
    assert metrics["median_inliers_wrong"] == pytest.approx(0.0)
    assert metrics["queries"] == 3


def test_the_sweep_refuses_to_run_without_its_control() -> None:
    truths = {"q1": "Q1"}
    evidence = (QueryEvidence("q1", (Candidate("Q1", 0.9, 5, True),), baseline_rank=1),)

    with pytest.raises(RerankAblationError, match=r"must include weight 0\.0"):
        summarize(evidence, truths, (0.1, 0.2), (1,))
    with pytest.raises(RerankAblationError, match="cannot be negative"):
        summarize(evidence, truths, (0.0, -0.1), (1,))
    with pytest.raises(RerankAblationError, match="no queries were scored"):
        summarize((), truths, (0.0,), (1,))


def test_a_query_whose_answer_is_outside_the_top_k_is_counted_not_dropped() -> None:
    truths = {"q1": "Q9"}
    evidence = (QueryEvidence("q1", (Candidate("Q1", 0.9, 0, False),), baseline_rank=0),)

    metrics = {m.name: m.value for m in summarize(evidence, truths, (0.0,), (1,))}

    assert metrics["truth_outside_top_k"] == 1
    assert metrics["weight_0_top_1"] == 0.0
    # It contributes zero reciprocal rank rather than leaving the denominator,
    # so the column stays comparable with the baseline.
    assert metrics["weight_0_mrr"] == 0.0


def test_evidence_is_gathered_for_the_top_candidates_only(tmp_path: Path) -> None:
    manifest = _manifest_with_images(tmp_path)
    seed_embedding_cache(tmp_path / "embeddings", manifest)
    matrix = load_embedding_matrix(manifest, FAKE_BACKBONE, tmp_path / "embeddings")
    split = build_evaluation_split(manifest, datetime.now(UTC))

    evidence = collect_evidence(split, manifest, matrix, FeatureCache(200), top_k=2)

    assert len(evidence) == len(split.folds)
    assert all(len(item.candidates) == 2 for item in evidence)
    # Each class's images share a texture, so the correct statue verifies and the
    # others do not.
    correct, wrong = separation(evidence)
    assert correct > wrong

    with pytest.raises(RerankAblationError, match="at least two candidates"):
        collect_evidence(split, manifest, matrix, FeatureCache(200), top_k=1)


def _by_two_photographers(manifest: DatasetManifest) -> DatasetManifest:
    """Credit each class's third image to a second photographer."""
    return manifest.model_copy(
        update={
            "images": tuple(
                image.model_copy(
                    update={"author": "Bruno" if image.image_id.endswith("-2") else "Ada"}
                )
                for image in manifest.images
            )
        }
    )


def test_the_disjoint_arm_scores_the_same_queries_on_fewer_references(tmp_path: Path) -> None:
    """Both arms must cover one query set, or the two columns are not comparable."""
    manifest = _by_two_photographers(_manifest_with_images(tmp_path))
    seed_embedding_cache(tmp_path / "embeddings", manifest)
    matrix = load_embedding_matrix(manifest, FAKE_BACKBONE, tmp_path / "embeddings")
    split = build_evaluation_split(manifest, datetime.now(UTC))
    cache = FeatureCache(200)

    everything = collect_evidence(split, manifest, matrix, cache, top_k=2, only_answerable=True)
    disjoint = collect_evidence(split, manifest, matrix, cache, top_k=2, photographer_disjoint=True)

    # Every class has two photographers, so nothing is dropped and the arms match.
    assert len(everything) == len(disjoint) == len(split.folds)
    assert [e.query_image_id for e in everything] == [d.query_image_id for d in disjoint]
    # Features are described once and reused across both arms.
    assert len(cache) == len(manifest.images)


def test_a_single_photographer_class_is_dropped_from_both_arms(tmp_path: Path) -> None:
    manifest = _manifest_with_images(tmp_path)  # one shared author throughout
    seed_embedding_cache(tmp_path / "embeddings", manifest)
    matrix = load_embedding_matrix(manifest, FAKE_BACKBONE, tmp_path / "embeddings")
    split = build_evaluation_split(manifest, datetime.now(UTC))

    disjoint = collect_evidence(
        split, manifest, matrix, cache=FeatureCache(200), top_k=2, photographer_disjoint=True
    )

    # No class can be answered without its only photographer.
    assert disjoint == ()


def test_each_arm_keeps_its_own_control() -> None:
    """A gain must be read against the regime that produced it.

    The disjoint arm's baseline is not the ordinary one, so prefixing the metrics
    per arm is what stops the two being compared to the wrong control.
    """
    truths = {"q1": "Q1"}
    arms = {
        "all": (QueryEvidence("q1", (Candidate("Q1", 0.9, 40, True),), baseline_rank=1),),
        "disjoint": (QueryEvidence("q1", (Candidate("Q2", 0.7, 0, False),), baseline_rank=0),),
    }

    metrics = {m.name: m.value for m in summarize_arms(arms, truths, (0.0,), (1,))}

    assert metrics["all_weight_0_top_1"] == 1.0
    assert metrics["disjoint_weight_0_top_1"] == 0.0
    assert metrics["all_median_inliers_correct"] == 40.0
    assert metrics["disjoint_truth_outside_top_k"] == 1.0
    # A single unnamed arm keeps the unprefixed names, so the existing artifacts
    # and their schema are unchanged.
    plain = {m.name for m in summarize_arms({"": arms["all"]}, truths, (0.0,), (1,))}
    assert "weight_0_top_1" in plain


def test_a_missing_photograph_stops_the_run(tmp_path: Path) -> None:
    """Geometry reads the pixels, so a vector-only dataset is not enough."""
    manifest = _manifest_with_images(tmp_path)
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(manifest.model_dump(mode="json")), encoding="utf-8")
    split_path = tmp_path / "splits" / "leave-one-out.json"
    write_evaluation_split(split_path, build_evaluation_split(manifest, datetime.now(UTC)))
    seed_embedding_cache(tmp_path / "embeddings", manifest)
    manifest.images[0].local_path.unlink()

    config = load_config(
        [
            "experiment=rerank_ablation",
            f"paths.manifest_path={manifest_path}",
            f"paths.evaluation_split_path={split_path}",
            f"paths.embeddings_dir={tmp_path / 'embeddings'}",
            f"backbone.model_id={FAKE_BACKBONE.model_id}",
            f"backbone.revision={FAKE_BACKBONE.revision}",
            f"backbone.preprocessing_id={FAKE_BACKBONE.preprocessing_id}",
        ]
    )
    with pytest.raises(RerankAblationError, match="geometric verification reads"):
        run_rerank_ablation(config)


def test_run_requires_the_rerank_experiment_group() -> None:
    with pytest.raises(RerankAblationError, match="requires experiment=rerank_ablation"):
        run_rerank_ablation(load_config(["experiment=baseline"]))


def test_packaged_defaults_are_usable() -> None:
    experiment = load_config(["experiment=rerank_ablation"]).experiment
    assert isinstance(experiment, RerankAblationConfig)
    assert 0.0 in experiment.weights, "the sweep is read against its own control"
    assert experiment.top_k >= 2
    # Off by default: the composed protocol is a deliberate second run.
    assert experiment.photographer_disjoint is False
    for broken, message in (
        ({"weights": (0.1,)}, r"must include weight 0\.0"),
        ({"weights": (0.0, -1.0)}, "cannot be negative"),
        ({"top_k_metrics": (0,)}, "must be positive"),
    ):
        with pytest.raises(ValueError, match=message):
            RerankAblationConfig.model_validate(
                {
                    "kind": "rerank_ablation",
                    "seed": 1,
                    "top_k": 10,
                    "max_keypoints": 100,
                    "weights": (0.0, 0.1),
                    "top_k_metrics": (1,),
                    **broken,
                }
            )


def test_cli_sweeps_and_writes_the_artifact(tmp_path: Path) -> None:
    manifest = _manifest_with_images(tmp_path)
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(manifest.model_dump(mode="json")), encoding="utf-8")
    split_path = tmp_path / "splits" / "leave-one-out.json"
    write_evaluation_split(split_path, build_evaluation_split(manifest, datetime.now(UTC)))
    seed_embedding_cache(tmp_path / "embeddings", manifest)

    result = runner.invoke(
        app,
        [
            "experiment",
            "rerank",
            *[
                f"-o{value}"
                for value in (
                    f"paths.manifest_path={manifest_path}",
                    f"paths.evaluation_split_path={split_path}",
                    f"paths.embeddings_dir={tmp_path / 'embeddings'}",
                    f"paths.results_dir={tmp_path / 'results'}",
                    f"backbone.model_id={FAKE_BACKBONE.model_id}",
                    f"backbone.revision={FAKE_BACKBONE.revision}",
                    f"backbone.preprocessing_id={FAKE_BACKBONE.preprocessing_id}",
                    "experiment.top_k=3",
                    "experiment.max_keypoints=120",
                    "logging.json_output=false",
                )
            ],
        ],
    )

    assert result.exit_code == 0, result.output
    assert "weight_0_top_1" in result.output
    assert "median_inliers_correct" in result.output
    written = json.loads((tmp_path / "results" / "rerank_ablation-dinov2.json").read_text())
    assert written["experiment"] == "rerank_ablation"

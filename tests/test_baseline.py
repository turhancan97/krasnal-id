"""Embedding-store access and full-pool baseline evaluation."""

import json
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pytest
from typer.testing import CliRunner

from krasnal_id.cli import app
from krasnal_id.config import BackboneConfig, load_config
from krasnal_id.data_pipeline.build_manifest import canonical_json_sha256
from krasnal_id.data_pipeline.build_split import build_evaluation_split, write_evaluation_split
from krasnal_id.embeddings.cache import EmbeddingCache
from krasnal_id.embeddings.store import (
    EmbeddingMatrix,
    EmbeddingStoreError,
    cache_key_for,
    load_embedding_matrix,
)
from krasnal_id.experiments.artifacts import experiment_result_path, write_experiment_result
from krasnal_id.experiments.baseline_accuracy import (
    BaselineExperimentError,
    evaluate_baseline,
    evaluate_fold,
    load_evaluation_inputs,
    run_baseline,
)
from krasnal_id.experiments.contracts import ExperimentResult, MetricSummary
from krasnal_id.models import DatasetManifest, DwarfRecord, ImageRecord

_BACKBONE = BackboneConfig(
    name="dinov2",
    model_id="fake/model",
    revision="fake-revision",
    preprocessing_id="fake-processor",
    batch_size=4,
)


def _image_record(image_id: str, dwarf_id: str, digest_seed: str) -> ImageRecord:
    return ImageRecord(
        image_id=image_id,
        dwarf_id=dwarf_id,
        local_path=Path(f"data/images/{dwarf_id}/{image_id}.jpg"),
        source_url="https://commons.wikimedia.org/wiki/File:Example.jpg",
        author="Author",
        license="CC BY-SA 4.0",
        license_url="https://creativecommons.org/licenses/by-sa/4.0/",
        sha256=(digest_seed * 64)[:64],
        width=800,
        height=600,
        acquired_at="2026-08-23T12:00:00Z",
    )


def _manifest(dwarf_count: int = 3, per_dwarf: int = 3) -> DatasetManifest:
    dwarfs = tuple(
        DwarfRecord(
            dwarf_id=f"Q{index}",
            display_name=f"Dwarf {index}",
            wikidata_url=f"https://www.wikidata.org/wiki/Q{index}",
            commons_category=f"Dwarf {index}",
        )
        for index in range(dwarf_count)
    )
    images = tuple(
        _image_record(f"image-{dwarf}-{position}", f"Q{dwarf}", f"{dwarf}{position}")
        for dwarf in range(dwarf_count)
        for position in range(per_dwarf)
    )
    return DatasetManifest(
        schema_version="1.0",
        source_query_sha256="a" * 64,
        staging_sha256="b" * 64,
        image_review_sha256="c" * 64,
        generated_at="2026-08-23T12:00:00Z",
        minimum_images_per_dwarf=3,
        dwarfs=dwarfs,
        images=images,
    )


def _tight_cluster_vector(dwarf_index: int, position: int, dwarf_count: int) -> np.ndarray:
    """Place each dwarf on its own axis, with a small per-image perturbation."""
    vector = np.zeros(dwarf_count + 1, dtype=np.float32)
    vector[dwarf_index] = 1.0
    vector[dwarf_count] = 0.01 * (position + 1)
    return vector / np.linalg.norm(vector)


def _seed_cache(
    cache_root: Path,
    manifest: DatasetManifest,
    vector_for: object,
    skip: tuple[str, ...] = (),
) -> None:
    cache = EmbeddingCache(cache_root)
    dwarf_ids = sorted({image.dwarf_id for image in manifest.images})
    for record in manifest.images:
        if record.image_id in skip:
            continue
        position = [
            image.image_id for image in manifest.images if image.dwarf_id == record.dwarf_id
        ].index(record.image_id)
        vector = vector_for(dwarf_ids.index(record.dwarf_id), position, len(dwarf_ids))  # type: ignore[operator]
        cache.store(cache_key_for(record, _BACKBONE), np.asarray(vector, dtype=np.float32))


def test_embedding_matrix_is_ordered_and_addressable(tmp_path: Path) -> None:
    manifest = _manifest()
    _seed_cache(tmp_path, manifest, _tight_cluster_vector)

    matrix = load_embedding_matrix(manifest, _BACKBONE, tmp_path)

    assert matrix.image_ids == tuple(sorted(matrix.image_ids))
    assert len(matrix.image_ids) == len(matrix.dwarf_ids) == matrix.vectors.shape[0] == 9
    assert matrix.vectors.dtype == np.dtype(np.float32)
    # Every row must stay paired with its own dwarf after the sort.
    for image_id, dwarf_id in zip(matrix.image_ids, matrix.dwarf_ids, strict=True):
        assert image_id.startswith(f"image-{dwarf_id[1:]}-")

    vectors, dwarf_ids = matrix.rows_for(["image-2-0", "image-0-1"])
    assert dwarf_ids == ("Q2", "Q0")
    np.testing.assert_allclose(vectors[0], matrix.vector_for("image-2-0"))
    with pytest.raises(EmbeddingStoreError, match="has no cached vector"):
        matrix.index_of("absent")


def test_missing_vectors_name_the_extraction_command(tmp_path: Path) -> None:
    manifest = _manifest()
    _seed_cache(tmp_path, manifest, _tight_cluster_vector, skip=("image-1-1",))

    with pytest.raises(EmbeddingStoreError) as error:
        load_embedding_matrix(manifest, _BACKBONE, tmp_path)

    assert "1 of 9 images have no cached dinov2 vector" in str(error.value)
    assert "image-1-1" in str(error.value)
    assert "embeddings extract" in str(error.value)


def test_empty_manifest_and_mixed_dimensions_are_rejected(tmp_path: Path) -> None:
    empty = DatasetManifest(
        schema_version="1.0",
        source_query_sha256="a" * 64,
        staging_sha256="b" * 64,
        image_review_sha256="c" * 64,
        generated_at="2026-08-23T12:00:00Z",
        minimum_images_per_dwarf=3,
        dwarfs=(),
        images=(),
    )
    with pytest.raises(EmbeddingStoreError, match="no images"):
        load_embedding_matrix(empty, _BACKBONE, tmp_path)

    manifest = _manifest()
    cache = EmbeddingCache(tmp_path)
    for index, record in enumerate(manifest.images):
        size = 4 if index else 3
        vector = np.zeros(size, dtype=np.float32)
        vector[0] = 1.0
        cache.store(cache_key_for(record, _BACKBONE), vector)

    with pytest.raises(EmbeddingStoreError, match="mixed dimensions"):
        load_embedding_matrix(manifest, _BACKBONE, tmp_path)


def test_fold_separates_dwarf_rank_from_image_rank() -> None:
    # image-b and image-c are nearer the query than any image of its own dwarf Q0,
    # but they belong to a single dwarf, so the dwarf rank stays ahead of the image rank.
    matrix = EmbeddingMatrix(
        image_ids=("image-a", "image-b", "image-c", "query"),
        dwarf_ids=("Q0", "Q1", "Q1", "Q0"),
        vectors=np.asarray(
            [[0.5, 0.5], [0.9, 0.1], [0.8, 0.2], [1.0, 0.0]],
            dtype=np.float32,
        ),
    )

    outcome = evaluate_fold("query", "Q0", ("image-a", "image-b", "image-c"), matrix)

    assert outcome.image_rank == 3
    assert outcome.dwarf_rank == 2
    assert outcome.candidate_dwarfs == 2


def test_fold_rejects_a_dwarf_with_no_reference_image() -> None:
    matrix = EmbeddingMatrix(
        image_ids=("image-a", "query"),
        dwarf_ids=("Q1", "Q0"),
        vectors=np.asarray([[0.0, 1.0], [1.0, 0.0]], dtype=np.float32),
    )

    with pytest.raises(BaselineExperimentError, match="no reference image"):
        evaluate_fold("query", "Q0", ("image-a",), matrix)


def test_perfectly_separated_embeddings_score_a_flawless_baseline(tmp_path: Path) -> None:
    manifest = _manifest()
    _seed_cache(tmp_path, manifest, _tight_cluster_vector)
    matrix = load_embedding_matrix(manifest, _BACKBONE, tmp_path)
    split = build_evaluation_split(manifest, datetime.now(UTC))

    metrics = {metric.name: metric for metric in evaluate_baseline(split, matrix, (1, 5))}

    assert metrics["top_1"].value == pytest.approx(1.0)
    assert metrics["mrr"].value == pytest.approx(1.0)
    assert metrics["image_mrr"].value == pytest.approx(1.0)
    assert metrics["evaluated_folds"].value == pytest.approx(9.0)
    assert metrics["candidate_dwarfs"].value == pytest.approx(3.0)
    # A proportion carries Wilson error bars that bracket it; a rank average does not.
    assert metrics["top_1"].lower_bound is not None
    assert metrics["top_1"].lower_bound < 1.0
    assert metrics["top_1"].upper_bound == pytest.approx(1.0)
    assert metrics["mrr"].lower_bound is None


def test_indistinguishable_embeddings_score_by_deterministic_tie_break(tmp_path: Path) -> None:
    manifest = _manifest()
    _seed_cache(tmp_path, manifest, lambda *_: np.asarray([1.0, 0.0], dtype=np.float32))
    matrix = load_embedding_matrix(manifest, _BACKBONE, tmp_path)
    split = build_evaluation_split(manifest, datetime.now(UTC))

    first = evaluate_baseline(split, matrix, (1,))
    second = evaluate_baseline(split, matrix, (1,))

    assert first == second
    accuracy = next(metric for metric in first if metric.name == "top_1")
    # Chance level, not a crash: identical vectors rank purely by image ID.
    assert 0.0 <= accuracy.value <= 1.0
    lower, upper = accuracy.lower_bound, accuracy.upper_bound
    assert lower is not None and upper is not None and lower < accuracy.value < upper


def test_baseline_rejects_invalid_top_k_and_an_empty_split(tmp_path: Path) -> None:
    manifest = _manifest()
    _seed_cache(tmp_path, manifest, _tight_cluster_vector)
    matrix = load_embedding_matrix(manifest, _BACKBONE, tmp_path)
    split = build_evaluation_split(manifest, datetime.now(UTC))

    with pytest.raises(BaselineExperimentError, match="must be positive"):
        evaluate_baseline(split, matrix, (1, 0))
    with pytest.raises(BaselineExperimentError, match="no folds"):
        evaluate_baseline(split.model_copy(update={"folds": ()}), matrix, (1,))


def test_a_split_built_for_another_manifest_is_refused(tmp_path: Path) -> None:
    manifest = _manifest()
    manifest_path = tmp_path / "manifest.json"
    split_path = tmp_path / "split.json"
    manifest_path.write_text(json.dumps(manifest.model_dump(mode="json")), encoding="utf-8")
    write_evaluation_split(split_path, build_evaluation_split(manifest, datetime.now(UTC)))

    loaded_manifest, loaded_split = load_evaluation_inputs(manifest_path, split_path)
    assert loaded_split.manifest_sha256 == canonical_json_sha256(
        loaded_manifest.model_dump(mode="json")
    )

    changed = manifest.model_copy(update={"images": manifest.images[:-1]})
    manifest_path.write_text(json.dumps(changed.model_dump(mode="json")), encoding="utf-8")
    with pytest.raises(BaselineExperimentError, match="rebuild it with"):
        load_evaluation_inputs(manifest_path, split_path)


def test_unreadable_and_invalid_inputs_are_reported(tmp_path: Path) -> None:
    manifest_path = tmp_path / "manifest.json"
    split_path = tmp_path / "split.json"

    with pytest.raises(BaselineExperimentError, match="invalid manifest"):
        load_evaluation_inputs(manifest_path, split_path)

    manifest_path.write_text("{}", encoding="utf-8")
    with pytest.raises(BaselineExperimentError, match="invalid evaluation inputs"):
        load_evaluation_inputs(manifest_path, split_path)


def test_run_baseline_requires_the_baseline_experiment_group() -> None:
    with pytest.raises(BaselineExperimentError, match="requires experiment=baseline"):
        run_baseline(load_config(["experiment=confusion"]))


def test_result_artifact_is_named_by_experiment_and_backbone(tmp_path: Path) -> None:
    result = ExperimentResult(
        experiment="baseline",
        backbone="clip",
        created_at=datetime.now(UTC),
        seed=42,
        metrics=(MetricSummary(name="top_1", value=0.5, lower_bound=0.2, upper_bound=0.8),),
    )
    path = experiment_result_path(tmp_path, result)
    assert path.name == "baseline-clip.json"

    write_experiment_result(path, result)
    reloaded = ExperimentResult.model_validate(json.loads(path.read_text(encoding="utf-8")))
    assert reloaded == result
    assert not list(tmp_path.glob(".*.tmp"))

    # A rerun must overwrite in place rather than accumulate files.
    write_experiment_result(path, result)
    assert len(list(tmp_path.glob("*.json"))) == 1


def test_cli_baseline_reports_metrics_and_fails_without_embeddings(tmp_path: Path) -> None:
    manifest = _manifest()
    manifest_path = tmp_path / "manifest.json"
    split_path = tmp_path / "splits" / "leave-one-out.json"
    manifest_path.write_text(json.dumps(manifest.model_dump(mode="json")), encoding="utf-8")
    write_evaluation_split(split_path, build_evaluation_split(manifest, datetime.now(UTC)))
    overrides = [
        f"paths.manifest_path={manifest_path}",
        f"paths.evaluation_split_path={split_path}",
        f"paths.embeddings_dir={tmp_path / 'embeddings'}",
        f"paths.results_dir={tmp_path / 'results'}",
        f"backbone.model_id={_BACKBONE.model_id}",
        f"backbone.revision={_BACKBONE.revision}",
        f"backbone.preprocessing_id={_BACKBONE.preprocessing_id}",
    ]
    runner = CliRunner()

    missing = runner.invoke(app, ["experiment", "baseline", *[f"-o{value}" for value in overrides]])
    assert missing.exit_code == 2
    assert "Baseline experiment error" in missing.output

    _seed_cache(tmp_path / "embeddings", manifest, _tight_cluster_vector)
    result = runner.invoke(app, ["experiment", "baseline", *[f"-o{value}" for value in overrides]])
    assert result.exit_code == 0, result.output
    assert "top_1: 1.0000" in result.output
    assert "95% CI" in result.output
    assert "mrr: 1.0000" in result.output

    written = ExperimentResult.model_validate(
        json.loads((tmp_path / "results" / "baseline-dinov2.json").read_text(encoding="utf-8"))
    )
    assert written.seed == 42
    assert {metric.name for metric in written.metrics} >= {"top_1", "top_5", "mrr"}

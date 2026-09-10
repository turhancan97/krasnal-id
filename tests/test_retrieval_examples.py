"""Query-and-nearest-neighbour contact sheets, and the rule that picks them."""

import json
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import numpy.typing as npt
import pytest
from typer.testing import CliRunner

from helpers import (
    FAKE_BACKBONE,
    materialise_images,
    seed_embedding_cache,
    synthetic_manifest,
)
from krasnal_id.cli import app
from krasnal_id.config import BackboneConfig, VisualizationExperimentConfig, load_config
from krasnal_id.data_pipeline.build_split import build_evaluation_split, write_evaluation_split
from krasnal_id.embeddings.store import EmbeddingMatrix, load_embedding_matrix
from krasnal_id.models import DatasetManifest
from krasnal_id.viz.embedding_plot import VisualizationError
from krasnal_id.viz.retrieval_examples import (
    create_retrieval_examples_plot,
    group_for,
    rank_dwarfs,
    render_examples,
    select_examples,
)

runner = CliRunner()


def _names(manifest: DatasetManifest) -> dict[str, str]:
    return {dwarf.dwarf_id: dwarf.display_name for dwarf in manifest.dwarfs}


def _paths(manifest: DatasetManifest) -> dict[str, Path]:
    return {image.image_id: image.local_path for image in manifest.images}


def _matrix(
    manifest: DatasetManifest, root: Path, config: BackboneConfig = FAKE_BACKBONE
) -> EmbeddingMatrix:
    return load_embedding_matrix(manifest, config, root)


def _confusing_vector(dwarf_index: int, position: int, dwarf_count: int) -> npt.NDArray[np.float32]:
    """Put every dwarf on one axis, so the ranking is near-arbitrary.

    `tight_cluster_vector` separates the classes perfectly, which is the right
    default for accuracy tests and useless here: it can only ever produce the
    "both right" group.
    """
    vector = np.zeros(dwarf_count + 1, dtype=np.float32)
    vector[0] = 1.0
    vector[dwarf_index + 1 if dwarf_index + 1 <= dwarf_count else dwarf_count] = 0.02 * (
        position + 1
    )
    return np.asarray(vector / np.linalg.norm(vector), dtype=np.float32)


def test_candidates_are_distinct_dwarves_ranked_by_their_best_photograph(tmp_path: Path) -> None:
    manifest = synthetic_manifest(dwarf_count=4, per_dwarf=3)
    seed_embedding_cache(tmp_path, manifest)
    matrix = _matrix(manifest, tmp_path)
    fold = build_evaluation_split(manifest, datetime.now(UTC)).folds[0]

    matches = rank_dwarfs(
        fold.query_image_id,
        fold.query_dwarf_id,
        fold.reference_image_ids,
        matrix,
        _names(manifest),
        _paths(manifest),
        top_k=3,
    )

    assert len(matches) == 3
    # One row per dwarf, never two photographs of the same one.
    assert len({match.dwarf_id for match in matches}) == 3
    assert [match.rank for match in matches] == [1, 2, 3]
    # Well-separated vectors put the query's own dwarf first, and only it is correct.
    assert matches[0].dwarf_id == fold.query_dwarf_id
    assert [match.correct for match in matches] == [True, False, False]
    # Similarity must decrease down the list.
    scores = [match.cosine_similarity for match in matches]
    assert scores == sorted(scores, reverse=True)


def test_a_candidate_carries_the_photograph_that_matched_it(tmp_path: Path) -> None:
    manifest = synthetic_manifest(dwarf_count=3, per_dwarf=3)
    seed_embedding_cache(tmp_path, manifest)
    fold = build_evaluation_split(manifest, datetime.now(UTC)).folds[0]

    matches = rank_dwarfs(
        fold.query_image_id,
        fold.query_dwarf_id,
        fold.reference_image_ids,
        _matrix(manifest, tmp_path),
        _names(manifest),
        _paths(manifest),
    )

    paths = _paths(manifest)
    for match in matches:
        # The figure draws this file, so it has to be the one that was ranked.
        assert match.local_path == paths[match.image_id]
        assert match.image_id in fold.reference_image_ids
        assert match.display_name == _names(manifest)[match.dwarf_id]


def test_the_correct_dwarf_outside_the_top_five_leaves_no_rank() -> None:
    assert group_for({"dinov2": 1, "clip": 1}, "dinov2") == "both right"
    assert group_for({"dinov2": None, "clip": 4}, "dinov2") == "both wrong"
    assert group_for({"dinov2": 1, "clip": 3}, "dinov2") == "dinov2 only"
    assert group_for({"dinov2": 2, "clip": 1}, "dinov2") == "dinov2 missed"


def test_only_rank_one_counts_as_right() -> None:
    # Second place is a miss for this figure, the same way top-1 accuracy counts it.
    assert group_for({"dinov2": 2, "clip": 2}, "dinov2") == "both wrong"


def test_each_group_is_represented_by_the_first_query_in_image_id_order(
    tmp_path: Path,
) -> None:
    manifest = synthetic_manifest(dwarf_count=5, per_dwarf=3)
    seed_embedding_cache(tmp_path / "a", manifest)
    seed_embedding_cache(tmp_path / "b", manifest, vector_for=_confusing_vector)
    split = build_evaluation_split(manifest, datetime.now(UTC))
    matrices = {
        "dinov2": _matrix(manifest, tmp_path / "a"),
        "clip": _matrix(manifest, tmp_path / "b"),
    }

    examples = select_examples(manifest, split, matrices, "dinov2")

    assert examples
    ordered = sorted(split.folds, key=lambda fold: fold.query_image_id)
    for example in examples:
        # Every fold before the chosen one must belong to a different group.
        earlier = ordered[: [f.query_image_id for f in ordered].index(example.query_image_id)]
        chosen_groups = {other.group for other in examples}
        assert example.group in chosen_groups
        assert all(fold.query_image_id < example.query_image_id for fold in earlier)
    # Groups are unique, and each example carries one row per backbone.
    assert len({example.group for example in examples}) == len(examples)
    for example in examples:
        # The primary leads, so the headline backbone is the row read first.
        assert [row.backbone for row in example.rows] == ["dinov2", "clip"]


def test_selection_is_deterministic(tmp_path: Path) -> None:
    manifest = synthetic_manifest(dwarf_count=5, per_dwarf=3)
    seed_embedding_cache(tmp_path / "a", manifest)
    seed_embedding_cache(tmp_path / "b", manifest, vector_for=_confusing_vector)
    split = build_evaluation_split(manifest, datetime.now(UTC))
    matrices = {
        "dinov2": _matrix(manifest, tmp_path / "a"),
        "clip": _matrix(manifest, tmp_path / "b"),
    }

    first = select_examples(manifest, split, matrices, "dinov2")
    second = select_examples(manifest, split, matrices, "dinov2")

    assert [example.query_image_id for example in first] == [
        example.query_image_id for example in second
    ]
    assert [example.group for example in first] == [example.group for example in second]


def test_an_empty_split_is_refused(tmp_path: Path) -> None:
    manifest = synthetic_manifest(dwarf_count=3, per_dwarf=3)
    seed_embedding_cache(tmp_path, manifest)
    split = build_evaluation_split(manifest, datetime.now(UTC)).model_copy(update={"folds": ()})

    with pytest.raises(VisualizationError, match="no folds"):
        select_examples(manifest, split, {"dinov2": _matrix(manifest, tmp_path)}, "dinov2")


def test_rendering_writes_a_figure(tmp_path: Path) -> None:
    manifest = materialise_images(
        synthetic_manifest(dwarf_count=4, per_dwarf=3), tmp_path / "images"
    )
    seed_embedding_cache(tmp_path / "vectors", manifest)
    split = build_evaluation_split(manifest, datetime.now(UTC))
    examples = select_examples(
        manifest, split, {"dinov2": _matrix(manifest, tmp_path / "vectors")}, "dinov2"
    )

    path = render_examples(examples, tmp_path / "out" / "retrieval-examples.jpg")

    assert path.is_file()
    assert path.stat().st_size > 0


def test_rendering_nothing_is_refused(tmp_path: Path) -> None:
    with pytest.raises(VisualizationError, match="no retrieval examples"):
        render_examples((), tmp_path / "figure.jpg")


def test_a_missing_photograph_is_named(tmp_path: Path) -> None:
    # The manifest points at files that were never written, which is exactly what
    # a stale manifest looks like; the error has to say which file.
    manifest = synthetic_manifest(dwarf_count=3, per_dwarf=3)
    seed_embedding_cache(tmp_path / "vectors", manifest)
    split = build_evaluation_split(manifest, datetime.now(UTC))
    examples = select_examples(
        manifest, split, {"dinov2": _matrix(manifest, tmp_path / "vectors")}, "dinov2"
    )

    with pytest.raises(VisualizationError, match="could not read"):
        render_examples(examples, tmp_path / "figure.jpg")


def test_the_visualization_experiment_group_is_required() -> None:
    with pytest.raises(VisualizationError, match="experiment=visualization"):
        create_retrieval_examples_plot(load_config(["experiment=baseline"]))


def test_packaged_defaults_compare_both_backbones() -> None:
    experiment = load_config(["experiment=visualization"]).experiment
    assert isinstance(experiment, VisualizationExperimentConfig)
    # The figure is a comparison; a default that drew one backbone would not be one.
    assert experiment.backbones == ("dinov2", "clip")


def test_a_backbone_with_no_cached_vectors_is_named(tmp_path: Path) -> None:
    manifest = materialise_images(
        synthetic_manifest(dwarf_count=3, per_dwarf=3), tmp_path / "images"
    )
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(manifest.model_dump(mode="json")), encoding="utf-8")
    split_path = tmp_path / "splits" / "leave-one-out.json"
    write_evaluation_split(split_path, build_evaluation_split(manifest, datetime.now(UTC)))
    # One backbone's vectors are cached; the other's are simply absent.
    seed_embedding_cache(tmp_path / "vectors", manifest)

    config = load_config(
        [
            "experiment=visualization",
            f"paths.manifest_path={manifest_path}",
            f"paths.evaluation_split_path={split_path}",
            f"paths.embeddings_dir={tmp_path / 'vectors'}",
            f"paths.results_dir={tmp_path / 'results'}",
            f"backbone.model_id={FAKE_BACKBONE.model_id}",
            f"backbone.revision={FAKE_BACKBONE.revision}",
            f"backbone.preprocessing_id={FAKE_BACKBONE.preprocessing_id}",
        ]
    )

    with pytest.raises(Exception, match="cached clip vector"):
        create_retrieval_examples_plot(config)


def test_cli_writes_the_contact_sheet(tmp_path: Path) -> None:
    manifest = materialise_images(
        synthetic_manifest(dwarf_count=4, per_dwarf=3), tmp_path / "images"
    )
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(manifest.model_dump(mode="json")), encoding="utf-8")
    split_path = tmp_path / "splits" / "leave-one-out.json"
    write_evaluation_split(split_path, build_evaluation_split(manifest, datetime.now(UTC)))
    seed_embedding_cache(tmp_path / "vectors", manifest)

    result = runner.invoke(
        app,
        [
            "visualize",
            "retrieval-examples",
            *[
                f"-o{value}"
                for value in (
                    f"paths.manifest_path={manifest_path}",
                    f"paths.evaluation_split_path={split_path}",
                    f"paths.embeddings_dir={tmp_path / 'vectors'}",
                    f"paths.results_dir={tmp_path / 'results'}",
                    f"backbone.model_id={FAKE_BACKBONE.model_id}",
                    f"backbone.revision={FAKE_BACKBONE.revision}",
                    f"backbone.preprocessing_id={FAKE_BACKBONE.preprocessing_id}",
                    # Only one backbone's vectors are cached in this test.
                    "experiment.backbones=[dinov2]",
                    "logging.json_output=false",
                )
            ],
        ],
    )

    assert result.exit_code == 0, result.output
    assert (tmp_path / "results" / "retrieval-examples.jpg").is_file()

"""Embedding projection, label placement, and figure output."""

from datetime import UTC, datetime
from itertools import pairwise
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from typer.testing import CliRunner

from helpers import FAKE_BACKBONE, seed_embedding_cache, synthetic_manifest
from krasnal_id.cli import app
from krasnal_id.config import load_config
from krasnal_id.data_pipeline.build_manifest import write_dataset_manifest
from krasnal_id.data_pipeline.build_split import build_evaluation_split, write_evaluation_split
from krasnal_id.embeddings.store import load_embedding_matrix
from krasnal_id.viz import embedding_plot
from krasnal_id.viz.embedding_plot import (
    VisualizationError,
    build_plot,
    create_embedding_plot,
    declutter_labels,
    figure_path,
    import_optional_analysis,
    most_entangled,
    project_embeddings,
    render_projection,
)


def test_figure_path_is_named_by_method_and_backbone() -> None:
    assert figure_path(Path("results"), "umap", "clip").name == "embeddings-umap-clip.png"
    assert figure_path(Path("results"), "tsne", "dinov2").name == "embeddings-tsne-dinov2.png"


def test_overlapping_labels_are_separated_and_isolated_ones_are_left_alone() -> None:
    # Three centroids stacked in one column, plus one far away.
    centroids = np.asarray(
        [[1.0, 1.0], [1.02, 0.99], [1.01, 1.005], [9.0, -4.0]],
        dtype=np.float32,
    )

    placed = declutter_labels(centroids, (10.0, 6.0))

    stacked = sorted(placed[:3, 1])
    gaps = [round(b - a, 6) for a, b in pairwise(stacked)]
    assert all(gap >= 6.0 * 0.035 - 1e-6 for gap in gaps)
    # The x coordinate is never moved, so a label stays over its own cluster.
    np.testing.assert_allclose(placed[:, 0], centroids[:, 0])
    # The isolated label is untouched.
    np.testing.assert_allclose(placed[3], centroids[3])


def test_projection_rejects_tiny_inputs_and_unknown_methods() -> None:
    with pytest.raises(VisualizationError, match="at least three vectors"):
        project_embeddings(np.zeros((2, 4), dtype=np.float32), "tsne", 42)
    with pytest.raises(VisualizationError, match="unsupported projection method"):
        project_embeddings(np.eye(4, dtype=np.float32), "pca", 42)


def test_missing_optional_dependencies_name_the_extra() -> None:
    with pytest.raises(VisualizationError, match="uv sync --extra analysis"):
        import_optional_analysis("krasnal_id_absent_analysis_module")


def test_umap_branch_is_configured_for_the_sample_size(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, Any] = {}

    class FakeReducer:
        def __init__(self, **kwargs: Any) -> None:
            captured.update(kwargs)

        def fit_transform(self, vectors: np.ndarray) -> np.ndarray:
            return np.zeros((vectors.shape[0], 2), dtype=np.float32)

    class FakeUmap:
        UMAP = FakeReducer

    monkeypatch.setattr(
        embedding_plot,
        "import_optional_analysis",
        lambda name: FakeUmap if name == "umap" else pytest.fail(f"unexpected import {name}"),
    )

    projected = project_embeddings(np.eye(5, dtype=np.float32), "umap", 42)

    assert projected.shape == (5, 2)
    # n_neighbors must stay below the sample count, and the seed must be passed
    # through or the figure stops being reproducible.
    assert captured["n_neighbors"] == 4
    assert captured["random_state"] == 42
    assert captured["metric"] == "cosine"


def test_a_projection_returning_the_wrong_shape_is_rejected(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class BadReducer:
        def __init__(self, **kwargs: Any) -> None:
            pass

        def fit_transform(self, vectors: np.ndarray) -> np.ndarray:
            return np.zeros((vectors.shape[0], 3), dtype=np.float32)

    class FakeUmap:
        UMAP = BadReducer

    monkeypatch.setattr(embedding_plot, "import_optional_analysis", lambda name: FakeUmap)

    with pytest.raises(VisualizationError, match="returned shape"):
        project_embeddings(np.eye(5, dtype=np.float32), "umap", 42)


def test_tsne_projects_real_vectors_deterministically() -> None:
    pytest.importorskip("sklearn.manifold")
    rng = np.random.default_rng(3)
    vectors = rng.normal(size=(30, 8)).astype(np.float32)
    vectors /= np.linalg.norm(vectors, axis=1, keepdims=True)

    first = project_embeddings(vectors, "tsne", 42)
    again = project_embeddings(vectors, "tsne", 42)

    assert first.shape == (30, 2)
    np.testing.assert_allclose(first, again)


def test_a_figure_is_written_for_every_class(tmp_path: Path) -> None:
    pytest.importorskip("matplotlib")
    pytest.importorskip("sklearn.manifold")
    manifest = synthetic_manifest(dwarf_count=5)
    seed_embedding_cache(tmp_path, manifest)
    matrix = load_embedding_matrix(manifest, FAKE_BACKBONE, tmp_path)

    path = build_plot(manifest, matrix, "tsne", 42, "dinov2", tmp_path / "results")

    assert path == figure_path(tmp_path / "results", "tsne", "dinov2")
    assert path.is_file()
    assert path.stat().st_size > 1000


def test_create_plot_requires_the_visualization_experiment_group() -> None:
    with pytest.raises(VisualizationError, match="requires experiment=visualization"):
        create_embedding_plot(load_config(["experiment=baseline"]))


def test_cli_visualization_writes_a_figure_and_fails_without_embeddings(tmp_path: Path) -> None:
    pytest.importorskip("matplotlib")
    pytest.importorskip("sklearn.manifold")
    manifest = synthetic_manifest(dwarf_count=5)
    manifest_path = tmp_path / "manifest.json"
    split_path = tmp_path / "splits" / "leave-one-out.json"
    write_dataset_manifest(manifest_path, manifest)
    write_evaluation_split(split_path, build_evaluation_split(manifest, datetime.now(UTC)))
    overrides = [
        f"paths.manifest_path={manifest_path}",
        f"paths.evaluation_split_path={split_path}",
        f"paths.embeddings_dir={tmp_path / 'embeddings'}",
        f"paths.results_dir={tmp_path / 'results'}",
        f"backbone.model_id={FAKE_BACKBONE.model_id}",
        f"backbone.revision={FAKE_BACKBONE.revision}",
        f"backbone.preprocessing_id={FAKE_BACKBONE.preprocessing_id}",
        "experiment.method=tsne",
        "logging.json_output=false",
    ]
    arguments = ["visualize", "embeddings", *[f"-o{value}" for value in overrides]]
    runner = CliRunner()

    missing = runner.invoke(app, arguments)
    assert missing.exit_code == 2
    assert "Embedding visualization error" in missing.output

    seed_embedding_cache(tmp_path / "embeddings", manifest)
    result = runner.invoke(app, arguments)
    assert result.exit_code == 0, result.output
    assert "Embedding visualization complete" in result.output
    assert (tmp_path / "results" / "embeddings-tsne-dinov2.png").is_file()


def test_entanglement_ranks_the_classes_sitting_on_top_of_each_other() -> None:
    import numpy as np

    # Two centroids nearly coincide; the third is far away.
    centroids = np.asarray([[0.0, 0.0], [0.1, 0.0], [10.0, 10.0]], dtype=np.float32)

    assert most_entangled(centroids, 2) == [0, 1]
    assert most_entangled(centroids, 3) == [0, 1, 2]
    # Degenerate inputs stay usable rather than raising.
    assert most_entangled(np.zeros((1, 2), dtype=np.float32), 5) == [0]
    assert most_entangled(np.zeros((0, 2), dtype=np.float32), 5) == []


def test_a_crowded_projection_labels_only_its_budget(tmp_path: Path) -> None:
    import numpy as np

    rng = np.random.default_rng(0)
    classes = tuple(f"Q{i + 1}" for i in range(40) for _ in range(2))
    projected = rng.normal(size=(len(classes), 2)).astype(np.float32)

    path = render_projection(
        projected,
        classes,
        {c: c for c in classes},
        "crowded",
        tmp_path / "crowded.png",
        label_budget=5,
    )

    assert path.is_file()
    # Under the budget every class is still named, which keeps small pools unchanged.
    small = tuple(f"Q{i + 1}" for i in range(3) for _ in range(2))
    assert render_projection(
        projected[: len(small)],
        small,
        {c: c for c in small},
        "small",
        tmp_path / "small.png",
        label_budget=5,
    ).is_file()

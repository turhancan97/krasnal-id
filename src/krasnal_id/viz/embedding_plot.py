"""UMAP and t-SNE embedding-space visualization."""

import importlib
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt

from krasnal_id.config import AppConfig, VisualizationExperimentConfig
from krasnal_id.embeddings.store import EmbeddingMatrix, load_embedding_matrix
from krasnal_id.experiments.baseline_accuracy import (
    BaselineExperimentError,
    load_evaluation_inputs,
)
from krasnal_id.models import DatasetManifest

# tab20 supplies twenty distinct colors, so classes beyond that are separated by
# marker shape as well as color rather than silently reusing an identical style.
_MARKERS = ("o", "^", "s", "D")
_PALETTE_SIZE = 20


class VisualizationError(ValueError):
    """Raised when a projection cannot be configured or written."""


def import_optional_analysis(module_name: str) -> Any:
    """Import an optional analysis dependency only when plotting is requested."""
    try:
        return importlib.import_module(module_name)
    except ImportError as error:
        raise VisualizationError(
            "analysis dependencies are required for embedding visualization; "
            "run uv sync --extra analysis"
        ) from error


def project_embeddings(
    vectors: npt.NDArray[np.float32],
    method: str,
    seed: int,
) -> npt.NDArray[np.float32]:
    """Project normalized embeddings into two dimensions deterministically."""
    if vectors.ndim != 2 or vectors.shape[0] < 3:
        raise VisualizationError(
            f"a projection needs at least three vectors, got shape {vectors.shape}"
        )

    if method == "umap":
        umap = import_optional_analysis("umap")
        # n_neighbors cannot reach the sample count, and a fixed random_state makes
        # UMAP single-threaded, which is what makes the figure reproducible.
        reducer = umap.UMAP(
            n_components=2,
            n_neighbors=min(15, vectors.shape[0] - 1),
            metric="cosine",
            random_state=seed,
        )
    elif method == "tsne":
        manifold = import_optional_analysis("sklearn.manifold")
        reducer = manifold.TSNE(
            n_components=2,
            # Perplexity must stay below the sample count.
            perplexity=min(30.0, max(5.0, (vectors.shape[0] - 1) / 3)),
            metric="cosine",
            init="pca",
            random_state=seed,
        )
    else:
        raise VisualizationError(f"unsupported projection method: {method}")

    projected = np.asarray(reducer.fit_transform(np.asarray(vectors, dtype=np.float32)))
    if projected.shape != (vectors.shape[0], 2):
        raise VisualizationError(
            f"{method} returned shape {projected.shape} for {vectors.shape[0]} vectors"
        )
    return np.asarray(projected, dtype=np.float32)


def declutter_labels(
    centroids: npt.NDArray[np.float32],
    span: tuple[float, float],
) -> npt.NDArray[np.float32]:
    """Nudge overlapping class labels apart along the vertical axis.

    Close clusters would otherwise stack their labels into an unreadable pile, which
    is exactly what happens in the densest part of a projection. Labels are pushed
    down in reading order until each clears the last one placed near its column, and
    a leader line keeps every displaced label attributable to its own cluster.
    """
    width, height = span
    minimum_gap = height * 0.035
    column_width = width * 0.14

    placed = np.array(centroids, dtype=np.float32)
    order = np.lexsort((placed[:, 0], -placed[:, 1]))
    for position, index in enumerate(order):
        for previous in order[:position]:
            same_column = abs(placed[index, 0] - placed[previous, 0]) < column_width
            overlapping = abs(placed[index, 1] - placed[previous, 1]) < minimum_gap
            if same_column and overlapping:
                placed[index, 1] = placed[previous, 1] - minimum_gap
    return placed


def render_projection(
    projected: npt.NDArray[np.float32],
    dwarf_ids: tuple[str, ...],
    display_names: dict[str, str],
    title: str,
    path: Path,
) -> Path:
    """Draw and save the projection, labeling each class at its centroid."""
    matplotlib = import_optional_analysis("matplotlib")
    # Select a non-interactive backend before pyplot binds one, so the figure
    # renders on headless machines and in CI.
    matplotlib.use("Agg")
    pyplot = import_optional_analysis("matplotlib.pyplot")

    classes = sorted(set(dwarf_ids))
    colormap = pyplot.get_cmap("tab20")
    figure, axes = pyplot.subplots(figsize=(11, 9))
    ids = np.asarray(dwarf_ids)

    centroids = np.asarray(
        [projected[np.flatnonzero(ids == dwarf_id)].mean(axis=0) for dwarf_id in classes],
        dtype=np.float32,
    )
    span = (
        float(np.ptp(projected[:, 0])) or 1.0,
        float(np.ptp(projected[:, 1])) or 1.0,
    )
    label_positions = declutter_labels(centroids, span)

    for index, dwarf_id in enumerate(classes):
        rows = np.flatnonzero(ids == dwarf_id)
        points = projected[rows]
        color = colormap(index % _PALETTE_SIZE)
        axes.scatter(
            points[:, 0],
            points[:, 1],
            color=color,
            marker=_MARKERS[(index // _PALETTE_SIZE) % len(_MARKERS)],
            s=42,
            alpha=0.85,
            edgecolors="white",
            linewidths=0.5,
        )
        # A legend of twenty-plus entries is unreadable, so each class is named
        # near its own points, with a leader line when it had to be moved.
        axes.annotate(
            display_names.get(dwarf_id, dwarf_id),
            xy=tuple(centroids[index]),
            xytext=tuple(label_positions[index]),
            fontsize=7,
            ha="center",
            va="center",
            color="black",
            bbox={"boxstyle": "round,pad=0.2", "facecolor": "white", "alpha": 0.75, "lw": 0},
            arrowprops={"arrowstyle": "-", "color": color, "lw": 0.6, "alpha": 0.8},
        )

    axes.set_title(title)
    axes.set_xlabel("component 1")
    axes.set_ylabel("component 2")
    axes.spines["top"].set_visible(False)
    axes.spines["right"].set_visible(False)
    figure.tight_layout()

    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        figure.savefig(path, dpi=200)
    except OSError as error:
        raise VisualizationError(f"could not write figure {path}: {error}") from error
    finally:
        pyplot.close(figure)
    return path


def figure_path(results_dir: Path, method: str, backbone: str) -> Path:
    """Return the deterministic figure path for one projection."""
    return results_dir / f"embeddings-{method}-{backbone}.png"


def build_plot(
    manifest: DatasetManifest,
    matrix: EmbeddingMatrix,
    method: str,
    seed: int,
    backbone: str,
    results_dir: Path,
) -> Path:
    """Project and render the cached vectors for one backbone."""
    projected = project_embeddings(matrix.vectors, method, seed)
    display_names = {dwarf.dwarf_id: dwarf.display_name for dwarf in manifest.dwarfs}
    classes = len(set(matrix.dwarf_ids))
    title = (
        f"{backbone} embeddings, {method.upper()} projection "
        f"({classes} dwarves, {len(matrix.image_ids)} images)"
    )
    return render_projection(
        projected,
        matrix.dwarf_ids,
        display_names,
        title,
        figure_path(results_dir, method, backbone),
    )


def create_embedding_plot(config: AppConfig) -> Path:
    """Create and save the configured two-dimensional embedding projection."""
    if not isinstance(config.experiment, VisualizationExperimentConfig):
        raise VisualizationError(
            f"visualization requires experiment=visualization, got {config.experiment.kind}"
        )

    try:
        manifest, _ = load_evaluation_inputs(
            config.paths.manifest_path,
            config.paths.evaluation_split_path,
        )
    except BaselineExperimentError as error:
        raise VisualizationError(str(error)) from error

    matrix = load_embedding_matrix(manifest, config.backbone, config.paths.embeddings_dir)
    return build_plot(
        manifest,
        matrix,
        config.experiment.method,
        config.experiment.seed,
        config.backbone.name,
        config.paths.results_dir,
    )

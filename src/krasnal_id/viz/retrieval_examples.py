"""Query-and-nearest-neighbour contact sheets, one row per backbone.

Every other figure here plots an aggregate. This one shows the retrieval itself:
a query photograph beside the five dwarves each backbone ranks highest for it,
which is the thing the accuracy numbers are an average over.

The queries are chosen by a stated rule rather than by eye, because a
hand-picked example of a model succeeding is worth nothing. Each fold is scored
under both backbones exactly as `experiment baseline` scores it, folds are
grouped by which backbones got them right, and the first fold in image-ID order
represents its group. Image IDs order by Commons page ID, which is unrelated to
anything either model sees, so "first" is arbitrary in the way a sample should
be.
"""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

from krasnal_id.config import AppConfig, VisualizationExperimentConfig, backbone_config
from krasnal_id.embeddings.store import EmbeddingMatrix, load_embedding_matrix
from krasnal_id.experiments.baseline_accuracy import (
    BaselineExperimentError,
    load_evaluation_inputs,
)
from krasnal_id.models import DatasetManifest, EvaluationSplit
from krasnal_id.retrieval.knn import cosine_knn
from krasnal_id.viz.embedding_plot import VisualizationError, import_optional_analysis

# Five, because that is what top-5 accuracy measures and what the demo shows.
_TOP_K = 5
# Long side of each rendered cell, in pixels. Large enough to tell two dwarves
# apart in a README at full width, small enough that the figure stays portable.
_CELL_PIXELS = 320
_CORRECT = "#1f7a5c"
_WRONG = "#cf4832"
# Longest display name a cell label can hold before it runs into the next cell.
_NAME_LIMIT = 22


@dataclass(frozen=True, slots=True)
class ExampleMatch:
    """One ranked candidate dwarf, and whether it is the right one."""

    rank: int
    dwarf_id: str
    display_name: str
    image_id: str
    local_path: Path
    cosine_similarity: float
    correct: bool


@dataclass(frozen=True, slots=True)
class ExampleRow:
    """One backbone's ranked candidates for a single query."""

    backbone: str
    matches: tuple[ExampleMatch, ...]

    @property
    def dwarf_rank(self) -> int | None:
        """Where the correct dwarf landed, or None if outside the top five."""
        for match in self.matches:
            if match.correct:
                return match.rank
        return None


@dataclass(frozen=True, slots=True)
class RetrievalExample:
    """One query photograph and every backbone's answer to it."""

    group: str
    query_image_id: str
    query_dwarf_id: str
    query_display_name: str
    query_local_path: Path
    rows: tuple[ExampleRow, ...]


def rank_dwarfs(
    query_image_id: str,
    query_dwarf_id: str,
    reference_image_ids: tuple[str, ...],
    matrix: EmbeddingMatrix,
    names: Mapping[str, str],
    paths: Mapping[str, Path],
    top_k: int = _TOP_K,
) -> tuple[ExampleMatch, ...]:
    """Rank distinct dwarves by their best-matching reference photograph.

    The collapse to distinct dwarves is `evaluate_fold`'s, because a figure that
    ranked images would not illustrate the accuracy the text quotes.
    """
    vectors, dwarf_ids = matrix.rows_for(reference_image_ids)
    ranked = cosine_knn(
        query_image_id,
        matrix.vector_for(query_image_id),
        vectors,
        reference_image_ids,
        dwarf_ids,
        top_k=len(reference_image_ids),
    )

    seen: dict[str, ExampleMatch] = {}
    for match in ranked.matches:
        if match.dwarf_id in seen:
            continue
        seen[match.dwarf_id] = ExampleMatch(
            rank=len(seen) + 1,
            dwarf_id=match.dwarf_id,
            display_name=names.get(match.dwarf_id, match.dwarf_id),
            image_id=match.image_id,
            local_path=paths[match.image_id],
            cosine_similarity=match.cosine_similarity,
            correct=match.dwarf_id == query_dwarf_id,
        )
        if len(seen) == top_k:
            break
    return tuple(seen.values())


def group_for(ranks: Mapping[str, int | None], primary: str) -> str:
    """Name the agreement pattern a fold falls into.

    Three groups, because they are the three things this figure is for: the
    ordinary case, the gap between the backbones that most of these findings are
    about, and the lookalike families neither one separates.
    """
    correct = {name for name, rank in ranks.items() if rank == 1}
    if len(correct) == len(ranks):
        return "both right"
    if not correct:
        return "both wrong"
    if primary in correct:
        return f"{primary} only"
    return f"{primary} missed"


def select_examples(
    manifest: DatasetManifest,
    split: EvaluationSplit,
    matrices: Mapping[str, EmbeddingMatrix],
    primary: str,
) -> tuple[RetrievalExample, ...]:
    """Pick one representative query per agreement group, by a stated rule."""
    if not split.folds:
        raise VisualizationError("split contains no folds")
    names = {dwarf.dwarf_id: dwarf.display_name for dwarf in manifest.dwarfs}
    paths = {image.image_id: image.local_path for image in manifest.images}
    # The primary leads, so the row a reader looks at first is the one the
    # project's headline number belongs to; alphabetical order put CLIP there.
    ordered = [primary, *sorted(name for name in matrices if name != primary)]

    # A stable, readable order: the ordinary case, then the disagreements, then
    # the failures. Groups that did not occur are simply absent.
    order = (
        "both right",
        f"{primary} only",
        f"{primary} missed",
        "both wrong",
    )

    chosen: dict[str, RetrievalExample] = {}
    # Image-ID order, so "the first in its group" is a rule and not a choice.
    for fold in sorted(split.folds, key=lambda fold: fold.query_image_id):
        # Every group already has its representative, and ranking the remaining
        # folds cannot change one: the rule is first-in-order, and they are later.
        if len(chosen) == len(order):
            break
        rows = tuple(
            ExampleRow(
                backbone=name,
                matches=rank_dwarfs(
                    fold.query_image_id,
                    fold.query_dwarf_id,
                    fold.reference_image_ids,
                    matrices[name],
                    names,
                    paths,
                ),
            )
            for name in ordered
        )
        group = group_for({row.backbone: row.dwarf_rank for row in rows}, primary)
        if group in chosen:
            continue
        chosen[group] = RetrievalExample(
            group=group,
            query_image_id=fold.query_image_id,
            query_dwarf_id=fold.query_dwarf_id,
            query_display_name=names.get(fold.query_dwarf_id, fold.query_dwarf_id),
            query_local_path=paths[fold.query_image_id],
            rows=rows,
        )

    if not chosen:
        raise VisualizationError("no folds could be grouped into examples")
    return tuple(chosen[group] for group in order if group in chosen)


def _short(name: str, limit: int = _NAME_LIMIT) -> str:
    """Trim a display name to what fits one cell.

    Commons category names run to "Concierge dwarf at the entrance of ..." and a
    handful of those are enough to overwrite the neighbouring cell's label.
    """
    return name if len(name) <= limit else f"{name[: limit - 1].rstrip()}…"


def _load_cell(local_path: Path) -> object:
    """Read one photograph, downscaled to the cell size before it is drawn."""
    image_module = import_optional_analysis("PIL.Image")
    try:
        with image_module.open(local_path) as handle:
            image = handle.convert("RGB")
            image.thumbnail((_CELL_PIXELS, _CELL_PIXELS))
            return image.copy()
    except OSError as error:
        raise VisualizationError(f"could not read {local_path}: {error}") from error


def render_examples(examples: Sequence[RetrievalExample], path: Path) -> Path:
    """Draw every example as a query column beside one row per backbone."""
    if not examples:
        raise VisualizationError("no retrieval examples to draw")

    matplotlib = import_optional_analysis("matplotlib")
    matplotlib.use("Agg")
    pyplot = import_optional_analysis("matplotlib.pyplot")

    rows_per_example = len(examples[0].rows)
    total_rows = rows_per_example * len(examples)
    figure = pyplot.figure(figsize=(2.05 * (_TOP_K + 1), 2.35 * total_rows))
    # The query spans its example's rows; the candidates get one row each.
    grid = figure.add_gridspec(
        total_rows,
        _TOP_K + 1,
        hspace=0.52,
        # Wide enough that the rotated backbone label clears the query
        # image on its left and the long names clear each other.
        wspace=0.17,
        left=0.012,
        right=0.988,
        top=0.93,
        bottom=0.02,
    )

    for index, example in enumerate(examples):
        top = index * rows_per_example
        query_axes = figure.add_subplot(grid[top : top + rows_per_example, 0])
        query_axes.imshow(_load_cell(example.query_local_path))
        # Group above, subject below: as one two-line title it ran past the
        # column and collided with the first candidate's label.
        query_axes.set_title(example.group, fontsize=11, fontweight="bold")
        query_axes.set_xlabel(
            f"query · {_short(example.query_display_name)}", fontsize=9.5, labelpad=6
        )
        for spine in query_axes.spines.values():
            spine.set_linewidth(1.6)
        query_axes.set_xticks([])
        query_axes.set_yticks([])

        for offset, row in enumerate(example.rows):
            for column, match in enumerate(row.matches, start=1):
                axes = figure.add_subplot(grid[top + offset, column])
                axes.imshow(_load_cell(match.local_path))
                colour = _CORRECT if match.correct else _WRONG
                axes.set_title(
                    f"{match.rank}. {_short(match.display_name)}\n{match.cosine_similarity:.3f}",
                    fontsize=8.5,
                    color=colour,
                    loc="left",
                )
                for spine in axes.spines.values():
                    spine.set_color(colour)
                    spine.set_linewidth(2.0 if match.correct else 1.0)
                axes.set_xticks([])
                axes.set_yticks([])
                if column == 1:
                    axes.set_ylabel(row.backbone, fontsize=10, fontweight="bold", labelpad=3)

    figure.suptitle(
        "Query photograph and the five dwarves each backbone ranks highest\n"
        "green is the correct statue, red is a wrong one",
        fontsize=12,
        y=0.985,
    )

    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        # JPEG, where every other figure here is a PNG. The others are line
        # plots, which PNG stores exactly and small; this one is 44 photographs,
        # which PNG stores exactly and at 3.7 MB. At quality 88 it is under a
        # tenth of that and the difference is invisible at this cell size.
        figure.savefig(path, dpi=150, pil_kwargs={"quality": 88, "optimize": True})
    except OSError as error:
        raise VisualizationError(f"could not write figure {path}: {error}") from error
    finally:
        pyplot.close(figure)
    return path


def create_retrieval_examples_plot(config: AppConfig) -> Path:
    """Draw one contact sheet covering every backbone the config names."""
    if not isinstance(config.experiment, VisualizationExperimentConfig):
        raise VisualizationError(
            f"visualization requires experiment=visualization, got {config.experiment.kind}"
        )

    try:
        manifest, split = load_evaluation_inputs(
            config.paths.manifest_path, config.paths.evaluation_split_path
        )
    except BaselineExperimentError as error:
        raise VisualizationError(str(error)) from error

    wanted = {config.backbone.name, *config.experiment.backbones}
    matrices = {
        name: load_embedding_matrix(
            manifest,
            config.backbone if name == config.backbone.name else backbone_config(name),
            Path(config.paths.embeddings_dir),
        )
        for name in sorted(wanted)
    }
    primary = matrices[config.backbone.name]
    for name, matrix in matrices.items():
        if matrix.image_ids != primary.image_ids:
            raise VisualizationError(
                f"the {name} vectors are not in the same row order as "
                f"{config.backbone.name}'s; re-run embeddings extract for both"
            )

    examples = select_examples(manifest, split, matrices, config.backbone.name)
    return render_examples(examples, config.paths.results_dir / "retrieval-examples.jpg")

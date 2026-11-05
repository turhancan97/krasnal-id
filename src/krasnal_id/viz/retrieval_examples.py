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
from krasnal_id.viz import figure_style as style
from krasnal_id.viz.embedding_plot import VisualizationError, import_optional_analysis

# Five, because that is what top-5 accuracy measures and what the demo shows.
_TOP_K = 5
# Long side of each rendered cell, in pixels. Large enough to tell two dwarves
# apart in a README at full width, small enough that the figure stays portable.
_CELL_PIXELS = 360
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
    """Read one photograph and letterbox it into a square cell.

    Letterboxed rather than cropped: a centre crop of a portrait frame can cut the
    statue off, and a sheet whose cells differ in shape reads as untidy. The pad
    is a near-white neutral, so it reads as the cell rather than as the photograph.
    """
    image_module = import_optional_analysis("PIL.Image")
    try:
        with image_module.open(local_path) as handle:
            image = handle.convert("RGB")
            image.thumbnail((_CELL_PIXELS, _CELL_PIXELS))
            cell = image_module.new("RGB", (_CELL_PIXELS, _CELL_PIXELS), (246, 248, 250))
            cell.paste(
                image, ((_CELL_PIXELS - image.width) // 2, (_CELL_PIXELS - image.height) // 2)
            )
            return cell
    except OSError as error:
        raise VisualizationError(f"could not read {local_path}: {error}") from error


def _backbone_name(backbone: str) -> str:
    return style.BACKBONE_LABEL.get(backbone, backbone)


def _group_name(group: str, primary: str) -> str:
    """The agreement group as a reader-facing phrase."""
    named = {
        "both right": "Both right",
        "both wrong": "Both wrong",
        f"{primary} only": f"Only {_backbone_name(primary)} right",
        f"{primary} missed": f"{_backbone_name(primary)} wrong",
    }
    return named.get(group, group)


def _frame(axes: object, correct: bool | None) -> None:
    """Frame one cell: green and heavier when correct, grey when wrong, ink for the query."""
    colour = style.INK if correct is None else (style.CORRECT if correct else style.WRONG)
    width = 1.0 if correct is None else (1.8 if correct else 0.8)
    for spine in axes.spines.values():  # type: ignore[attr-defined]
        spine.set_visible(True)
        spine.set_color(colour)
        spine.set_linewidth(width)
    axes.set_xticks([])  # type: ignore[attr-defined]
    axes.set_yticks([])  # type: ignore[attr-defined]
    axes.grid(False)  # type: ignore[attr-defined]


def _badge(axes: object, correct: bool) -> None:
    """A check or cross in the cell's corner, beside the frame colour.

    Correctness never rests on hue alone. Set in DejaVu Sans because TeX Gyre
    Heros has no check-mark glyph, and matplotlib substitutes a wrong one rather
    than falling back -- the first render showed a boxed aeroplane instead.
    """
    axes.text(  # type: ignore[attr-defined]
        0.95,
        0.95,
        "\N{CHECK MARK}" if correct else "\N{BALLOT X}",
        transform=axes.transAxes,  # type: ignore[attr-defined]
        ha="right",
        va="top",
        fontsize=style.SIZE_LABEL,
        fontweight="bold",
        fontfamily="DejaVu Sans",
        color="white",
        bbox={
            "boxstyle": "circle,pad=0.18",
            "facecolor": style.CORRECT if correct else style.WRONG,
            "edgecolor": "none",
        },
    )


def _setup() -> object:
    matplotlib = import_optional_analysis("matplotlib")
    matplotlib.use("Agg")
    pyplot = import_optional_analysis("matplotlib.pyplot")
    style.apply(pyplot)
    return pyplot


def _write(pyplot: object, figure: object, path: Path, dpi: int) -> Path:
    try:
        style.save(figure, path, dpi)
    except OSError as error:
        raise VisualizationError(f"could not write figure {path}: {error}") from error
    finally:
        pyplot.close(figure)  # type: ignore[attr-defined]
    return path


def render_examples(
    examples: Sequence[RetrievalExample],
    path: Path,
    dpi: int = 150,
    width: float = style.CVWW.full,
) -> Path:
    """Draw every example as a query column beside one row per backbone.

    Drawn at `width` inches -- the CVWW full text width by default -- in the shared
    figure style. The density is a parameter because the sheet has two readers: the
    repository's copy is read on a screen at 150 dpi, the manuscript's at 300.
    """
    if not examples:
        raise VisualizationError("no retrieval examples to draw")
    pyplot = _setup()

    primary = examples[0].rows[0].backbone
    rows_per_example = len(examples[0].rows)
    total_rows = rows_per_example * len(examples)
    cell = width / (_TOP_K + 1)
    figure = pyplot.figure(  # type: ignore[attr-defined]
        figsize=(width, cell * 1.32 * total_rows + 0.1), layout="constrained"
    )
    grid = figure.add_gridspec(total_rows, _TOP_K + 1, hspace=0.02, wspace=0.02)

    for index, example in enumerate(examples):
        top = index * rows_per_example
        query_axes = figure.add_subplot(grid[top : top + rows_per_example, 0])
        query_axes.imshow(_load_cell(example.query_local_path))
        _frame(query_axes, None)
        query_axes.set_title(
            _group_name(example.group, primary),
            fontsize=style.SIZE_LABEL,
            fontweight="bold",
            color=style.INK,
        )
        query_axes.set_xlabel(
            f"Query\n{_short(example.query_display_name)}",
            fontsize=style.SIZE_ANNOTATION,
            color=style.INK_SECONDARY,
            labelpad=3,
        )
        for offset, row in enumerate(example.rows):
            for column, match in enumerate(row.matches, start=1):
                axes = figure.add_subplot(grid[top + offset, column])
                axes.imshow(_load_cell(match.local_path))
                _frame(axes, match.correct)
                _badge(axes, match.correct)
                axes.set_title(
                    f"{match.rank}. {_short(match.display_name, 18)}\n"
                    f"{match.cosine_similarity:.3f}",
                    fontsize=style.SIZE_ANNOTATION - 0.5,
                    color=style.INK if match.correct else style.INK_SECONDARY,
                    loc="left",
                    pad=2,
                )
                if column == 1:
                    axes.set_ylabel(
                        _backbone_name(row.backbone),
                        fontsize=style.SIZE_LABEL,
                        fontweight="bold",
                        color=style.INK,
                        labelpad=3,
                    )
    return _write(pyplot, figure, path, dpi)


def render_top_one(
    examples: Sequence[RetrievalExample],
    path: Path,
    dpi: int = 150,
    width: float = style.CVWW.column,
) -> Path:
    """Draw each example as one row: the query, then each backbone's first answer.

    The compact form a two-column paper has room for, drawn at one column's width.
    The examples are the ones `select_examples` chose, so the sheet shows the same
    queries as the top-five version and differs only in how much of each ranking.
    """
    if not examples:
        raise VisualizationError("no retrieval examples to draw")
    pyplot = _setup()

    primary = examples[0].rows[0].backbone
    columns = 1 + len(examples[0].rows)
    cell = width / columns
    figure, axes_grid = pyplot.subplots(  # type: ignore[attr-defined]
        len(examples),
        columns,
        figsize=(width, cell * 1.22 * len(examples) + 0.05),
        squeeze=False,
        layout="constrained",
        gridspec_kw={"wspace": 0.02, "hspace": 0.02},
    )
    for index, example in enumerate(examples):
        query = axes_grid[index][0]
        query.imshow(_load_cell(example.query_local_path))
        _frame(query, None)
        query.set_title("Query", fontsize=style.SIZE_ANNOTATION, color=style.INK, pad=2)
        query.set_ylabel(
            _group_name(example.group, primary),
            fontsize=style.SIZE_ANNOTATION,
            fontweight="bold",
            color=style.INK,
            labelpad=2,
        )
        for column, row in enumerate(example.rows, start=1):
            first = row.matches[0]
            axes = axes_grid[index][column]
            axes.imshow(_load_cell(first.local_path))
            _frame(axes, first.correct)
            _badge(axes, first.correct)
            axes.set_title(
                f"{_backbone_name(row.backbone)}\n{_short(first.display_name, 16)}",
                fontsize=style.SIZE_ANNOTATION - 0.5,
                color=style.INK if first.correct else style.INK_SECONDARY,
                pad=2,
            )
    return _write(pyplot, figure, path, dpi)


def create_retrieval_examples_plot(
    config: AppConfig,
    groups: Sequence[str] | None = None,
    output: Path | None = None,
    dpi: int = 150,
    top_k: int = _TOP_K,
    width_cm: float | None = None,
) -> Path:
    """Draw one contact sheet covering every backbone the config names.

    `width_cm` is the width the sheet will be printed at. It defaults to one CVWW
    column for the top-1 sheet and the full text width for the top-5 one; a
    narrower width is drawn narrower rather than scaled, so its type stays the
    size every other figure uses.

    `groups` keeps only the named agreement groups, in the order the selection
    rule already puts them. The manuscript has room for three rows where the
    repository shows four, and dropping one there rather than cropping the image
    keeps the figure regenerable.
    """
    if not isinstance(config.experiment, VisualizationExperimentConfig):
        raise VisualizationError(
            f"visualization requires experiment=visualization, got {config.experiment.kind}"
        )
    # Checked before any data is read, so a bad request fails the same way with or
    # without a dataset on disk.
    if top_k not in (1, _TOP_K):
        raise VisualizationError(f"the sheet draws the top {_TOP_K} or the top 1, not {top_k}")
    if width_cm is not None and width_cm <= 0:
        raise VisualizationError(f"the sheet width must be positive, got {width_cm} cm")

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
    if groups is not None:
        wanted_groups = set(groups)
        unknown = wanted_groups - {example.group for example in examples}
        if unknown:
            raise VisualizationError(
                f"no query fell into {sorted(unknown)}; the groups that occurred are "
                f"{sorted(example.group for example in examples)}"
            )
        examples = tuple(example for example in examples if example.group in wanted_groups)
    if top_k == 1:
        return render_top_one(
            examples,
            output or config.paths.results_dir / "retrieval-examples-top1.jpg",
            dpi,
            width=style.CVWW.column if width_cm is None else width_cm / 2.54,
        )
    return render_examples(
        examples,
        output or config.paths.results_dir / "retrieval-examples.jpg",
        dpi,
        width=style.CVWW.full if width_cm is None else width_cm / 2.54,
    )

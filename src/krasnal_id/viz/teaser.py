"""The paper's page-1 teaser: one query against a family of lookalikes.

The paper asks which statue a photograph shows, and page 1 is where a reader decides what
the paper is about, so the teaser answers the question visually: a query photograph,
then every member of one near-identical family with the cosine similarity of its
best-matching photograph, the correct statue marked.

**The query is chosen by a rule, never by eye**, and the caption states the rule:
among the family's leave-one-out queries that the backbone answers correctly, the
one whose winning margin over the next statue is smallest. That is the hardest case
the model still gets right -- the point the figure makes -- and a stated rule is
what keeps a teaser from being a cherry-pick. Ties go to the lower image ID.
"""

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from krasnal_id.config import AppConfig, VisualizationExperimentConfig
from krasnal_id.embeddings.store import EmbeddingMatrix, load_embedding_matrix
from krasnal_id.experiments.baseline_accuracy import (
    BaselineExperimentError,
    load_evaluation_inputs,
)
from krasnal_id.models import DatasetManifest, EvaluationSplit
from krasnal_id.viz import figure_style as style
from krasnal_id.viz.embedding_plot import VisualizationError
from krasnal_id.viz.retrieval_examples import _badge, _frame, _load_cell, _setup, _write


@dataclass(frozen=True, slots=True)
class Candidate:
    """One family member, represented by its best-matching reference photograph."""

    dwarf_id: str
    display_name: str
    local_path: Path
    similarity: float
    correct: bool


@dataclass(frozen=True, slots=True)
class Teaser:
    """The chosen query and the family it is ranked against."""

    query_image_id: str
    query_local_path: Path
    margin: float
    candidates: tuple[Candidate, ...]


def select_teaser(
    manifest: DatasetManifest,
    split: EvaluationSplit,
    matrix: EmbeddingMatrix,
    family: Sequence[str],
) -> Teaser:
    """The family query answered correctly by the smallest margin, with its candidates."""
    by_name = {dwarf.display_name: dwarf.dwarf_id for dwarf in manifest.dwarfs}
    unknown = [name for name in family if name not in by_name]
    if unknown:
        raise VisualizationError(f"no statue in the manifest is named {unknown}")
    if len(set(family)) < 2:
        raise VisualizationError("a lookalike family needs at least two statues")
    members = [by_name[name] for name in family]
    names = {by_name[name]: name for name in family}
    paths = {image.image_id: Path(image.local_path) for image in manifest.images}

    best: tuple[float, str, Teaser] | None = None
    for fold in split.folds:
        if fold.query_dwarf_id not in members:
            continue
        vectors, dwarf_ids = matrix.rows_for(fold.reference_image_ids)
        similarities = vectors @ matrix.vector_for(fold.query_image_id)
        statue_best: dict[str, tuple[float, str]] = {}
        for image_id, dwarf_id, value in zip(
            fold.reference_image_ids, dwarf_ids, similarities, strict=True
        ):
            current = statue_best.get(dwarf_id)
            if current is None or value > current[0]:
                statue_best[dwarf_id] = (float(value), image_id)
        truth = fold.query_dwarf_id
        rivals = [score for dwarf, (score, _) in statue_best.items() if dwarf != truth]
        margin = statue_best[truth][0] - max(rivals)
        if margin <= 0:
            continue  # answered wrongly: the teaser shows the hardest *correct* case
        key = (margin, fold.query_image_id)
        if best is not None and key >= (best[0], best[1]):
            continue
        candidates = tuple(
            sorted(
                (
                    Candidate(
                        dwarf_id=member,
                        display_name=names[member],
                        local_path=paths[statue_best[member][1]],
                        similarity=statue_best[member][0],
                        correct=member == truth,
                    )
                    for member in members
                ),
                key=lambda candidate: -candidate.similarity,
            )
        )
        best = (
            margin,
            fold.query_image_id,
            Teaser(fold.query_image_id, paths[fold.query_image_id], margin, candidates),
        )
    if best is None:
        raise VisualizationError(f"the backbone answers no query of {list(family)} correctly")
    return best[2]


def _label(name: str, prefix: str) -> str:
    """A family member's distinguishing word: the family shares the rest of its name."""
    return name.removeprefix(prefix).strip() or name


def render_teaser(teaser: Teaser, path: Path, dpi: int, width: float) -> Path:
    """Draw the query, then the family in similarity order, at the given print width."""
    pyplot = _setup()
    columns = 1 + len(teaser.candidates)
    cell = width / columns
    figure, axes_row = pyplot.subplots(  # type: ignore[attr-defined]
        1,
        columns,
        figsize=(width, cell * 1.24),
        squeeze=False,
        layout="constrained",
        gridspec_kw={"wspace": 0.02},
    )
    names = [candidate.display_name for candidate in teaser.candidates]
    shared = names[0]
    for name in names[1:]:
        while not name.startswith(shared):
            shared = shared[:-1]
    query = axes_row[0][0]
    query.imshow(_load_cell(teaser.query_local_path))
    _frame(query, None)
    query.set_title("Query", fontsize=style.SIZE_ANNOTATION, color=style.INK, pad=2)
    for axes, candidate in zip(axes_row[0][1:], teaser.candidates, strict=True):
        axes.imshow(_load_cell(candidate.local_path))
        _frame(axes, candidate.correct)
        _badge(axes, candidate.correct)
        axes.set_title(
            f"{_label(candidate.display_name, shared)}\n{candidate.similarity:.3f}",
            fontsize=style.SIZE_ANNOTATION,
            color=style.INK if candidate.correct else style.INK_SECONDARY,
            pad=2,
        )
    return _write(pyplot, figure, path, dpi)


def create_teaser_selection(config: AppConfig, family: Sequence[str]) -> Teaser:
    """The teaser's query and candidates, without drawing anything."""
    if not isinstance(config.experiment, VisualizationExperimentConfig):
        raise VisualizationError(
            f"the teaser requires experiment=visualization, got {config.experiment.kind}"
        )
    try:
        manifest, split = load_evaluation_inputs(
            config.paths.manifest_path, config.paths.evaluation_split_path
        )
    except BaselineExperimentError as error:
        raise VisualizationError(str(error)) from error
    matrix = load_embedding_matrix(manifest, config.backbone, Path(config.paths.embeddings_dir))
    return select_teaser(manifest, split, matrix, family)


def create_teaser(
    config: AppConfig,
    family: Sequence[str],
    output: Path,
    dpi: int = 300,
    width_cm: float | None = None,
) -> tuple[Path, Teaser]:
    """Select the teaser's query from the data and draw it."""
    teaser = create_teaser_selection(config, family)
    width = style.CVWW.column if width_cm is None else width_cm / 2.54
    return render_teaser(teaser, output, dpi, width), teaser


__all__ = [
    "Candidate",
    "Teaser",
    "create_teaser",
    "create_teaser_selection",
    "render_teaser",
    "select_teaser",
]

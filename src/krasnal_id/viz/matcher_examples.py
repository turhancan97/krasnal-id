"""Qualitative matcher examples for the paper's supplement, chosen by stated rules.

A correspondence figure is the one place a reader can see *why* a learned matcher
re-ranks where SIFT does not. It is also the figure most easily cherry-picked, so
every pair here is chosen by a rule the caption repeats:

* **A rescue.** Among the photographer-disjoint queries whose correct statue cosine
  ranks below first, the one where DISK+LightGlue's verified matches favour the
  correct statue over every wrong candidate by the widest margin. Ties go to the
  lower image ID. It is the clearest case of the mechanism working.
* **A failure** is the teaser's pair (`viz.teaser`), chosen by the teaser's own rule.

Each candidate is shown by the photograph the re-ranking actually matched: that
statue's best-cosine reference once the query's photographer is withheld, which is
exactly how `experiment matcher-rerank` chose it. The selection reads inlier counts
from the journal; drawing re-runs the matcher, so a learned matcher's drawn count
can differ from the journal's by a match or two.
"""

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

from krasnal_id.embeddings.store import EmbeddingMatrix
from krasnal_id.experiments.rerank_ablation import QueryEvidence
from krasnal_id.models import DatasetManifest, EvaluationSplit
from krasnal_id.photographers import authors_by_image
from krasnal_id.viz.embedding_plot import VisualizationError


@dataclass(frozen=True, slots=True)
class Rescue:
    """One query cosine gets wrong and geometry gets right."""

    query_image_id: str
    correct_dwarf_id: str
    cosine_choice_dwarf_id: str
    correct_inliers: int
    best_wrong_inliers: int


def select_rescue(evidence: Mapping[str, QueryEvidence]) -> Rescue:
    """The cosine miss whose correct statue geometry favours by the widest margin."""
    best: tuple[int, str, Rescue] | None = None
    for query, item in evidence.items():
        if not item.candidates or item.candidates[0].correct:
            continue  # cosine already right, or nothing to re-rank
        truths = [c for c in item.candidates if c.correct]
        if not truths:
            continue  # the correct statue never reached the shortlist
        wrong = max(c.inliers for c in item.candidates if not c.correct)
        margin = truths[0].inliers - wrong
        if margin <= 0:
            continue
        key = (-margin, query)
        if best is not None and key >= (best[0], best[1]):
            continue
        best = (
            -margin,
            query,
            Rescue(
                query_image_id=query,
                correct_dwarf_id=truths[0].dwarf_id,
                cosine_choice_dwarf_id=item.candidates[0].dwarf_id,
                correct_inliers=truths[0].inliers,
                best_wrong_inliers=wrong,
            ),
        )
    if best is None:
        raise VisualizationError("no cosine miss is rescued by geometry in this journal")
    return best[2]


def matched_photograph(
    query_image_id: str,
    dwarf_id: str,
    split: EvaluationSplit,
    manifest: DatasetManifest,
    matrix: EmbeddingMatrix,
) -> Path:
    """The statue photograph the re-ranking matched: best cosine, photographer withheld."""
    authors = authors_by_image(manifest)
    fold = next((f for f in split.folds if f.query_image_id == query_image_id), None)
    if fold is None:
        raise VisualizationError(f"{query_image_id} is not a query of this split")
    photographer = authors[query_image_id]
    dwarf_of = dict(zip(matrix.image_ids, matrix.dwarf_ids, strict=True))
    options = [
        image_id
        for image_id in fold.reference_image_ids
        if dwarf_of[image_id] == dwarf_id and authors[image_id] != photographer
    ]
    if not options:
        raise VisualizationError(
            f"{dwarf_id} has no reference outside {photographer}'s photographs"
        )
    query = matrix.vector_for(query_image_id)
    best = max(options, key=lambda image_id: (float(matrix.vector_for(image_id) @ query), image_id))
    paths = {image.image_id: Path(image.local_path) for image in manifest.images}
    return paths[best]

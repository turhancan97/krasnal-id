"""The page-1 teaser: its query is chosen by a stated rule, never by eye."""

from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import numpy.typing as npt
import pytest

from helpers import FAKE_BACKBONE, seed_embedding_cache, synthetic_manifest
from krasnal_id.data_pipeline.build_split import build_evaluation_split
from krasnal_id.embeddings.store import load_embedding_matrix
from krasnal_id.experiments.baseline_accuracy import evaluate_fold
from krasnal_id.viz.embedding_plot import VisualizationError
from krasnal_id.viz.teaser import select_teaser


def _lookalike_vector(dwarf_index: int, position: int, dwarf_count: int) -> npt.NDArray[np.float32]:
    """Statues 0-2 share one design plus a small mark of their own; the rest stand apart.

    Dimension 0 is the shared design, dimensions 1-3 the three marks, and every
    other statue gets a dimension of its own beyond them.
    """
    rng = np.random.default_rng(97 * dwarf_index + position)
    vector = np.zeros(dwarf_count + 4, dtype=np.float32)
    if dwarf_index < 3:
        vector[0] = 1.0
        vector[1 + dwarf_index] = 0.35
    else:
        vector[4 + dwarf_index] = 1.0
    vector += rng.normal(0.0, 0.25, size=vector.shape).astype(np.float32)
    return np.asarray(vector / np.linalg.norm(vector), dtype=np.float32)


def test_the_hardest_correct_family_query_is_chosen(tmp_path: Path) -> None:
    manifest = synthetic_manifest(dwarf_count=6, per_dwarf=4)
    seed_embedding_cache(tmp_path, manifest, vector_for=_lookalike_vector)
    matrix = load_embedding_matrix(manifest, FAKE_BACKBONE, tmp_path)
    split = build_evaluation_split(manifest, datetime.now(UTC))
    family = ["Dwarf 1", "Dwarf 2", "Dwarf 3"]

    teaser = select_teaser(manifest, split, matrix, family)

    # Every member is shown, in similarity order, with exactly one marked correct.
    assert {c.display_name for c in teaser.candidates} == set(family)
    assert [c.similarity for c in teaser.candidates] == sorted(
        (c.similarity for c in teaser.candidates), reverse=True
    )
    assert sum(c.correct for c in teaser.candidates) == 1
    assert teaser.margin > 0

    # No other correctly answered family query wins by a smaller margin.
    members = {"Q1", "Q2", "Q3"}
    for fold in split.folds:
        if fold.query_dwarf_id not in members:
            continue
        outcome = evaluate_fold(
            fold.query_image_id, fold.query_dwarf_id, fold.reference_image_ids, matrix
        )
        if outcome.dwarf_rank != 1:
            continue
        vectors, dwarf_ids = matrix.rows_for(fold.reference_image_ids)
        scores = vectors @ matrix.vector_for(fold.query_image_id)
        best: dict[str, float] = {}
        for dwarf, score in zip(dwarf_ids, scores, strict=True):
            best[dwarf] = max(best.get(dwarf, -2.0), float(score))
        margin = best[fold.query_dwarf_id] - max(
            v for d, v in best.items() if d != fold.query_dwarf_id
        )
        assert margin >= teaser.margin - 1e-6


def test_a_family_the_manifest_does_not_hold_is_refused(tmp_path: Path) -> None:
    manifest = synthetic_manifest(dwarf_count=3, per_dwarf=3)
    seed_embedding_cache(tmp_path, manifest)
    matrix = load_embedding_matrix(manifest, FAKE_BACKBONE, tmp_path)
    split = build_evaluation_split(manifest, datetime.now(UTC))

    with pytest.raises(VisualizationError, match="no statue in the manifest"):
        select_teaser(manifest, split, matrix, ["Dwarf 1", "Nobody"])
    with pytest.raises(VisualizationError, match="at least two statues"):
        select_teaser(manifest, split, matrix, ["Dwarf 1"])

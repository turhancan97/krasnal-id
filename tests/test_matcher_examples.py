"""The supplement's matcher examples are chosen by a stated rule, never by eye."""

from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pytest

from helpers import FAKE_BACKBONE, seed_embedding_cache, synthetic_manifest
from krasnal_id.data_pipeline.build_split import build_evaluation_split
from krasnal_id.embeddings.store import load_embedding_matrix
from krasnal_id.experiments.rerank_ablation import Candidate, QueryEvidence
from krasnal_id.viz.embedding_plot import VisualizationError
from krasnal_id.viz.matcher_examples import matched_photograph, select_rescue


def _evidence(query: str, *candidates: tuple[str, int, bool]) -> QueryEvidence:
    rows = tuple(
        Candidate(dwarf_id=d, cosine=1.0 - i / 10, inliers=n, correct=c)
        for i, (d, n, c) in enumerate(candidates)
    )
    rank = next((i + 1 for i, row in enumerate(rows) if row.correct), 0)
    return QueryEvidence(query_image_id=query, candidates=rows, baseline_rank=rank)


def test_the_widest_geometric_lead_on_a_cosine_miss_is_chosen() -> None:
    evidence = {
        # Cosine is right: never a rescue, however large its lead.
        "a": _evidence("a", ("Q1", 500, True), ("Q2", 0, False)),
        # A miss geometry leads on by 20.
        "b": _evidence("b", ("Q2", 10, False), ("Q1", 30, True), ("Q3", 4, False)),
        # A miss geometry leads on by 40: the winner.
        "c": _evidence("c", ("Q3", 2, False), ("Q4", 50, True), ("Q5", 10, False)),
        # A miss geometry also gets wrong.
        "d": _evidence("d", ("Q2", 60, False), ("Q1", 30, True)),
        # The correct statue never reached the shortlist.
        "e": _evidence("e", ("Q2", 1, False), ("Q3", 0, False)),
    }
    rescue = select_rescue(evidence)
    assert rescue.query_image_id == "c"
    assert rescue.correct_dwarf_id == "Q4"
    assert rescue.cosine_choice_dwarf_id == "Q3"
    assert (rescue.correct_inliers, rescue.best_wrong_inliers) == (50, 10)


def test_ties_go_to_the_lower_image_id_and_no_rescue_is_refused() -> None:
    tied = {
        "z": _evidence("z", ("Q2", 0, False), ("Q1", 9, True)),
        "m": _evidence("m", ("Q2", 0, False), ("Q1", 9, True)),
    }
    assert select_rescue(tied).query_image_id == "m"
    with pytest.raises(VisualizationError, match="no cosine miss"):
        select_rescue({"a": _evidence("a", ("Q1", 5, True), ("Q2", 9, False))})


def test_the_matched_photograph_withholds_the_query_photographer(tmp_path: Path) -> None:
    manifest = synthetic_manifest(dwarf_count=3, per_dwarf=3)
    images = list(manifest.images)
    # The query's photographer also took one photograph of the candidate statue.
    query = images[0]
    candidate = [image for image in images if image.dwarf_id == "Q2"]
    images[images.index(candidate[0])] = candidate[0].model_copy(update={"author": query.author})
    images[images.index(candidate[1])] = candidate[1].model_copy(update={"author": "Someone"})
    images[images.index(candidate[2])] = candidate[2].model_copy(update={"author": "Other"})
    manifest = manifest.model_copy(update={"images": tuple(images)})
    seed_embedding_cache(tmp_path, manifest)
    matrix = load_embedding_matrix(manifest, FAKE_BACKBONE, tmp_path)
    split = build_evaluation_split(manifest, datetime.now(UTC))

    chosen = matched_photograph(query.image_id, "Q2", split, manifest, matrix)

    allowed = {Path(candidate[1].local_path), Path(candidate[2].local_path)}
    assert chosen in allowed
    scores = {
        Path(image.local_path): float(
            matrix.vector_for(image.image_id) @ matrix.vector_for(query.image_id)
        )
        for image in candidate[1:]
    }
    assert scores[chosen] == pytest.approx(max(scores.values()))
    assert np.isfinite(scores[chosen])
    with pytest.raises(VisualizationError, match="not a query"):
        matched_photograph("nope", "Q2", split, manifest, matrix)


def test_a_pair_is_drawn_at_the_query_height() -> None:
    pytest.importorskip("cv2")
    from krasnal_id.viz.match_plot import _side_by_side

    left = np.zeros((100, 80, 3), dtype=np.uint8)
    right = np.zeros((200, 50, 3), dtype=np.uint8)
    canvas, scale = _side_by_side(left, right, gap=10)
    assert scale == 0.5
    assert canvas.shape == (100, 80 + 10 + 25, 3)

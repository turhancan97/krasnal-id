"""Cosine k-NN retrieval behavior, determinism, and input validation."""

import numpy as np
import pytest

from krasnal_id.retrieval.knn import RetrievalError, cosine_knn


def _vector(*values: float) -> np.ndarray:
    return np.asarray(values, dtype=np.float32)


def test_ranks_references_by_descending_cosine_similarity() -> None:
    query = _vector(1.0, 0.0)
    references = np.asarray(
        [[0.0, 1.0], [1.0, 0.0], [0.7071068, 0.7071068], [-1.0, 0.0]],
        dtype=np.float32,
    )

    result = cosine_knn(
        "query-1",
        query,
        references,
        ("orthogonal", "identical", "diagonal", "opposite"),
        ("Q2", "Q1", "Q3", "Q4"),
        top_k=4,
    )

    assert result.query_image_id == "query-1"
    assert [match.image_id for match in result.matches] == [
        "identical",
        "diagonal",
        "orthogonal",
        "opposite",
    ]
    assert [match.rank for match in result.matches] == [1, 2, 3, 4]
    assert [match.dwarf_id for match in result.matches] == ["Q1", "Q3", "Q2", "Q4"]
    similarities = [match.cosine_similarity for match in result.matches]
    assert similarities[0] == pytest.approx(1.0)
    assert similarities[1] == pytest.approx(0.7071068, abs=1e-6)
    assert similarities[2] == pytest.approx(0.0)
    assert similarities[3] == pytest.approx(-1.0)
    assert similarities == sorted(similarities, reverse=True)


def test_identical_similarities_break_ties_by_ascending_image_id() -> None:
    query = _vector(1.0, 0.0)
    references = np.tile(_vector(1.0, 0.0), (3, 1))
    ids = ("image-c", "image-a", "image-b")

    result = cosine_knn("query-1", query, references, ids, ("Q1", "Q2", "Q3"), top_k=3)
    reversed_result = cosine_knn(
        "query-1",
        query,
        np.tile(_vector(1.0, 0.0), (3, 1)),
        tuple(reversed(ids)),
        ("Q3", "Q2", "Q1"),
        top_k=3,
    )

    ranked = [match.image_id for match in result.matches]
    assert ranked == ["image-a", "image-b", "image-c"]
    assert [match.image_id for match in reversed_result.matches] == ranked
    # The dwarf ID must still travel with its own image, not with the rank.
    assert {(m.image_id, m.dwarf_id) for m in result.matches} == {
        ("image-c", "Q1"),
        ("image-a", "Q2"),
        ("image-b", "Q3"),
    }


def test_top_k_truncates_and_a_small_pool_returns_every_reference() -> None:
    query = _vector(1.0, 0.0, 0.0)
    references = np.asarray(
        [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
        dtype=np.float32,
    )
    ids = ("image-1", "image-2", "image-3")
    dwarfs = ("Q1", "Q2", "Q3")

    assert len(cosine_knn("query-1", query, references, ids, dwarfs, top_k=1).matches) == 1
    assert len(cosine_knn("query-1", query, references, ids, dwarfs, top_k=3).matches) == 3
    # A pool smaller than top_k yields the whole pool rather than an error.
    assert len(cosine_knn("query-1", query, references, ids, dwarfs, top_k=25).matches) == 3


def test_unnormalized_inputs_yield_a_true_cosine_ranking() -> None:
    scaled = cosine_knn(
        "query-1",
        _vector(3.0, 4.0),
        np.asarray([[6.0, 8.0], [-30.0, -40.0]], dtype=np.float32),
        ("same-direction", "opposite"),
        ("Q1", "Q2"),
        top_k=2,
    )

    assert scaled.matches[0].image_id == "same-direction"
    assert scaled.matches[0].cosine_similarity == pytest.approx(1.0)
    assert scaled.matches[1].cosine_similarity == pytest.approx(-1.0)


def test_similarities_stay_inside_the_validated_range() -> None:
    # A duplicated high-dimensional vector is the case where an unclipped dot
    # product drifts above 1.0 and would fail RetrievalMatch validation.
    rng = np.random.default_rng(17)
    query = rng.normal(size=1024).astype(np.float32)
    query /= np.linalg.norm(query)
    references = np.tile(query, (4, 1))

    result = cosine_knn(
        "query-1",
        query,
        references,
        ("a", "b", "c", "d"),
        ("Q1", "Q1", "Q2", "Q2"),
        top_k=4,
    )

    for match in result.matches:
        assert -1.0 <= match.cosine_similarity <= 1.0
        assert match.cosine_similarity == pytest.approx(1.0)


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"query_image_id": ""}, "query_image_id must not be empty"),
        ({"top_k": 0}, "top_k must be positive"),
        ({"top_k": -3}, "top_k must be positive"),
        ({"reference_dwarf_ids": ("Q1",)}, "dwarf IDs"),
        ({"reference_image_ids": (), "reference_dwarf_ids": ()}, "at least one reference"),
        (
            {"reference_image_ids": ("image-1", "image-1")},
            "reference image IDs must be unique",
        ),
        ({"reference_image_ids": ("image-1", "")}, "must not be empty"),
        ({"reference_dwarf_ids": ("Q1", "")}, "must not be empty"),
        ({"query_image_id": "image-2"}, "cannot appear in its reference set"),
        ({"query": np.zeros(2, dtype=np.float32)}, "positive finite norm"),
        ({"query": np.asarray([[1.0, 0.0]], dtype=np.float32)}, "one-dimensional"),
        ({"query": _vector(np.nan, 1.0)}, "non-finite"),
        ({"query": _vector(1.0, 0.0, 0.0)}, "does not match query dimension"),
        ({"references": np.zeros((2, 2), dtype=np.float32)}, "positive finite norm"),
        ({"references": np.zeros((0, 2), dtype=np.float32)}, "two-dimensional"),
        (
            {"references": np.asarray([[np.nan, 1.0], [1.0, 0.0]], dtype=np.float32)},
            "non-finite",
        ),
        (
            {"references": np.asarray([[1.0, 0.0]], dtype=np.float32)},
            "rows for 2 reference IDs",
        ),
    ],
)
def test_invalid_inputs_are_rejected(kwargs: dict[str, object], message: str) -> None:
    arguments: dict[str, object] = {
        "query_image_id": "query-1",
        "query": _vector(1.0, 0.0),
        "references": np.asarray([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32),
        "reference_image_ids": ("image-1", "image-2"),
        "reference_dwarf_ids": ("Q1", "Q2"),
        "top_k": 2,
    }
    arguments.update(kwargs)

    with pytest.raises(RetrievalError, match=message):
        cosine_knn(**arguments)  # type: ignore[arg-type]

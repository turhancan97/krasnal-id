"""The vector-only reviewer checks of AGENTS.md section 7.15."""

import math
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import numpy.typing as npt
import pytest

from helpers import FAKE_BACKBONE, seed_embedding_cache, synthetic_manifest
from krasnal_id.config import (
    GeoAblationConfig,
    OpenSetExperimentConfig,
    RecallCurveConfig,
    WorkshopValidationConfig,
    load_config,
)
from krasnal_id.data_pipeline.build_split import build_evaluation_split
from krasnal_id.embeddings.store import EmbeddingMatrix, load_embedding_matrix
from krasnal_id.experiments.baseline_accuracy import evaluate_fold
from krasnal_id.experiments.geo_ablation import located_inputs, measure_geographic_pool
from krasnal_id.experiments.open_set import (
    score_known_queries,
    score_unknown_queries,
    summarize_open_set,
)
from krasnal_id.experiments.pool_size_ablation import measure_pool_size, resolve_pool_sizes
from krasnal_id.experiments.recall_curve import ARMS, DISJOINT, measure_arm
from krasnal_id.experiments.reviewer_checks import (
    families_part,
    geo_noise_part,
    geo_permutation_part,
    groups_part,
    lookalike_families,
    mutual_pairs,
    null_bounds_part,
    rank_with_prior,
    rejection_signals_part,
    soft_prior_part,
    statue_ranking,
    weighted_auroc,
)
from krasnal_id.experiments.workshop_validation import WorkshopValidationError
from krasnal_id.geometry import haversine_metres, offset_metres
from krasnal_id.models import DatasetManifest, EvaluationSplit
from krasnal_id.statistics import separability_auroc

DWARFS = 8


def _noisy_vector(dwarf_index: int, position: int, dwarf_count: int) -> npt.NDArray[np.float32]:
    """Statues that overlap, so the ranking makes mistakes a prior could fix."""
    rng = np.random.default_rng(1000 * dwarf_index + position)
    vector = np.zeros(dwarf_count + 4, dtype=np.float32)
    vector[dwarf_index] = 1.0
    vector[(dwarf_index + 1) % dwarf_count] = 0.6  # each statue leans on its neighbour
    vector += rng.normal(0.0, 0.45, size=vector.shape).astype(np.float32)
    return np.asarray(vector / np.linalg.norm(vector), dtype=np.float32)


def _located_world(
    tmp_path: Path,
) -> tuple[DatasetManifest, EvaluationSplit, EmbeddingMatrix]:
    # Statues along a line at uneven spacing (at least 300 m), so no statue has two
    # neighbours at exactly the same distance for a tiny move to reorder.
    places = {
        f"Q{i}": offset_metres((51.11, 17.03), 300.0 * i + 41.0 * i * i, math.pi / 2)
        for i in range(1, DWARFS + 1)
    }
    manifest = synthetic_manifest(dwarf_count=DWARFS, per_dwarf=4, coordinates=places)
    seed_embedding_cache(tmp_path, manifest, vector_for=_noisy_vector)
    matrix = load_embedding_matrix(manifest, FAKE_BACKBONE, tmp_path)
    return manifest, build_evaluation_split(manifest, datetime.now(UTC)), matrix


def _settings(**overrides: object) -> WorkshopValidationConfig:
    base = load_config(["experiment=workshop_validation"]).experiment
    assert isinstance(base, WorkshopValidationConfig)
    return base.model_copy(update={"bootstrap_resamples": 200, "folds": 2, **overrides})


def _released_geo(
    manifest: DatasetManifest, split: EvaluationSplit, matrix: EmbeddingMatrix
) -> dict[str, float]:
    geo_settings = load_config(["experiment=geo_ablation"]).experiment
    assert isinstance(geo_settings, GeoAblationConfig)
    located, dwarf_ids, _, locations = located_inputs(manifest, split, matrix)
    released: dict[str, float] = {}
    for size in resolve_pool_sizes(geo_settings.pool_sizes, len(dwarf_ids)):
        released[f"geo_top_1_pool_{size}"] = measure_geographic_pool(
            located, matrix, size, locations
        ).top_1
        released[f"random_top_1_pool_{size}"] = measure_pool_size(
            located, matrix, size, geo_settings.seeds, dwarf_ids
        ).top_1
    return released


def test_an_offset_is_the_distance_it_was_asked_for() -> None:
    for bearing in (0.0, 1.0, math.pi, 4.0):
        moved = offset_metres((51.11, 17.03), 150.0, bearing)
        assert haversine_metres((51.11, 17.03), moved) == pytest.approx(150.0, abs=1e-3)


def test_the_statue_ranking_is_the_baselines_to_the_tie(tmp_path: Path) -> None:
    _, split, matrix = _located_world(tmp_path)
    for fold in split.folds:
        ranking = statue_ranking(
            fold.query_image_id, fold.query_dwarf_id, fold.reference_image_ids, matrix
        )
        outcome = evaluate_fold(
            fold.query_image_id, fold.query_dwarf_id, fold.reference_image_ids, matrix
        )
        assert ranking.truth + 1 == outcome.dwarf_rank
        assert rank_with_prior(ranking, np.zeros_like(ranking.best)) == outcome.dwarf_rank
        # A bonus on the truth alone, big enough, always wins.
        bonus = np.zeros_like(ranking.best)
        bonus[ranking.truth] = 5.0
        assert rank_with_prior(ranking, bonus) == 1


def test_the_weighted_auroc_is_the_auroc_and_respects_multiplicity() -> None:
    known = np.asarray([0.9, 0.7, 0.7, 0.4])
    unknown = np.asarray([0.8, 0.7, 0.2])
    ones_k, ones_u = np.ones(4), np.ones(3)
    assert weighted_auroc(known, unknown, ones_k, ones_u) == pytest.approx(
        separability_auroc(known, unknown)
    )
    doubled = weighted_auroc(known, unknown, np.asarray([2.0, 1, 1, 0]), ones_u)
    assert doubled == pytest.approx(separability_auroc(np.asarray([0.9, 0.9, 0.7, 0.7]), unknown))


def test_the_soft_prior_reproduces_its_control_and_reports_held_out(tmp_path: Path) -> None:
    manifest, split, matrix = _located_world(tmp_path)
    released = _released_geo(manifest, split, matrix)
    metrics = {
        m.name: m for m in soft_prior_part("dinov2", split, manifest, matrix, released, _settings())
    }
    none = metrics["dinov2_soft_prior_none_top_1"]
    assert none.value == pytest.approx(released[f"geo_top_1_pool_{DWARFS}"])
    for error in (10, 25, 50):
        label = f"dinov2_soft_prior_gps{error}"
        # The zero weight of the sweep is the control, whatever the GPS error.
        assert metrics[f"{label}_w0_top_1"].value == pytest.approx(none.value)
        gain = metrics[f"{label}_gain"]
        assert gain.value == pytest.approx(metrics[f"{label}_top_1"].value - none.value)
        assert gain.lower_bound is not None and gain.upper_bound is not None
        assert {f"{label}_fold_0_weight", f"{label}_fold_1_weight"} <= set(metrics)
        # Statues at least 300 m apart, user within 50 m: location alone finds every one.
        assert metrics[f"{label}_location_only_top_1"].value == pytest.approx(1.0)

    drifted = {**released, f"geo_top_1_pool_{DWARFS}": 0.0}
    with pytest.raises(WorkshopValidationError, match="inputs have changed"):
        soft_prior_part("dinov2", split, manifest, matrix, drifted, _settings())


def test_vanishing_noise_reproduces_the_released_geographic_gap(tmp_path: Path) -> None:
    manifest, split, matrix = _located_world(tmp_path)
    released = _released_geo(manifest, split, matrix)
    settings = _settings(noise_metres=(0.001, 150.0), noise_seeds=(0, 1))
    metrics = {
        m.name: m for m in geo_noise_part("dinov2", split, manifest, matrix, released, settings)
    }
    for size in (2, 3, 5):
        gap = metrics[f"dinov2_geo_minus_random_pool_{size}_noise_0.001"]
        expected = released[f"geo_top_1_pool_{size}"] - released[f"random_top_1_pool_{size}"]
        assert gap.value == pytest.approx(expected)
        assert f"dinov2_geo_minus_random_pool_{size}_noise_150" in metrics


def test_rejection_signals_reproduce_the_released_cosine_result(tmp_path: Path) -> None:
    _, split, matrix = _located_world(tmp_path)
    open_set = load_config(["experiment=open_set"]).experiment
    assert isinstance(open_set, OpenSetExperimentConfig)
    released = {
        m.name: m.value
        for m in summarize_open_set(
            score_known_queries(split, matrix), score_unknown_queries(matrix), open_set
        )
    }
    metrics = {
        m.name: m
        for m in rejection_signals_part("dinov2", split, matrix, released, _settings(knn_k=2))
    }
    assert metrics["dinov2_reject_cosine_auroc"].value == pytest.approx(released["auroc"])
    assert metrics["dinov2_reject_cosine_false_acceptance_90"].value == pytest.approx(
        released["target_90_false_acceptance"]
    )
    for signal in ("margin", "knn"):
        auroc = metrics[f"dinov2_reject_{signal}_auroc"]
        assert 0.0 <= auroc.value <= 1.0
        assert auroc.lower_bound is not None and auroc.lower_bound <= auroc.value
        difference = metrics[f"dinov2_reject_{signal}_minus_cosine_auroc"]
        assert difference.value == pytest.approx(
            auroc.value - metrics["dinov2_reject_cosine_auroc"].value
        )

    with pytest.raises(WorkshopValidationError, match="inputs have changed"):
        rejection_signals_part(
            "dinov2", split, matrix, {**released, "auroc": 0.0}, _settings(knn_k=2)
        )


def test_merging_a_group_counts_its_members_as_correct(tmp_path: Path) -> None:
    _, split, matrix = _located_world(tmp_path)
    top_1 = sum(
        evaluate_fold(f.query_image_id, f.query_dwarf_id, f.reference_image_ids, matrix).dwarf_rank
        == 1
        for f in split.folds
    ) / len(split.folds)
    released = {"top_1": top_1}
    everything = {"Q1": tuple(f"Q{i}" for i in range(2, DWARFS + 1))}
    metrics = {
        m.name: m.value
        for m in groups_part("dinov2", split, matrix, released, _settings(groups=everything))
    }
    # One family holding every statue: every answer is right.
    assert metrics["dinov2_groups_merged_top_1"] == pytest.approx(1.0)
    assert metrics["dinov2_groups_merged_gain"] == pytest.approx(1.0 - top_1)
    assert metrics["dinov2_groups_changed_queries"] == pytest.approx(
        round((1.0 - top_1) * len(split.folds))
    )
    with pytest.raises(WorkshopValidationError, match="unknown statue"):
        groups_part("dinov2", split, matrix, released, _settings(groups={"Q1": ("Q99",)}))


def test_the_reviewer_settings_refuse_what_would_measure_nothing() -> None:
    base = load_config(["experiment=workshop_validation"]).experiment
    assert isinstance(base, WorkshopValidationConfig)
    raw = base.model_dump()
    for bad in (
        {"soft_prior_weights": (0.1, 0.2)},
        {"gps_error_m": (0.0,)},
        {"noise_metres": (0.0,)},
        {"wide_matchers": ("sift",), "wide_top_k": None},
        {"extra_pairs": (("loftr", "loftr"),)},
    ):
        with pytest.raises(ValueError):
            WorkshopValidationConfig.model_validate({**raw, **bad})


# ---------------------------------------------------------------- section 7.16


def test_the_permutation_test_reproduces_then_reports_a_p_value(tmp_path: Path) -> None:
    manifest, split, matrix = _located_world(tmp_path)
    released = _released_geo(manifest, split, matrix)
    settings = _settings(permutation_count=19, permutation_pool_sizes=(2, 3, 50))
    metrics = {
        m.name: m
        for m in geo_permutation_part("dinov2", split, manifest, matrix, released, settings)
    }
    for size in (2, 3):
        p_value = metrics[f"dinov2_geo_permutation_pool_{size}_p_value"].value
        # Counting the real layout, a p-value is a multiple of 1/20 and never zero.
        assert 0.05 <= p_value <= 1.0
        assert round(p_value * 20) == pytest.approx(p_value * 20)
    # A pool as large as the whole set has nothing to permute, so it is skipped.
    assert "dinov2_geo_permutation_pool_50_p_value" not in metrics
    assert metrics["dinov2_geo_permutation_permutations"].value == 19

    drifted = {**released, "geo_top_1_pool_2": 0.0}
    with pytest.raises(WorkshopValidationError, match="inputs have changed"):
        geo_permutation_part("dinov2", split, manifest, matrix, drifted, settings)


def _matrix(dwarfs: tuple[str, ...], vectors: list[list[float]]) -> EmbeddingMatrix:
    return EmbeddingMatrix(
        image_ids=tuple(f"i{n}" for n in range(len(dwarfs))),
        dwarf_ids=dwarfs,
        vectors=np.asarray(vectors, dtype=np.float32),
    )


def test_families_need_enough_backbones_to_agree() -> None:
    # A and B are each other's nearest neighbours; so are C and D.
    paired = _matrix(("A", "B", "C", "D"), [[1, 0.1, 0], [1, 0, 0.1], [0, 1, 0], [0.1, 1, 0]])
    # Here A pairs with C instead, and B with D.
    crossed = _matrix(("A", "B", "C", "D"), [[1, 0, 0], [0, 1, 0], [1, 0.1, 0], [0, 1, 0.1]])
    assert mutual_pairs(paired, 1) == {("A", "B"), ("C", "D")}

    agreed = lookalike_families(
        {"x": paired, "y": paired, "z": crossed}, minimum_votes=2, neighbours=1
    )
    assert agreed["A"] == agreed["B"] != agreed["C"] == agreed["D"]
    # With one vote each pairing counts, and the two chains join all four statues.
    chained = lookalike_families({"x": paired, "z": crossed}, minimum_votes=1, neighbours=1)
    assert len(set(chained.values())) == 1
    assert lookalike_families({"x": paired, "z": crossed}, minimum_votes=3, neighbours=1) == {}


def test_three_neighbours_let_a_trio_form_one_family() -> None:
    """The defect the first run hit: mutual nearest neighbours split a trio."""
    # A, B and C are three lookalikes; D stands apart.
    trio = _matrix(
        ("A", "B", "C", "D"), [[1, 0.1, 0.05], [1, 0.05, 0.1], [1, 0.12, 0.12], [0, 0, 1]]
    )
    split = lookalike_families({"x": trio, "y": trio}, minimum_votes=2, neighbours=1)
    assert len({split.get(s) for s in "ABC"}) > 1  # one pair and a leftover
    joined = lookalike_families({"x": trio, "y": trio}, minimum_votes=2, neighbours=2)
    assert joined["A"] == joined["B"] == joined["C"]


def test_family_errors_are_scored_against_the_other_backbones(tmp_path: Path) -> None:
    _, split, matrix = _located_world(tmp_path)
    matrices = {"dinov2": matrix, "clip": matrix, "dinov3": matrix}
    metrics = {m.name: m for m in families_part(split, matrices, _settings())}
    for name in matrices:
        share = metrics[f"{name}_families_error_share"]
        assert 0.0 <= share.value <= 1.0
        assert share.lower_bound is not None and share.upper_bound is not None
        assert metrics[f"{name}_families_statues"].value <= DWARFS
        errors = metrics[f"{name}_families_errors"].value
        assert errors == sum(
            evaluate_fold(
                f.query_image_id, f.query_dwarf_id, f.reference_image_ids, matrix
            ).dwarf_rank
            != 1
            for f in split.folds
        )


def test_null_bounds_reproduce_the_released_recall_and_bound_the_gap(tmp_path: Path) -> None:
    manifest, split, matrix = _located_world(tmp_path)
    # Two photographers per statue, so every query is answerable cross-photographer.
    manifest = manifest.model_copy(
        update={
            "images": tuple(
                image.model_copy(
                    update={"author": "Bo" if image.image_id.endswith(("-1", "-3")) else "Al"}
                )
                for image in manifest.images
            )
        }
    )
    matrices = {"dinov2": matrix, "dinov2-large": matrix}
    recall = load_config(["experiment=recall_curve"]).experiment
    assert isinstance(recall, RecallCurveConfig)
    arm = next(a for a in ARMS if a.name == DISJOINT)
    ranks, total = measure_arm(
        arm,
        split,
        manifest,
        matrices,
        "dinov2",
        recall.model_copy(update={"fuse_backbones": (), "expansion_neighbours": ()}),
    )
    released = {
        f"disjoint_r_at_{k}": sum(0 < r <= k for r in ranks["plain"]) / total for k in (1, 10)
    }
    settings = _settings(null_bound_backbones=("dinov2-large",), null_bound_cutoffs=(1, 10))
    metrics = {m.name: m for m in null_bounds_part(split, manifest, matrices, released, settings)}
    gap = metrics["null_dinov2-large_minus_dinov2_disjoint_r_at_1"]
    # The same vectors under two names: nothing to find, and nothing hidden.
    assert gap.value == 0.0 and gap.lower_bound == gap.upper_bound == 0.0

    with pytest.raises(WorkshopValidationError, match="inputs have changed"):
        null_bounds_part(split, manifest, matrices, {**released, "disjoint_r_at_1": 0.0}, settings)

"""Workshop re-checks from journalled evidence: held-out weights and statue intervals."""

import json
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pytest
from typer.testing import CliRunner

from helpers import FAKE_BACKBONE, seed_embedding_cache, synthetic_manifest
from krasnal_id.bootstrap import (
    BootstrapError,
    generator,
    per_statue,
    percentile_interval,
    resampled_rate,
    statue_draws,
)
from krasnal_id.cli import app
from krasnal_id.config import (
    PhotographerGapConfig,
    PoolSizeAblationConfig,
    WorkshopValidationConfig,
    load_config,
)
from krasnal_id.data_pipeline.build_split import build_evaluation_split, write_evaluation_split
from krasnal_id.embeddings.store import EmbeddingMatrix, load_embedding_matrix
from krasnal_id.experiments import workshop_validation
from krasnal_id.experiments.photographer_gap import evaluate_conditions
from krasnal_id.experiments.pool_size_ablation import (
    measure_pool_size,
    points_per_doubling,
    resolve_pool_sizes,
)
from krasnal_id.experiments.rerank_ablation import Candidate, QueryEvidence, evidence_to_row
from krasnal_id.experiments.workshop_validation import (
    COSINE,
    REPRODUCTION_TOLERANCE,
    WorkshopValidationError,
    check_reproduced,
    cross_validate,
    eligible_queries,
    journal_path,
    load_journal_strictly,
    mean_by_query,
    paired_gap,
    photographer_part,
    pool_part,
    rate_interval,
    released_metrics,
    run_workshop_validation,
    slope_interval,
    statue_folds,
    summarize_held_out,
)
from krasnal_id.models import DatasetManifest, EvaluationSplit

runner = CliRunner()


def _settings(**overrides: object) -> WorkshopValidationConfig:
    config = load_config(
        ["experiment=workshop_validation", *(f"experiment.{k}={v}" for k, v in overrides.items())]
    ).experiment
    assert isinstance(config, WorkshopValidationConfig)
    return config


def _two_photographers(manifest: DatasetManifest) -> DatasetManifest:
    """Every class has two photographers, so every query is eligible."""
    return manifest.model_copy(
        update={
            "images": tuple(
                image.model_copy(update={"author": "Bo" if image.image_id.endswith("-2") else "Al"})
                for image in manifest.images
            )
        }
    )


def _evidence(
    query: str, truth: str, wrong: str, *, truth_inliers: int, wrong_inliers: int, truth_first: bool
) -> QueryEvidence:
    """A two-candidate shortlist whose order cosine and geometry may disagree about."""
    right = Candidate(truth, 0.80 if truth_first else 0.79, truth_inliers, True)
    other = Candidate(wrong, 0.79 if truth_first else 0.80, wrong_inliers, False)
    candidates = (right, other) if truth_first else (other, right)
    return QueryEvidence(query, candidates, baseline_rank=1 if truth_first else 2)


def _world(
    tmp_path: Path, dwarfs: int = 6
) -> tuple[DatasetManifest, EvaluationSplit, dict[str, str], dict[str, dict[str, QueryEvidence]]]:
    """A manifest, split and two matchers' evidence in which geometry fixes cosine.

    Cosine puts the wrong statue first for every query; SIFT finds a little
    geometric evidence for the truth and the learned matcher a lot, so a positive
    weight rescues the queries for the learned matcher only.
    """
    manifest = _two_photographers(synthetic_manifest(dwarf_count=dwarfs, per_dwarf=3))
    split = build_evaluation_split(manifest, datetime.now(UTC))
    truths = {fold.query_image_id: fold.query_dwarf_id for fold in split.folds}
    statues = sorted(set(truths.values()))
    evidence: dict[str, dict[str, QueryEvidence]] = {"sift": {}, "disk-lightglue": {}}
    for query, truth in truths.items():
        wrong = statues[(statues.index(truth) + 1) % len(statues)]
        evidence["sift"][query] = _evidence(
            query, truth, wrong, truth_inliers=1, wrong_inliers=1, truth_first=False
        )
        evidence["disk-lightglue"][query] = _evidence(
            query, truth, wrong, truth_inliers=30, wrong_inliers=0, truth_first=False
        )
    return manifest, split, truths, evidence


def _write_journals(
    root: Path, split: EvaluationSplit, evidence: dict[str, dict[str, QueryEvidence]]
) -> Path:
    journals = root / "journals"
    journals.mkdir(parents=True, exist_ok=True)
    for matcher, items in evidence.items():
        path = journal_path(journals, split, "dinov2", matcher, _settings())
        path.write_text(
            "".join(json.dumps(evidence_to_row(item)) + "\n" for item in items.values()),
            encoding="utf-8",
        )
    return journals


# ---------------------------------------------------------------- folds


def test_folds_are_deterministic_near_equal_and_keep_statues_whole() -> None:
    statues = [f"Q{i}" for i in range(1, 23)]
    folds = statue_folds(statues, 5, seed=42)

    assert folds == statue_folds(list(reversed(statues)), 5, seed=42)
    assert folds != statue_folds(statues, 5, seed=7)
    sizes = np.bincount(list(folds.values()))
    assert sizes.max() - sizes.min() <= 1
    assert set(folds) == set(statues)
    with pytest.raises(WorkshopValidationError, match="cannot fill"):
        statue_folds(["Q1", "Q2"], 5, seed=42)


def test_weights_are_chosen_without_the_statues_they_judge(tmp_path: Path) -> None:
    _, _, truths, evidence = _world(tmp_path)
    fold_of = statue_folds(sorted(set(truths.values())), 3, seed=1)

    held_out, chosen = cross_validate(evidence, truths, fold_of, (0.0, 0.1, 1.0))

    # Every query is scored exactly once, in the fold its statue belongs to.
    assert sorted(item.query_image_id for item in held_out) == sorted(truths)
    assert all(item.fold == fold_of[item.dwarf_id] for item in held_out)
    for row in chosen:
        held = [item for item in held_out if item.fold == row.fold]
        assert row.held_out_queries == len(held)
        assert row.held_out_statues == len({item.dwarf_id for item in held})
    # Weight zero is cosine by construction, which puts every truth second here.
    assert all(item.ranks[COSINE] == 2 for item in held_out)
    # The learned matcher's evidence rescues every query once any weight is chosen.
    assert all(item.ranks["disk-lightglue"] == 1 for item in held_out)
    assert {row.weight for row in chosen if row.matcher == "disk-lightglue"} == {0.1}


def test_a_weight_that_gains_nothing_resolves_to_the_control(tmp_path: Path) -> None:
    _, _, truths, evidence = _world(tmp_path)
    fold_of = statue_folds(sorted(set(truths.values())), 3, seed=1)

    held_out, chosen = cross_validate(evidence, truths, fold_of, (0.0, 0.5, 1.0))

    # SIFT's evidence is level, so every weight ties and the smallest one wins.
    assert {row.weight for row in chosen if row.matcher == "sift"} == {0.0}
    assert all(item.ranks["sift"] == item.ranks[COSINE] for item in held_out)


def test_journals_that_disagree_on_the_shortlist_are_not_paired(tmp_path: Path) -> None:
    _, _, truths, evidence = _world(tmp_path)
    query = next(iter(truths))
    truth, wrong = (
        c.dwarf_id for c in sorted(evidence["sift"][query].candidates, key=lambda c: not c.correct)
    )
    # Same statues, different cosine scores: this shortlist ranks the truth first,
    # the other matcher's ranks it second, so the two cannot be paired.
    evidence["sift"][query] = _evidence(
        query, truth, wrong, truth_inliers=1, wrong_inliers=1, truth_first=True
    )
    fold_of = statue_folds(sorted(set(truths.values())), 3, seed=1)

    with pytest.raises(WorkshopValidationError, match="disagree on the cosine ranking"):
        cross_validate(evidence, truths, fold_of, (0.0, 0.1))


# ---------------------------------------------------------------- journals


def test_only_complete_well_formed_journals_are_accepted(tmp_path: Path) -> None:
    manifest, split, _, evidence = _world(tmp_path)
    expected = eligible_queries(split, manifest)
    journals = _write_journals(tmp_path, split, evidence)
    path = journal_path(journals, split, "dinov2", "sift", _settings())

    assert set(load_journal_strictly(path, expected)) == set(expected)

    lines = path.read_text(encoding="utf-8").splitlines()
    path.write_text("\n".join(lines[:-1]) + "\n", encoding="utf-8")
    with pytest.raises(WorkshopValidationError, match="1 missing"):
        load_journal_strictly(path, expected)

    path.write_text("\n".join([*lines, lines[0]]) + "\n", encoding="utf-8")
    with pytest.raises(WorkshopValidationError, match="repeats query"):
        load_journal_strictly(path, expected)

    path.write_text("\n".join([*lines[:-1], "{not json"]) + "\n", encoding="utf-8")
    with pytest.raises(WorkshopValidationError, match="not a valid evidence row"):
        load_journal_strictly(path, expected)

    with pytest.raises(WorkshopValidationError, match="missing"):
        load_journal_strictly(tmp_path / "absent.jsonl", expected)


def test_a_journal_for_another_split_is_not_found(tmp_path: Path) -> None:
    _, split, _, _ = _world(tmp_path)
    moved = split.model_copy(update={"manifest_sha256": "b" * 64})

    assert journal_path(tmp_path, split, "dinov2", "sift", _settings()) != journal_path(
        tmp_path, moved, "dinov2", "sift", _settings()
    )


def test_single_photographer_statues_are_not_eligible(tmp_path: Path) -> None:
    manifest = synthetic_manifest(dwarf_count=3)  # one author throughout
    split = build_evaluation_split(manifest, datetime.now(UTC))

    assert eligible_queries(split, manifest) == ()


def test_the_journal_identity_matches_the_matcher_sweep_that_wrote_it() -> None:
    sweep = load_config(["experiment=matcher_rerank"]).experiment
    settings = _settings()
    assert settings.matchers == sweep.matchers  # type: ignore[union-attr]
    assert settings.rerank_top_k == sweep.top_k  # type: ignore[union-attr]
    assert settings.max_keypoints == sweep.max_keypoints  # type: ignore[union-attr]
    assert settings.learned_keypoints == sweep.learned_keypoints  # type: ignore[union-attr]
    assert settings.weights == sweep.weights  # type: ignore[union-attr]
    with pytest.raises(ValueError, match="zero control weight"):
        WorkshopValidationConfig.model_validate({**settings.model_dump(), "weights": (0.1, 0.2)})


# ---------------------------------------------------------------- bootstrap


def test_statue_draws_are_reproducible_and_keyed_by_label() -> None:
    first = statue_draws(4, 50, generator(42, "a"))
    assert np.array_equal(first, statue_draws(4, 50, generator(42, "a")))
    assert not np.array_equal(first, statue_draws(4, 50, generator(42, "b")))
    with pytest.raises(BootstrapError):
        statue_draws(0, 10, generator(1, "x"))


def test_rates_are_image_weighted_over_unequal_statues() -> None:
    # Statue 0 has one query (a hit), statue 1 has three (all misses).
    outcomes = {"q0": 1.0, "q1": 0.0, "q2": 0.0, "q3": 0.0}
    statue_of = {"q0": "S0", "q1": "S1", "q2": "S1", "q3": "S1"}
    sums, counts = per_statue(outcomes, statue_of, ["S0", "S1"])
    draws = np.asarray([[0, 1], [0, 0], [1, 1]], dtype=np.int64)

    rates = resampled_rate(sums, counts, draws)
    assert rates.tolist() == pytest.approx([0.25, 1.0, 0.0])
    with pytest.raises(BootstrapError, match="no statue"):
        per_statue({"q9": 1.0}, statue_of, ["S0", "S1"])


def test_an_identical_pair_of_methods_has_a_zero_width_gain(tmp_path: Path) -> None:
    _, _, truths, evidence = _world(tmp_path)
    evidence["disk-lightglue"] = dict(evidence["sift"])
    fold_of = statue_folds(sorted(set(truths.values())), 3, seed=1)
    held_out, _ = cross_validate(evidence, truths, fold_of, (0.0, 0.1))

    metrics = {
        m.name: m
        for m in summarize_held_out(held_out, (COSINE, "sift", "disk-lightglue"), (1,), 200, 3)
    }
    gain = metrics["cv_disk-lightglue_vs_sift_top_1_gain"]
    assert gain.value == 0.0
    assert gain.lower_bound == gain.upper_bound == 0.0
    assert metrics["cv_disk-lightglue_vs_sift_top_1_p_value"].value == 1.0
    assert percentile_interval(np.asarray([1.0, 2.0, 3.0])) == pytest.approx((1.05, 2.95))


def test_the_learned_gain_and_its_paired_counts_are_reported(tmp_path: Path) -> None:
    _, _, truths, evidence = _world(tmp_path)
    fold_of = statue_folds(sorted(set(truths.values())), 3, seed=1)
    held_out, _ = cross_validate(evidence, truths, fold_of, (0.0, 0.1))

    metrics = {
        m.name: m.value
        for m in summarize_held_out(held_out, (COSINE, "sift", "disk-lightglue"), (1,), 200, 3)
    }

    assert metrics["cv_cosine_top_1"] == 0.0
    assert metrics["cv_disk-lightglue_top_1"] == 1.0
    assert metrics["cv_disk-lightglue_vs_cosine_top_1_gain"] == 1.0
    assert metrics["cv_disk-lightglue_vs_cosine_top_1_wins"] == len(truths)
    assert metrics["cv_disk-lightglue_vs_cosine_top_1_losses"] == 0
    assert metrics["cv_queries"] == len(truths)
    with pytest.raises(WorkshopValidationError, match="no held-out"):
        summarize_held_out((), (COSINE,), (1,), 100, 1)


# ---------------------------------------------------------------- end to end


def _paths(tmp_path: Path, manifest: DatasetManifest, split: EvaluationSplit) -> list[str]:
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(manifest.model_dump(mode="json")), encoding="utf-8")
    split_path = tmp_path / "split.json"
    write_evaluation_split(split_path, split)
    return [
        f"-opaths.manifest_path={manifest_path}",
        f"-opaths.evaluation_split_path={split_path}",
        f"-opaths.results_dir={tmp_path}",
        "-oexperiment.folds=3",
        "-oexperiment.bootstrap_resamples=200",
        "-oexperiment.parts=[rerank]",
        "-ologging.json_output=false",
    ]


def test_the_validation_never_builds_a_matcher_or_loads_a_model(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    manifest, split, _, evidence = _world(tmp_path)
    _write_journals(tmp_path, split, evidence)

    def forbidden(*args: object, **kwargs: object) -> None:
        raise AssertionError("the validation must not construct a matcher or a model")

    monkeypatch.setattr("krasnal_id.retrieval.matchers.create_matcher", forbidden)
    monkeypatch.setattr("krasnal_id.embeddings.extract.create_backbone", forbidden)

    result = runner.invoke(
        app, ["experiment", "workshop-validation", *_paths(tmp_path, manifest, split)]
    )

    assert result.exit_code == 0, result.output
    assert "cv_disk-lightglue_vs_cosine_top_1_gain" in result.output
    written = tmp_path / "cvww_revision" / "workshop_validation-dinov2.json"
    payload = json.loads(written.read_text(encoding="utf-8"))
    assert len(payload["fold_weights"]) == 2 * 3
    # Nothing is written beside the released artifacts.
    assert not (tmp_path / "workshop_validation-dinov2.json").exists()


def test_missing_evidence_is_reported_with_the_command_that_writes_it(tmp_path: Path) -> None:
    manifest, split, _, _ = _world(tmp_path)

    result = runner.invoke(
        app, ["experiment", "workshop-validation", *_paths(tmp_path, manifest, split)]
    )

    assert result.exit_code == 2
    assert "experiment matcher-rerank" in result.output


def test_run_requires_its_own_experiment_group(tmp_path: Path) -> None:
    with pytest.raises(WorkshopValidationError, match="requires experiment=workshop_validation"):
        run_workshop_validation(load_config(["experiment=baseline"]), tmp_path)
    assert workshop_validation.COSINE == "cosine"


# ---------------------------------------------------------------- recomputed parts


def test_a_recomputation_must_reproduce_the_released_number() -> None:
    released = {"top_1": 0.9314}
    check_reproduced("baseline", 0.9314, released, "top_1")
    check_reproduced("baseline", 0.9314 + REPRODUCTION_TOLERANCE / 2, released, "top_1")
    with pytest.raises(WorkshopValidationError, match="inputs have changed"):
        check_reproduced("baseline", 0.93, released, "top_1")
    with pytest.raises(WorkshopValidationError, match="has no metric"):
        check_reproduced("baseline", 0.93, released, "top_5")


def test_released_artifacts_are_read_by_metric_name(tmp_path: Path) -> None:
    (tmp_path / "baseline-dinov2.json").write_text(
        json.dumps({"metrics": [{"name": "top_1", "value": 0.5}]}), encoding="utf-8"
    )
    assert released_metrics(tmp_path, "baseline-dinov2") == {"top_1": 0.5}
    with pytest.raises(WorkshopValidationError, match="unreadable"):
        released_metrics(tmp_path, "absent")


def test_a_seed_stacked_arm_is_averaged_per_query() -> None:
    assert mean_by_query(["a", "b", "a", "b"], [1.0, 0.0, 0.0, 0.0]) == {"a": 0.5, "b": 0.0}


def test_paired_gaps_and_rates_resample_statues() -> None:
    statue_of = {"q1": "S1", "q2": "S1", "q3": "S2"}
    better = {"q1": 1.0, "q2": 1.0, "q3": 1.0}
    worse = {"q1": 0.0, "q2": 1.0, "q3": 1.0}

    gap = paired_gap("gap", better, worse, statue_of, 500, 1)
    assert gap.value == pytest.approx(1 / 3)
    # S2 contributes no difference, so a resample of only S2 gives zero.
    assert gap.lower_bound == pytest.approx(0.0)
    assert rate_interval("rate", worse, statue_of, 500, 1).value == pytest.approx(2 / 3)
    with pytest.raises(WorkshopValidationError, match="same queries"):
        paired_gap("gap", better, {"q1": 1.0}, statue_of, 500, 1)


def test_the_slope_interval_refits_the_whole_curve() -> None:
    statue_of = {f"q{i}": f"S{i}" for i in range(8)}
    sizes = (2, 4, 8)
    # Accuracy falls exactly one point (0.01) per doubling for every statue.
    curves = [{q: 1.0 - 0.01 * step for q in statue_of} for step in range(len(sizes))]

    slope = slope_interval("slope", sizes, curves, statue_of, 300, 2)

    assert slope.value == pytest.approx(-1.0)
    # Every statue decays identically, so every resample does too.
    assert slope.lower_bound == pytest.approx(-1.0)
    assert slope.upper_bound == pytest.approx(-1.0)


def _cached(tmp_path: Path) -> tuple[DatasetManifest, EvaluationSplit, EmbeddingMatrix]:
    manifest = _two_photographers(synthetic_manifest(dwarf_count=4, per_dwarf=3))
    seed_embedding_cache(tmp_path, manifest)
    matrix = load_embedding_matrix(manifest, FAKE_BACKBONE, tmp_path)
    return manifest, build_evaluation_split(manifest, datetime.now(UTC)), matrix


def test_the_photographer_part_reproduces_then_resamples(tmp_path: Path) -> None:
    manifest, split, matrix = _cached(tmp_path)
    settings = load_config(["experiment=photographer_gap"]).experiment
    assert isinstance(settings, PhotographerGapConfig)
    outcomes, _ = evaluate_conditions(split, manifest, matrix, settings.seeds)
    released = {
        f"{name}_top_1": outcome.top_k_hits(1) / len(outcome.ranks)
        for name, outcome in outcomes.items()
    }

    metrics = {
        m.name: m for m in photographer_part("dinov2", split, manifest, matrix, released, 200, 5)
    }

    raw = metrics["dinov2_photographer_raw_gap"]
    assert raw.value == pytest.approx(released["standard_top_1"] - released["disjoint_top_1"])
    attributable = metrics["dinov2_photographer_attributable"]
    assert attributable.value == pytest.approx(
        released["control_top_1"] - released["disjoint_top_1"]
    )
    drifted = {**released, "standard_top_1": 0.0}
    with pytest.raises(WorkshopValidationError, match="inputs have changed"):
        photographer_part("dinov2", split, manifest, matrix, drifted, 200, 5)


def test_the_pool_part_reproduces_the_published_slope(tmp_path: Path) -> None:
    _, split, matrix = _cached(tmp_path)
    settings = load_config(["experiment=pool_size_ablation"]).experiment
    assert isinstance(settings, PoolSizeAblationConfig)
    dwarf_ids = tuple(sorted(set(matrix.dwarf_ids)))
    measured = tuple(
        measure_pool_size(split, matrix, size, settings.seeds, dwarf_ids)
        for size in resolve_pool_sizes(settings.pool_sizes, len(dwarf_ids))
    )
    released = {f"top_1_pool_{m.pool_size}": m.top_1 for m in measured}
    released["top_1_points_per_doubling"] = points_per_doubling(measured)

    (slope,) = pool_part("dinov2", split, matrix, released, 200, 5)

    assert slope.value == pytest.approx(released["top_1_points_per_doubling"])
    # The per-query outcomes are the aggregate's own: their mean is the seed mean.
    for m in measured:
        assert sum(m.query_top_1) / len(m.query_top_1) == pytest.approx(m.top_1)

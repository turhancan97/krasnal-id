"""Does the model recognise the statue, or the photographer who shot it?"""

import json
from datetime import UTC, datetime
from pathlib import Path

import pytest
from typer.testing import CliRunner

from helpers import FAKE_BACKBONE, seed_embedding_cache, synthetic_manifest
from krasnal_id.cli import app
from krasnal_id.config import PhotographerGapConfig, load_config
from krasnal_id.data_pipeline.build_split import build_evaluation_split, write_evaluation_split
from krasnal_id.embeddings.store import load_embedding_matrix
from krasnal_id.experiments.photographer_gap import (
    CONTROL,
    DISJOINT,
    STANDARD,
    ConditionOutcome,
    Coverage,
    PhotographerGapError,
    disjoint_references,
    evaluate_conditions,
    measure_coverage,
    run_photographer_gap,
    summarize,
)
from krasnal_id.models import DatasetManifest

runner = CliRunner()


def _with_authors(manifest: DatasetManifest, authors: dict[str, str]) -> DatasetManifest:
    """Credit each image to a named photographer, defaulting to a shared one."""
    return manifest.model_copy(
        update={
            "images": tuple(
                image.model_copy(update={"author": authors.get(image.image_id, "Anonymous")})
                for image in manifest.images
            )
        }
    )


def _two_photographers_each() -> DatasetManifest:
    """Three classes of three images, each class shot by two photographers."""
    manifest = synthetic_manifest(dwarf_count=3, per_dwarf=3)
    authors = {}
    for dwarf in (1, 2, 3):
        authors[f"image-{dwarf}-0"] = "Ada"
        authors[f"image-{dwarf}-1"] = "Ada"
        authors[f"image-{dwarf}-2"] = "Bruno"
    return _with_authors(manifest, authors)


def test_a_single_photographer_class_cannot_be_asked_about() -> None:
    """Its queries are unanswerable, not hard: no correct reference survives."""
    manifest = _with_authors(
        synthetic_manifest(dwarf_count=3, per_dwarf=3),
        # Q1 is Ada's alone; Q2 and Q3 are shared.
        {
            "image-1-0": "Ada",
            "image-1-1": "Ada",
            "image-1-2": "Ada",
            "image-2-0": "Ada",
            "image-2-1": "Bruno",
            "image-2-2": "Bruno",
            "image-3-0": "Ada",
            "image-3-1": "Bruno",
            "image-3-2": "Cleo",
        },
    )
    split = build_evaluation_split(manifest, datetime.now(UTC))

    coverage = measure_coverage(manifest, split)

    assert coverage.single_photographer_classes == 1
    assert coverage.unanswerable_queries == 3
    assert coverage.answerable_queries == 6
    assert coverage.classes == 3
    assert coverage.photographers == 3


def test_the_disjoint_arm_withholds_the_photographer_and_the_control_matches_its_size(
    tmp_path: Path,
) -> None:
    manifest = _two_photographers_each()
    seed_embedding_cache(tmp_path, manifest)
    matrix = load_embedding_matrix(manifest, FAKE_BACKBONE, tmp_path)
    split = build_evaluation_split(manifest, datetime.now(UTC))

    outcomes, coverage = evaluate_conditions(split, manifest, matrix, (11, 23))

    # Every query is answerable: each class has two photographers.
    assert coverage.answerable_queries == 9
    assert coverage.unanswerable_queries == 0
    assert len(outcomes[STANDARD].ranks) == 9
    assert len(outcomes[DISJOINT].ranks) == 9
    # The control pools one run per seed over the same queries.
    assert len(outcomes[CONTROL].ranks) == 18

    # A query by Ada loses Ada's other five images, so it keeps three references
    # and one of its own; the control is cut to the same shape.
    assert outcomes[DISJOINT].candidate_classes[0] <= outcomes[STANDARD].candidate_classes[0]


def test_the_control_keeps_the_same_number_of_correct_references(tmp_path: Path) -> None:
    """The control is what makes the result mean anything.

    Withholding a photographer removes distractors too, and this project's
    headline finding is that accuracy rises as the pool shrinks. If the control
    were not size-matched, that inflation would hide the penalty.
    """
    manifest = _two_photographers_each()
    seed_embedding_cache(tmp_path, manifest)
    matrix = load_embedding_matrix(manifest, FAKE_BACKBONE, tmp_path)
    split = build_evaluation_split(manifest, datetime.now(UTC))

    outcomes, _ = evaluate_conditions(split, manifest, matrix, (11,))

    # Same queries, same count, so the two rates are directly comparable.
    assert len(outcomes[CONTROL].ranks) == len(outcomes[DISJOINT].ranks)


def test_a_query_that_only_its_own_photographer_can_answer_is_dropped(tmp_path: Path) -> None:
    manifest = _with_authors(
        synthetic_manifest(dwarf_count=3, per_dwarf=3),
        {
            # Q1 is Ada's alone, so its queries cannot be answered by anyone else.
            "image-1-0": "Ada",
            "image-1-1": "Ada",
            "image-1-2": "Ada",
            "image-2-0": "Ada",
            "image-2-1": "Bruno",
            "image-2-2": "Bruno",
            "image-3-0": "Cleo",
            "image-3-1": "Bruno",
            "image-3-2": "Cleo",
        },
    )
    seed_embedding_cache(tmp_path, manifest)
    matrix = load_embedding_matrix(manifest, FAKE_BACKBONE, tmp_path)
    split = build_evaluation_split(manifest, datetime.now(UTC))

    outcomes, coverage = evaluate_conditions(split, manifest, matrix, (11,))

    # Q1's three queries are unanswerable; the other six classes share "Anonymous",
    # so their queries lose every reference of their own class too.
    assert coverage.unanswerable_queries == 3
    assert coverage.answerable_queries == 6
    # Dropped, never scored as failures: counting them would measure the
    # dataset's coverage and call it the model's weakness.
    assert len(outcomes[STANDARD].ranks) == 6
    assert len(outcomes[DISJOINT].ranks) == 6


def test_every_class_having_one_photographer_is_refused(tmp_path: Path) -> None:
    manifest = synthetic_manifest(dwarf_count=3, per_dwarf=3)  # one shared author
    seed_embedding_cache(tmp_path, manifest)
    matrix = load_embedding_matrix(manifest, FAKE_BACKBONE, tmp_path)
    split = build_evaluation_split(manifest, datetime.now(UTC))

    with pytest.raises(PhotographerGapError, match="no query can be answered"):
        evaluate_conditions(split, manifest, matrix, (11,))


def test_bad_inputs_are_refused(tmp_path: Path) -> None:
    manifest = _two_photographers_each()
    seed_embedding_cache(tmp_path, manifest)
    matrix = load_embedding_matrix(manifest, FAKE_BACKBONE, tmp_path)
    split = build_evaluation_split(manifest, datetime.now(UTC))

    with pytest.raises(PhotographerGapError, match="no folds"):
        evaluate_conditions(split.model_copy(update={"folds": ()}), manifest, matrix, (11,))
    with pytest.raises(PhotographerGapError, match="at least one seed"):
        evaluate_conditions(split, manifest, matrix, ())


def test_the_gap_is_decomposed_rather_than_reported_whole() -> None:
    """Total drop splits into "fewer references" and "the same photographer"."""
    outcomes = {
        # 4 of 4 correct under the ordinary protocol.
        STANDARD: ConditionOutcome(STANDARD, (1, 1, 1, 1), (10, 10, 10, 10)),
        # 1 of 4 once the photographer is withheld.
        DISJOINT: ConditionOutcome(DISJOINT, (1, 3, 4, 6), (8, 8, 8, 8)),
        # 2 of 4 with the same number of references, chosen at random.
        CONTROL: ConditionOutcome(CONTROL, (1, 1, 5, 7), (9, 9, 9, 9)),
    }
    coverage = Coverage(
        answerable_queries=4,
        unanswerable_queries=2,
        single_photographer_classes=1,
        classes=5,
        photographers=3,
    )

    metrics = {m.name: m.value for m in summarize(outcomes, coverage, (1, 5))}

    assert metrics["standard_top_1"] == pytest.approx(1.0)
    assert metrics["disjoint_top_1"] == pytest.approx(0.25)
    assert metrics["control_top_1"] == pytest.approx(0.5)
    # The whole drop...
    assert metrics["top_1_gap"] == pytest.approx(0.75)
    # ...of which only this part is not explained by having fewer references.
    assert metrics["attributable_top_1_gap"] == pytest.approx(0.25)
    assert metrics["unanswerable_queries"] == pytest.approx(2.0)
    assert metrics["single_photographer_classes"] == pytest.approx(1.0)
    assert metrics["disjoint_median_candidate_classes"] == pytest.approx(8.0)

    with pytest.raises(PhotographerGapError, match="must be positive"):
        summarize(outcomes, coverage, (0,))


def test_the_disjoint_set_contains_nothing_by_the_query_photographer() -> None:
    """The mechanism the arm rests on, asserted directly.

    Whether withholding a photographer *costs* accuracy is an empirical question
    the real dataset answers; on crafted vectors it can come out either way, so
    testing the inequality would only test the fixture.
    """
    authors = {
        "a1": "Ada",
        "a2": "Ada",
        "b1": "Bruno",
        "c1": "Cleo",
    }

    kept = disjoint_references(("a1", "a2", "b1", "c1"), authors, "Ada")

    assert kept == ("b1", "c1")
    assert not any(authors[image_id] == "Ada" for image_id in kept)
    # A photographer who contributed nothing to this fold changes nothing.
    assert disjoint_references(("b1", "c1"), authors, "Ada") == ("b1", "c1")


def test_run_requires_the_photographer_gap_experiment_group() -> None:
    with pytest.raises(PhotographerGapError, match="requires experiment=photographer_gap"):
        run_photographer_gap(load_config(["experiment=baseline"]))


def test_packaged_defaults_are_usable() -> None:
    experiment = load_config(["experiment=photographer_gap"]).experiment
    assert isinstance(experiment, PhotographerGapConfig)
    assert 1 in experiment.top_k
    assert len(experiment.seeds) > 1, "the control samples, so one seed hides its spread"
    with pytest.raises(ValueError, match="top_k values must be positive"):
        PhotographerGapConfig.model_validate(
            {"kind": "photographer_gap", "seeds": (1,), "top_k": (0,)}
        )
    with pytest.raises(ValueError, match="seeds cannot contain duplicates"):
        PhotographerGapConfig.model_validate(
            {"kind": "photographer_gap", "seeds": (1, 1), "top_k": (1,)}
        )


def test_cli_reports_the_decomposition(tmp_path: Path) -> None:
    manifest = _two_photographers_each()
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(manifest.model_dump(mode="json")), encoding="utf-8")
    split_path = tmp_path / "splits" / "leave-one-out.json"
    write_evaluation_split(split_path, build_evaluation_split(manifest, datetime.now(UTC)))
    seed_embedding_cache(tmp_path / "embeddings", manifest)

    result = runner.invoke(
        app,
        [
            "experiment",
            "photographer-gap",
            *[
                f"-o{value}"
                for value in (
                    f"paths.manifest_path={manifest_path}",
                    f"paths.evaluation_split_path={split_path}",
                    f"paths.embeddings_dir={tmp_path / 'embeddings'}",
                    f"paths.results_dir={tmp_path / 'results'}",
                    f"backbone.model_id={FAKE_BACKBONE.model_id}",
                    f"backbone.revision={FAKE_BACKBONE.revision}",
                    f"backbone.preprocessing_id={FAKE_BACKBONE.preprocessing_id}",
                    "logging.json_output=false",
                )
            ],
        ],
    )

    assert result.exit_code == 0, result.output
    for expected in ("standard_top_1", "disjoint_top_1", "control_top_1", "attributable_top_1_gap"):
        assert expected in result.output
    written = json.loads((tmp_path / "results" / "photographer_gap-dinov2.json").read_text())
    assert written["experiment"] == "photographer_gap"

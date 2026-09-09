"""Geometry as an open-set rejection signal, and the control that makes it readable."""

import json
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("cv2")

from typer.testing import CliRunner

from helpers import FAKE_BACKBONE, seed_embedding_cache, synthetic_manifest
from krasnal_id.cli import app
from krasnal_id.config import OpenSetGeometryConfig, load_config
from krasnal_id.data_pipeline.build_split import build_evaluation_split, write_evaluation_split
from krasnal_id.embeddings.store import load_embedding_matrix
from krasnal_id.experiments.open_set_geometry import (
    DISJOINT,
    SIGNALS,
    STANDARD,
    OpenSetGeometryError,
    QueryEvidence,
    best_balanced_accuracy,
    calibrate_leave_one_class_out,
    collect_known_arm,
    collect_unknown_arm,
    measure_signal,
    run_open_set_geometry,
    summarize,
)
from krasnal_id.models import DatasetManifest

runner = CliRunner()


def _textured(path: Path, seed: int, size: int = 160) -> None:
    """Write an image with enough texture for SIFT to describe.

    Flat colour yields no keypoints at all, so noise is the cheapest thing a
    detector can actually work on.
    """
    from PIL import Image

    rng = np.random.default_rng(seed)
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(rng.integers(0, 256, size=(size, size, 3), dtype=np.uint8)).save(path)


def _manifest_with_images(
    tmp_path: Path,
    dwarfs: int = 4,
    per_dwarf: int = 3,
    *,
    authors: dict[str, str] | None = None,
) -> DatasetManifest:
    """A manifest whose images exist, one distinct texture per class.

    Same seed within a class, so a class's images verify geometrically against each
    other and against nothing else — the separation the experiment is looking for,
    made unambiguous so a test can assert on its direction.
    """
    manifest = synthetic_manifest(dwarf_count=dwarfs, per_dwarf=per_dwarf)
    records = []
    for image in manifest.images:
        path = tmp_path / "images" / image.dwarf_id / f"{image.image_id}.png"
        _textured(path, seed=int(image.dwarf_id.removeprefix("Q")))
        update: dict[str, object] = {"local_path": path, "width": 160, "height": 160}
        if authors is not None:
            update["author"] = authors.get(image.image_id, "Author")
        records.append(image.model_copy(update=update))
    return manifest.model_copy(update={"images": tuple(records)})


def _settings(**overrides: object) -> OpenSetGeometryConfig:
    """Build the experiment group directly, so a unit test needs no Hydra."""
    values: dict[str, object] = {
        "kind": "open_set_geometry",
        "seed": 42,
        "top_k": 2,
        "max_keypoints": 300,
        "blend_weight": 0.05,
        "target_known_acceptance": 0.9,
        "photographer_disjoint": False,
    }
    values.update(overrides)
    return OpenSetGeometryConfig.model_validate(values)


def _evidence(dwarf: str, *, present: bool, cosine: float, inliers: int) -> QueryEvidence:
    """One synthetic query's evidence, with both signals set independently."""
    return QueryEvidence(
        query_image_id=f"{dwarf}-{present}-{cosine}",
        query_dwarf_id=dwarf,
        present=present,
        top_cosine=cosine,
        top_inliers=inliers,
        best_inliers=inliers,
        best_blended=cosine + 0.05 * min(inliers, 30) / 30,
    )


def test_every_signal_is_named_and_an_unknown_one_is_refused() -> None:
    """`score` is a dispatch on a string, so a typo must not read as a zero."""
    row = _evidence("Q1", present=True, cosine=0.8, inliers=12)

    assert row.score("cosine") == pytest.approx(0.8)
    assert row.score("inliers_top_1") == pytest.approx(12.0)
    assert row.score("inliers_best") == pytest.approx(12.0)
    assert row.score("blended") == pytest.approx(0.8 + 0.05 * 12 / 30)
    with pytest.raises(OpenSetGeometryError, match="unknown rejection signal"):
        row.score("inliers")


def test_a_signal_that_separates_perfectly_scores_one_and_a_useless_one_scores_half() -> None:
    """The two anchors that make every intermediate AUROC readable."""
    known = tuple(_evidence(f"Q{i}", present=True, cosine=0.9, inliers=20) for i in range(1, 5))
    unknown = tuple(_evidence(f"Q{i}", present=False, cosine=0.5, inliers=0) for i in range(1, 5))

    separating = measure_signal(known, unknown, "inliers_best", STANDARD, 0.9)
    assert separating.auroc == pytest.approx(1.0)
    assert separating.false_acceptance == pytest.approx(0.0)
    assert separating.in_sample_balanced_accuracy == pytest.approx(1.0)

    # Identical scores on both arms: ties count as half, so the signal carries
    # nothing and must say so rather than looking mildly informative.
    flat_unknown = tuple(
        _evidence(f"Q{i}", present=False, cosine=0.9, inliers=20) for i in range(1, 5)
    )
    useless = measure_signal(known, flat_unknown, "inliers_best", STANDARD, 0.9)
    assert useless.auroc == pytest.approx(0.5)
    assert useless.in_sample_balanced_accuracy == pytest.approx(0.5)


def test_a_threshold_is_never_fitted_on_the_statue_it_judges() -> None:
    """The calibration leak that would make false acceptance look better than it is."""
    known = (
        _evidence("Q1", present=True, cosine=0.99, inliers=30),
        _evidence("Q1", present=True, cosine=0.98, inliers=29),
        _evidence("Q2", present=True, cosine=0.10, inliers=1),
        _evidence("Q2", present=True, cosine=0.11, inliers=2),
    )

    thresholds = calibrate_leave_one_class_out(known, "cosine", 0.9)

    # Q1's threshold comes from Q2's low scores and Q2's from Q1's high ones, so
    # each is calibrated on data it did not produce.
    assert thresholds["Q1"] < 0.5
    assert thresholds["Q2"] > 0.5
    with pytest.raises(OpenSetGeometryError, match="at least two statues"):
        calibrate_leave_one_class_out(known[:2], "cosine", 0.9)
    with pytest.raises(OpenSetGeometryError, match="must lie in"):
        calibrate_leave_one_class_out(known, "cosine", 1.0)


def test_an_unknown_statue_with_no_threshold_of_its_own_is_rejected_not_skipped() -> None:
    """Dropping it instead would shrink the denominator and flatter the signal."""
    known = (
        _evidence("Q1", present=True, cosine=0.9, inliers=20),
        _evidence("Q2", present=True, cosine=0.9, inliers=20),
    )
    # Q9 never appears in the known arm, so no threshold is fitted for it.
    unknown = (_evidence("Q9", present=False, cosine=1.0, inliers=30),)

    row = measure_signal(known, unknown, "cosine", STANDARD, 0.9)

    assert row.unknown_queries == 1
    assert row.false_acceptance == pytest.approx(0.0)


def test_the_best_balanced_accuracy_is_the_best_any_threshold_reaches() -> None:
    known = np.array([0.9, 0.8, 0.7], dtype=np.float64)
    unknown = np.array([0.2, 0.1, 0.0], dtype=np.float64)

    assert best_balanced_accuracy(known, unknown) == pytest.approx(1.0)
    # Perfectly interleaved: no threshold does better than a coin flip on one side.
    assert best_balanced_accuracy(
        np.array([0.5, 0.3], dtype=np.float64), np.array([0.4, 0.2], dtype=np.float64)
    ) == pytest.approx(0.75)


def test_the_summary_names_the_gain_over_the_control_and_keeps_both_ends() -> None:
    rows = (
        measure_signal(
            tuple(_evidence(f"Q{i}", present=True, cosine=0.9, inliers=20) for i in range(1, 4)),
            tuple(_evidence(f"Q{i}", present=False, cosine=0.5, inliers=0) for i in range(1, 4)),
            signal,
            STANDARD,
            0.9,
        )
        for signal in SIGNALS
    )
    metrics = {metric.name: metric for metric in summarize(tuple(rows))}

    assert "standard_cosine_auroc" in metrics
    gain = metrics["standard_best_geometric_auroc_gain"]
    # The control and the winner are both kept, so a zero gain can be told apart
    # from two signals that are both useless.
    assert gain.lower_bound is not None and gain.upper_bound is not None
    assert gain.value == pytest.approx(gain.upper_bound - gain.lower_bound)
    with pytest.raises(OpenSetGeometryError, match="empty signal comparison"):
        summarize(())


def test_the_unknown_arm_never_reaches_its_own_statue(tmp_path: Path) -> None:
    """If it could, the query would not be open-set at all."""
    manifest = _manifest_with_images(tmp_path)
    seed_embedding_cache(tmp_path / "embeddings", manifest)
    matrix = load_embedding_matrix(manifest, FAKE_BACKBONE, tmp_path / "embeddings")
    from krasnal_id.retrieval.rerank import FeatureCache

    paths = {image.image_id: Path(image.local_path) for image in manifest.images}
    arm = collect_unknown_arm(matrix, FeatureCache(300), paths, _settings())

    assert len(arm) == len(manifest.images)
    assert all(not row.present for row in arm)
    # Same-class images share a texture here, so reaching one's own class would
    # show up as high inlier counts. It cannot, because the class is absent.
    assert max(row.best_inliers for row in arm) < 30


def test_withholding_a_photographer_drops_the_queries_it_makes_unanswerable(
    tmp_path: Path,
) -> None:
    """A statue one person documented has no correct answer once they are removed."""
    # Q1 is documented by one person; every other class by two.
    authors = {
        image.image_id: ("Solo" if image.dwarf_id == "Q1" else f"Person {index % 2}")
        for index, image in enumerate(synthetic_manifest(dwarf_count=3, per_dwarf=3).images)
    }
    manifest = _manifest_with_images(tmp_path, dwarfs=3, authors=authors)
    seed_embedding_cache(tmp_path / "embeddings", manifest)
    matrix = load_embedding_matrix(manifest, FAKE_BACKBONE, tmp_path / "embeddings")
    split = build_evaluation_split(manifest, datetime.now(UTC))
    from krasnal_id.retrieval.rerank import FeatureCache

    paths = {image.image_id: Path(image.local_path) for image in manifest.images}
    by_class = {
        dwarf: {authors[i.image_id] for i in manifest.images if i.dwarf_id == dwarf}
        for dwarf in {image.dwarf_id for image in manifest.images}
    }
    withheld = collect_known_arm(
        split,
        matrix,
        FeatureCache(300),
        paths,
        _settings(photographer_disjoint=True),
        authors=authors,
        answerable=by_class,
    )

    assert {row.query_dwarf_id for row in withheld} == {"Q2", "Q3"}
    # And the same withholding applies to the unknown arm, or the two arms would
    # search galleries of different sizes.
    unknown = collect_unknown_arm(
        matrix,
        FeatureCache(300),
        paths,
        _settings(photographer_disjoint=True),
        authors=authors,
        answerable=by_class,
    )
    assert {row.query_dwarf_id for row in unknown} == {"Q2", "Q3"}
    with pytest.raises(OpenSetGeometryError, match="per-class photographer sets"):
        collect_known_arm(split, matrix, FeatureCache(300), paths, _settings(), authors=authors)


def test_the_experiment_reports_both_conditions_and_every_signal(tmp_path: Path) -> None:
    authors = {
        image.image_id: f"Person {index % 2}"
        for index, image in enumerate(synthetic_manifest(dwarf_count=4, per_dwarf=3).images)
    }
    manifest = _manifest_with_images(tmp_path, dwarfs=4, authors=authors)
    seed_embedding_cache(tmp_path / "embeddings", manifest)
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(manifest.model_dump_json(), encoding="utf-8")
    split_path = tmp_path / "split.json"
    write_evaluation_split(split_path, build_evaluation_split(manifest, datetime.now(UTC)))

    config = load_config(
        [
            "experiment=open_set_geometry",
            f"paths.manifest_path={manifest_path}",
            f"paths.evaluation_split_path={split_path}",
            f"paths.embeddings_dir={tmp_path / 'embeddings'}",
            f"backbone.name={FAKE_BACKBONE.name}",
            f"backbone.model_id={FAKE_BACKBONE.model_id}",
            f"backbone.revision={FAKE_BACKBONE.revision}",
            f"backbone.preprocessing_id={FAKE_BACKBONE.preprocessing_id}",
            "experiment.top_k=2",
            "experiment.max_keypoints=300",
        ]
    )
    result = run_open_set_geometry(config)

    assert result.experiment == "open_set_geometry"
    assert {row.condition for row in result.signals} == {STANDARD, DISJOINT}
    assert {row.signal for row in result.signals if row.condition == STANDARD} == set(SIGNALS)
    # The configuration is recorded, so two runs at different top_k are told apart.
    assert result.configuration is not None
    assert result.configuration["top_k"] == 2
    # Every row must name a real, non-empty population: a condition that silently
    # emptied an arm would otherwise be reported as a measurement.
    assert all(row.known_queries > 0 and row.unknown_queries > 0 for row in result.signals)


def test_the_cosine_control_is_computed_on_the_identical_population(tmp_path: Path) -> None:
    """The property that makes a geometric difference readable as the signal.

    Every signal is derived from one pass over one query set, so the control cannot
    drift from the rows it is compared against — the failure that would make the
    whole comparison meaningless.
    """
    manifest = _manifest_with_images(tmp_path, dwarfs=4)
    seed_embedding_cache(tmp_path / "embeddings", manifest)
    matrix = load_embedding_matrix(manifest, FAKE_BACKBONE, tmp_path / "embeddings")
    split = build_evaluation_split(manifest, datetime.now(UTC))
    from krasnal_id.retrieval.rerank import FeatureCache

    paths = {image.image_id: Path(image.local_path) for image in manifest.images}
    cache = FeatureCache(300)
    known = collect_known_arm(split, matrix, cache, paths, _settings())
    unknown = collect_unknown_arm(matrix, cache, paths, _settings())

    rows = [measure_signal(known, unknown, signal, STANDARD, 0.9) for signal in SIGNALS]

    assert len({(row.known_queries, row.unknown_queries) for row in rows}) == 1
    # A blend weight of zero must reduce the blended signal to the control exactly,
    # which is the analogue of section 7.6's weight-zero arm.
    zero = collect_known_arm(split, matrix, cache, paths, _settings(blend_weight=0.0))
    assert [row.best_blended for row in zero] == pytest.approx([row.top_cosine for row in zero])


def test_the_experiment_refuses_a_mismatched_group_and_missing_pixels(tmp_path: Path) -> None:
    config = load_config(["experiment=open_set"])
    with pytest.raises(OpenSetGeometryError, match="requires experiment=open_set_geometry"):
        run_open_set_geometry(config)

    # A manifest pointing at absent files must fail before any feature is computed,
    # not part-way through an hour of homographies.
    manifest = synthetic_manifest(dwarf_count=3, per_dwarf=3)
    seed_embedding_cache(tmp_path / "embeddings", manifest)
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(manifest.model_dump_json(), encoding="utf-8")
    split_path = tmp_path / "split.json"
    write_evaluation_split(split_path, build_evaluation_split(manifest, datetime.now(UTC)))
    absent = load_config(
        [
            "experiment=open_set_geometry",
            f"paths.manifest_path={manifest_path}",
            f"paths.evaluation_split_path={split_path}",
            f"paths.embeddings_dir={tmp_path / 'embeddings'}",
            f"backbone.name={FAKE_BACKBONE.name}",
            f"backbone.model_id={FAKE_BACKBONE.model_id}",
            f"backbone.revision={FAKE_BACKBONE.revision}",
            f"backbone.preprocessing_id={FAKE_BACKBONE.preprocessing_id}",
        ]
    )
    with pytest.raises(OpenSetGeometryError, match=r"image\(s\) are missing"):
        run_open_set_geometry(absent)


def test_the_command_writes_an_artifact_and_prints_every_signal(tmp_path: Path) -> None:
    manifest = _manifest_with_images(tmp_path, dwarfs=4)
    seed_embedding_cache(tmp_path / "embeddings", manifest)
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(manifest.model_dump_json(), encoding="utf-8")
    split_path = tmp_path / "split.json"
    write_evaluation_split(split_path, build_evaluation_split(manifest, datetime.now(UTC)))
    results = tmp_path / "results"

    outcome = runner.invoke(
        app,
        [
            "experiment",
            "open-set-geometry",
            "--override",
            f"paths.manifest_path={manifest_path}",
            "--override",
            f"paths.evaluation_split_path={split_path}",
            "--override",
            f"paths.embeddings_dir={tmp_path / 'embeddings'}",
            "--override",
            f"paths.results_dir={results}",
            "--override",
            f"backbone.name={FAKE_BACKBONE.name}",
            "--override",
            f"backbone.model_id={FAKE_BACKBONE.model_id}",
            "--override",
            f"backbone.revision={FAKE_BACKBONE.revision}",
            "--override",
            f"backbone.preprocessing_id={FAKE_BACKBONE.preprocessing_id}",
            "--override",
            "experiment.top_k=2",
            "--override",
            "experiment.max_keypoints=300",
            "--override",
            "experiment.photographer_disjoint=false",
        ],
    )

    assert outcome.exit_code == 0, outcome.output
    written = results / f"open_set_geometry-{FAKE_BACKBONE.name}.json"
    payload = json.loads(written.read_text(encoding="utf-8"))
    assert {row["signal"] for row in payload["signals"]} == set(SIGNALS)
    for signal in SIGNALS:
        assert signal in outcome.output


def test_the_command_exits_two_when_the_pixels_are_absent(tmp_path: Path) -> None:
    """Configuration and input problems exit 2, as every other experiment does."""
    manifest = synthetic_manifest(dwarf_count=3, per_dwarf=3)
    seed_embedding_cache(tmp_path / "embeddings", manifest)
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(manifest.model_dump_json(), encoding="utf-8")
    split_path = tmp_path / "split.json"
    write_evaluation_split(split_path, build_evaluation_split(manifest, datetime.now(UTC)))

    outcome = runner.invoke(
        app,
        [
            "experiment",
            "open-set-geometry",
            "--override",
            f"paths.manifest_path={manifest_path}",
            "--override",
            f"paths.evaluation_split_path={split_path}",
            "--override",
            f"paths.embeddings_dir={tmp_path / 'embeddings'}",
            "--override",
            f"paths.results_dir={tmp_path / 'results'}",
            "--override",
            f"backbone.name={FAKE_BACKBONE.name}",
            "--override",
            f"backbone.model_id={FAKE_BACKBONE.model_id}",
            "--override",
            f"backbone.revision={FAKE_BACKBONE.revision}",
            "--override",
            f"backbone.preprocessing_id={FAKE_BACKBONE.preprocessing_id}",
        ],
    )

    assert outcome.exit_code == 2
    assert "Geometric open-set rejection error" in outcome.output

"""Real coordinate-based candidate pools, against randomly sampled ones."""

import json
from datetime import UTC, datetime
from pathlib import Path

import pytest
from typer.testing import CliRunner

from helpers import FAKE_BACKBONE, seed_embedding_cache, synthetic_manifest
from krasnal_id.cli import app
from krasnal_id.config import load_config
from krasnal_id.data_pipeline.build_split import build_evaluation_split, write_evaluation_split
from krasnal_id.embeddings.store import load_embedding_matrix
from krasnal_id.experiments.contracts import ExperimentResult
from krasnal_id.experiments.geo_ablation import (
    GeoAblationError,
    GeoMeasurement,
    dwarf_locations,
    measure_geographic_pool,
    nearest_pool,
    run_geo_ablation,
    summarize_geo,
)
from krasnal_id.geometry import haversine_metres

# A degree of longitude at the equator, and Wrocław's market square.
_EQUATOR_DEGREE_METRES = 111195.0


def _line(count: int, spacing_degrees: float = 0.001) -> dict[str, tuple[float, float]]:
    """Place dwarves in a west-to-east line, Q1 westmost."""
    return {f"Q{i}": (51.11, 17.03 + (i - 1) * spacing_degrees) for i in range(1, count + 1)}


def test_haversine_matches_known_distances() -> None:
    assert haversine_metres((0.0, 0.0), (0.0, 0.0)) == pytest.approx(0.0)
    assert haversine_metres((0.0, 0.0), (0.0, 1.0)) == pytest.approx(
        _EQUATOR_DEGREE_METRES, rel=0.001
    )
    # Symmetric, and a degree of latitude is the same length anywhere.
    assert haversine_metres((51.0, 17.0), (52.0, 17.0)) == pytest.approx(
        haversine_metres((52.0, 17.0), (51.0, 17.0))
    )
    assert haversine_metres((51.0, 17.0), (52.0, 17.0)) == pytest.approx(111195.0, rel=0.001)


def test_locations_cover_whichever_dwarves_are_placed(tmp_path: Path) -> None:
    located = synthetic_manifest(dwarf_count=3, coordinates=_line(3))
    assert set(dwarf_locations(located)) == {"Q1", "Q2", "Q3"}

    # A Commons-only class never carries P625 coordinates, so a partly placed
    # manifest is the normal case rather than an error; the arm scopes to the
    # located subset and reports what fraction it covers.
    partial = synthetic_manifest(dwarf_count=3, coordinates=_line(2))
    assert set(dwarf_locations(partial)) == {"Q1", "Q2"}

    empty = synthetic_manifest(dwarf_count=3).model_copy(update={"dwarfs": ()})
    with pytest.raises(GeoAblationError, match="no dwarves"):
        dwarf_locations(empty)

    with pytest.raises(GeoAblationError, match="none of the 3 dwarves carry coordinates"):
        dwarf_locations(synthetic_manifest(dwarf_count=3))


def test_the_geographic_arm_reports_what_fraction_it_covers(tmp_path: Path) -> None:
    # Three of five placed: the result must say so rather than implying it measured
    # the whole pool.
    manifest = synthetic_manifest(dwarf_count=5, coordinates=_line(3))
    seed_embedding_cache(tmp_path, manifest)
    matrix = load_embedding_matrix(manifest, FAKE_BACKBONE, tmp_path)
    split = build_evaluation_split(manifest, datetime.now(UTC))
    locations = dwarf_locations(manifest)
    located_split = split.model_copy(
        update={"folds": tuple(f for f in split.folds if f.query_dwarf_id in locations)}
    )

    geo = (measure_geographic_pool(located_split, matrix, 2, locations),)
    metrics = {
        m.name: m.value
        for m in summarize_geo(
            geo, {2: (0.5, 0.4, 0.6)}, len(located_split.folds), 3, coverage=(3, 5)
        )
    }

    assert metrics["located_dwarfs"] == pytest.approx(3.0)
    assert metrics["manifest_dwarfs"] == pytest.approx(5.0)
    assert metrics["located_fraction"] == pytest.approx(0.6)
    # Only the placed dwarves' folds were scored.
    assert metrics["evaluated_folds"] == pytest.approx(float(len(located_split.folds)))
    assert len(located_split.folds) < len(split.folds)

    with pytest.raises(GeoAblationError, match="cannot outnumber"):
        summarize_geo(geo, {2: (0.5, 0.4, 0.6)}, 1, 3, coverage=(9, 5))


def test_a_pool_holds_the_query_and_its_nearest_neighbours() -> None:
    locations = _line(5)

    pool, radius = nearest_pool("Q1", 3, locations)

    # The query comes first, then the two nearest going east.
    assert pool == ("Q1", "Q2", "Q3")
    assert radius == pytest.approx(haversine_metres(locations["Q1"], locations["Q3"]))
    # A middle dwarf reaches out in both directions.
    assert set(nearest_pool("Q3", 3, locations)[0]) == {"Q3", "Q2", "Q4"}
    assert nearest_pool("Q1", 5, locations)[0] == ("Q1", "Q2", "Q3", "Q4", "Q5")


def test_equal_distances_break_by_dwarf_id() -> None:
    # Statues really do share a spot in this dataset, so exact ties are not
    # hypothetical. Q2 is listed first to show insertion order does not decide.
    locations = {"Q0": (51.11, 17.03), "Q2": (51.12, 17.04), "Q1": (51.12, 17.04)}

    first = nearest_pool("Q0", 2, locations)
    again = nearest_pool("Q0", 2, locations)

    assert first == again
    assert first[0] == ("Q0", "Q1")


def test_pool_sizes_outside_the_located_set_are_rejected() -> None:
    locations = _line(3)

    with pytest.raises(GeoAblationError, match="at least two"):
        nearest_pool("Q1", 1, locations)
    with pytest.raises(GeoAblationError, match="exceeds the 3 located dwarves"):
        nearest_pool("Q1", 4, locations)
    with pytest.raises(GeoAblationError, match="has no coordinates"):
        nearest_pool("Q9", 2, locations)


def test_a_geographic_measurement_is_exact_and_reports_its_radius(tmp_path: Path) -> None:
    manifest = synthetic_manifest(dwarf_count=5, coordinates=_line(5))
    seed_embedding_cache(tmp_path, manifest)
    matrix = load_embedding_matrix(manifest, FAKE_BACKBONE, tmp_path)
    split = build_evaluation_split(manifest, datetime.now(UTC))
    locations = dwarf_locations(manifest)

    first = measure_geographic_pool(split, matrix, 3, locations)
    again = measure_geographic_pool(split, matrix, 3, locations)

    # Nothing samples, so repeated runs are identical rather than a mean.
    assert first == again
    assert first.pool_size == 3
    assert first.top_1 == pytest.approx(1.0)
    assert 0 < first.median_radius_metres <= first.max_radius_metres
    # A wider pool has to reach further.
    assert measure_geographic_pool(split, matrix, 5, locations).median_radius_metres > (
        first.median_radius_metres
    )

    with pytest.raises(GeoAblationError, match="no folds"):
        measure_geographic_pool(split.model_copy(update={"folds": ()}), matrix, 3, locations)


def test_co_located_lookalikes_make_a_geographic_pool_harder(tmp_path: Path) -> None:
    # Q1 and Q2 share a spot and share an embedding axis; the others are far away
    # and easy. Proximity therefore always pools the one confusable neighbour.
    coordinates = {
        "Q1": (51.11, 17.03),
        "Q2": (51.11, 17.03),
        "Q3": (51.20, 17.30),
        "Q4": (51.30, 17.40),
    }
    manifest = synthetic_manifest(dwarf_count=4, coordinates=coordinates)

    def paired(dwarf_index: int, position: int, dwarf_count: int) -> object:
        import numpy as np

        vector = np.zeros(dwarf_count + 1, dtype=np.float32)
        vector[0 if dwarf_index < 2 else dwarf_index] = 1.0
        if dwarf_index < 2:
            vector[-1] = 0.02 * position + 0.01 * (dwarf_index % 2)
        return np.asarray(vector / np.linalg.norm(vector), dtype=np.float32)

    seed_embedding_cache(tmp_path, manifest, paired)
    matrix = load_embedding_matrix(manifest, FAKE_BACKBONE, tmp_path)
    split = build_evaluation_split(manifest, datetime.now(UTC))
    locations = dwarf_locations(manifest)

    geo = measure_geographic_pool(split, matrix, 2, locations)

    # Every Q1/Q2 query is pooled with its lookalike, so accuracy is far from perfect
    # even though the pool holds only two candidates.
    assert geo.top_1 < 1.0
    # The co-located pair needs no radius at all to be pooled together.
    assert nearest_pool("Q1", 2, locations) == (("Q1", "Q2"), 0.0)


def test_summary_pairs_both_arms_and_records_the_advantage() -> None:
    geo = (
        GeoMeasurement(
            pool_size=2, top_1=0.90, mrr=0.95, median_radius_metres=120.0, max_radius_metres=800.0
        ),
        GeoMeasurement(
            pool_size=5, top_1=0.80, mrr=0.88, median_radius_metres=310.0, max_radius_metres=2400.0
        ),
    )
    random_arm = {2: (0.95, 0.94, 0.96), 5: (0.78, 0.75, 0.81)}

    metrics = {m.name: m for m in summarize_geo(geo, random_arm, folds=100, dwarf_count=9)}

    assert metrics["geo_top_1_pool_2"].value == pytest.approx(0.90)
    assert metrics["random_top_1_pool_2"].value == pytest.approx(0.95)
    assert metrics["random_top_1_pool_2"].lower_bound == pytest.approx(0.94)
    # Proximity costs five points at a pool of two and gains two at five.
    assert metrics["geo_advantage_pool_2"].value == pytest.approx(-0.05)
    assert metrics["geo_advantage_pool_5"].value == pytest.approx(0.02)
    assert metrics["geo_radius_metres_pool_5"].value == pytest.approx(310.0)
    assert metrics["geo_radius_metres_pool_5"].upper_bound == pytest.approx(2400.0)
    assert metrics["evaluated_folds"].value == pytest.approx(100.0)
    assert metrics["candidate_dwarfs"].value == pytest.approx(9.0)

    with pytest.raises(GeoAblationError, match="no pool sizes"):
        summarize_geo((), {}, 1, 1)


def test_run_requires_the_geo_experiment_group() -> None:
    with pytest.raises(GeoAblationError, match="requires experiment=geo_ablation"):
        run_geo_ablation(load_config(["experiment=baseline"]))


def test_cli_geo_ablation_reports_both_arms_and_needs_coordinates(tmp_path: Path) -> None:
    manifest = synthetic_manifest(dwarf_count=4, coordinates=_line(4))
    manifest_path = tmp_path / "manifest.json"
    split_path = tmp_path / "splits" / "leave-one-out.json"
    manifest_path.write_text(json.dumps(manifest.model_dump(mode="json")), encoding="utf-8")
    write_evaluation_split(split_path, build_evaluation_split(manifest, datetime.now(UTC)))
    overrides = [
        f"paths.manifest_path={manifest_path}",
        f"paths.evaluation_split_path={split_path}",
        f"paths.embeddings_dir={tmp_path / 'embeddings'}",
        f"paths.results_dir={tmp_path / 'results'}",
        f"backbone.model_id={FAKE_BACKBONE.model_id}",
        f"backbone.revision={FAKE_BACKBONE.revision}",
        f"backbone.preprocessing_id={FAKE_BACKBONE.preprocessing_id}",
        "experiment.pool_sizes=[2,3]",
        "logging.json_output=false",
    ]
    arguments = ["experiment", "geo-ablation", *[f"-o{value}" for value in overrides]]
    runner = CliRunner()

    missing = runner.invoke(app, arguments)
    assert missing.exit_code == 2
    assert "Geographic ablation error" in missing.output

    seed_embedding_cache(tmp_path / "embeddings", manifest)
    result = runner.invoke(app, arguments)
    assert result.exit_code == 0, result.output
    assert "geo_top_1_pool_2" in result.output
    assert "random_top_1_pool_2" in result.output
    assert "geo_advantage_pool_2" in result.output
    assert "located_fraction" in result.output
    assert " m median" in result.output

    written = ExperimentResult.model_validate(
        json.loads((tmp_path / "results" / "geo_ablation-dinov2.json").read_text(encoding="utf-8"))
    )
    assert written.experiment == "geo_ablation"

    # A manifest where nothing is placed cannot be pooled by proximity at all.
    unplaced = synthetic_manifest(dwarf_count=4)
    manifest_path.write_text(json.dumps(unplaced.model_dump(mode="json")), encoding="utf-8")
    write_evaluation_split(split_path, build_evaluation_split(unplaced, datetime.now(UTC)))
    failed = runner.invoke(app, arguments)
    assert failed.exit_code == 2
    assert "none of the 4 dwarves carry coordinates" in failed.output

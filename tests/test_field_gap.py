"""Field photographs as queries: staging them, and the domain gap they measure."""

import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from PIL import Image
from typer.testing import CliRunner

from helpers import FAKE_BACKBONE, seed_embedding_cache, synthetic_manifest
from krasnal_id.cli import app
from krasnal_id.config import FieldGapExperimentConfig, load_config
from krasnal_id.data_pipeline.build_split import build_evaluation_split, write_evaluation_split
from krasnal_id.data_pipeline.field_queries import (
    FieldCohort,
    FieldQueryError,
    FieldQueryManifest,
    FieldRouteFile,
    field_query_manifest_path,
    load_field_query_manifest,
    load_field_route,
    queries_by_cohort,
    stage_field_queries,
    write_field_query_manifest,
)
from krasnal_id.embeddings.cache import EmbeddingCache
from krasnal_id.embeddings.store import cache_key_for, load_embedding_matrix
from krasnal_id.experiments.field_gap import (
    FieldGapError,
    StatueRanks,
    bootstrap_gaps,
    collect_statue_ranks,
    gap,
    run_field_gap,
    score_commons_disjoint_queries,
    score_commons_queries,
    score_field_queries,
    summarize_classes,
    summarize_field_gap,
)
from krasnal_id.models import DatasetManifest

runner = CliRunner()

ROUTE: dict[str, Any] = {
    "schema_version": "1.0",
    "entries": [
        {
            "dwarf_id": "Q1",
            "display_name": "Dwarf 1",
            "cohort": "confusable",
            "recorded_top_1_errors": 7,
            "tier": "core",
        },
        {
            "dwarf_id": "Q2",
            "display_name": "Dwarf 2",
            "cohort": "control",
            "recorded_top_1_errors": 0,
            "tier": "core",
        },
        {
            "dwarf_id": "Q3",
            "display_name": "Dwarf 3",
            "cohort": "control",
            "recorded_top_1_errors": 1,
            "tier": "extended",
        },
    ],
}


def _write_photograph(path: Path, color: tuple[int, int, int]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (8, 8), color).save(path)


def _field_directories(root: Path, counts: dict[str, int]) -> Path:
    """Lay out one directory of photographs per statue, as the guide asks for."""
    for index, (dwarf_id, count) in enumerate(sorted(counts.items())):
        for position in range(count):
            _write_photograph(
                root / dwarf_id / f"IMG_{position}.png", (10 + index * 5, 20 + position, 30)
            )
    return root


def _route(payload: dict[str, Any] | None = None) -> FieldRouteFile:
    return FieldRouteFile.model_validate(payload or ROUTE)


def _stage(tmp_path: Path, counts: dict[str, int], manifest: DatasetManifest) -> FieldQueryManifest:
    root = _field_directories(tmp_path / "field-queries", counts)
    return stage_field_queries(root, manifest, _route(), datetime.now(UTC))


def _seed_field_vectors(cache_root: Path, staged: FieldQueryManifest, dwarf_count: int) -> None:
    """Give each field photograph a vector near, but not on, its statue's axis.

    The perturbation is what a field photograph is: recognisably the same statue,
    further from it than another Commons upload of the same statue would be.
    """
    cache = EmbeddingCache(cache_root)
    dwarf_ids = sorted({query.dwarf_id for query in staged.queries})
    for position, query in enumerate(staged.queries):
        vector = np.zeros(dwarf_count + 1, dtype=np.float32)
        vector[dwarf_ids.index(query.dwarf_id)] = 1.0
        vector[dwarf_count] = 0.5 + 0.1 * position
        cache.store(
            cache_key_for(query, FAKE_BACKBONE),
            np.asarray(vector / np.linalg.norm(vector), dtype=np.float32),
        )


def test_the_route_needs_unique_statues_and_both_cohorts() -> None:
    route = _route()
    assert route.entry_for("Q1") is not None
    assert route.entry_for("absent") is None

    with pytest.raises(ValueError, match="cannot appear on the route twice"):
        FieldRouteFile.model_validate(
            {"schema_version": "1.0", "entries": [ROUTE["entries"][0], ROUTE["entries"][0]]}
        )
    # A drop measured without controls is confounded, so a route without them is refused.
    with pytest.raises(ValueError, match="needs both cohorts"):
        FieldRouteFile.model_validate({"schema_version": "1.0", "entries": [ROUTE["entries"][0]]})
    assert route.entry_for("Q2").recorded_top_1_errors == 0  # type: ignore[union-attr]


def test_the_tracked_route_matches_the_dataset_and_the_guide() -> None:
    route = load_field_route(Path("data/field-route.json"))
    ids = {entry.dwarf_id for entry in route.entries}

    assert len(ids) == len(route.entries)
    assert {entry.cohort for entry in route.entries} == set(FieldCohort)
    # The core route is a complete experiment on its own, as `AGENTS.md` 5.8 requires:
    # the eleven statues of the confused families, plus controls in the same streets.
    core = [entry for entry in route.entries if entry.tier == "core"]
    assert len(core) == 19
    assert sum(1 for entry in core if entry.cohort == FieldCohort.CONFUSABLE) == 11
    assert sum(1 for entry in core if entry.cohort == FieldCohort.CONTROL) == 8
    # Cohort is family membership, not an error count, so four controls carry errors.
    # That biases towards finding no difference between the cohorts, which is why the
    # count is recorded beside the label rather than left to the reader to discover.
    contaminated = [
        entry
        for entry in core
        if entry.cohort == FieldCohort.CONTROL and entry.recorded_top_1_errors
    ]
    assert len(contaminated) == 4
    assert max(entry.recorded_top_1_errors for entry in contaminated) == 3
    assert all(entry.recorded_top_1_errors >= 1 for entry in core if entry.cohort == "confusable")


def test_photographs_are_staged_by_the_directory_they_sit_in(tmp_path: Path) -> None:
    manifest = synthetic_manifest(dwarf_count=3)
    staged = _stage(tmp_path, {"Q1": 2, "Q2": 3}, manifest)

    assert [query.image_id for query in staged.queries] == [
        "Q1/IMG_0.png",
        "Q1/IMG_1.png",
        "Q2/IMG_0.png",
        "Q2/IMG_1.png",
        "Q2/IMG_2.png",
    ]
    assert {query.dwarf_id for query in staged.queries} == {"Q1", "Q2"}
    grouped = queries_by_cohort(staged.queries)
    assert len(grouped[FieldCohort.CONFUSABLE]) == 2
    assert len(grouped[FieldCohort.CONTROL]) == 3
    # The manifest is never rebuilt to include them: staging only records what it read.
    assert staged.manifest_sha256 and staged.route_sha256


def test_staging_refuses_photographs_it_cannot_attribute(tmp_path: Path) -> None:
    manifest = synthetic_manifest(dwarf_count=3)

    with pytest.raises(FieldQueryError, match="does not exist"):
        stage_field_queries(tmp_path / "absent", manifest, _route())

    empty = tmp_path / "empty"
    empty.mkdir()
    with pytest.raises(FieldQueryError, match="no field photographs are staged"):
        stage_field_queries(empty, manifest, _route())

    unknown = _field_directories(tmp_path / "unknown", {"Q9": 1})
    with pytest.raises(FieldQueryError, match="names no dwarf in the manifest"):
        stage_field_queries(unknown, manifest, _route())

    # In the dataset, but nobody decided its cohort before the walk.
    off_route = _field_directories(tmp_path / "off-route", {"Q3": 1})
    two_statues = {"schema_version": "1.0", "entries": ROUTE["entries"][:2]}
    with pytest.raises(FieldQueryError, match="not on the reviewed route"):
        stage_field_queries(off_route, manifest, _route(two_statues))

    broken = tmp_path / "broken"
    (broken / "Q1").mkdir(parents=True)
    (broken / "Q1" / "IMG_0.png").write_text("not an image", encoding="utf-8")
    with pytest.raises(FieldQueryError, match="cannot be decoded"):
        stage_field_queries(broken, manifest, _route())


def test_a_reference_copied_into_the_query_set_is_refused(tmp_path: Path) -> None:
    root = _field_directories(tmp_path / "field-queries", {"Q1": 1})
    copied = root / "Q1" / "IMG_0.png"
    manifest = synthetic_manifest(dwarf_count=3)
    manifest = manifest.model_copy(
        update={
            "images": (
                manifest.images[0].model_copy(
                    update={"sha256": hashlib.sha256(copied.read_bytes()).hexdigest()}
                ),
                *manifest.images[1:],
            )
        }
    )

    with pytest.raises(FieldQueryError, match="byte-identical to reference image"):
        stage_field_queries(root, manifest, _route())


def test_dotfiles_are_skipped_rather_than_failing_the_run(tmp_path: Path) -> None:
    root = _field_directories(tmp_path / "field-queries", {"Q1": 1, "Q2": 1})
    (root / "Q1" / ".DS_Store").write_text("junk", encoding="utf-8")
    (root / "unrelated-file.txt").write_text("junk", encoding="utf-8")

    staged = stage_field_queries(root, synthetic_manifest(dwarf_count=3), _route())

    assert len(staged.queries) == 2


def test_a_stale_query_manifest_is_detected(tmp_path: Path) -> None:
    manifest = synthetic_manifest(dwarf_count=3)
    staged = _stage(tmp_path, {"Q1": 1, "Q2": 1}, manifest)
    path = tmp_path / "field-queries.json"
    write_field_query_manifest(path, staged)

    assert load_field_query_manifest(path, manifest).queries == staged.queries

    rebuilt = manifest.model_copy(update={"minimum_images_per_dwarf": 4})
    with pytest.raises(FieldQueryError, match="was staged against manifest"):
        load_field_query_manifest(path, rebuilt)

    path.write_text("{}", encoding="utf-8")
    with pytest.raises(FieldQueryError, match="unusable field query manifest"):
        load_field_query_manifest(path, manifest)
    # A manifest that was never staged reports how to stage one, not just its absence.
    with pytest.raises(FieldQueryError, match="stage the photographs"):
        load_field_query_manifest(tmp_path / "absent.json", manifest)
    with pytest.raises(FieldQueryError, match="invalid field route"):
        load_field_route(tmp_path / "absent.json")


def test_field_queries_are_ranked_against_the_unchanged_reference_set(tmp_path: Path) -> None:
    manifest = synthetic_manifest(dwarf_count=3)
    staged = _stage(tmp_path, {"Q1": 2, "Q2": 2}, manifest)
    seed_embedding_cache(tmp_path / "cache", manifest)
    _seed_field_vectors(tmp_path / "cache", staged, dwarf_count=3)
    matrix = load_embedding_matrix(manifest, FAKE_BACKBONE, tmp_path / "cache")
    cache = EmbeddingCache(tmp_path / "cache")
    vectors = {
        query.image_id: cache.load(cache_key_for(query, FAKE_BACKBONE)) for query in staged.queries
    }

    scored = score_field_queries(staged, matrix, vectors)  # type: ignore[arg-type]

    assert len(scored) == 4
    # Nothing is withheld: a field photograph is not in the manifest to begin with.
    assert all(rank >= 1 for _, rank in scored.values())
    assert all(rank == 1 for _, rank in scored.values())


def test_the_commons_side_covers_only_the_statues_that_were_photographed(
    tmp_path: Path,
) -> None:
    manifest = synthetic_manifest(dwarf_count=3)
    seed_embedding_cache(tmp_path, manifest)
    matrix = load_embedding_matrix(manifest, FAKE_BACKBONE, tmp_path)
    split = build_evaluation_split(manifest, datetime.now(UTC))

    ranks = score_commons_queries(split, matrix, frozenset({"Q1", "Q2"}))

    assert set(ranks) == {"Q1", "Q2"}
    assert sum(len(values) for values in ranks.values()) == 6
    with pytest.raises(FieldGapError, match="nothing to compare"):
        score_commons_queries(split, matrix, frozenset({"Q9"}))


def _statue(
    dwarf_id: str,
    cohort: FieldCohort,
    field: tuple[int, ...],
    commons: tuple[int, ...],
    disjoint: tuple[int, ...] = (),
) -> StatueRanks:
    return StatueRanks(
        dwarf_id=dwarf_id, cohort=cohort, field=field, commons=commons, commons_disjoint=disjoint
    )


def test_the_gap_is_reported_per_cohort_in_the_direction_that_reads_correctly() -> None:
    statues = (
        _statue("Q1", FieldCohort.CONFUSABLE, (4, 6), (1, 2), (2, 3)),
        _statue("Q2", FieldCohort.CONTROL, (1, 1), (1, 1), (1, 1)),
    )

    metrics = {metric.name: metric for metric in summarize_field_gap(statues, (1, 5), 200, 7)}

    assert metrics["field_top_1"].value == pytest.approx(0.5)
    assert metrics["commons_top_1"].value == pytest.approx(0.75)
    # Positive means the field photographs did worse, which is the expected direction.
    assert metrics["top_1_gap"].value == pytest.approx(0.25)
    # A drop concentrated on the confusable clusters is a different finding from a
    # uniform one, and only the cohorts make the difference visible.
    assert metrics["confusable_top_1_gap"].value == pytest.approx(0.5)
    assert metrics["control_top_1_gap"].value == pytest.approx(0.0)
    assert metrics["cohort_top_1_gap_difference"].value == pytest.approx(0.5)
    assert metrics["field_confusable_top_5"].value == pytest.approx(0.5)
    assert metrics["field_queries"].value == pytest.approx(4.0)
    assert metrics["field_classes"].value == pytest.approx(2.0)
    # Against the photographer-disjoint folds: Q1 drops to 0 of 2 there too.
    assert metrics["commons_disjoint_top_1"].value == pytest.approx(0.5)
    assert metrics["disjoint_top_1_gap"].value == pytest.approx(0.0)
    assert metrics["disjoint_classes"].value == pytest.approx(2.0)

    with pytest.raises(FieldGapError, match="no photographed statues"):
        summarize_field_gap((), (1,), 200, 7)
    with pytest.raises(FieldGapError, match="must be positive"):
        summarize_field_gap(statues, (0,), 200, 7)
    with pytest.raises(FieldGapError, match="resamples must be positive"):
        summarize_field_gap(statues, (1,), 0, 7)


def test_a_cohort_with_no_queries_reports_no_gap() -> None:
    statues = (_statue("Q1", FieldCohort.CONFUSABLE, (1, 2), (1, 1, 2)),)

    names = {metric.name for metric in summarize_field_gap(statues, (1,), 200, 7)}

    assert "confusable_top_1_gap" in names
    assert "control_top_1_gap" not in names
    assert "cohort_top_1_gap_difference" not in names
    # A statue nobody else photographed cannot be asked cross-photographer at all.
    assert "disjoint_top_1_gap" not in names


def test_the_disjoint_gap_covers_only_statues_answerable_without_their_photographer() -> None:
    statues = (
        _statue("Q1", FieldCohort.CONTROL, (1, 1), (1, 1), (1, 1)),
        # Single photographer: in the standard gap, absent from the disjoint one.
        _statue("Q2", FieldCohort.CONTROL, (2, 2), (1, 1), ()),
    )

    metrics = {metric.name: metric.value for metric in summarize_field_gap(statues, (1,), 200, 7)}

    assert metrics["top_1_gap"] == pytest.approx(0.5)
    # Field side restricted to Q1 as well, so both sides cover one set of statues.
    assert metrics["disjoint_top_1_gap"] == pytest.approx(0.0)
    assert metrics["disjoint_classes"] == pytest.approx(1.0)
    assert metrics["commons_disjoint_queries"] == pytest.approx(2.0)


def test_intervals_resample_statues_not_photographs() -> None:
    # Two statues, each perfectly consistent within itself: one the field arm
    # always gets, one it always misses. Photographs of one statue are not
    # independent, so the interval has to reach both extremes.
    statues = (
        _statue("Q1", FieldCohort.CONTROL, (1,) * 5, (1,) * 5),
        _statue("Q2", FieldCohort.CONTROL, (9,) * 5, (1,) * 5),
    )

    draws = bootstrap_gaps(statues, "commons", 1, 2000, np.random.default_rng(3))
    metric = next(
        metric
        for metric in summarize_field_gap(statues, (1,), 2000, 3)
        if metric.name == "top_1_gap"
    )

    assert gap(statues, "commons", 1) == pytest.approx(0.5)
    assert set(np.unique(draws)) == {0.0, 0.5, 1.0}
    assert metric.lower_bound == pytest.approx(0.0)
    assert metric.upper_bound == pytest.approx(1.0)
    # Reproducible from the configured seed.
    again = next(
        metric
        for metric in summarize_field_gap(statues, (1,), 2000, 3)
        if metric.name == "top_1_gap"
    )
    assert again == metric


def test_every_photographed_statue_is_reported_worst_first() -> None:
    manifest = synthetic_manifest(dwarf_count=3)
    statues = (
        _statue("Q1", FieldCohort.CONFUSABLE, (3, 5), (1, 1, 1), (1, 2)),
        _statue("Q2", FieldCohort.CONTROL, (1, 1), (1, 2, 1)),
    )

    rows = summarize_classes(statues, manifest)

    assert [row.dwarf_id for row in rows] == ["Q1", "Q2"]
    assert rows[0].field_top_1_hits == 0
    assert rows[0].field_mean_rank == pytest.approx(4.0)
    assert rows[0].commons_top_1_hits == 3
    assert rows[0].commons_disjoint_queries == 2
    assert rows[0].commons_disjoint_top_1_hits == 1
    assert rows[0].cohort == "confusable"
    assert rows[1].field_top_1_hits == 2
    assert rows[1].commons_disjoint_queries == 0


def test_statue_ranks_are_collected_and_a_missing_fold_is_refused(tmp_path: Path) -> None:
    manifest = synthetic_manifest(dwarf_count=3)
    staged = _stage(tmp_path, {"Q1": 2, "Q2": 1}, manifest)
    scored = {query.image_id: (query, 1 + index) for index, query in enumerate(staged.queries)}

    statues = collect_statue_ranks(scored, {"Q1": [1, 1], "Q2": [2]}, {"Q1": [1]})

    assert [statue.dwarf_id for statue in statues] == ["Q1", "Q2"]
    assert statues[0].field == (1, 2)
    assert statues[0].commons_disjoint == (1,)
    assert statues[1].commons_disjoint == ()
    assert statues[1].cohort == FieldCohort.CONTROL
    with pytest.raises(FieldGapError, match="no leave-one-out fold"):
        collect_statue_ranks(scored, {"Q1": [1, 1]}, {})
    with pytest.raises(FieldGapError, match="unknown query origin"):
        statues[0].ranks("elsewhere")


def test_the_disjoint_folds_withhold_the_query_photographer(tmp_path: Path) -> None:
    manifest = synthetic_manifest(dwarf_count=3)
    # Q1 and Q2 each have two photographers; Q3 has only one.
    authors = {
        "image-1-0": "Ada",
        "image-1-1": "Ada",
        "image-1-2": "Ben",
        "image-2-0": "Ada",
        "image-2-1": "Cy",
        "image-2-2": "Cy",
    }
    manifest = manifest.model_copy(
        update={
            "images": tuple(
                image.model_copy(update={"author": authors.get(image.image_id, "Dee")})
                for image in manifest.images
            )
        }
    )
    seed_embedding_cache(tmp_path, manifest)
    matrix = load_embedding_matrix(manifest, FAKE_BACKBONE, tmp_path)
    split = build_evaluation_split(manifest, datetime.now(UTC))

    ranks = score_commons_disjoint_queries(split, matrix, manifest, frozenset({"Q1", "Q2", "Q3"}))

    # Every fold of an answerable statue is scored; the single-photographer one is not.
    assert set(ranks) == {"Q1", "Q2"}
    assert len(ranks["Q1"]) == 3
    assert len(ranks["Q2"]) == 3


def test_a_heic_photograph_is_refused_with_the_fix(tmp_path: Path) -> None:
    root = tmp_path / "field-queries"
    (root / "Q1").mkdir(parents=True)
    (root / "Q1" / "IMG_0001.HEIC").write_bytes(b"ftypheic not decodable here")

    with pytest.raises(FieldQueryError, match="Most Compatible"):
        stage_field_queries(root, synthetic_manifest(dwarf_count=3), _route())


def test_run_requires_the_field_gap_experiment_group(tmp_path: Path) -> None:
    manifest = synthetic_manifest(dwarf_count=3)
    staged = _stage(tmp_path, {"Q1": 1, "Q2": 1}, manifest)
    with pytest.raises(FieldGapError, match="requires experiment=field_gap"):
        run_field_gap(load_config(["experiment=baseline"]), staged)


def test_packaged_field_gap_defaults_are_usable() -> None:
    config = load_config(["experiment=field_gap"])
    assert isinstance(config.experiment, FieldGapExperimentConfig)
    assert 1 in config.experiment.top_k
    assert config.paths.field_route_path == Path("data/field-route.json")
    assert config.paths.field_queries_dir == Path("data/field-queries")
    assert config.experiment.bootstrap_resamples >= 1000
    with pytest.raises(ValueError, match="top_k values must be positive"):
        FieldGapExperimentConfig.model_validate(
            {"kind": "field_gap", "seed": 1, "top_k": (0,), "bootstrap_resamples": 1000}
        )


def test_cli_stages_embeds_and_scores_the_field_photographs(tmp_path: Path) -> None:
    manifest = synthetic_manifest(dwarf_count=3)
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(manifest.model_dump(mode="json")), encoding="utf-8")
    split_path = tmp_path / "splits" / "leave-one-out.json"
    write_evaluation_split(split_path, build_evaluation_split(manifest, datetime.now(UTC)))
    seed_embedding_cache(tmp_path / "embeddings", manifest)
    route_path = tmp_path / "field-route.json"
    route_path.write_text(json.dumps(ROUTE), encoding="utf-8")
    field_dir = _field_directories(tmp_path / "field-queries", {"Q1": 2, "Q2": 2})

    paths = [
        f"-opaths.{name}"
        for name in (
            f"data_dir={tmp_path}",
            f"manifest_path={manifest_path}",
            f"evaluation_split_path={split_path}",
            f"embeddings_dir={tmp_path / 'embeddings'}",
            f"results_dir={tmp_path / 'results'}",
            f"field_route_path={route_path}",
            f"field_queries_dir={field_dir}",
        )
    ]
    backbone = [
        f"-obackbone.{name}"
        for name in (
            f"model_id={FAKE_BACKBONE.model_id}",
            f"revision={FAKE_BACKBONE.revision}",
            f"preprocessing_id={FAKE_BACKBONE.preprocessing_id}",
        )
    ]

    staged_run = runner.invoke(
        app, ["data", "field-queries", *paths, "-ologging.json_output=false"]
    )
    assert staged_run.exit_code == 0, staged_run.output
    assert "photographs=4 statues=2 of 3" in staged_run.output
    assert "confusable: 2 photographs" in staged_run.output

    # The experiment refuses to embed anything itself, so it fails until extraction ran.
    arguments = ["experiment", "field-gap", *paths, *backbone, "-ologging.json_output=false"]
    missing = runner.invoke(app, arguments)
    assert missing.exit_code == 2
    assert "no cached" in missing.output

    staged = load_field_query_manifest(field_query_manifest_path(tmp_path), manifest)
    _seed_field_vectors(tmp_path / "embeddings", staged, dwarf_count=3)

    result = runner.invoke(app, arguments)
    assert result.exit_code == 0, result.output
    assert "field_top_1" in result.output
    assert "commons_top_1" in result.output
    assert "top_1_gap" in result.output
    assert "confusable_top_1_gap" in result.output
    # One photographer behind every synthetic reference: nothing can be asked
    # cross-photographer, which is reported as a count rather than a fake gap.
    assert "disjoint_classes: +0.0000" in result.output
    assert "disjoint_top_1_gap" not in result.output
    assert "photographer-disjoint" in result.output

    written = json.loads((tmp_path / "results" / "field_gap-dinov2.json").read_text())
    assert written["experiment"] == "field_gap"
    assert {row["dwarf_id"] for row in written["classes"]} == {"Q1", "Q2"}


def test_cli_field_query_failures_are_reported(tmp_path: Path) -> None:
    route_path = tmp_path / "field-route.json"
    route_path.write_text(json.dumps(ROUTE), encoding="utf-8")
    arguments = [
        "data",
        "field-queries",
        f"-opaths.data_dir={tmp_path}",
        f"-opaths.manifest_path={tmp_path / 'absent.json'}",
        f"-opaths.field_route_path={route_path}",
        f"-opaths.field_queries_dir={tmp_path / 'field-queries'}",
        "-ologging.json_output=false",
    ]

    failure = runner.invoke(app, arguments)

    assert failure.exit_code == 2
    assert "Field query error" in failure.output


def test_cli_extract_can_target_the_field_queries(tmp_path: Path) -> None:
    manifest = synthetic_manifest(dwarf_count=3)
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(manifest.model_dump(mode="json")), encoding="utf-8")
    staged = _stage(tmp_path, {"Q1": 1, "Q2": 1}, manifest)
    write_field_query_manifest(field_query_manifest_path(tmp_path), staged)
    _seed_field_vectors(tmp_path / "embeddings", staged, dwarf_count=3)

    result = runner.invoke(
        app,
        [
            "embeddings",
            "extract",
            "--field-queries",
            f"-opaths.data_dir={tmp_path}",
            f"-opaths.manifest_path={manifest_path}",
            f"-opaths.embeddings_dir={tmp_path / 'embeddings'}",
            f"-obackbone.model_id={FAKE_BACKBONE.model_id}",
            f"-obackbone.revision={FAKE_BACKBONE.revision}",
            f"-obackbone.preprocessing_id={FAKE_BACKBONE.preprocessing_id}",
            "-ologging.json_output=false",
        ],
    )

    # Every vector is already cached, so no backbone is loaded and nothing is computed.
    assert result.exit_code == 0, result.output
    assert "source=field-queries" in result.output
    assert "reused=2 computed=0" in result.output

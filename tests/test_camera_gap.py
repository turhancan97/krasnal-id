"""Phone-versus-camera query comparison and the metadata behind it."""

import json
from datetime import UTC, datetime
from pathlib import Path

import pytest
from typer.testing import CliRunner

from helpers import FAKE_BACKBONE, seed_embedding_cache, synthetic_manifest
from krasnal_id.cli import app
from krasnal_id.config import CameraGapExperimentConfig, load_config
from krasnal_id.data_pipeline.build_split import build_evaluation_split, write_evaluation_split
from krasnal_id.data_pipeline.camera_metadata import (
    CameraMetadataError,
    CameraMetadataFile,
    camera_from_page,
    collect_cameras,
    fetch_camera_metadata,
    load_camera_metadata,
    request_parameters,
)
from krasnal_id.embeddings.store import load_embedding_matrix
from krasnal_id.experiments.camera_gap import (
    CameraGapError,
    GroupOutcome,
    classify_camera,
    group_queries,
    run_camera_gap,
    summarize_camera_gap,
)

runner = CliRunner()


def _metadata(cameras: dict[str, str | None]) -> CameraMetadataFile:
    return CameraMetadataFile(
        schema_version="1.0",
        endpoint="https://commons.wikimedia.org/w/api.php",  # type: ignore[arg-type]
        retrieved_at=datetime.now(UTC),
        cameras=cameras,
    )


def test_phones_and_cameras_are_told_apart() -> None:
    for phone in (
        "Apple iPhone XR",
        "samsung SM-G935F",
        "Xiaomi Redmi Note 8 Pro",
        "Google Pixel 7",
        "motorola Moto C",
    ):
        assert classify_camera(phone) == "phone", phone
    for camera in ("Canon EOS 400D DIGITAL", "NIKON D700", "Panasonic DMC-LS80", "SONY DSLR-A350"):
        assert classify_camera(camera) == "camera", camera
    # Absent EXIF is its own bucket, never silently folded into either group.
    assert classify_camera(None) == "unknown"
    assert classify_camera("   ") == "unknown"


def test_a_page_yields_its_make_and_model_joined() -> None:
    page = {
        "pageid": 7,
        "imageinfo": [
            {
                "commonmetadata": [
                    {"name": "Make", "value": "Apple"},
                    {"name": "Model", "value": "iPhone XR"},
                ]
            }
        ],
    }
    assert camera_from_page(page) == (7, "Apple iPhone XR")
    # A page Commons has no camera for is recorded as known-absent, not skipped.
    assert camera_from_page({"pageid": 8, "imageinfo": [{"commonmetadata": []}]}) == (8, None)
    assert camera_from_page({"pageid": 9}) == (9, None)
    assert camera_from_page({"no": "pageid"}) is None


def test_batches_are_folded_and_api_errors_surface() -> None:
    cameras = collect_cameras(
        (
            {
                "query": {
                    "pages": [
                        {
                            "pageid": 1,
                            "imageinfo": [
                                {"commonmetadata": [{"name": "Model", "value": "iPhone 12"}]}
                            ],
                        }
                    ]
                }
            },
            {"query": {"pages": [{"pageid": 2, "imageinfo": [{"commonmetadata": []}]}]}},
        )
    )
    assert cameras == {"1": "iPhone 12", "2": None}

    with pytest.raises(CameraMetadataError, match="Commons API error"):
        collect_cameras(({"error": {"code": "badpageid"}},))
    with pytest.raises(CameraMetadataError, match="not an object"):
        collect_cameras(([],))


def test_requests_are_batched_by_page_id() -> None:
    manifest = synthetic_manifest(dwarf_count=3)
    seen: list[dict[str, str]] = []

    def session(parameters: dict[str, str]) -> object:
        seen.append(parameters)
        ids = parameters["pageids"].split("|")
        return {
            "query": {
                "pages": [
                    {
                        "pageid": int(i),
                        "imageinfo": [
                            {"commonmetadata": [{"name": "Model", "value": "Canon EOS 5D"}]}
                        ],
                    }
                    for i in ids
                ]
            }
        }

    placed = manifest.model_copy(
        update={
            "images": tuple(
                image.model_copy(update={"commons_page_id": index + 1})
                for index, image in enumerate(manifest.images)
            )
        }
    )
    metadata = fetch_camera_metadata(placed, load_config().data, session)

    assert len(metadata.cameras) == len(placed.images)
    assert all("pageids" in call for call in seen)
    assert set(request_parameters((1, 2))) >= {"action", "prop", "iiprop", "pageids"}

    with pytest.raises(CameraMetadataError, match="no manifest image carries"):
        fetch_camera_metadata(manifest, load_config().data, session)


def test_reading_rejects_a_malformed_artifact(tmp_path: Path) -> None:
    with pytest.raises(CameraMetadataError, match="invalid camera metadata"):
        load_camera_metadata(tmp_path / "absent.json")
    broken = tmp_path / "broken.json"
    broken.write_text("{}", encoding="utf-8")
    with pytest.raises(CameraMetadataError, match="invalid camera metadata"):
        load_camera_metadata(broken)


def test_queries_are_grouped_by_the_camera_that_took_them(tmp_path: Path) -> None:
    manifest = synthetic_manifest(dwarf_count=4)
    manifest = manifest.model_copy(
        update={
            "images": tuple(
                image.model_copy(update={"commons_page_id": index + 1})
                for index, image in enumerate(manifest.images)
            )
        }
    )
    seed_embedding_cache(tmp_path, manifest)
    matrix = load_embedding_matrix(manifest, FAKE_BACKBONE, tmp_path)
    split = build_evaluation_split(manifest, datetime.now(UTC))
    cameras: dict[str, str | None] = {"1": "Apple iPhone XR", "2": "Canon EOS 5D"}

    outcomes = {o.group: o for o in group_queries(split, manifest, matrix, _metadata(cameras))}

    assert outcomes["phone"].ranks and outcomes["camera"].ranks
    # Every image without metadata lands in unknown rather than being dropped.
    assert len(outcomes["unknown"].ranks) == len(split.folds) - 2
    assert sum(len(o.ranks) for o in outcomes.values()) == len(split.folds)

    with pytest.raises(CameraGapError, match="no folds"):
        group_queries(split.model_copy(update={"folds": ()}), manifest, matrix, _metadata({}))


def test_the_gap_is_reported_in_the_direction_that_reads_correctly() -> None:
    outcomes = (
        GroupOutcome(group="phone", ranks=(1, 3, 4, 5), references_per_class=(7, 7, 7, 7)),
        GroupOutcome(group="camera", ranks=(1, 1, 1, 2), references_per_class=(6, 6, 6, 6)),
    )

    metrics = {m.name: m.value for m in summarize_camera_gap(outcomes, (1, 5))}

    assert metrics["phone_top_1"] == pytest.approx(0.25)
    assert metrics["camera_top_1"] == pytest.approx(0.75)
    # Positive means phone queries did worse, which is the expected direction.
    assert metrics["top_1_gap"] == pytest.approx(0.5)
    # The first confound a reader reaches for is reported beside the result.
    assert metrics["phone_median_references"] == pytest.approx(7.0)
    assert metrics["camera_median_references"] == pytest.approx(6.0)

    # With only one group present there is no gap to report.
    solo = {m.name for m in summarize_camera_gap((outcomes[0],), (1,))}
    assert not {name for name in solo if name.endswith("_gap")}
    with pytest.raises(CameraGapError, match="no queries were grouped"):
        summarize_camera_gap((), (1,))


def test_run_requires_the_camera_gap_experiment_group() -> None:
    with pytest.raises(CameraGapError, match="requires experiment=camera_gap"):
        run_camera_gap(load_config(["experiment=baseline"]), _metadata({}))


def test_packaged_camera_gap_defaults_are_usable() -> None:
    experiment = load_config(["experiment=camera_gap"]).experiment
    assert isinstance(experiment, CameraGapExperimentConfig)
    assert 1 in experiment.top_k
    with pytest.raises(ValueError, match="top_k values must be positive"):
        CameraGapExperimentConfig.model_validate({"kind": "camera_gap", "seed": 1, "top_k": (0,)})


def test_cli_camera_gap_reports_groups_and_fails_without_metadata(tmp_path: Path) -> None:
    manifest = synthetic_manifest(dwarf_count=4)
    manifest = manifest.model_copy(
        update={
            "images": tuple(
                image.model_copy(update={"commons_page_id": index + 1})
                for index, image in enumerate(manifest.images)
            )
        }
    )
    manifest_path = tmp_path / "manifest.json"
    split_path = tmp_path / "splits" / "leave-one-out.json"
    manifest_path.write_text(json.dumps(manifest.model_dump(mode="json")), encoding="utf-8")
    write_evaluation_split(split_path, build_evaluation_split(manifest, datetime.now(UTC)))
    seed_embedding_cache(tmp_path / "embeddings", manifest)
    arguments = [
        "experiment",
        "camera-gap",
        *[
            f"-o{v}"
            for v in (
                f"paths.manifest_path={manifest_path}",
                f"paths.evaluation_split_path={split_path}",
                f"paths.embeddings_dir={tmp_path / 'embeddings'}",
                f"paths.results_dir={tmp_path / 'results'}",
                f"paths.discovery_dir={tmp_path / 'discovery'}",
                f"backbone.model_id={FAKE_BACKBONE.model_id}",
                f"backbone.revision={FAKE_BACKBONE.revision}",
                f"backbone.preprocessing_id={FAKE_BACKBONE.preprocessing_id}",
                "logging.json_output=false",
            )
        ],
    ]

    missing = runner.invoke(app, arguments)
    assert missing.exit_code == 2
    assert "Camera gap error" in missing.output

    (tmp_path / "discovery").mkdir(parents=True, exist_ok=True)
    (tmp_path / "discovery" / "camera-metadata.json").write_text(
        json.dumps(
            _metadata({"1": "Apple iPhone XR", "2": "Canon EOS 5D"}).model_dump(mode="json")
        ),
        encoding="utf-8",
    )
    result = runner.invoke(app, arguments)
    assert result.exit_code == 0, result.output
    assert "phone_top_1" in result.output
    assert "camera_top_1" in result.output
    assert "top_1_gap" in result.output

"""Gradio demonstration bundle, callback behavior, and environment setup."""

import hashlib
import json
import os
from pathlib import Path
from typing import Any

import pytest
from PIL import Image
from typer.testing import CliRunner

from helpers import FAKE_BACKBONE, seed_embedding_cache, synthetic_manifest
from krasnal_id.cli import app
from krasnal_id.config import AppConfig, load_config
from krasnal_id.demo import app as demo_app
from krasnal_id.demo.app import (
    DemoBundle,
    DemoError,
    build_interface,
    ensure_private_temp_dir,
    identify,
    import_optional_demo,
    launch,
    load_bundle,
)


def _prepare(tmp_path: Path, dwarf_count: int = 3) -> tuple[AppConfig, Path]:
    """Write a manifest with real image files and seed its embedding cache."""
    manifest = synthetic_manifest(dwarf_count=dwarf_count)
    images = []
    for index, record in enumerate(manifest.images):
        path = tmp_path / "images" / record.dwarf_id / f"{record.image_id}.png"
        path.parent.mkdir(parents=True, exist_ok=True)
        Image.new("RGB", (8, 8), (index * 7 % 256, 40, 60)).save(path)
        # Record the file's real hash so an upload of it resolves to a cached
        # vector, the way a real dataset image does.
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        images.append(record.model_copy(update={"local_path": path, "sha256": digest}))
    manifest = manifest.model_copy(update={"images": tuple(images)})

    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(manifest.model_dump(mode="json")), encoding="utf-8")
    cache_root = tmp_path / "embeddings"
    seed_embedding_cache(cache_root, manifest)

    config = load_config(
        [
            f"paths.manifest_path={manifest_path}",
            f"paths.embeddings_dir={cache_root}",
            f"backbone.model_id={FAKE_BACKBONE.model_id}",
            f"backbone.revision={FAKE_BACKBONE.revision}",
            f"backbone.preprocessing_id={FAKE_BACKBONE.preprocessing_id}",
        ]
    )
    return config, manifest_path


def test_a_configured_temp_dir_is_respected(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("GRADIO_TEMP_DIR", "/somewhere/explicit")
    assert ensure_private_temp_dir() == Path("/somewhere/explicit")


def test_an_unset_temp_dir_becomes_per_user(monkeypatch: pytest.MonkeyPatch) -> None:
    # A shared /tmp/gradio fails outright when another account created it first,
    # so the default must not be shared between users.
    monkeypatch.delenv("GRADIO_TEMP_DIR", raising=False)

    path = ensure_private_temp_dir()

    assert path.is_dir()
    assert path.name != "gradio"
    assert os.environ["GRADIO_TEMP_DIR"] == str(path)


def test_missing_demo_dependencies_name_the_extra() -> None:
    with pytest.raises(DemoError, match="uv sync --extra demo"):
        import_optional_demo("krasnal_id_absent_demo_module")


def test_bundle_loads_once_and_resolves_reference_images(tmp_path: Path) -> None:
    config, _ = _prepare(tmp_path)

    bundle = load_bundle(config, top_k=2)

    assert bundle.top_k == 2
    assert len(bundle.matrix.image_ids) == len(bundle.manifest.images)
    first = bundle.manifest.images[0]
    assert bundle.image_path_for(first.image_id) == first.local_path
    assert bundle.image_path_for("absent") is None


def test_a_deleted_reference_file_is_skipped_rather_than_offered(tmp_path: Path) -> None:
    config, _ = _prepare(tmp_path)
    bundle = load_bundle(config, top_k=2)
    missing = bundle.manifest.images[0]
    missing.local_path.unlink()

    assert bundle.image_path_for(missing.image_id) is None


def test_bundle_rejects_bad_top_k_and_missing_artifacts(tmp_path: Path) -> None:
    config, _ = _prepare(tmp_path)

    with pytest.raises(DemoError, match="top_k must be positive"):
        load_bundle(config, top_k=0)

    empty = load_config(
        [
            f"paths.manifest_path={tmp_path / 'absent.json'}",
            f"paths.embeddings_dir={tmp_path / 'embeddings'}",
        ]
    )
    with pytest.raises(DemoError, match="invalid manifest"):
        load_bundle(empty, top_k=2)


def test_identify_ranks_an_uploaded_photograph(tmp_path: Path) -> None:
    config, _ = _prepare(tmp_path)
    bundle = load_bundle(config, top_k=2)
    query = bundle.manifest.images[0]

    rows, gallery, status = identify(bundle, str(query.local_path))

    assert len(rows) == 2
    assert [row[0] for row in rows] == ["1", "2"]
    # Rank, name, QID, similarity.
    assert all(len(row) == 4 for row in rows)
    assert rows[0][2].startswith("Q")
    assert 0.0 <= float(rows[0][3]) <= 1.0
    assert len(gallery) == 2
    assert all(Path(path).is_file() for path, _ in gallery)
    assert "reference photos" in status


def test_identify_explains_a_withheld_self_match(tmp_path: Path) -> None:
    config, _ = _prepare(tmp_path)
    bundle = load_bundle(config, top_k=2)
    query = bundle.manifest.images[0]
    # A copy of a dataset photo carries the same content, so its own reference
    # must be withheld rather than returned as a perfect self-match.
    uploaded = tmp_path / "uploaded.png"
    uploaded.write_bytes(query.local_path.read_bytes())

    rows, gallery, status = identify(bundle, str(uploaded))

    assert "already in the dataset" in status
    assert query.image_id in status
    assert rows
    # The withheld photograph must not come back as its own closest match.
    assert str(query.local_path) not in [path for path, _ in gallery]


def test_identify_reports_problems_instead_of_raising(tmp_path: Path) -> None:
    config, _ = _prepare(tmp_path)
    bundle = load_bundle(config, top_k=2)

    # A Gradio callback that raises shows a stack trace, not an explanation.
    assert identify(bundle, None) == ([], [], "Upload a photograph to identify it.")
    assert identify(bundle, "") == ([], [], "Upload a photograph to identify it.")

    corrupt = tmp_path / "corrupt.png"
    corrupt.write_bytes(b"not an image")
    rows, gallery, status = identify(bundle, str(corrupt))
    assert rows == []
    assert gallery == []
    assert status.startswith("Could not identify this photograph")

    rows, _, status = identify(bundle, str(tmp_path / "absent.png"))
    assert rows == []
    assert "does not exist" in status


def test_interface_is_built_without_analytics(tmp_path: Path) -> None:
    pytest.importorskip("gradio")
    config, _ = _prepare(tmp_path)

    interface = build_interface(load_bundle(config, top_k=3))

    # Launching a local research demo must not report to an external service.
    assert interface.analytics_enabled is False
    assert interface.title == "Krasnal-ID"


def test_launch_builds_the_interface_and_passes_its_options(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    pytest.importorskip("gradio")
    config, _ = _prepare(tmp_path)
    captured: dict[str, Any] = {}

    def fake_launch(self: Any, **kwargs: Any) -> None:
        captured.update(kwargs)

    monkeypatch.setattr(demo_app, "build_interface", lambda bundle: _Recorder(captured, bundle))

    launch(config, top_k=3, share=True, server_port=7999)

    assert captured["share"] is True
    assert captured["server_port"] == 7999
    assert captured["top_k"] == 3


class _Recorder:
    """Stands in for a Gradio interface so no server is started in tests."""

    def __init__(self, captured: dict[str, Any], bundle: DemoBundle) -> None:
        self._captured = captured
        self._captured["top_k"] = bundle.top_k

    def launch(self, **kwargs: Any) -> None:
        self._captured.update(kwargs)


def test_cli_demo_reports_a_missing_reference_set(tmp_path: Path) -> None:
    runner = CliRunner()

    result = runner.invoke(
        app,
        [
            "demo",
            f"-opaths.manifest_path={tmp_path / 'absent.json'}",
            f"-opaths.embeddings_dir={tmp_path / 'embeddings'}",
            "-ologging.json_output=false",
        ],
    )

    assert result.exit_code == 2
    assert "Demo error" in result.output


def test_cli_demo_launches_with_its_options(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config, manifest_path = _prepare(tmp_path)
    captured: dict[str, Any] = {}

    def fake_launch(
        config: AppConfig, top_k: int = 5, share: bool = False, server_port: int | None = None
    ) -> None:
        captured.update(
            {"top_k": top_k, "share": share, "server_port": server_port, "config": config}
        )

    monkeypatch.setattr("krasnal_id.cli.launch_demo", fake_launch)
    runner = CliRunner()

    result = runner.invoke(
        app,
        [
            "demo",
            "--top-k",
            "3",
            "--port",
            "7999",
            f"-opaths.manifest_path={manifest_path}",
            f"-opaths.embeddings_dir={config.paths.embeddings_dir}",
            "-ologging.json_output=false",
        ],
    )

    assert result.exit_code == 0, result.output
    assert captured["top_k"] == 3
    assert captured["server_port"] == 7999
    assert captured["share"] is False

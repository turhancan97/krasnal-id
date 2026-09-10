"""The Kaggle dataset directory, and the constraints Kaggle enforces on upload."""

import csv
import json
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pytest
from typer.testing import CliRunner

from helpers import (
    FAKE_BACKBONE,
    materialise_images,
    seed_embedding_cache,
    synthetic_manifest,
)
from krasnal_id.cli import app
from krasnal_id.config import AppConfig, load_config
from krasnal_id.data_pipeline.build_split import build_evaluation_split, write_evaluation_split
from krasnal_id.export.kaggle import (
    ALLOWED_LICENSES,
    KEYWORDS,
    LICENSE_NAME,
    SUBTITLE,
    SUBTITLE_LIMITS,
    TITLE,
    TITLE_LIMITS,
    KaggleExportError,
    build_kaggle_export,
    fold_payloads,
    render_metadata,
    validate_dataset_id,
)
from krasnal_id.export.rows import build_image_rows
from krasnal_id.models import DatasetManifest, EvaluationSplit

runner = CliRunner()


def _prepared(tmp_path: Path, dwarfs: int = 3) -> tuple[DatasetManifest, EvaluationSplit]:
    """Build a manifest with real files, its split, and its cached vectors."""
    manifest = synthetic_manifest(dwarf_count=dwarfs, per_dwarf=3)
    manifest = materialise_images(manifest, tmp_path / "images")
    split = build_evaluation_split(manifest, datetime.now(UTC))
    seed_embedding_cache(tmp_path / "embeddings", manifest)
    return manifest, split


def _config(tmp_path: Path, **overrides: str) -> AppConfig:
    settings = {
        "paths.kaggle_export_dir": str(tmp_path / "export"),
        "paths.embeddings_dir": str(tmp_path / "embeddings"),
        **overrides,
    }
    return load_config([f"{key}={value}" for key, value in settings.items()])


def _build(tmp_path: Path, **kwargs: object) -> tuple[Path, DatasetManifest]:
    manifest, split = _prepared(tmp_path)
    result = build_kaggle_export(
        _config(tmp_path),
        manifest,
        split,
        backbones=(FAKE_BACKBONE,),
        **kwargs,  # type: ignore[arg-type]
    )
    return result.paths.root, manifest


def _rows(path: Path) -> list[dict[str, str]]:
    return list(csv.DictReader(path.open(encoding="utf-8")))


def test_the_packaged_title_and_subtitle_fit_kaggles_limits() -> None:
    # Kaggle rejects the upload rather than trimming, and the upload is 676 MB.
    low, high = TITLE_LIMITS
    assert low <= len(TITLE) <= high, TITLE
    low, high = SUBTITLE_LIMITS
    assert low <= len(SUBTITLE) <= high, SUBTITLE


def test_the_licence_is_one_kaggle_accepts_and_claims_nothing_false() -> None:
    assert LICENSE_NAME in ALLOWED_LICENSES
    # This corpus has ten licences across four families. Any specific name would
    # be false of most of it; "other" is Kaggle's escape hatch for exactly that.
    assert LICENSE_NAME == "other"


def test_a_dataset_id_must_be_owner_and_slug() -> None:
    assert validate_dataset_id("someone/wroclaw-dwarves") == ("someone", "wroclaw-dwarves")
    for bad, message in (
        ("wroclaw-dwarves", "username/dataset-slug"),
        ("someone/", "username/dataset-slug"),
        ("/wroclaw", "username/dataset-slug"),
        ("someone/ab", "3-50"),
        ("someone/" + "a" * 51, "3-50"),
        ("someone/has spaces", "letters, digits and hyphens"),
        ("someone/under_score", "letters, digits and hyphens"),
    ):
        with pytest.raises(KaggleExportError, match=message):
            validate_dataset_id(bad)


def test_the_metadata_is_the_shape_kaggle_documents(tmp_path: Path) -> None:
    root, _ = _build(tmp_path)
    payload = json.loads((root / "dataset-metadata.json").read_text(encoding="utf-8"))

    assert set(payload) >= {"title", "id", "licenses"}
    assert payload["licenses"] == [{"name": "other"}]
    assert len(payload["licenses"]) == 1, "Kaggle takes exactly one licence"
    assert payload["id"] == "turhancan97/wroclaw-dwarves"
    assert payload["keywords"] == list(KEYWORDS)
    assert {resource["path"] for resource in payload["resources"]} >= {
        "images",
        "images.csv",
        "classes.csv",
        "folds.csv",
    }


def test_the_description_states_the_per_file_terms_the_licence_field_cannot(
    tmp_path: Path,
) -> None:
    root, _ = _build(tmp_path)
    description = json.loads((root / "dataset-metadata.json").read_text(encoding="utf-8"))[
        "description"
    ]

    # "Other (specified in description)" is only honest if the description does
    # in fact specify them.
    assert "CC BY-SA" in description
    assert "images.csv" in description
    assert "freedom of panorama" in description
    assert "grant no rights in the sculptures" in description
    # And the removal path, which is what makes the disclosure credible.
    assert "image-review.json" in description


def test_every_photograph_is_copied_under_its_dwarf(tmp_path: Path) -> None:
    root, manifest = _build(tmp_path)
    rows = _rows(root / "images.csv")

    assert len(rows) == len(manifest.images)
    for row in rows:
        copied = root / row["file_path"]
        assert copied.is_file(), row["file_path"]
        assert copied.parent.name == row["dwarf_id"]
    # Nothing beyond what the table lists.
    on_disk = {str(p.relative_to(root)) for p in (root / "images").rglob("*") if p.is_file()}
    assert on_disk == {row["file_path"] for row in rows}


def test_the_copied_bytes_are_the_manifests_bytes(tmp_path: Path) -> None:
    import hashlib

    root, manifest = _build(tmp_path)
    digests = {image.image_id: image.sha256 for image in manifest.images}
    for row in _rows(root / "images.csv"):
        payload = (root / row["file_path"]).read_bytes()
        assert hashlib.sha256(payload).hexdigest() == digests[row["image_id"]]


def test_the_vectors_are_row_aligned_with_the_table(tmp_path: Path) -> None:
    root, _ = _build(tmp_path)
    rows = _rows(root / "images.csv")
    vectors = np.load(root / f"embeddings_{FAKE_BACKBONE.name}.npy")

    # A .npy carries no keys, so the only thing tying a vector to a photograph is
    # its position. If that is wrong every downstream number is wrong silently.
    assert vectors.shape[0] == len(rows)
    assert vectors.dtype == np.float32


def test_vectors_out_of_order_are_refused(tmp_path: Path, monkeypatch: object) -> None:
    manifest, split = _prepared(tmp_path)
    config = _config(tmp_path)

    from krasnal_id.embeddings import store
    from krasnal_id.export import kaggle as kaggle_module

    real = store.load_embedding_matrix

    def shuffled(*args: object, **kwargs: object) -> object:
        matrix = real(*args, **kwargs)  # type: ignore[arg-type]
        return type(matrix)(
            image_ids=tuple(reversed(matrix.image_ids)),
            dwarf_ids=tuple(reversed(matrix.dwarf_ids)),
            vectors=matrix.vectors,
        )

    monkeypatch.setattr(kaggle_module, "load_embedding_matrix", shuffled)  # type: ignore[attr-defined]
    with pytest.raises(KaggleExportError, match=r"not in the order images\.csv lists"):
        build_kaggle_export(config, manifest, split, backbones=(FAKE_BACKBONE,))


def test_the_classes_table_carries_the_label_the_vectors_use(tmp_path: Path) -> None:
    root, manifest = _build(tmp_path)
    classes = _rows(root / "classes.csv")
    images = _rows(root / "images.csv")

    assert len(classes) == len(manifest.dwarfs)
    labels = {row["dwarf_id"]: int(row["label"]) for row in classes}
    # Labels are dense, ordered, and agree between the two tables.
    assert sorted(labels.values()) == list(range(len(classes)))
    for row in images:
        assert int(row["label"]) == labels[row["dwarf_id"]]
    counted = {row["dwarf_id"]: 0 for row in classes}
    for row in images:
        counted[row["dwarf_id"]] += 1
    assert {row["dwarf_id"]: int(row["images"]) for row in classes} == counted


def test_folds_are_described_by_rule_rather_than_written_out(tmp_path: Path) -> None:
    root, manifest = _build(tmp_path)
    folds = _rows(root / "folds.csv")

    assert len(folds) == len(manifest.images)
    # Writing the reference sets out would be n^2 identifiers; the rule holds, so
    # only the count is stored.
    assert all(int(row["reference_count"]) == len(manifest.images) - 1 for row in folds)
    assert "reference_image_ids" not in folds[0]


def test_a_split_that_is_not_leave_one_out_is_refused(tmp_path: Path) -> None:
    manifest, split = _prepared(tmp_path)
    rows = build_image_rows(manifest)
    # Drop one reference from one fold: the compact table's stated rule no longer
    # describes it, so publishing that table would state something false.
    first = split.folds[0]
    narrowed = first.model_copy(update={"reference_image_ids": first.reference_image_ids[:-1]})
    broken = split.model_copy(update={"folds": (narrowed, *split.folds[1:])})

    with pytest.raises(KaggleExportError, match="not leave-one-out"):
        fold_payloads(broken, rows)


def test_the_rights_documents_are_the_ones_the_hub_copy_publishes(tmp_path: Path) -> None:
    root, manifest = _build(tmp_path)

    # Same generated artifacts, so the two platforms cannot disagree about a
    # photographer; section 5.11's obligation is per file, not per platform.
    assert (root / "LICENSES.md").read_text(encoding="utf-8").startswith("# Licences")
    assert (root / "ATTRIBUTION.md").read_text(encoding="utf-8").startswith("# Attribution")
    credits = _rows(root / "credits.csv")
    assert len(credits) == len(manifest.images)
    assert all(row["attribution_text"] for row in credits)


def test_an_image_missing_its_attribution_is_refused(tmp_path: Path) -> None:
    manifest, _ = _prepared(tmp_path)
    stripped = manifest.images[0].model_copy(update={"author": "   "})
    manifest = manifest.model_copy(update={"images": (stripped, *manifest.images[1:])})
    # Rebuilt, because editing the manifest changes its hash and the split check
    # would otherwise fire first and hide what this test is about.
    split = build_evaluation_split(manifest, datetime.now(UTC))

    with pytest.raises(KaggleExportError, match="has no author"):
        build_kaggle_export(_config(tmp_path), manifest, split, backbones=(FAKE_BACKBONE,))


def test_a_split_built_for_another_manifest_is_refused(tmp_path: Path) -> None:
    manifest, split = _prepared(tmp_path)
    stale = split.model_copy(update={"manifest_sha256": "0" * 64})

    with pytest.raises(KaggleExportError, match="rebuild it with"):
        build_kaggle_export(_config(tmp_path), manifest, stale, backbones=(FAKE_BACKBONE,))


def test_the_receipt_records_what_was_written(tmp_path: Path) -> None:
    root, manifest = _build(tmp_path)
    receipt = json.loads((root / "provenance.json").read_text(encoding="utf-8"))

    assert receipt["platform"] == "kaggle"
    assert receipt["counts"]["images"] == len(manifest.images)
    assert receipt["images"]["included"] is True
    assert receipt["images"]["bytes"] > 0
    written = {entry["path"] for entry in receipt["files"]}
    assert {"images.csv", "classes.csv", "folds.csv", "dataset-metadata.json"} <= written
    # The 1,691 photographs are digested by the manifest already; listing them
    # here would make the receipt bigger than the table it describes.
    assert not any(entry["path"].startswith("images/") for entry in receipt["files"])


def test_the_pixels_can_be_left_out(tmp_path: Path) -> None:
    root, _ = _build(tmp_path, with_images=False)

    assert not (root / "images").exists()
    assert (root / "images.csv").is_file()
    payload = json.loads((root / "dataset-metadata.json").read_text(encoding="utf-8"))
    assert "images" not in {resource["path"] for resource in payload["resources"]}
    # Every row still names where it came from, so the corpus stays refetchable.
    assert all(row["source_url"] for row in _rows(root / "images.csv"))


def test_the_vectors_can_be_left_out(tmp_path: Path) -> None:
    root, _ = _build(tmp_path, with_embeddings=False)

    assert not (root / f"embeddings_{FAKE_BACKBONE.name}.npy").exists()
    payload = json.loads((root / "dataset-metadata.json").read_text(encoding="utf-8"))
    assert not any(r["path"].startswith("embeddings_") for r in payload["resources"])


def test_a_bad_dataset_id_is_refused_before_anything_is_written(tmp_path: Path) -> None:
    manifest, split = _prepared(tmp_path)
    config = _config(tmp_path)

    with pytest.raises(KaggleExportError, match="username/dataset-slug"):
        build_kaggle_export(
            config, manifest, split, backbones=(FAKE_BACKBONE,), dataset_id="no-slash"
        )
    # Refused *before*: the 676 MB copy must not have started.
    assert not (config.paths.kaggle_export_dir / "images").exists()


def test_render_metadata_refuses_a_licence_kaggle_does_not_know() -> None:
    from krasnal_id.export import kaggle as kaggle_module

    original = kaggle_module.LICENSE_NAME
    try:
        kaggle_module.LICENSE_NAME = "CC-BY-WHATEVER"
        with pytest.raises(KaggleExportError, match="not a licence name Kaggle accepts"):
            render_metadata("someone/wroclaw-dwarves", "text", ["dinov2"], with_images=True)
    finally:
        kaggle_module.LICENSE_NAME = original


def test_cli_builds_the_directory_and_prints_the_upload_command(tmp_path: Path) -> None:
    manifest, split = _prepared(tmp_path)
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(manifest.model_dump(mode="json")), encoding="utf-8")
    split_path = tmp_path / "splits" / "leave-one-out.json"
    write_evaluation_split(split_path, split)

    result = runner.invoke(
        app,
        [
            "data",
            "export-kaggle",
            "--dataset-id",
            "someone/wroclaw-dwarves",
            *[
                f"-o{value}"
                for value in (
                    f"paths.manifest_path={manifest_path}",
                    f"paths.evaluation_split_path={split_path}",
                    f"paths.embeddings_dir={tmp_path / 'embeddings'}",
                    f"paths.kaggle_export_dir={tmp_path / 'export'}",
                    f"backbone.model_id={FAKE_BACKBONE.model_id}",
                    f"backbone.revision={FAKE_BACKBONE.revision}",
                    f"backbone.preprocessing_id={FAKE_BACKBONE.preprocessing_id}",
                    "export.backbones=[dinov2]",
                    "logging.json_output=false",
                )
            ],
        ],
    )

    assert result.exit_code == 0, result.output
    assert "Kaggle export complete" in result.output
    # Publishing stays a human step: a Kaggle slug cannot be renamed once made.
    assert "kaggle datasets create -p" in result.output
    assert "kaggle datasets version -p" in result.output
    assert "someone/wroclaw-dwarves" in result.output
    assert (tmp_path / "export" / "dataset-metadata.json").is_file()

"""Publishing the dataset: the schemas, the rights columns, and the push."""

import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path

import pytest

pytest.importorskip("pyarrow")

import pyarrow as pa
import pyarrow.parquet as pq
import yaml
from typer.testing import CliRunner

from helpers import (
    FAKE_BACKBONE,
    materialise_images,
    seed_embedding_cache,
    synthetic_manifest,
)
from krasnal_id.cli import app
from krasnal_id.config import AppConfig, load_config
from krasnal_id.data_pipeline.build_split import (
    build_evaluation_split,
    write_evaluation_split,
)
from krasnal_id.export import schema as s
from krasnal_id.export.card import (
    license_link_uri,
    measure,
    render_attribution,
    render_credits_csv,
)
from krasnal_id.export.huggingface import (
    ExportPaths,
    HuggingFaceExportError,
    build_export,
)
from krasnal_id.export.push import (
    PushConfigurationError,
    PushError,
    push_export,
    validate_card,
)
from krasnal_id.export.rows import (
    RowError,
    attribution_line,
    build_image_rows,
    commons_filename,
    normalise_license_url,
)
from krasnal_id.export.tables import TableWriteError, write_table
from krasnal_id.models import DatasetManifest, EvaluationSplit

runner = CliRunner()
NAMES = ("C-a-dwarf", "Q1", "Q2")


def _prepared(
    tmp_path: Path, *, unmodified: tuple[str, ...] = (), dwarfs: int = 3
) -> tuple[DatasetManifest, EvaluationSplit]:
    """Build a manifest with real files, its split, and its cached vectors."""
    manifest = synthetic_manifest(dwarf_count=dwarfs, per_dwarf=3)
    manifest = materialise_images(manifest, tmp_path / "images", unmodified=unmodified)
    split = build_evaluation_split(manifest, datetime.now(UTC))
    seed_embedding_cache(tmp_path / "embeddings", manifest)
    return manifest, split


def _config(tmp_path: Path, **overrides: str) -> AppConfig:
    settings = {
        "paths.huggingface_export_dir": str(tmp_path / "export"),
        "paths.embeddings_dir": str(tmp_path / "embeddings"),
        **overrides,
    }
    return load_config([f"{key}={value}" for key, value in settings.items()])


def _read(path: Path) -> pa.Table:
    return pq.read_table(path)


def test_every_schema_matches_what_datasets_would_build() -> None:
    """The oracle: the library that reads the artifact agrees with our schema.

    This is why the export can hand-write parquet at all. If `datasets` derives a
    different arrow schema from our declared features than we wrote, the Image
    column silently stops decoding, and nothing else in the suite would notice.
    """
    datasets = pytest.importorskip("datasets")

    for columns in (
        s.image_columns(NAMES),
        s.metadata_columns(NAMES),
        s.class_columns(NAMES),
        s.fold_columns(NAMES),
        s.embedding_columns(4),
    ):
        features = datasets.Features.from_dict(s.hf_features(columns))
        assert features.arrow_schema.equals(s.arrow_schema(columns))
        # The card carries the list form, which datasets reads through a
        # different code path and rejects if it is given the dict.
        assert s.hf_features_yaml(columns) == features._to_yaml_list()


def test_the_image_column_is_the_struct_the_hub_expects() -> None:
    columns = {column.name: column for column in s.image_columns(NAMES)}
    image = columns["image"].arrow_type

    assert image.num_fields == 2
    assert image.field(0).name == "bytes"
    assert image.field(1).name == "path"
    assert s.hf_features_yaml((columns["image"],)) == [{"name": "image", "dtype": "image"}]


def test_licence_urls_collapse_onto_one_form_per_licence() -> None:
    """Commons supplies several spellings of the same deed."""
    variants = (
        "https://creativecommons.org/licenses/by-sa/4.0",
        "https://creativecommons.org/licenses/by-sa/4.0/",
        "https://creativecommons.org/licenses/by-sa/4.0/deed.en",
        "https://creativecommons.org/licenses/by-sa/4.0/legalcode",
    )
    assert len({normalise_license_url(url) for url in variants}) == 1
    assert normalise_license_url(variants[0]).endswith("by-sa/4.0/")


def test_a_commons_filename_is_recovered_from_its_page_url() -> None:
    assert (
        commons_filename("https://commons.wikimedia.org/wiki/File:Krasnal-Wroc%C5%82aw.jpg")
        == "Krasnal-Wrocław.jpg"
    )
    with pytest.raises(RowError, match="cannot read a Commons filename"):
        commons_filename("https://commons.wikimedia.org/wiki/File:")


def test_a_credit_line_names_the_modification_only_when_there_was_one() -> None:
    common = {
        "commons_file": "Dwarf.jpg",
        "author": "A Photographer",
        "license_url": "https://creativecommons.org/licenses/by-sa/4.0/",
        "source_url": "https://commons.wikimedia.org/wiki/File:Dwarf.jpg",
    }
    modified = attribution_line(license_name="CC BY-SA 4.0", modified=True, **common)
    untouched = attribution_line(license_name="CC BY-SA 4.0", modified=False, **common)
    public = attribution_line(license_name="Public domain", modified=False, **common)

    assert "modified:" in modified
    assert "modified:" not in untouched
    assert "licensed CC BY-SA 4.0" in untouched
    # A public-domain label grants nothing, so it is not described as a licence.
    assert "licensed" not in public


def test_modification_is_measured_per_file_not_asserted(tmp_path: Path) -> None:
    """153 of the real files were never resized; saying otherwise is a false claim."""
    manifest, _ = _prepared(tmp_path, unmodified=("image-1-0", "image-2-1"))

    rows = {row.record.image_id: row for row in build_image_rows(manifest)}

    assert rows["image-1-0"].modified is False
    assert rows["image-1-0"].modification == ""
    assert rows["image-1-1"].modified is True
    assert rows["image-1-1"].modification
    assert sum(row.modified for row in rows.values()) == len(rows) - 2


def test_a_file_that_disagrees_with_the_manifest_is_refused(tmp_path: Path) -> None:
    """These bytes are about to be published under someone's name."""
    manifest, _ = _prepared(tmp_path)
    manifest.images[0].local_path.write_bytes(b"not the photograph the manifest describes")

    with pytest.raises(RowError, match="checksum mismatch"):
        build_image_rows(manifest)


def test_a_missing_file_is_refused(tmp_path: Path) -> None:
    manifest, _ = _prepared(tmp_path)
    manifest.images[0].local_path.unlink()

    with pytest.raises(RowError, match="is missing"):
        build_image_rows(manifest)


def test_the_export_writes_every_config(tmp_path: Path) -> None:
    manifest, split = _prepared(tmp_path)
    config = _config(tmp_path)

    result = build_export(config, manifest, split, backbones=(FAKE_BACKBONE,))
    paths = ExportPaths(root=config.paths.huggingface_export_dir)

    assert result.images == 9
    assert result.classes == 3
    assert result.folds == 9
    assert result.backbones == ("dinov2",)
    for artifact in (
        paths.card,
        paths.licenses,
        paths.attribution,
        paths.credits,
        paths.provenance,
    ):
        assert artifact.is_file(), artifact

    images = _read(next(paths.config_dir("default").glob("*.parquet")))
    assert images.num_rows == 9
    assert images.schema.field("image").type.num_fields == 2
    # The path an extracted file keeps, so it stays traceable to its Commons page.
    assert images.column("image")[0]["path"].as_py().endswith(".jpg")
    assert _read(next(paths.config_dir("classes").glob("*.parquet"))).num_rows == 3
    assert _read(next(paths.config_dir("leave_one_out").glob("*.parquet"))).num_rows == 9
    assert _read(next(paths.config_dir("embeddings_dinov2").glob("*.parquet"))).num_rows == 9


def test_no_published_row_may_lack_its_attribution(tmp_path: Path) -> None:
    """The section 5.11 obligation, as something the writer refuses to violate."""
    manifest, split = _prepared(tmp_path)
    config = _config(tmp_path)
    build_export(config, manifest, split, backbones=(FAKE_BACKBONE,))

    table = _read(
        next(
            ExportPaths(root=config.paths.huggingface_export_dir)
            .config_dir("default")
            .glob("*.parquet")
        )
    )

    for column in ("author", "license", "license_url", "source_url", "commons_file"):
        values = table.column(column).to_pylist()
        assert all(values), f"{column} is empty on some row"
    assert table.column("attribution_text").null_count == 0


def test_a_hole_in_a_required_column_is_refused() -> None:
    columns = s.class_columns(NAMES)
    rows = [
        {
            "dwarf_id": None,
            "label": 0,
            "display_name": "x",
            "commons_category": "x",
            "wikidata_url": None,
            "latitude": None,
            "longitude": None,
            "coordinate_source": None,
            "image_count": 1,
        }
    ]

    with pytest.raises(TableWriteError, match="has no dwarf_id"):
        write_table(Path("unused.parquet"), rows, columns, row_group_rows=10)


def test_an_unplaced_class_reads_as_null_never_as_a_real_place(tmp_path: Path) -> None:
    """(0, 0) is a point in the Gulf of Guinea, not a missing value."""
    manifest, split = _prepared(tmp_path)
    config = _config(tmp_path)
    build_export(config, manifest, split, backbones=(FAKE_BACKBONE,))

    classes = _read(
        next(
            ExportPaths(root=config.paths.huggingface_export_dir)
            .config_dir("classes")
            .glob("*.parquet")
        )
    )

    assert classes.column("latitude").null_count == 3
    assert 0.0 not in classes.column("latitude").to_pylist()
    assert classes.column("coordinate_source").null_count == 3
    assert classes.column("image_count").to_pylist() == [3, 3, 3]


def test_a_stale_split_stops_the_export(tmp_path: Path) -> None:
    manifest, split = _prepared(tmp_path)

    with pytest.raises(HuggingFaceExportError, match="was built for manifest"):
        build_export(
            _config(tmp_path),
            manifest,
            split.model_copy(update={"manifest_sha256": "0" * 64}),
            backbones=(FAKE_BACKBONE,),
        )


def test_a_missing_vector_names_the_command_that_makes_it(tmp_path: Path) -> None:
    manifest = synthetic_manifest(dwarf_count=3, per_dwarf=3)
    manifest = materialise_images(manifest, tmp_path / "images")
    split = build_evaluation_split(manifest, datetime.now(UTC))
    seed_embedding_cache(tmp_path / "embeddings", manifest, skip=("image-1-0",))

    with pytest.raises(HuggingFaceExportError, match="embeddings extract"):
        build_export(_config(tmp_path), manifest, split, backbones=(FAKE_BACKBONE,))


def test_embeddings_can_be_left_out(tmp_path: Path) -> None:
    manifest, split = _prepared(tmp_path)
    config = _config(tmp_path)

    result = build_export(
        config, manifest, split, backbones=(FAKE_BACKBONE,), with_embeddings=False
    )
    paths = ExportPaths(root=config.paths.huggingface_export_dir)

    assert result.backbones == ()
    assert not paths.config_dir("embeddings_dinov2").exists()
    assert "embeddings_dinov2" not in paths.card.read_text(encoding="utf-8")


def test_the_vector_width_comes_from_the_matrix(tmp_path: Path) -> None:
    """Nothing hardcodes 768: the fake backbone's vectors are four-dimensional."""
    manifest, split = _prepared(tmp_path)
    config = _config(tmp_path)
    build_export(config, manifest, split, backbones=(FAKE_BACKBONE,))

    table = _read(
        next(
            ExportPaths(root=config.paths.huggingface_export_dir)
            .config_dir("embeddings_dinov2")
            .glob("*.parquet")
        )
    )

    assert table.schema.field("embedding").type.list_size == 4
    assert table.schema.metadata[b"krasnal_id.backbone.revision"] == b"fake-revision"


def test_folds_never_include_their_own_query(tmp_path: Path) -> None:
    manifest, split = _prepared(tmp_path)
    config = _config(tmp_path)
    build_export(config, manifest, split, backbones=(FAKE_BACKBONE,))

    folds = _read(
        next(
            ExportPaths(root=config.paths.huggingface_export_dir)
            .config_dir("leave_one_out")
            .glob("*.parquet")
        )
    )

    for query, references, count in zip(
        folds.column("query_image_id").to_pylist(),
        folds.column("reference_image_ids").to_pylist(),
        folds.column("reference_count").to_pylist(),
        strict=True,
    ):
        assert query not in references
        assert count == len(references) == 8
    assert folds.column("fold_index").to_pylist() == list(range(9))


def test_sharding_is_derived_rather_than_assumed() -> None:
    assert s.shard_plan(100, 10, 1000).shard_count == 1
    assert s.shard_plan(1000, 10, 500).shard_count == 2
    # More shards than rows is impossible, however small the target.
    tiny = s.shard_plan(10_000, 4, 1)
    assert tiny.shard_count == 4
    assert tiny.ranges == ((0, 1), (1, 2), (2, 3), (3, 4))
    # Equal slices must never leave a trailing empty shard.
    for total in range(1, 40):
        for target in (1, 3, 7, 100):
            plan = s.shard_plan(total * 10, total, target)
            assert all(stop > start for start, stop in plan.ranges)
            assert plan.ranges[-1][1] == total
            assert len(plan.ranges) == plan.shard_count

    with pytest.raises(ValueError, match="zero rows"):
        s.shard_plan(10, 0, 10)
    with pytest.raises(ValueError, match="target must be positive"):
        s.shard_plan(10, 10, 0)


def test_a_large_corpus_is_written_across_shards(tmp_path: Path) -> None:
    manifest, split = _prepared(tmp_path)
    config = _config(tmp_path, **{"export.shard_target_bytes": "1"})

    result = build_export(config, manifest, split, backbones=(FAKE_BACKBONE,))
    shards = sorted(
        ExportPaths(root=config.paths.huggingface_export_dir)
        .config_dir("default")
        .glob("*.parquet")
    )

    assert result.shards == len(shards) == 9
    assert shards[0].name == "reference-00000-of-00009.parquet"
    written = [row for shard in shards for row in _read(shard).column("image_id").to_pylist()]
    assert sorted(written) == sorted(image.image_id for image in manifest.images)


def test_the_card_declares_what_was_written(tmp_path: Path) -> None:
    manifest, split = _prepared(tmp_path)
    config = _config(tmp_path)
    build_export(config, manifest, split, backbones=(FAKE_BACKBONE,))
    paths = ExportPaths(root=config.paths.huggingface_export_dir)

    text = paths.card.read_text(encoding="utf-8")
    front = yaml.safe_load(text.split("---")[1])

    # A mixed collection cannot honestly claim one SPDX licence.
    assert front["license"] == "other"
    assert front["license_link"].endswith("/LICENSES.md")
    declared = {entry["config_name"] for entry in front["configs"]}
    assert declared == {
        "default",
        "metadata",
        "classes",
        "leave_one_out",
        "embeddings_dinov2",
    }
    default_entry = next(entry for entry in front["configs"] if entry.get("default"))
    assert default_entry["config_name"] == "default"
    for entry in front["configs"]:
        name = entry["config_name"]
        assert list(paths.config_dir(name).glob("*.parquet")), name
    # Nothing here is trained, so nothing is called train.
    assert all(
        file["split"] in {"reference", "test"}
        for entry in front["configs"]
        for file in entry["data_files"]
    )
    for statement in ("freedom-of-panorama", "Removal requests", "photographers contributed"):
        assert statement in text


def test_the_card_and_licences_state_the_measured_split(tmp_path: Path) -> None:
    manifest, split = _prepared(tmp_path, unmodified=("image-1-0",))
    config = _config(tmp_path)
    build_export(config, manifest, split, backbones=(FAKE_BACKBONE,))
    paths = ExportPaths(root=config.paths.huggingface_export_dir)

    licences = paths.licenses.read_text(encoding="utf-8")

    assert "**8 of the 9 files are modified**" in licences
    assert "**1 are byte-identical to the Commons original**" in licences


def test_the_credit_ledger_covers_every_photograph(tmp_path: Path) -> None:
    manifest, _ = _prepared(tmp_path)
    rows = build_image_rows(manifest)

    ledger = render_credits_csv(rows).splitlines()
    facts = measure(rows, 3, "a" * 64, ())
    attribution = render_attribution(rows, facts)

    assert len(ledger) == len(rows) + 1
    assert ledger[0].startswith("image_id,dwarf_id,commons_file")
    # Grouping by photographer is what makes the corpus's concentration visible.
    assert attribution.count("## ") == facts.photographers


def test_the_receipt_traces_the_export_to_its_inputs(tmp_path: Path) -> None:
    manifest, split = _prepared(tmp_path)
    config = _config(tmp_path)
    result = build_export(config, manifest, split, backbones=(FAKE_BACKBONE,))
    paths = ExportPaths(root=config.paths.huggingface_export_dir)

    receipt = json.loads(paths.provenance.read_text(encoding="utf-8"))

    assert receipt["manifest"]["manifest_sha256"] == result.manifest_sha256 == split.manifest_sha256
    assert receipt["manifest"]["staging_sha256"] == manifest.staging_sha256
    assert receipt["backbones"][0]["revision"] == FAKE_BACKBONE.revision
    assert receipt["counts"] == {
        "dwarfs": 3,
        "images": 9,
        "folds": 9,
        "shards": 1,
        "modified": 9,
        "unmodified": 0,
        "photographers": 1,
    }
    # Every emitted file is digested, so a published dataset traces to bytes.
    listed = {entry["path"] for entry in receipt["files"]}
    assert "README.md" in listed and "credits.csv" in listed
    for entry in receipt["files"]:
        payload = (paths.root / entry["path"]).read_bytes()
        assert hashlib.sha256(payload).hexdigest() == entry["sha256"]
        assert len(payload) == entry["size_bytes"]


def test_two_builds_of_one_manifest_agree(tmp_path: Path) -> None:
    manifest, split = _prepared(tmp_path)
    stamp = datetime(2026, 9, 6, tzinfo=UTC)
    digests = []
    for name in ("first", "second"):
        config = _config(tmp_path, **{"paths.huggingface_export_dir": str(tmp_path / name)})
        build_export(config, manifest, split, backbones=(FAKE_BACKBONE,), generated_at=stamp)
        root = tmp_path / name
        digests.append(
            {
                path.relative_to(root).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
                for path in sorted(root.rglob("*"))
                if path.is_file()
            }
        )

    assert digests[0] == digests[1]


def test_a_failed_write_leaves_no_debris(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    manifest, split = _prepared(tmp_path)
    config = _config(tmp_path)

    def refuse(*args: object, **kwargs: object) -> None:
        raise OSError("no space left on device")

    monkeypatch.setattr("krasnal_id.atomic.os.replace", refuse)
    with pytest.raises((HuggingFaceExportError, TableWriteError)):
        build_export(config, manifest, split, backbones=(FAKE_BACKBONE,))

    leftovers = list(config.paths.huggingface_export_dir.rglob("*.tmp"))
    assert leftovers == []


class _FakeUploader:
    """Stands in for HfApi, so the push path is testable with no network."""

    def __init__(self, fail: bool = False) -> None:
        self.calls: list[tuple[str, dict[str, object]]] = []
        self.fail = fail

    def create_repo(self, repo_id: str, **kwargs: object) -> None:
        self.calls.append(("create_repo", {"repo_id": repo_id, **kwargs}))

    def upload_folder(self, **kwargs: object) -> str:
        self.calls.append(("upload_folder", kwargs))
        if self.fail:
            raise RuntimeError("429 Too Many Requests")
        return "https://huggingface.co/datasets/someone/something"


def test_a_push_uploads_the_directory_that_was_built(tmp_path: Path) -> None:
    directory = tmp_path / "export"
    directory.mkdir()
    uploader = _FakeUploader()

    outcome = push_export(
        directory,
        "someone/something",
        private=True,
        commit_message="Export abc",
        uploader=uploader,
    )

    create, upload = uploader.calls
    assert create[1] == {
        "repo_id": "someone/something",
        "repo_type": "dataset",
        "exist_ok": True,
        "private": True,
    }
    assert upload[1]["folder_path"] == str(directory)
    assert upload[1]["repo_type"] == "dataset"
    assert upload[1]["commit_message"] == "Export abc"
    # The token is the hub's business; nothing here passes or holds one.
    assert "token" not in create[1] and "token" not in upload[1]
    assert outcome.private is True


def test_push_refuses_before_reaching_the_network(tmp_path: Path) -> None:
    directory = tmp_path / "export"
    directory.mkdir()

    with pytest.raises(PushConfigurationError, match="not a Hugging Face repository id"):
        push_export(directory, "no-namespace", private=True, commit_message="x")
    with pytest.raises(PushConfigurationError, match="nothing to push"):
        push_export(tmp_path / "absent", "a/b", private=True, commit_message="x")
    with pytest.raises(PushConfigurationError, match="no Hugging Face token"):
        push_export(directory, "a/b", private=True, commit_message="x", token_check=lambda: False)
    with pytest.raises(PushError, match="upload to someone/something failed"):
        push_export(
            directory,
            "someone/something",
            private=True,
            commit_message="x",
            uploader=_FakeUploader(fail=True),
        )


def test_the_licence_link_is_an_absolute_uri() -> None:
    """The Hub's validator rejects a repository-relative link outright."""
    resolved = license_link_uri("turhancan97/wroclaw-dwarves", "LICENSES.md")

    assert resolved == (
        "https://huggingface.co/datasets/turhancan97/wroclaw-dwarves/blob/main/LICENSES.md"
    )
    # An absolute link is left alone.
    assert license_link_uri("a/b", "https://example.org/x") == "https://example.org/x"


def test_the_card_declares_a_link_the_hub_will_accept(tmp_path: Path) -> None:
    manifest, split = _prepared(tmp_path)
    config = _config(tmp_path)
    build_export(config, manifest, split, backbones=(FAKE_BACKBONE,))

    front = yaml.safe_load(
        ExportPaths(root=config.paths.huggingface_export_dir)
        .card.read_text(encoding="utf-8")
        .split("---")[1]
    )

    assert front["license_link"].startswith("https://")
    assert front["license_link"].endswith("/LICENSES.md")


def test_an_export_with_no_card_is_refused_before_anything_is_created(tmp_path: Path) -> None:
    """A repository must not be created for a directory that cannot be published."""
    empty = tmp_path / "empty"
    empty.mkdir()

    with pytest.raises(PushConfigurationError, match="no dataset card"):
        validate_card(empty)


def test_cli_builds_the_export_and_declines_to_publish(tmp_path: Path) -> None:
    manifest, split = _prepared(tmp_path)
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(manifest.model_dump(mode="json")), encoding="utf-8")
    split_path = tmp_path / "splits" / "leave-one-out.json"
    write_evaluation_split(split_path, split)

    arguments = [
        "data",
        "export-hf",
        *[
            f"-o{value}"
            for value in (
                f"paths.manifest_path={manifest_path}",
                f"paths.evaluation_split_path={split_path}",
                f"paths.embeddings_dir={tmp_path / 'embeddings'}",
                f"paths.huggingface_export_dir={tmp_path / 'export'}",
                f"paths.discovery_dir={tmp_path / 'discovery'}",
                f"backbone.model_id={FAKE_BACKBONE.model_id}",
                f"backbone.revision={FAKE_BACKBONE.revision}",
                f"backbone.preprocessing_id={FAKE_BACKBONE.preprocessing_id}",
                "export.backbones=[dinov2]",
                "logging.json_output=false",
            )
        ],
    ]

    result = runner.invoke(app, arguments)

    assert result.exit_code == 0, result.output
    assert "images=9 classes=3 folds=9 shards=1" in result.output
    assert "modified=9 unmodified=0" in result.output
    # Publishing is never implied by building.
    assert "not published" in result.output
    assert "--push" in result.output


def test_cli_reports_a_missing_manifest(tmp_path: Path) -> None:
    result = runner.invoke(
        app,
        [
            "data",
            "export-hf",
            f"-opaths.manifest_path={tmp_path / 'absent.json'}",
            f"-opaths.huggingface_export_dir={tmp_path / 'export'}",
            "-ologging.json_output=false",
        ],
    )

    assert result.exit_code == 2
    assert "Hugging Face export error" in result.output

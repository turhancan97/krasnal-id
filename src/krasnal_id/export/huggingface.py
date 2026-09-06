"""Building the Hugging Face dataset directory.

Reads the manifest, the split and the embedding cache; writes parquet, the card,
the licence inventory, the credit ledger and a provenance receipt. Nothing here
writes to `dwarfs.json`, `fetched-images.json`, the manifest or the split, so
running an export invalidates no published result.

`AGENTS.md` section 5.10's separation holds: extraction is the only thing that
writes vectors, so this reads the cache and refuses to compute a missing one.
"""

import hashlib
import json
from dataclasses import dataclass
from datetime import UTC, datetime
from importlib import metadata
from pathlib import Path

from krasnal_id.atomic import atomic_text
from krasnal_id.config import AppConfig, BackboneConfig
from krasnal_id.data_pipeline.build_manifest import canonical_json_sha256
from krasnal_id.embeddings.store import EmbeddingStoreError, load_embedding_matrix
from krasnal_id.export import card as card_module
from krasnal_id.export import schema as schema_module
from krasnal_id.export.rows import ImageRow, RowError, build_image_rows, class_names
from krasnal_id.export.tables import (
    TableWriteError,
    class_row_payloads,
    embedding_row_payloads,
    fold_row_payloads,
    image_row_payload,
    write_image_shards,
    write_table,
)
from krasnal_id.models import DatasetManifest, EvaluationSplit


class HuggingFaceExportError(ValueError):
    """Raised when the export cannot be built."""


@dataclass(frozen=True, slots=True)
class ExportPaths:
    """Where each artifact of the export lives."""

    root: Path

    @property
    def card(self) -> Path:
        """The dataset card."""
        return self.root / "README.md"

    @property
    def licenses(self) -> Path:
        """The licence inventory."""
        return self.root / "LICENSES.md"

    @property
    def attribution(self) -> Path:
        """The credits, grouped by photographer."""
        return self.root / "ATTRIBUTION.md"

    @property
    def credits(self) -> Path:
        """The machine-readable credit ledger."""
        return self.root / "credits.csv"

    @property
    def provenance(self) -> Path:
        """The provenance receipt."""
        return self.root / "provenance.json"

    def config_dir(self, config_name: str) -> Path:
        """The directory holding one config's data files."""
        return self.root / ("data" if config_name == schema_module.DEFAULT_CONFIG else config_name)


@dataclass(frozen=True, slots=True)
class ExportResult:
    """What one export run produced."""

    paths: ExportPaths
    images: int
    classes: int
    folds: int
    shards: int
    modified: int
    unmodified: int
    backbones: tuple[str, ...]
    manifest_sha256: str


def _package_version() -> str:
    """Return the installed package version rather than a module constant."""
    try:
        return metadata.version("krasnal-id")
    except metadata.PackageNotFoundError:  # pragma: no cover - always installed under uv
        return "unknown"


def _digest_file(path: Path) -> tuple[int, str]:
    """Return one emitted file's size and digest, for the receipt."""
    payload = path.read_bytes()
    return len(payload), hashlib.sha256(payload).hexdigest()


def _total_image_bytes(rows: tuple[ImageRow, ...]) -> int:
    """Project the export size from file sizes, without reading any image.

    JPEG is already entropy-coded, so no compression gain is assumed. Guessing
    one would under-estimate the shard size, which is the failure that produces a
    900 MB shard.
    """
    return sum(row.record.local_path.stat().st_size for row in rows)


def build_export(
    config: AppConfig,
    manifest: DatasetManifest,
    split: EvaluationSplit,
    *,
    backbones: tuple[BackboneConfig, ...],
    license_templates: dict[str, tuple[str, ...]] | None = None,
    generated_at: datetime | None = None,
    with_embeddings: bool = True,
) -> ExportResult:
    """Build the whole export directory."""
    export = config.export
    paths = ExportPaths(root=config.paths.huggingface_export_dir)
    stamp = generated_at or datetime.now(UTC)

    manifest_sha256 = canonical_json_sha256(manifest.model_dump(mode="json"))
    if split.manifest_sha256 != manifest_sha256:
        raise HuggingFaceExportError(
            f"the evaluation split was built for manifest {split.manifest_sha256[:12]} but the "
            f"manifest hashes to {manifest_sha256[:12]}; rebuild it with "
            "krasnal-id data build-split"
        )

    try:
        rows = build_image_rows(manifest, license_templates)
    except RowError as error:
        raise HuggingFaceExportError(str(error)) from error

    names = class_names(manifest)
    image_columns = schema_module.image_columns(names)
    metadata_columns = schema_module.metadata_columns(names)
    plan = schema_module.shard_plan(_total_image_bytes(rows), len(rows), export.shard_target_bytes)

    configs: list[tuple[str, str, str]] = []
    features: dict[str, list[dict[str, object]]] = {}
    emitted: list[Path] = []

    try:
        emitted.extend(
            write_image_shards(
                paths.config_dir(schema_module.DEFAULT_CONFIG),
                rows,
                image_columns,
                plan,
                split=schema_module.REFERENCE_SPLIT,
                row_group_rows=export.row_group_rows,
                with_image=True,
            )
        )
        configs.append(
            (
                schema_module.DEFAULT_CONFIG,
                schema_module.REFERENCE_SPLIT,
                f"data/{schema_module.REFERENCE_SPLIT}-*.parquet",
            )
        )
        features[schema_module.DEFAULT_CONFIG] = schema_module.hf_features_yaml(image_columns)

        for name, columns, payloads, split_name in (
            (
                schema_module.METADATA_CONFIG,
                metadata_columns,
                [image_row_payload(row, with_image=False) for row in rows],
                schema_module.REFERENCE_SPLIT,
            ),
            (
                schema_module.CLASSES_CONFIG,
                schema_module.class_columns(names),
                class_row_payloads(manifest, rows),
                schema_module.REFERENCE_SPLIT,
            ),
            (
                schema_module.FOLDS_CONFIG,
                schema_module.fold_columns(names),
                fold_row_payloads(split, rows),
                schema_module.FOLDS_SPLIT,
            ),
        ):
            filename = schema_module.shard_filename(split_name, 0, 1)
            path = paths.config_dir(name) / filename
            write_table(path, payloads, columns, row_group_rows=export.row_group_rows)
            emitted.append(path)
            configs.append((name, split_name, f"{name}/{filename}"))
            features[name] = schema_module.hf_features_yaml(columns)

        backbone_facts: list[tuple[str, str, str, int]] = []
        if with_embeddings:
            for backbone in backbones:
                matrix = load_embedding_matrix(manifest, backbone, config.paths.embeddings_dir)
                dimensions = int(matrix.vectors.shape[1])
                columns = schema_module.embedding_columns(dimensions)
                name = schema_module.embeddings_config_name(backbone.name)
                filename = schema_module.shard_filename(schema_module.REFERENCE_SPLIT, 0, 1)
                path = paths.config_dir(name) / filename
                write_table(
                    path,
                    list(
                        embedding_row_payloads(matrix.image_ids, matrix.dwarf_ids, matrix.vectors)
                    ),
                    columns,
                    row_group_rows=export.row_group_rows,
                    extra_metadata={
                        "backbone.name": backbone.name,
                        "backbone.model_id": backbone.model_id,
                        "backbone.revision": backbone.revision,
                        "backbone.preprocessing_id": backbone.preprocessing_id,
                    },
                )
                emitted.append(path)
                configs.append((name, schema_module.REFERENCE_SPLIT, f"{name}/{filename}"))
                features[name] = schema_module.hf_features_yaml(columns)
                backbone_facts.append(
                    (backbone.name, backbone.model_id, backbone.revision, dimensions)
                )

        facts = card_module.measure(rows, len(names), manifest_sha256, backbone_facts)

        for path, text in (
            (paths.credits, card_module.render_credits_csv(rows)),
            (paths.licenses, card_module.render_licenses(facts)),
            (paths.attribution, card_module.render_attribution(rows, facts)),
            (
                paths.card,
                card_module.render_card(
                    facts,
                    configs,
                    features,
                    repo_id=export.repo_id,
                    license_name=export.license_name,
                    license_link=export.license_link,
                ),
            ),
        ):
            with atomic_text(path, HuggingFaceExportError) as handle:
                handle.write(text)
            emitted.append(path)

        receipt = {
            "schema_version": schema_module.SCHEMA_VERSION,
            "generated_at": stamp.isoformat(),
            "krasnal_id_version": _package_version(),
            "repo_id": export.repo_id,
            "manifest": {
                "schema_version": manifest.schema_version,
                "manifest_sha256": manifest_sha256,
                "source_query_sha256": manifest.source_query_sha256,
                "staging_sha256": manifest.staging_sha256,
                "image_review_sha256": manifest.image_review_sha256,
                "generated_at": manifest.generated_at.isoformat(),
                "minimum_images_per_dwarf": manifest.minimum_images_per_dwarf,
            },
            "split": {
                "schema_version": split.schema_version,
                "strategy": split.strategy,
                "manifest_sha256": split.manifest_sha256,
                "generated_at": split.generated_at.isoformat(),
            },
            "backbones": [
                {
                    "name": name,
                    "model_id": model_id,
                    "revision": revision,
                    "dimensions": dimensions,
                }
                for name, model_id, revision, dimensions in backbone_facts
            ],
            "counts": {
                "dwarfs": len(names),
                "images": len(rows),
                "folds": len(split.folds),
                "shards": plan.shard_count,
                "modified": facts.modified,
                "unmodified": facts.unmodified,
                "photographers": facts.photographers,
            },
            "sharding": {
                "image_bytes_total": _total_image_bytes(rows),
                "shard_target_bytes": export.shard_target_bytes,
                "rows_per_shard": plan.rows_per_shard,
                "row_group_rows": export.row_group_rows,
                "compression": "zstd",
            },
            "files": [
                {
                    "path": path.relative_to(paths.root).as_posix(),
                    "size_bytes": size,
                    "sha256": digest,
                }
                for path, (size, digest) in sorted(
                    ((path, _digest_file(path)) for path in emitted),
                    key=lambda item: item[0].as_posix(),
                )
            ],
        }
        with atomic_text(paths.provenance, HuggingFaceExportError) as handle:
            json.dump(receipt, handle, ensure_ascii=False, indent=2, sort_keys=True)
            handle.write("\n")
    except (EmbeddingStoreError, TableWriteError) as error:
        raise HuggingFaceExportError(str(error)) from error

    return ExportResult(
        paths=paths,
        images=len(rows),
        classes=len(names),
        folds=len(split.folds),
        shards=plan.shard_count,
        modified=facts.modified,
        unmodified=facts.unmodified,
        backbones=tuple(name for name, _, _, _ in backbone_facts),
        manifest_sha256=manifest_sha256,
    )

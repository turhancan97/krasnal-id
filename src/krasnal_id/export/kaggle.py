"""Building the Kaggle dataset directory.

A separate writer rather than a second target for the Hugging Face one, because
the two platforms want different shapes and pretending otherwise would publish
something awkward on both. The Hub wants parquet with an image column that is a
`{bytes, path}` struct plus a declared feature type; Kaggle's data explorer
previews CSV and serves files, and its users open a dataset expecting a folder of
images beside a table describing them. Shipping the Hub's parquet here would give
a Kaggle user a column their tools cannot read.

What is *not* duplicated is everything the rights obligation rests on.
`build_image_rows` derives the same licence URL, SPDX identifier, per-file
modification flag and credit line, and `LICENSES.md`, `ATTRIBUTION.md` and
`credits.csv` are the same generated artifacts the Hub copy publishes. §5.11's
obligation is per file, so the two exports must not be able to disagree about a
photographer.

Nothing here writes to the staging chain, so running an export invalidates no
published result — the same guarantee `huggingface.py` gives.
"""

import csv
import hashlib
import io
import json
import shutil
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from importlib import metadata
from pathlib import Path

import numpy as np

from krasnal_id.atomic import atomic_bytes, atomic_text
from krasnal_id.config import AppConfig, BackboneConfig
from krasnal_id.data_pipeline.build_manifest import canonical_json_sha256
from krasnal_id.embeddings.store import EmbeddingStoreError, load_embedding_matrix
from krasnal_id.export import card as card_module
from krasnal_id.export.rows import ImageRow, RowError, build_image_rows, class_names
from krasnal_id.models import DatasetManifest, EvaluationSplit

# Kaggle validates these on upload and rejects the dataset rather than trimming,
# so they are checked here instead — a failure after 676 MB has been sent is a
# bad way to learn that a title is one character too long.
TITLE_LIMITS = (6, 50)
SUBTITLE_LIMITS = (20, 80)
SLUG_LIMITS = (3, 50)

TITLE = "Wrocław Dwarves: Fine-Grained Instance Retrieval"
SUBTITLE = "1,691 attributed photographs of 306 near-identical bronze statues"

# Kaggle accepts exactly one licence for a dataset, from this list. This corpus
# has ten across four families, so no single name is true of it: tagging it
# `CC-BY-SA-4.0` would assert 4.0 over the 3.0, 2.5 and 2.0 files and assert a
# licence at all over the 52 public-domain and 7 CC0 ones. `other` is Kaggle's
# "Other (specified in description)", which is the honest option and the one
# §5.11 anticipated. The per-file terms travel in images.csv and LICENSES.md.
LICENSE_NAME = "other"
ALLOWED_LICENSES = frozenset(
    {
        "CC0-1.0",
        "CC-BY-SA-3.0",
        "CC-BY-SA-4.0",
        "CC-BY-NC-SA-4.0",
        "GPL-2.0",
        "ODbL-1.0",
        "DbCL-1.0",
        "copyright-authors",
        "other",
        "unknown",
    }
)

# Kaggle matches keywords against its own tag vocabulary and ignores anything it
# does not recognise, so these are deliberately generic. Subject-specific tags
# are better added in the web UI, where the existing vocabulary is searchable.
KEYWORDS = ("computer vision", "image", "art", "europe")

# The columns a user needs to *use* the dataset: where the file is, what it
# shows, and the attribution they must carry when they redistribute it. The full
# rights ledger is credits.csv; author, licence and source appear in both because
# a user who only opens one table still needs them.
IMAGE_COLUMNS = (
    "file_path",
    "image_id",
    "dwarf_id",
    "dwarf_name",
    "label",
    "width",
    "height",
    "author",
    "license",
    "license_url",
    "source_url",
    "modified",
)

CLASS_COLUMNS = (
    "label",
    "dwarf_id",
    "display_name",
    "images",
    "wikidata_url",
    "commons_category",
    "latitude",
    "longitude",
    "coordinate_source",
)

FOLD_COLUMNS = (
    "fold_index",
    "query_image_id",
    "query_dwarf_id",
    "query_label",
    "reference_count",
)


class KaggleExportError(ValueError):
    """Raised when the export cannot be built."""


@dataclass(frozen=True, slots=True)
class KaggleExportPaths:
    """Where each artifact of the export lives."""

    root: Path

    @property
    def metadata(self) -> Path:
        """Kaggle's own dataset descriptor."""
        return self.root / "dataset-metadata.json"

    @property
    def images_dir(self) -> Path:
        """The photographs, one directory per dwarf."""
        return self.root / "images"

    @property
    def images(self) -> Path:
        """The table describing every photograph."""
        return self.root / "images.csv"

    @property
    def classes(self) -> Path:
        """The label vocabulary."""
        return self.root / "classes.csv"

    @property
    def folds(self) -> Path:
        """The leave-one-out evaluation protocol."""
        return self.root / "folds.csv"

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

    def embeddings(self, backbone: str) -> Path:
        """One backbone's cached vectors."""
        return self.root / f"embeddings_{backbone}.npy"


@dataclass(frozen=True, slots=True)
class KaggleExportResult:
    """What one export run produced."""

    paths: KaggleExportPaths
    dataset_id: str
    images: int
    classes: int
    folds: int
    modified: int
    unmodified: int
    backbones: tuple[str, ...]
    image_bytes: int
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


def validate_dataset_id(dataset_id: str) -> tuple[str, str]:
    """Split and check a Kaggle dataset id, before anything is written."""
    owner, separator, slug = dataset_id.partition("/")
    if not separator or not owner or not slug:
        raise KaggleExportError(
            f"'{dataset_id}' is not a Kaggle dataset id; expected username/dataset-slug"
        )
    low, high = SLUG_LIMITS
    if not low <= len(slug) <= high:
        raise KaggleExportError(
            f"the dataset slug '{slug}' is {len(slug)} characters; Kaggle requires {low}-{high}"
        )
    if not all(character.isalnum() or character == "-" for character in slug):
        raise KaggleExportError(
            f"the dataset slug '{slug}' must use letters, digits and hyphens only"
        )
    return owner, slug


def _check_text_limits(title: str, subtitle: str) -> None:
    """Refuse a title or subtitle Kaggle would reject on upload."""
    for label, value, (low, high) in (
        ("title", title, TITLE_LIMITS),
        ("subtitle", subtitle, SUBTITLE_LIMITS),
    ):
        if not low <= len(value) <= high:
            raise KaggleExportError(
                f"the {label} is {len(value)} characters; Kaggle requires {low}-{high}: {value!r}"
            )


def _csv(columns: Sequence[str], records: Sequence[dict[str, object]]) -> str:
    """Render one table, with the header Kaggle's preview reads."""
    buffer = io.StringIO()
    writer = csv.DictWriter(buffer, fieldnames=list(columns), lineterminator="\n")
    writer.writeheader()
    writer.writerows(records)
    return buffer.getvalue()


def image_payloads(rows: Sequence[ImageRow]) -> list[dict[str, object]]:
    """Describe every photograph, in the order the images table lists them."""
    return [
        {
            "file_path": f"images/{row.dwarf.dwarf_id}/{row.record.local_path.name}",
            "image_id": row.record.image_id,
            "dwarf_id": row.dwarf.dwarf_id,
            "dwarf_name": row.dwarf.display_name,
            "label": row.label,
            "width": row.record.width,
            "height": row.record.height,
            "author": row.record.author,
            "license": row.record.license,
            "license_url": row.license_url,
            "source_url": str(row.record.source_url),
            "modified": str(row.modified).lower(),
        }
        for row in rows
    ]


def class_payloads(manifest: DatasetManifest, rows: Sequence[ImageRow]) -> list[dict[str, object]]:
    """Describe every dwarf, with the label the embeddings are ordered by."""
    labels = {name: index for index, name in enumerate(class_names(manifest))}
    counts: dict[str, int] = {}
    for row in rows:
        counts[row.dwarf.dwarf_id] = counts.get(row.dwarf.dwarf_id, 0) + 1
    return [
        {
            "label": labels[dwarf.dwarf_id],
            "dwarf_id": dwarf.dwarf_id,
            "display_name": dwarf.display_name,
            "images": counts.get(dwarf.dwarf_id, 0),
            "wikidata_url": str(dwarf.wikidata_url) if dwarf.wikidata_url else "",
            "commons_category": dwarf.commons_category or "",
            "latitude": dwarf.coordinates.latitude if dwarf.coordinates else "",
            "longitude": dwarf.coordinates.longitude if dwarf.coordinates else "",
            "coordinate_source": str(dwarf.coordinate_source) if dwarf.coordinate_source else "",
        }
        for dwarf in sorted(manifest.dwarfs, key=lambda item: labels[item.dwarf_id])
    ]


def fold_payloads(split: EvaluationSplit, rows: Sequence[ImageRow]) -> list[dict[str, object]]:
    """Describe every fold, and verify the rule that lets it stay compact.

    Written out, each fold's reference set is 1,690 image IDs, and 1,691 of those
    is a 30 MB CSV nobody can read. Leave-one-out makes it unnecessary: the
    references are every image except the query. That is a property of the
    current split rather than a promise, so it is *checked* here and the export
    refuses rather than silently publishing a table whose stated rule is false.
    """
    labels = {row.record.image_id: row.label for row in rows}
    every = {row.record.image_id for row in rows}
    payloads: list[dict[str, object]] = []
    for index, fold in enumerate(sorted(split.folds, key=lambda item: item.query_image_id)):
        if fold.query_image_id not in labels:
            raise KaggleExportError(
                f"fold {fold.query_image_id} queries an image that is not in the manifest"
            )
        if set(fold.reference_image_ids) != every - {fold.query_image_id}:
            raise KaggleExportError(
                f"fold {fold.query_image_id} is not leave-one-out over the whole manifest, so "
                "folds.csv cannot describe it by rule; the reference sets would have to be "
                "written out in full"
            )
        payloads.append(
            {
                "fold_index": index,
                "query_image_id": fold.query_image_id,
                "query_dwarf_id": fold.query_dwarf_id,
                "query_label": labels[fold.query_image_id],
                "reference_count": len(fold.reference_image_ids),
            }
        )
    return payloads


def render_description(facts: card_module.CardFacts, dataset_id: str, images: bool) -> str:
    """Render the description Kaggle shows, including the per-file rights terms.

    Kaggle has one licence field and this corpus has ten licences, so the field
    says `other` and the specifics have to be *here* — that is what "Other
    (specified in description)" means, and leaving it vague would make the
    licence tag meaningless rather than merely coarse.
    """
    families: dict[str, int] = {}
    for name, _, _, count in facts.licenses:
        family = (
            "CC BY-SA"
            if name.startswith("CC BY-SA")
            else "CC BY"
            if name.startswith("CC BY")
            else "CC0"
            if name == "CC0"
            else "Public domain"
        )
        families[family] = families.get(family, 0) + count

    lines = [
        f"Which of Wroclaw's {facts.classes} bronze dwarf statues does a photograph show? Every "
        "statue is the same semantic category, so this is instance recognition rather than "
        "classification: what separates them is a hat, a tool, a pose.",
        "",
        f"{facts.images:,} photographs of {facts.classes} individual statues, sourced from "
        "Wikimedia Commons, every one carrying its photographer, licence and source URL. Frozen "
        "DINOv2 embeddings reach 93.1% top-1 across the whole pool with no fine-tuning; CLIP "
        "reaches 82.9%.",
        "",
        "## Files",
        "",
    ]
    if images:
        lines.append(f"- `images/` — the {facts.images:,} photographs, one directory per statue.")
    lines += [
        "- `images.csv` — one row per photograph: file path, class, dimensions, and the "
        "photographer, licence and source you must carry when redistributing it.",
        f"- `classes.csv` — the {facts.classes} statues, their labels and their coordinates "
        f"({facts.placed} placed, {facts.derived_positions} of those derived from the "
        "photographs' own camera positions).",
        "- `folds.csv` — the leave-one-out evaluation protocol. Every fold's reference set is "
        "every image except the query, so only the query is listed.",
        "- `embeddings_<backbone>.npy` — cached vectors, row-aligned with `images.csv`.",
        "- `credits.csv`, `ATTRIBUTION.md`, `LICENSES.md` — the full rights ledger.",
        "- `provenance.json` — the manifest hash and a digest of every file here.",
        "",
        "## Licensing",
        "",
        "**This is a collection of separately licensed photographs, not a single relicensed "
        "work.** Kaggle allows one licence field and this corpus has several, so the field says "
        "*Other* and the real terms are per file, in the `license` and `license_url` columns of "
        "`images.csv`:",
        "",
    ]
    lines += [
        f"- {family}: {count:,} images ({count / facts.images:.1%})"
        for family, count in sorted(families.items(), key=lambda item: -item[1])
    ]
    lines += [
        "",
        f"{facts.modified:,} of the {facts.images:,} files were downscaled from the Commons "
        f"original and {facts.unmodified:,} are byte-identical to it; `modified` says which, per "
        "file, because the ShareAlike terms require a change to be indicated and asserting one "
        "over an exact copy would be false.",
        "",
        "**The sculptures themselves are copyrighted.** They are contemporary works by living "
        "sculptors. Commons hosts photographs of them under Polish freedom of panorama; the "
        "Creative Commons licences here are granted by the photographers and cover the "
        "photographs only. They grant no rights in the sculptures depicted. A removal request "
        "for any photograph is honoured through the project's `data/image-review.json`: one "
        "exclusion entry, a rebuild, a new version of this dataset.",
        "",
        "## Identifiers are dataset-local",
        "",
        f"Built from manifest `{facts.manifest_sha256[:12]}`. `dwarf_id` is a Wikidata QID where "
        "one exists and a slug of the Commons category otherwise, so a renamed category changes "
        "the slug. Treat these as keys within this version, not as stable external identifiers.",
        "",
        "## Source",
        "",
        f"Code, method and full results: https://github.com/turhancan97/krasnal-id — "
        f"DOI {card_module.PROJECT_DOI}. The same corpus is published on Hugging Face as "
        "`turhancan97/wroclaw-dwarves`, in parquet with the images embedded.",
    ]
    return "\n".join(lines) + "\n"


def render_metadata(
    dataset_id: str,
    description: str,
    backbones: Sequence[str],
    *,
    with_images: bool,
) -> dict[str, object]:
    """Build Kaggle's `dataset-metadata.json`."""
    _check_text_limits(TITLE, SUBTITLE)
    validate_dataset_id(dataset_id)
    if LICENSE_NAME not in ALLOWED_LICENSES:
        raise KaggleExportError(f"'{LICENSE_NAME}' is not a licence name Kaggle accepts")

    resources: list[dict[str, object]] = [
        {
            "path": "images.csv",
            "description": (
                "One row per photograph: file path, class, dimensions, and the photographer, "
                "licence and source URL required for attribution."
            ),
        },
        {
            "path": "classes.csv",
            "description": "The statues, their integer labels, image counts and coordinates.",
        },
        {
            "path": "folds.csv",
            "description": (
                "The leave-one-out protocol. Each fold's reference set is every image except "
                "the query, so only the query is listed."
            ),
        },
        {
            "path": "credits.csv",
            "description": "The full rights ledger, one ready-to-paste credit line per file.",
        },
    ]
    resources += [
        {
            "path": f"embeddings_{name}.npy",
            "description": (
                f"Cached {name} vectors, one row per photograph in the order of images.csv."
            ),
        }
        for name in backbones
    ]
    if with_images:
        resources.insert(
            0,
            {
                "path": "images",
                "description": "The photographs, one directory per statue.",
            },
        )

    return {
        "title": TITLE,
        "subtitle": SUBTITLE,
        "id": dataset_id,
        "licenses": [{"name": LICENSE_NAME}],
        "keywords": list(KEYWORDS),
        "description": description,
        "resources": resources,
    }


def _copy_images(rows: Sequence[ImageRow], destination: Path) -> int:
    """Copy every photograph under its dwarf, returning the bytes written.

    `build_image_rows` has already verified each file against the digest the
    manifest records, so this copies bytes it knows are the right ones.
    """
    written = 0
    for row in rows:
        target = destination / row.dwarf.dwarf_id / row.record.local_path.name
        target.parent.mkdir(parents=True, exist_ok=True)
        try:
            shutil.copyfile(row.record.local_path, target)
        except OSError as error:
            raise KaggleExportError(f"could not copy {row.record.local_path}: {error}") from error
        written += target.stat().st_size
    return written


def build_kaggle_export(
    config: AppConfig,
    manifest: DatasetManifest,
    split: EvaluationSplit,
    *,
    backbones: tuple[BackboneConfig, ...],
    dataset_id: str | None = None,
    license_templates: dict[str, tuple[str, ...]] | None = None,
    generated_at: datetime | None = None,
    with_embeddings: bool = True,
    with_images: bool = True,
) -> KaggleExportResult:
    """Build the whole Kaggle export directory."""
    export = config.export
    target = dataset_id or export.kaggle_id
    validate_dataset_id(target)
    _check_text_limits(TITLE, SUBTITLE)

    paths = KaggleExportPaths(root=config.paths.kaggle_export_dir)
    stamp = generated_at or datetime.now(UTC)

    manifest_sha256 = canonical_json_sha256(manifest.model_dump(mode="json"))
    if split.manifest_sha256 != manifest_sha256:
        raise KaggleExportError(
            f"the evaluation split was built for manifest {split.manifest_sha256[:12]} but the "
            f"manifest hashes to {manifest_sha256[:12]}; rebuild it with "
            "krasnal-id data build-split"
        )

    try:
        rows = build_image_rows(manifest, license_templates)
    except RowError as error:
        raise KaggleExportError(str(error)) from error

    # The attribution columns are the whole obligation, so an empty one is
    # refused before anything is written rather than discovered by a reuser.
    for row in rows:
        for field, value in (
            ("author", row.record.author),
            ("license", row.record.license),
            ("license_url", row.license_url),
            ("source_url", str(row.record.source_url)),
        ):
            if not value.strip():
                raise KaggleExportError(
                    f"image {row.record.image_id} has no {field}; every published row must carry "
                    "one, because attribution is per file"
                )

    names = class_names(manifest)
    paths.root.mkdir(parents=True, exist_ok=True)
    emitted: list[Path] = []

    image_bytes = 0
    if with_images:
        if paths.images_dir.exists():
            shutil.rmtree(paths.images_dir)
        image_bytes = _copy_images(rows, paths.images_dir)

    tables = (
        (paths.images, IMAGE_COLUMNS, image_payloads(rows)),
        (paths.classes, CLASS_COLUMNS, class_payloads(manifest, rows)),
        (paths.folds, FOLD_COLUMNS, fold_payloads(split, rows)),
    )
    for path, columns, payloads in tables:
        with atomic_text(path, KaggleExportError) as handle:
            handle.write(_csv(columns, payloads))
        emitted.append(path)

    measured_backbones: list[tuple[str, str, str, int]] = []
    try:
        for backbone in backbones:
            matrix = load_embedding_matrix(manifest, backbone, config.paths.embeddings_dir)
            if matrix.image_ids != tuple(row.record.image_id for row in rows):
                raise KaggleExportError(
                    f"the {backbone.name} vectors are not in the order images.csv lists; "
                    "the .npy rows would not line up with the table"
                )
            measured_backbones.append(
                (backbone.name, backbone.model_id, backbone.revision, matrix.vectors.shape[1])
            )
            if with_embeddings:
                path = paths.embeddings(backbone.name)
                with atomic_bytes(path, KaggleExportError) as handle:
                    np.save(handle, np.asarray(matrix.vectors, dtype=np.float32))
                emitted.append(path)
    except EmbeddingStoreError as error:
        raise KaggleExportError(str(error)) from error

    facts = card_module.measure(rows, len(names), manifest_sha256, measured_backbones)

    description = render_description(facts, target, images=with_images)
    documents = (
        (paths.licenses, card_module.render_licenses(facts)),
        (paths.attribution, card_module.render_attribution(rows, facts)),
        (paths.credits, card_module.render_credits_csv(rows)),
    )
    for path, text in documents:
        with atomic_text(path, KaggleExportError) as handle:
            handle.write(text)
        emitted.append(path)

    metadata_payload = render_metadata(
        target,
        description,
        [name for name, _, _, _ in measured_backbones] if with_embeddings else [],
        with_images=with_images,
    )
    with atomic_text(paths.metadata, KaggleExportError) as handle:
        json.dump(metadata_payload, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    emitted.append(paths.metadata)

    receipt = {
        "generated_at": stamp.isoformat(),
        "generator": f"krasnal-id {_package_version()}",
        "platform": "kaggle",
        "dataset_id": target,
        "manifest_sha256": manifest_sha256,
        "staging_sha256": manifest.staging_sha256,
        "counts": {
            "images": len(rows),
            "classes": len(names),
            "folds": len(split.folds),
            "modified": facts.modified,
            "unmodified": facts.unmodified,
            "photographers": facts.photographers,
        },
        "images": {"included": with_images, "bytes": image_bytes},
        "backbones": [
            {"name": name, "model_id": model, "revision": revision, "dimensions": dimensions}
            for name, model, revision, dimensions in measured_backbones
        ],
        # The photographs are digested by the manifest already and there are
        # 1,691 of them; listing every one here would make the receipt larger
        # than the table it describes.
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
    with atomic_text(paths.provenance, KaggleExportError) as handle:
        json.dump(receipt, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")

    return KaggleExportResult(
        paths=paths,
        dataset_id=target,
        images=len(rows),
        classes=len(names),
        folds=len(split.folds),
        modified=facts.modified,
        unmodified=facts.unmodified,
        backbones=tuple(name for name, _, _, _ in measured_backbones),
        image_bytes=image_bytes,
        manifest_sha256=manifest_sha256,
    )

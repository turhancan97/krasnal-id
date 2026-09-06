"""Writing the described tables to parquet, one row group at a time.

Each writer takes rows and a column description and emits a file whose arrow
schema is exactly `schema.arrow_schema(columns)` — including the `huggingface`
key in the file's own metadata, so a bare `load_dataset("parquet", ...)` and the
Hub's viewer both decode the Image column without consulting the card.

Required columns are checked before anything is written. The attribution and
licence columns are among them, which is how the section 5.11 obligation becomes
something the writer refuses to violate rather than something the card promises.
"""

import json
from collections.abc import Iterator, Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from krasnal_id.atomic import staged_path
from krasnal_id.export.rows import ImageRow
from krasnal_id.export.schema import (
    Column,
    ShardPlan,
    arrow_schema,
    hf_features,
    required_columns,
    shard_filename,
)
from krasnal_id.models import DatasetManifest, EvaluationSplit

Row = Mapping[str, Any]


class TableWriteError(ValueError):
    """Raised when a table cannot be assembled or written."""


def _validate_required(rows: Sequence[Row], columns: tuple[Column, ...]) -> None:
    """Refuse to write a table with a hole in a column that must not have one."""
    for name in required_columns(columns):
        for index, row in enumerate(rows):
            value = row.get(name)
            if value is None or (isinstance(value, str) and not value and name != "modification"):
                raise TableWriteError(
                    f"row {index} has no {name}; every published row must carry it"
                )


def table_metadata(
    columns: tuple[Column, ...], extra: Mapping[str, str] | None = None
) -> dict[bytes, bytes]:
    """Build the parquet key-value metadata, features included.

    The `huggingface` key is what `datasets` writes itself and what makes the
    Image column decode when the file is read directly rather than through the
    card, so it is not optional decoration.
    """
    metadata = {
        b"huggingface": json.dumps(
            {"info": {"features": hf_features(columns)}}, sort_keys=True
        ).encode("utf-8")
    }
    for key, value in (extra or {}).items():
        metadata[f"krasnal_id.{key}".encode()] = value.encode("utf-8")
    return metadata


def write_table(
    path: Path,
    rows: Sequence[Row],
    columns: tuple[Column, ...],
    *,
    row_group_rows: int,
    extra_metadata: Mapping[str, str] | None = None,
) -> int:
    """Write one described table atomically, returning the rows written."""
    _validate_required(rows, columns)
    schema = arrow_schema(columns).with_metadata(table_metadata(columns, extra_metadata))

    with (
        staged_path(path, TableWriteError) as staged,
        pq.ParquetWriter(staged, schema, compression="zstd", compression_level=3) as writer,
    ):
        for start in range(0, len(rows), row_group_rows):
            batch = rows[start : start + row_group_rows]
            writer.write_table(pa.Table.from_pylist([dict(row) for row in batch], schema=schema))
    return len(rows)


def image_row_payload(row: ImageRow, *, with_image: bool) -> dict[str, Any]:
    """Flatten one reference photograph into its published columns."""
    record, dwarf = row.record, row.dwarf
    payload: dict[str, Any] = {
        "image_id": record.image_id,
        "dwarf_id": record.dwarf_id,
        "label": row.label,
        "display_name": dwarf.display_name,
        "commons_category": dwarf.commons_category,
        "wikidata_url": str(dwarf.wikidata_url) if dwarf.wikidata_url else None,
        "author": record.author,
        "license": record.license,
        "license_url": row.license_url,
        "license_spdx": row.license_spdx,
        "license_template": row.license_template,
        "source_url": str(record.source_url),
        "commons_file": row.commons_file,
        "modified": row.modified,
        "modification": row.modification,
        "attribution_text": row.attribution_text,
        "sha256": row.stored_sha256,
        "commons_sha1": record.commons_sha1,
        "width": record.width,
        "height": record.height,
        "acquired_at": record.acquired_at,
        "source_revision_at": record.source_revision_at,
        "commons_page_id": record.commons_page_id,
        "image_latitude": record.coordinates.latitude if record.coordinates else None,
        "image_longitude": record.coordinates.longitude if record.coordinates else None,
        "dwarf_latitude": dwarf.coordinates.latitude if dwarf.coordinates else None,
        "dwarf_longitude": dwarf.coordinates.longitude if dwarf.coordinates else None,
        "dwarf_coordinate_source": (
            str(dwarf.coordinate_source) if dwarf.coordinate_source else None
        ),
    }
    if with_image:
        payload["image"] = {
            "bytes": record.local_path.read_bytes(),
            "path": row.image_path,
        }
    return payload


def write_image_shards(
    directory: Path,
    rows: tuple[ImageRow, ...],
    columns: tuple[Column, ...],
    plan: ShardPlan,
    *,
    split: str,
    row_group_rows: int,
    with_image: bool,
) -> tuple[Path, ...]:
    """Write the reference table across its planned shards.

    Image bytes are read a row group at a time and released, so peak memory is
    the row group rather than the corpus.
    """
    written: list[Path] = []
    for index, (start, stop) in enumerate(plan.ranges):
        path = directory / shard_filename(split, index, plan.shard_count)
        payloads = [image_row_payload(row, with_image=with_image) for row in rows[start:stop]]
        write_table(path, payloads, columns, row_group_rows=row_group_rows)
        written.append(path)
    return tuple(written)


def class_row_payloads(
    manifest: DatasetManifest,
    rows: tuple[ImageRow, ...],
) -> list[dict[str, Any]]:
    """Describe every statue once, with how many photographs it has."""
    labels = {row.dwarf.dwarf_id: row.label for row in rows}
    counts: dict[str, int] = {}
    for row in rows:
        counts[row.dwarf.dwarf_id] = counts.get(row.dwarf.dwarf_id, 0) + 1

    payloads: list[dict[str, Any]] = []
    for dwarf in sorted(manifest.dwarfs, key=lambda item: item.dwarf_id):
        payloads.append(
            {
                "dwarf_id": dwarf.dwarf_id,
                "label": labels.get(dwarf.dwarf_id, 0),
                "display_name": dwarf.display_name,
                "commons_category": dwarf.commons_category,
                "wikidata_url": str(dwarf.wikidata_url) if dwarf.wikidata_url else None,
                "latitude": dwarf.coordinates.latitude if dwarf.coordinates else None,
                "longitude": dwarf.coordinates.longitude if dwarf.coordinates else None,
                "coordinate_source": (
                    str(dwarf.coordinate_source) if dwarf.coordinate_source else None
                ),
                "image_count": counts.get(dwarf.dwarf_id, 0),
            }
        )
    return payloads


def fold_row_payloads(
    split: EvaluationSplit,
    rows: tuple[ImageRow, ...],
) -> list[dict[str, Any]]:
    """Describe every leave-one-out fold, in query order."""
    labels = {row.record.image_id: row.label for row in rows}
    payloads: list[dict[str, Any]] = []
    for index, fold in enumerate(sorted(split.folds, key=lambda item: item.query_image_id)):
        if fold.query_image_id not in labels:
            raise TableWriteError(
                f"fold {fold.query_image_id} queries an image that is not in the manifest"
            )
        payloads.append(
            {
                "fold_index": index,
                "query_image_id": fold.query_image_id,
                "query_dwarf_id": fold.query_dwarf_id,
                "query_label": labels[fold.query_image_id],
                "reference_image_ids": list(fold.reference_image_ids),
                "reference_count": len(fold.reference_image_ids),
            }
        )
    return payloads


def embedding_row_payloads(
    image_ids: Sequence[str],
    dwarf_ids: Sequence[str],
    vectors: np.ndarray[Any, Any],
) -> Iterator[dict[str, Any]]:
    """Describe one backbone's vectors, in the matrix's own row order."""
    if not (len(image_ids) == len(dwarf_ids) == vectors.shape[0]):
        raise TableWriteError(
            f"misaligned embeddings: {len(image_ids)} ids against {vectors.shape[0]} rows"
        )
    for image_id, dwarf_id, vector in zip(image_ids, dwarf_ids, vectors, strict=True):
        yield {
            "image_id": image_id,
            "dwarf_id": dwarf_id,
            "embedding": [float(value) for value in vector],
        }

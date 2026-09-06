"""Arrow schemas, Hugging Face feature declarations, and the shard plan.

Pure: no filesystem, no network, no configuration loading. Everything here is a
function of the row counts and the class list, which is what makes the schema
directly testable and the sharding arithmetic checkable without writing a file.

The features dict and the arrow schema are derived from **one** description per
config, because a hand-written parquet Image column only decodes if the two agree
— the struct in the file and the `dtype: image` in the card. Deriving both from
the same table is what stops them drifting apart.
"""

import math
from dataclasses import dataclass
from typing import Final

import pyarrow as pa

SCHEMA_VERSION: Final = "1.0"

DEFAULT_CONFIG: Final = "default"
METADATA_CONFIG: Final = "metadata"
CLASSES_CONFIG: Final = "classes"
FOLDS_CONFIG: Final = "leave_one_out"

# The photographs are references to retrieve against, not training data. Naming
# the split `train` would invite exactly the fine-tuning this benchmark warns
# against, on a median of four images per class.
REFERENCE_SPLIT: Final = "reference"
FOLDS_SPLIT: Final = "test"

# The Hugging Face Image feature at rest: raw bytes plus a path that survives
# extraction, so a JPEG on someone's disk is still traceable to its Commons page.
IMAGE_TYPE: Final = pa.struct([pa.field("bytes", pa.binary()), pa.field("path", pa.string())])
TIMESTAMP_TYPE: Final = pa.timestamp("us", tz="UTC")


def embeddings_config_name(backbone_name: str) -> str:
    """Return the config name carrying one backbone's vectors."""
    return f"embeddings_{backbone_name}"


@dataclass(frozen=True, slots=True)
class Column:
    """One column, described once for both arrow and Hugging Face.

    `required` is a build-time contract, not an arrow flag. The `datasets`
    library models every field as nullable and has no way to express otherwise,
    so a non-nullable arrow field would not match the features it derives — and
    the features are what make the Image column decode. Nullability therefore
    lives here as something the writer asserts before writing, and the tests
    assert after reading, rather than as a schema constraint that would put the
    published file out of step with the library that reads it.
    """

    name: str
    arrow_type: pa.DataType
    feature: object
    required: bool = True

    def field(self) -> pa.Field:
        """Return the arrow field, nullable to match `datasets`' own features."""
        return pa.field(self.name, self.arrow_type, nullable=True)


def _string(name: str, *, nullable: bool = False) -> Column:
    return Column(name, pa.string(), {"dtype": "string", "_type": "Value"}, not nullable)


def _int(name: str, bits: int = 64, *, nullable: bool = False) -> Column:
    arrow = pa.int32() if bits == 32 else pa.int64()
    return Column(name, arrow, {"dtype": f"int{bits}", "_type": "Value"}, not nullable)


def _float(name: str, *, nullable: bool = False) -> Column:
    return Column(name, pa.float64(), {"dtype": "float64", "_type": "Value"}, not nullable)


def _bool(name: str) -> Column:
    return Column(name, pa.bool_(), {"dtype": "bool", "_type": "Value"})


def _timestamp(name: str, *, nullable: bool = False) -> Column:
    return Column(
        name,
        TIMESTAMP_TYPE,
        {"dtype": "timestamp[us, tz=UTC]", "_type": "Value"},
        not nullable,
    )


def _rights_columns() -> tuple[Column, ...]:
    """The columns that discharge the attribution and licence obligations.

    Grouped together deliberately: `AGENTS.md` section 5.11 makes these per-file
    duties, and a reader auditing compliance should find them in one place rather
    than scattered among the geometry.
    """
    return (
        _string("author"),
        _string("license"),
        _string("license_url"),
        _string("license_spdx", nullable=True),
        _string("license_template", nullable=True),
        _string("source_url"),
        _string("commons_file"),
        # Derived per file by comparing the stored bytes with the Commons digest,
        # because 153 of the 1,691 files were never resized and asserting
        # modification over them would be a false statement in a rights field.
        _bool("modified"),
        _string("modification"),
        _string("attribution_text"),
    )


def image_columns(class_names: tuple[str, ...]) -> tuple[Column, ...]:
    """Describe the flat reference table, with or without the pixels."""
    label = Column(
        "label",
        pa.int64(),
        {"names": list(class_names), "_type": "ClassLabel"},
    )
    return (
        Column("image", IMAGE_TYPE, {"_type": "Image"}),
        _string("image_id"),
        _string("dwarf_id"),
        label,
        _string("display_name"),
        _string("commons_category"),
        _string("wikidata_url", nullable=True),
        *_rights_columns(),
        _string("sha256"),
        _string("commons_sha1", nullable=True),
        _int("width", 32),
        _int("height", 32),
        _timestamp("acquired_at"),
        _timestamp("source_revision_at", nullable=True),
        _int("commons_page_id", nullable=True),
        # Flat nullable pairs rather than a nested struct: every consumer wants
        # lat/lon, and an unplaced class must read null, never (0, 0), which is a
        # real point in the Gulf of Guinea.
        _float("image_latitude", nullable=True),
        _float("image_longitude", nullable=True),
        _float("dwarf_latitude", nullable=True),
        _float("dwarf_longitude", nullable=True),
        _string("dwarf_coordinate_source", nullable=True),
    )


def metadata_columns(class_names: tuple[str, ...]) -> tuple[Column, ...]:
    """Describe the pixel-free view of the reference table."""
    return tuple(column for column in image_columns(class_names) if column.name != "image")


def class_columns(class_names: tuple[str, ...]) -> tuple[Column, ...]:
    """Describe the per-statue table."""
    return (
        _string("dwarf_id"),
        Column("label", pa.int64(), {"names": list(class_names), "_type": "ClassLabel"}),
        _string("display_name"),
        _string("commons_category"),
        _string("wikidata_url", nullable=True),
        _float("latitude", nullable=True),
        _float("longitude", nullable=True),
        _string("coordinate_source", nullable=True),
        _int("image_count", 32),
    )


def fold_columns(class_names: tuple[str, ...]) -> tuple[Column, ...]:
    """Describe the leave-one-out folds."""
    return (
        _int("fold_index", 32),
        _string("query_image_id"),
        _string("query_dwarf_id"),
        Column("query_label", pa.int64(), {"names": list(class_names), "_type": "ClassLabel"}),
        Column(
            "reference_image_ids",
            pa.list_(pa.field("item", pa.string(), nullable=True)),
            {"feature": {"dtype": "string", "_type": "Value"}, "_type": "Sequence"},
        ),
        _int("reference_count", 32),
    )


def embedding_columns(dimensions: int) -> tuple[Column, ...]:
    """Describe one backbone's vector table.

    The width comes from the loaded matrix, never a literal, which is what lets
    the tests run on four-dimensional fake vectors and still prove the shape.
    """
    if dimensions <= 0:
        raise ValueError(f"embedding dimensions must be positive, got {dimensions}")
    return (
        _string("image_id"),
        _string("dwarf_id"),
        Column(
            "embedding",
            pa.list_(pa.field("item", pa.float32(), nullable=True), dimensions),
            {
                "feature": {"dtype": "float32", "_type": "Value"},
                "length": dimensions,
                "_type": "Sequence",
            },
        ),
    )


def arrow_schema(columns: tuple[Column, ...]) -> pa.Schema:
    """Return the arrow schema for a described table."""
    return pa.schema([column.field() for column in columns])


def hf_features(columns: tuple[Column, ...]) -> dict[str, object]:
    """Return the Hugging Face features dict for a described table."""
    return {column.name: column.feature for column in columns}


def hf_features_yaml(columns: tuple[Column, ...]) -> list[dict[str, object]]:
    """Return the features in the *list* form a dataset card carries.

    The card's `dataset_info.features` is a YAML list, not the dict used in the
    parquet metadata: `datasets` reads the two through different code paths and
    rejects a dict outright. Both are derived from the same column description so
    they cannot drift, and the oracle test asserts this matches what `datasets`
    itself would emit.
    """
    entries: list[dict[str, object]] = []
    for column in columns:
        feature = column.feature
        if not isinstance(feature, dict):  # pragma: no cover - every feature is a dict
            raise TypeError(f"column {column.name} has no feature description")
        kind = feature.get("_type")
        if kind == "Image":
            entries.append({"name": column.name, "dtype": "image"})
        elif kind == "ClassLabel":
            names = feature["names"]
            assert isinstance(names, list)
            entries.append(
                {
                    "name": column.name,
                    "dtype": {"class_label": {"names": {str(i): n for i, n in enumerate(names)}}},
                }
            )
        elif kind == "Sequence":
            inner = feature["feature"]
            assert isinstance(inner, dict)
            entry: dict[str, object] = {"name": column.name, "list": inner["dtype"]}
            if "length" in feature:
                entry["length"] = feature["length"]
            entries.append(entry)
        else:
            entries.append({"name": column.name, "dtype": feature["dtype"]})
    return entries


def required_columns(columns: tuple[Column, ...]) -> tuple[str, ...]:
    """Return the columns no row may leave empty.

    The attribution and licence columns are in here, which is how the section
    5.11 obligation becomes something the writer refuses to violate rather than
    something the card promises.
    """
    return tuple(column.name for column in columns if column.required)


@dataclass(frozen=True, slots=True)
class ShardPlan:
    """How many files a table is written across, and which rows go where."""

    shard_count: int
    rows_per_shard: int
    ranges: tuple[tuple[int, int], ...]


def shard_plan(total_bytes: int, row_count: int, target_bytes: int) -> ShardPlan:
    """Split `row_count` rows into contiguous, equal-sized shards.

    Contiguous equal-row slices rather than greedy byte packing: the row-to-shard
    mapping is then a function of the row index alone, reproducible without
    reading a single file, and it matches what `Dataset.shard(contiguous=True)`
    does, so a consumer's mental model is the standard one.
    """
    if row_count <= 0:
        raise ValueError("cannot shard zero rows")
    if target_bytes <= 0:
        raise ValueError(f"shard target must be positive, got {target_bytes}")

    shard_count = max(1, math.ceil(total_bytes / target_bytes))
    shard_count = min(shard_count, row_count)
    rows_per_shard = math.ceil(row_count / shard_count)
    # Equal slices can leave the final shard empty (10 rows over 4 shards is
    # 3+3+3+1, but 10 over 6 would be 2+2+2+2+2+0), so re-derive the count from
    # the rows each shard actually takes.
    shard_count = math.ceil(row_count / rows_per_shard)
    ranges = tuple(
        (start, min(start + rows_per_shard, row_count))
        for start in range(0, row_count, rows_per_shard)
    )
    return ShardPlan(shard_count=shard_count, rows_per_shard=rows_per_shard, ranges=ranges)


def shard_filename(split: str, index: int, total: int) -> str:
    """Return the Hub's conventional shard name."""
    return f"{split}-{index:05d}-of-{total:05d}.parquet"

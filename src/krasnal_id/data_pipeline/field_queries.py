"""Staging for field photographs, which are queries and never references.

`AGENTS.md` section 5.8 fixes the contract these files live under: photographs
taken in Wroclaw go in `data/field-queries/<dwarf_id>/`, the manifest is not
rebuilt to include them, and the directory name is what ties a photograph to a
statue. Admitting them as references would destroy the comparison they exist to
make, so nothing here writes to the manifest, the staging file, or the split.

Two artifacts meet here. `data/field-route.json` is tracked and human-reviewed:
it names the statues on the route and files each one as a confusable cluster
member or a control, and it is fixed before any photograph is scored so the
cohorts cannot be chosen after seeing the result. The query manifest this module
writes is generated, records the reference manifest and route it was built
against, and is the only input the experiment reads.
"""

import hashlib
import json
import os
from collections.abc import Iterable
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path
from tempfile import NamedTemporaryFile

from PIL import Image, UnidentifiedImageError
from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

from krasnal_id.data_pipeline.build_manifest import canonical_json_sha256
from krasnal_id.models import DWARF_ID_PATTERN, DatasetManifest

SCHEMA_VERSION = "1.0"


class FieldQueryError(ValueError):
    """Raised when the route, the photographs, or the query manifest are unusable."""


class FieldCohort(StrEnum):
    """Why a statue is on the route.

    `confusable` means the statue belongs to one of the named families both
    backbones confuse on clean photographs; `control` means everything else on the
    route. Both are required. A uniform drop and a drop concentrated on the
    confusable families are different findings, and only the second is visible if
    controls are present.

    Control does not mean "never confused". Four core controls carry one to three
    recorded top-1 errors, which is why `recorded_top_1_errors` sits beside the
    cohort: it biases the comparison towards finding no difference between the
    cohorts, so it understates a concentrated drop rather than inventing one.
    """

    CONFUSABLE = "confusable"
    CONTROL = "control"


class FieldRouteEntry(BaseModel):
    """One statue on the route, with the cohort it was assigned before shooting."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    dwarf_id: str = Field(pattern=DWARF_ID_PATTERN)
    display_name: str = Field(min_length=1)
    cohort: FieldCohort
    # The evidence behind the cohort, not its definition: how many top-1 errors this
    # statue's Commons photographs already draw. Recorded so a reviewer can see how
    # clean the control arm actually is instead of taking the label on trust.
    recorded_top_1_errors: int = Field(ge=0)
    # The core route is a complete experiment on its own; the extended one is
    # walked only if there is time. Recorded so a partial walk is self-describing.
    tier: str = Field(pattern=r"^(?:core|extended)$")


class FieldRouteFile(BaseModel):
    """The tracked, reviewed route: which statues, and which are controls."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: str = Field(pattern=r"^\d+\.\d+$")
    entries: tuple[FieldRouteEntry, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def validate_entries(self) -> "FieldRouteFile":
        """Require unique statues and both cohorts."""
        ids = [entry.dwarf_id for entry in self.entries]
        if len(set(ids)) != len(ids):
            raise ValueError("a statue cannot appear on the route twice")
        cohorts = {entry.cohort for entry in self.entries}
        if cohorts != set(FieldCohort):
            raise ValueError(
                "the route needs both cohorts: a drop measured without controls is "
                "confounded by hard statues also standing somewhere awkward"
            )
        return self

    def entry_for(self, dwarf_id: str) -> FieldRouteEntry | None:
        """Return the route entry for one statue, if it is on the route."""
        return next((entry for entry in self.entries if entry.dwarf_id == dwarf_id), None)


class FieldQueryRecord(BaseModel):
    """One staged field photograph.

    Field names match `ImageRecord` where they overlap, so the same extraction
    loop and the same content-addressed cache key serve both without a branch.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    image_id: str = Field(min_length=1)
    dwarf_id: str = Field(pattern=DWARF_ID_PATTERN)
    local_path: Path
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    width: int = Field(gt=0)
    height: int = Field(gt=0)
    cohort: FieldCohort


class FieldQueryManifest(BaseModel):
    """Every staged field photograph, and the artifacts it was staged against."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: str = Field(pattern=r"^\d+\.\d+$")
    # Both are recorded so a rebuilt dataset or a re-reviewed route is detected
    # rather than silently mixed with photographs staged under the old one.
    manifest_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    route_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    generated_at: datetime
    queries: tuple[FieldQueryRecord, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def validate_queries(self) -> "FieldQueryManifest":
        """Require unique identifiers across the staged photographs."""
        ids = [query.image_id for query in self.queries]
        if len(set(ids)) != len(ids):
            raise ValueError("field query identifiers must be unique")
        return self


def field_query_manifest_path(data_dir: Path) -> Path:
    """Return the artifact path for the generated query manifest."""
    return data_dir / "field-queries.json"


def _sha256_file(path: Path) -> str:
    """Hash one photograph without loading all of it at once."""
    digest = hashlib.sha256()
    try:
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
    except OSError as error:
        raise FieldQueryError(f"could not read field photograph {path}: {error}") from error
    return digest.hexdigest()


def _photograph_files(directory: Path) -> tuple[Path, ...]:
    """Return the candidate photographs in one statue's directory.

    Dotfiles are skipped because operating systems leave them everywhere; anything
    else is required to decode, so a stray file is a loud error rather than a
    silently missing query.
    """
    return tuple(
        sorted(
            path for path in directory.iterdir() if path.is_file() and not path.name.startswith(".")
        )
    )


def stage_field_queries(
    field_queries_dir: Path,
    manifest: DatasetManifest,
    route: FieldRouteFile,
    generated_at: datetime | None = None,
) -> FieldQueryManifest:
    """Build the query manifest from the photographs currently on disk.

    Every statue directory must name a dwarf that is both in the dataset and on
    the reviewed route, and no photograph may be byte-identical to a reference:
    that would be a Commons upload copied into the query set, which measures the
    protocol rather than the domain gap.
    """
    if not field_queries_dir.is_dir():
        raise FieldQueryError(f"field query directory does not exist: {field_queries_dir}")

    known_dwarfs = {dwarf.dwarf_id for dwarf in manifest.dwarfs}
    reference_digests = {image.sha256: image.image_id for image in manifest.images}

    records: list[FieldQueryRecord] = []
    for directory in sorted(path for path in field_queries_dir.iterdir() if path.is_dir()):
        dwarf_id = directory.name
        files = _photograph_files(directory)
        if not files:
            continue
        if dwarf_id not in known_dwarfs:
            raise FieldQueryError(
                f"directory {dwarf_id} names no dwarf in the manifest; the directory name "
                "is the dataset ID that ties a photograph to a statue"
            )
        entry = route.entry_for(dwarf_id)
        if entry is None:
            raise FieldQueryError(
                f"dwarf {dwarf_id} has photographs but is not on the reviewed route; "
                "add it to the route file with its cohort before scoring it"
            )
        for path in files:
            digest = _sha256_file(path)
            if digest in reference_digests:
                raise FieldQueryError(
                    f"field photograph {path} is byte-identical to reference image "
                    f"{reference_digests[digest]}; field photographs are queries, never "
                    "references"
                )
            try:
                with Image.open(path) as image:
                    image.load()
                    width, height = image.size
            except (OSError, UnidentifiedImageError) as error:
                raise FieldQueryError(
                    f"field photograph cannot be decoded: {path}: {error}"
                ) from error
            records.append(
                FieldQueryRecord(
                    image_id=f"{dwarf_id}/{path.name}",
                    dwarf_id=dwarf_id,
                    local_path=path,
                    sha256=digest,
                    width=width,
                    height=height,
                    cohort=entry.cohort,
                )
            )

    if not records:
        raise FieldQueryError(
            f"no field photographs are staged under {field_queries_dir}; see "
            "data/field-guide.md for the shooting protocol"
        )

    return FieldQueryManifest(
        schema_version=SCHEMA_VERSION,
        manifest_sha256=canonical_json_sha256(manifest.model_dump(mode="json")),
        route_sha256=canonical_json_sha256(route.model_dump(mode="json")),
        generated_at=generated_at or datetime.now(UTC),
        queries=tuple(records),
    )


def _read_json(path: Path, label: str) -> object:
    """Read one required JSON artifact."""
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise FieldQueryError(f"invalid {label} {path}: {error}") from error


def load_field_route(path: Path) -> FieldRouteFile:
    """Read the tracked route, refusing anything malformed."""
    try:
        return FieldRouteFile.model_validate(_read_json(path, "field route"))
    except ValidationError as error:
        raise FieldQueryError(f"invalid field route {path}: {error}") from error


def load_field_query_manifest(path: Path, manifest: DatasetManifest) -> FieldQueryManifest:
    """Read the query manifest, requiring it to describe the current dataset."""
    try:
        staged = FieldQueryManifest.model_validate(_read_json(path, "field query manifest"))
    except (FieldQueryError, ValidationError) as error:
        raise FieldQueryError(
            f"unusable field query manifest {path}: {error}; stage the photographs "
            "with krasnal-id data field-queries"
        ) from error

    manifest_sha256 = canonical_json_sha256(manifest.model_dump(mode="json"))
    if staged.manifest_sha256 != manifest_sha256:
        raise FieldQueryError(
            f"field query manifest {path} was staged against manifest "
            f"{staged.manifest_sha256[:12]} but the current manifest hashes to "
            f"{manifest_sha256[:12]}; rebuild it with krasnal-id data field-queries"
        )
    return staged


def write_field_query_manifest(path: Path, staged: FieldQueryManifest) -> None:
    """Write the query manifest atomically as formatted JSON."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    try:
        with NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temporary_path = Path(handle.name)
            json.dump(
                staged.model_dump(mode="json"),
                handle,
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
            )
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_path, path)
    except OSError as error:
        raise FieldQueryError(f"could not write field query manifest {path}: {error}") from error
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


def queries_by_cohort(
    queries: Iterable[FieldQueryRecord],
) -> dict[FieldCohort, tuple[FieldQueryRecord, ...]]:
    """Group staged photographs by the cohort their statue was assigned."""
    grouped: dict[FieldCohort, list[FieldQueryRecord]] = {cohort: [] for cohort in FieldCohort}
    for query in queries:
        grouped[query.cohort].append(query)
    return {cohort: tuple(records) for cohort, records in grouped.items() if records}

"""Proposing which route statue each unsorted field photograph shows, from its GPS.

`AGENTS.md` section 5.8 names photo GPS as the intended sorting mechanism and is
equally firm that it is **a proposal, never an assignment**: the directory a
photograph sits in is what ties it to a statue. So nothing here moves, copies or
renames a file. It reads an inbox of photographs straight off the phone, reports
the nearest statues on the route, and prints the commands a person would run
after looking at the photograph.

GPS cannot do more than narrow, and on this route it often cannot even do that.
Grajek, Meloman and Grajek i Meloman share one coordinate, as do Slepak and
Gluchak and the four statues of the water installation; WrocLovek and Spioch
stand 5 m apart. A phone fix in the Old Town is good to roughly ten metres and the
route's own derived coordinates to about nine (section 5.7). A photograph is
therefore *proposed* only when exactly one route statue lies within the ambiguity
margin of the nearest, and otherwise reported with every candidate it could be.
"""

import json
import math
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path

from PIL import ExifTags, Image, UnidentifiedImageError
from pydantic import BaseModel, ConfigDict, Field

from krasnal_id.atomic import atomic_text
from krasnal_id.data_pipeline.field_queries import FieldRouteFile, decode_hint
from krasnal_id.geometry import haversine_metres
from krasnal_id.models import DatasetManifest

SCHEMA_VERSION = "1.0"

# EXIF tag numbers inside the GPS IFD (EXIF 2.3, section 4.6.6).
_GPS_LATITUDE_REF = 1
_GPS_LATITUDE = 2
_GPS_LONGITUDE_REF = 3
_GPS_LONGITUDE = 4
_DATETIME_ORIGINAL = 0x9003


class FieldSortError(ValueError):
    """Raised when the inbox, the route or the manifest cannot be used."""


class SortStatus(StrEnum):
    """What the GPS could say about one photograph."""

    # One route statue is clearly nearest; still a proposal, to be checked by eye.
    PROPOSED = "proposed"
    # Several route statues are within the margin of the nearest; look and decide.
    AMBIGUOUS = "ambiguous"
    # A fix, but nothing on the route within reach of it.
    OFF_ROUTE = "off_route"
    # No usable fix: GPS was off, or a transfer stripped the metadata.
    NO_GPS = "no_gps"
    # The file will not decode at all; staging would refuse it too.
    UNREADABLE = "unreadable"


class SortCandidate(BaseModel):
    """A route statue a photograph may show, and how far the fix is from it."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    dwarf_id: str = Field(min_length=1)
    display_name: str = Field(min_length=1)
    distance_m: float = Field(ge=0.0)


class SortProposal(BaseModel):
    """What the GPS proposes for one photograph in the inbox."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    path: Path
    status: SortStatus
    taken_at: str | None = None
    latitude: float | None = None
    longitude: float | None = None
    # Nearest first. One entry when proposed, several when ambiguous, and the
    # nearest statue even when it is out of reach, so an off-route fix still says
    # how far off it was.
    candidates: tuple[SortCandidate, ...] = ()
    detail: str | None = None


class SortReport(BaseModel):
    """Every proposal for one inbox, with the settings that produced it."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: str = Field(pattern=r"^\d+\.\d+$")
    generated_at: datetime
    inbox: Path
    max_distance_m: float = Field(gt=0.0)
    ambiguity_m: float = Field(ge=0.0)
    proposals: tuple[SortProposal, ...]

    def count(self, status: SortStatus) -> int:
        """Count the photographs that ended in one status."""
        return sum(1 for proposal in self.proposals if proposal.status == status)


@dataclass(frozen=True, slots=True)
class RouteStatue:
    """A statue on the route, where the dataset places it."""

    dwarf_id: str
    display_name: str
    latitude: float
    longitude: float


def field_sort_report_path(data_dir: Path) -> Path:
    """Return the artifact path for the generated proposal report."""
    return data_dir / "field-sort-proposal.json"


def route_statues(route: FieldRouteFile, manifest: DatasetManifest) -> tuple[RouteStatue, ...]:
    """Locate every statue on the route, refusing one the dataset cannot place.

    Every route statue is placed today (section 5.7). A statue without coordinates
    could never be proposed and would silently make its neighbours look
    unambiguous, so it is an error rather than a skip.
    """
    dwarfs = {dwarf.dwarf_id: dwarf for dwarf in manifest.dwarfs}
    statues: list[RouteStatue] = []
    for entry in route.entries:
        dwarf = dwarfs.get(entry.dwarf_id)
        if dwarf is None:
            raise FieldSortError(f"route statue {entry.dwarf_id} is not in the manifest")
        if dwarf.coordinates is None:
            raise FieldSortError(
                f"route statue {entry.dwarf_id} has no coordinates, so GPS cannot propose it "
                "and its neighbours would look less ambiguous than they are"
            )
        statues.append(
            RouteStatue(
                dwarf_id=entry.dwarf_id,
                display_name=entry.display_name,
                latitude=dwarf.coordinates.latitude,
                longitude=dwarf.coordinates.longitude,
            )
        )
    return tuple(statues)


def _degrees(value: object) -> float:
    """Convert an EXIF degrees/minutes/seconds triple to decimal degrees.

    Pillow reads a rational with a zero denominator as NaN rather than raising,
    so NaN is refused here explicitly instead of propagating into a distance.
    """
    if not isinstance(value, tuple) or len(value) != 3:
        raise ValueError(f"expected a degrees/minutes/seconds triple, got {value!r}")
    try:
        degrees, minutes, seconds = (float(part) for part in value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"non-numeric coordinate component in {value!r}") from error
    result = degrees + minutes / 60.0 + seconds / 3600.0
    if math.isnan(result):
        raise ValueError(f"undefined coordinate component in {value!r}")
    return result


def read_gps(image: Image.Image) -> tuple[float, float] | None:
    """Return the (latitude, longitude) a photograph was taken at, if it says.

    A zero rational in a degrees field reads as a division by zero inside Pillow,
    and a stripped tag reads as absent; both are "no fix" rather than errors,
    because a photograph without GPS is still a perfectly good query.
    """
    gps = image.getexif().get_ifd(ExifTags.IFD.GPSInfo)
    try:
        latitude = _degrees(gps[_GPS_LATITUDE])
        longitude = _degrees(gps[_GPS_LONGITUDE])
        latitude_ref = str(gps.get(_GPS_LATITUDE_REF, "N")).strip().upper()
        longitude_ref = str(gps.get(_GPS_LONGITUDE_REF, "E")).strip().upper()
    except (KeyError, ValueError, ZeroDivisionError):
        return None
    if latitude_ref == "S":
        latitude = -latitude
    if longitude_ref == "W":
        longitude = -longitude
    if not (-90.0 <= latitude <= 90.0 and -180.0 <= longitude <= 180.0):
        return None
    # A phone with no fix sometimes writes 0/0 rather than omitting the tag.
    if latitude == 0.0 and longitude == 0.0:
        return None
    return latitude, longitude


def read_taken_at(image: Image.Image) -> str | None:
    """Return the EXIF capture time verbatim, which orders a walk's photographs."""
    value = image.getexif().get_ifd(ExifTags.IFD.Exif).get(_DATETIME_ORIGINAL)
    if value is None:
        return None
    return str(value).strip() or None


def propose(
    fix: tuple[float, float],
    statues: Iterable[RouteStatue],
    max_distance_m: float,
    ambiguity_m: float,
) -> tuple[SortStatus, tuple[SortCandidate, ...]]:
    """Rank the route by distance from one fix and decide what it can claim."""
    ranked = sorted(
        (
            SortCandidate(
                dwarf_id=statue.dwarf_id,
                display_name=statue.display_name,
                distance_m=haversine_metres(fix, (statue.latitude, statue.longitude)),
            )
            for statue in statues
        ),
        key=lambda candidate: (candidate.distance_m, candidate.dwarf_id),
    )
    if not ranked:
        raise FieldSortError("the route has no statues to propose")
    nearest = ranked[0]
    if nearest.distance_m > max_distance_m:
        return SortStatus.OFF_ROUTE, (nearest,)
    within = tuple(
        candidate
        for candidate in ranked
        if candidate.distance_m <= max_distance_m
        and candidate.distance_m - nearest.distance_m <= ambiguity_m
    )
    if len(within) == 1:
        return SortStatus.PROPOSED, within
    return SortStatus.AMBIGUOUS, within


def _inbox_files(inbox: Path) -> tuple[Path, ...]:
    """Every non-hidden file under the inbox, in a stable order."""
    return tuple(
        sorted(
            path
            for path in inbox.rglob("*")
            if path.is_file() and not any(part.startswith(".") for part in path.parts)
        )
    )


def sort_field_inbox(
    inbox: Path,
    route: FieldRouteFile,
    manifest: DatasetManifest,
    max_distance_m: float,
    ambiguity_m: float,
    generated_at: datetime | None = None,
) -> SortReport:
    """Read every photograph in the inbox and propose a statue for each."""
    if not inbox.is_dir():
        raise FieldSortError(f"field inbox does not exist: {inbox}")
    if max_distance_m <= 0.0 or ambiguity_m < 0.0:
        raise FieldSortError("the sort distances must be positive")
    statues = route_statues(route, manifest)
    files = _inbox_files(inbox)
    if not files:
        raise FieldSortError(
            f"no photographs in {inbox}; copy the phone's files there unsorted, "
            "as data/field-guide.md describes"
        )

    proposals: list[SortProposal] = []
    for path in files:
        try:
            with Image.open(path) as image:
                image.load()
                fix = read_gps(image)
                taken_at = read_taken_at(image)
        except (OSError, UnidentifiedImageError) as error:
            proposals.append(
                SortProposal(
                    path=path,
                    status=SortStatus.UNREADABLE,
                    detail=f"{error}{decode_hint(path)}",
                )
            )
            continue
        if fix is None:
            proposals.append(
                SortProposal(
                    path=path,
                    status=SortStatus.NO_GPS,
                    taken_at=taken_at,
                    detail="no GPS fix in the file; sort it by eye",
                )
            )
            continue
        status, candidates = propose(fix, statues, max_distance_m, ambiguity_m)
        proposals.append(
            SortProposal(
                path=path,
                status=status,
                taken_at=taken_at,
                latitude=fix[0],
                longitude=fix[1],
                candidates=candidates,
            )
        )

    return SortReport(
        schema_version=SCHEMA_VERSION,
        generated_at=generated_at or datetime.now(UTC),
        inbox=inbox,
        max_distance_m=max_distance_m,
        ambiguity_m=ambiguity_m,
        proposals=tuple(proposals),
    )


def write_sort_report(path: Path, report: SortReport) -> None:
    """Write the proposal report atomically as formatted JSON."""
    with atomic_text(path, FieldSortError) as handle:
        json.dump(report.model_dump(mode="json"), handle, ensure_ascii=False, indent=2)
        handle.write("\n")

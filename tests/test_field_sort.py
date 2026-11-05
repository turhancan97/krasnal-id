"""GPS proposals for unsorted field photographs, which never move a file."""

import json
from pathlib import Path
from typing import Any

import pytest
from PIL import Image
from typer.testing import CliRunner

from helpers import synthetic_manifest
from krasnal_id.cli import app
from krasnal_id.data_pipeline.field_queries import FieldRouteFile
from krasnal_id.data_pipeline.field_sort import (
    FieldSortError,
    RouteStatue,
    SortStatus,
    field_sort_report_path,
    propose,
    read_gps,
    route_statues,
    sort_field_inbox,
    write_sort_report,
)

runner = CliRunner()

# Q1 and Q2 share a plinth, like Grajek and Meloman; Q3 stands ~220 m away.
PLACES = {"Q1": (51.1100, 17.0300), "Q2": (51.1100, 17.0300), "Q3": (51.1120, 17.0300)}
ROUTE: dict[str, Any] = {
    "schema_version": "1.0",
    "entries": [
        {
            "dwarf_id": dwarf_id,
            "display_name": f"Dwarf {dwarf_id[1:]}",
            "cohort": cohort,
            "recorded_top_1_errors": 0,
            "tier": "core",
        }
        for dwarf_id, cohort in (("Q1", "confusable"), ("Q2", "confusable"), ("Q3", "control"))
    ],
}


def _dms(value: float) -> tuple[float, float, float]:
    degrees = int(value)
    minutes_float = (value - degrees) * 60
    minutes = int(minutes_float)
    return (float(degrees), float(minutes), round((minutes_float - minutes) * 60, 4))


def _photograph(
    path: Path,
    fix: tuple[float, float] | None,
    taken_at: str | None = "2026:11:20 10:15:00",
) -> Path:
    """Write a JPEG the way a phone would: GPS and capture time in its EXIF."""
    path.parent.mkdir(parents=True, exist_ok=True)
    exif = Image.Exif()
    if fix is not None:
        exif[0x8825] = {
            1: "N" if fix[0] >= 0 else "S",
            2: _dms(abs(fix[0])),
            3: "E" if fix[1] >= 0 else "W",
            4: _dms(abs(fix[1])),
        }
    if taken_at is not None:
        exif[0x8769] = {0x9003: taken_at}
    Image.new("RGB", (8, 8), (120, 90, 60)).save(path, exif=exif)
    return path


def _manifest() -> Any:
    return synthetic_manifest(dwarf_count=3, coordinates=PLACES)


def _route() -> FieldRouteFile:
    return FieldRouteFile.model_validate(ROUTE)


def test_gps_is_read_from_a_phone_photograph(tmp_path: Path) -> None:
    path = _photograph(tmp_path / "IMG_1.jpg", (51.1100, 17.0300))
    with Image.open(path) as image:
        fix = read_gps(image)
    assert fix is not None
    assert fix[0] == pytest.approx(51.11, abs=1e-6)
    assert fix[1] == pytest.approx(17.03, abs=1e-6)

    southern = _photograph(tmp_path / "IMG_2.jpg", (-33.8, -70.6))
    with Image.open(southern) as image:
        assert read_gps(image) == pytest.approx((-33.8, -70.6), abs=1e-6)

    bare = _photograph(tmp_path / "IMG_3.jpg", None)
    with Image.open(bare) as image:
        assert read_gps(image) is None

    # Some phones write 0/0 for "no fix" instead of leaving the tag out.
    null_island = _photograph(tmp_path / "IMG_4.jpg", (0.0, 0.0))
    with Image.open(null_island) as image:
        assert read_gps(image) is None


def test_a_proposal_is_single_only_when_no_neighbour_is_within_the_margin() -> None:
    statues = (
        RouteStatue("Q1", "Dwarf 1", 51.1100, 17.0300),
        RouteStatue("Q2", "Dwarf 2", 51.1100, 17.0300),
        RouteStatue("Q3", "Dwarf 3", 51.1120, 17.0300),
    )

    status, candidates = propose((51.1120, 17.0300), statues, 100.0, 25.0)
    assert status == SortStatus.PROPOSED
    assert [candidate.dwarf_id for candidate in candidates] == ["Q3"]

    # GPS cannot tell apart two statues on one plinth, and must not pretend to.
    status, candidates = propose((51.1100, 17.0301), statues, 100.0, 25.0)
    assert status == SortStatus.AMBIGUOUS
    assert [candidate.dwarf_id for candidate in candidates] == ["Q1", "Q2"]

    status, candidates = propose((51.2000, 17.0300), statues, 100.0, 25.0)
    assert status == SortStatus.OFF_ROUTE
    assert candidates[0].dwarf_id == "Q3"
    assert candidates[0].distance_m > 9_000

    with pytest.raises(FieldSortError, match="no statues"):
        propose((51.1, 17.0), (), 100.0, 25.0)


def test_the_inbox_is_sorted_into_proposals_without_moving_anything(tmp_path: Path) -> None:
    inbox = tmp_path / "inbox"
    _photograph(inbox / "IMG_1.jpg", (51.1120, 17.0301))
    _photograph(inbox / "IMG_2.jpg", (51.1100, 17.0300))
    _photograph(inbox / "IMG_3.jpg", (51.3000, 17.0300))
    _photograph(inbox / "IMG_4.jpg", None, taken_at=None)
    (inbox / "IMG_5.HEIC").write_bytes(b"not decodable here")
    (inbox / ".hidden.jpg").write_bytes(b"skipped")
    before = sorted(path.name for path in inbox.iterdir())

    report = sort_field_inbox(inbox, _route(), _manifest(), 100.0, 25.0)

    statuses = {proposal.path.name: proposal.status for proposal in report.proposals}
    assert statuses == {
        "IMG_1.jpg": SortStatus.PROPOSED,
        "IMG_2.jpg": SortStatus.AMBIGUOUS,
        "IMG_3.jpg": SortStatus.OFF_ROUTE,
        "IMG_4.jpg": SortStatus.NO_GPS,
        "IMG_5.HEIC": SortStatus.UNREADABLE,
    }
    first = report.proposals[0]
    assert first.taken_at == "2026:11:20 10:15:00"
    assert first.candidates[0].dwarf_id == "Q3"
    heic = next(proposal for proposal in report.proposals if proposal.path.name == "IMG_5.HEIC")
    assert heic.detail is not None and "Most Compatible" in heic.detail
    assert report.count(SortStatus.PROPOSED) == 1
    assert sorted(path.name for path in inbox.iterdir()) == before

    path = tmp_path / "proposal.json"
    write_sort_report(path, report)
    assert json.loads(path.read_text(encoding="utf-8"))["proposals"][0]["status"] == "proposed"


def test_sorting_refuses_what_it_cannot_use(tmp_path: Path) -> None:
    with pytest.raises(FieldSortError, match="does not exist"):
        sort_field_inbox(tmp_path / "absent", _route(), _manifest(), 100.0, 25.0)
    empty = tmp_path / "empty"
    empty.mkdir()
    with pytest.raises(FieldSortError, match="no photographs"):
        sort_field_inbox(empty, _route(), _manifest(), 100.0, 25.0)
    with pytest.raises(FieldSortError, match="must be positive"):
        sort_field_inbox(empty, _route(), _manifest(), 0.0, 25.0)

    # An unplaced route statue would make its neighbours look unambiguous.
    unplaced = synthetic_manifest(dwarf_count=3, coordinates={"Q1": PLACES["Q1"]})
    with pytest.raises(FieldSortError, match="has no coordinates"):
        route_statues(_route(), unplaced)
    with pytest.raises(FieldSortError, match="not in the manifest"):
        route_statues(_route(), synthetic_manifest(dwarf_count=2, coordinates=PLACES))


def test_cli_prints_suggestions_and_moves_nothing(tmp_path: Path) -> None:
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(_manifest().model_dump(mode="json")), encoding="utf-8")
    route_path = tmp_path / "field-route.json"
    route_path.write_text(json.dumps(ROUTE), encoding="utf-8")
    inbox = tmp_path / "inbox"
    _photograph(inbox / "IMG_1.jpg", (51.1120, 17.0301))
    _photograph(inbox / "IMG_2.jpg", (51.1100, 17.0300))
    _photograph(inbox / "IMG_3.jpg", (51.3000, 17.0300))
    _photograph(inbox / "IMG_4.jpg", None)
    arguments = [
        "data",
        "field-sort",
        f"-opaths.data_dir={tmp_path}",
        f"-opaths.manifest_path={manifest_path}",
        f"-opaths.field_route_path={route_path}",
        f"-opaths.field_inbox_dir={inbox}",
        f"-opaths.field_queries_dir={tmp_path / 'field-queries'}",
        "-ologging.json_output=false",
    ]

    result = runner.invoke(app, arguments)

    assert result.exit_code == 0, result.output
    assert "proposed=1 ambiguous=1 off_route=1 no_gps=1 unreadable=0" in result.output
    assert "Nothing was moved" in result.output
    assert f"mv -n {inbox / 'IMG_1.jpg'} {tmp_path / 'field-queries' / 'Q3'}/" in result.output
    assert "ambiguous, decide by eye between Dwarf 1 [Q1]" in result.output
    assert "off the route" in result.output
    assert "no_gps" in result.output
    assert not (tmp_path / "field-queries").exists()
    assert field_sort_report_path(tmp_path).is_file()

    missing = runner.invoke(app, [*arguments[:5], f"-opaths.field_inbox_dir={tmp_path / 'absent'}"])
    assert missing.exit_code == 2
    assert "Field sort error" in missing.output


def test_packaged_sort_settings_are_usable() -> None:
    from krasnal_id.config import load_config

    config = load_config()
    assert config.paths.field_inbox_dir == Path("data/field-inbox")
    assert config.thresholds.field_sort_max_distance_m > config.thresholds.field_sort_ambiguity_m

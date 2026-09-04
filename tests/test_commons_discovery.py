"""Commons-first discovery of per-dwarf categories Wikidata has no item for."""

import json
import re
from collections.abc import Callable
from pathlib import Path
from typing import Any

import httpx
import pytest
from typer.testing import CliRunner

from krasnal_id.cli import app
from krasnal_id.config import load_config
from krasnal_id.data_pipeline.commons_discovery import (
    CACHE_METADATA_FILENAME,
    MAX_CATEGORY_PAGES,
    RAW_RESPONSE_FILENAME,
    CommonsDiscoveryError,
    category_slug,
    commons_discovery_paths,
    discover_commons_categories,
    display_name_for,
    fetch_pages,
    merge_discovery_records,
    normalize_categories,
    query_sha256,
    request_parameters,
    titles_from_pages,
)
from krasnal_id.data_pipeline.wikidata_query import (
    QUERY_SHA256,
    USER_AGENT_ENV_VAR,
    effective_query_sha256,
    query_dwarfs,
)
from krasnal_id.models import DWARF_ID_PATTERN, AuditDisposition, AuditReason, DwarfRecord

runner = CliRunner()


def _config(**overrides: object) -> Any:
    config = load_config().data
    return config.model_copy(update=overrides) if overrides else config


def _page(titles: list[str], continuation: str | None = None) -> dict[str, Any]:
    payload: dict[str, Any] = {"query": {"categorymembers": [{"title": title} for title in titles]}}
    if continuation is not None:
        payload["continue"] = {"cmcontinue": continuation}
    return payload


def _writer(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")


def test_slugs_fold_polish_letters_and_stay_valid_identifiers() -> None:
    # 'ł' has no NFKD decomposition, so without an explicit rule "Wrocław" would
    # silently slug to "wrocaw".
    assert category_slug("Binio dwarf, Wrocław") == "binio-dwarf-wroclaw"
    assert category_slug("Abruzjusz dwarf, Wroclaw") == "abruzjusz-dwarf-wroclaw"
    assert category_slug("Śpiewak Operowy dwarf, Wrocław") == "spiewak-operowy-dwarf-wroclaw"
    assert category_slug("100matolog dwarf, Wrocław") == "100matolog-dwarf-wroclaw"

    for title in ("Binio dwarf, Wrocław", "3TESLUŚ dwarf, Wrocław", "Alpinki dwarfs, Wrocław"):
        assert re.match(DWARF_ID_PATTERN, f"C-{category_slug(title)}")

    with pytest.raises(CommonsDiscoveryError, match="no ASCII characters"):
        category_slug("・・・")


def test_display_names_come_off_the_title_pattern() -> None:
    assert display_name_for("Binio dwarf, Wrocław") == "Binio"
    # Both plurals mark a group installation rather than a naming mistake, and
    # Commons really does use both spellings.
    assert display_name_for("Alpinki dwarfs, Wrocław") == "Alpinki"
    assert display_name_for("Aquaparczki dwarves, Wrocław") == "Aquaparczki"
    assert display_name_for("Abruzjusz dwarf, Wroclaw") == "Abruzjusz"
    # Anything else is for the reviewer to name.
    assert display_name_for("Dwarves in Wrocław by artist") is None
    assert display_name_for("dwarf, Wrocław") is None


def test_normalization_emits_records_without_wikidata_provenance() -> None:
    records, audit = normalize_categories(
        (
            "Category:Binio dwarf, Wrocław",
            "Category:Alpinki_dwarfs,_Wrocław",
            "Category:Binio dwarf, Wrocław",
        )
    )

    assert [record.dwarf_id for record in records] == [
        "C-alpinki-dwarfs-wroclaw",
        "C-binio-dwarf-wroclaw",
    ]
    assert records[0].commons_category == "Alpinki dwarfs, Wrocław"
    assert records[0].display_name == "Alpinki"
    # No Wikidata item means no URL and, since coordinates are P625, no coordinates.
    assert all(record.wikidata_url is None for record in records)
    assert all(record.coordinates is None for record in records)
    assert audit == ()


def test_an_unexpected_title_is_flagged_but_still_offered_for_review() -> None:
    records, audit = normalize_categories(("Category:Dwarves in Wrocław by artist",))

    assert len(records) == 1
    # The full title becomes the display name until review corrects it.
    assert records[0].display_name == "Dwarves in Wrocław by artist"
    assert audit[0].disposition is AuditDisposition.WARNING
    assert audit[0].reason is AuditReason.UNEXPECTED_CATEGORY_NAME


def test_a_slug_collision_is_excluded_rather_than_merged() -> None:
    # Two statues sharing one identifier would share one image directory.
    records, audit = normalize_categories(
        ("Category:Abruzjusz dwarf, Wrocław", "Category:Abruzjusz dwarf, Wroclaw")
    )

    assert len(records) == 1
    assert audit[0].disposition is AuditDisposition.EXCLUDED
    assert audit[0].reason is AuditReason.DUPLICATE_CATEGORY_SLUG
    assert "already taken by" in audit[0].details


def test_page_parsing_reports_api_errors_and_bad_shapes() -> None:
    assert titles_from_pages((_page(["Category:A dwarf, Wrocław"]),)) == (
        "Category:A dwarf, Wrocław",
    )
    with pytest.raises(CommonsDiscoveryError, match="not a JSON object"):
        titles_from_pages(([],))
    with pytest.raises(CommonsDiscoveryError, match="Commons API error"):
        titles_from_pages(({"error": {"code": "badtitle"}},))
    with pytest.raises(CommonsDiscoveryError, match="invalid Commons category response"):
        titles_from_pages(({"query": {"categorymembers": [{"title": ""}]}},))


def test_parameters_and_hash_bind_the_root_category() -> None:
    parameters = request_parameters("Category:Root", "next-token")

    assert parameters["cmtitle"] == "Category:Root"
    assert parameters["cmtype"] == "subcat"
    assert parameters["cmcontinue"] == "next-token"
    assert "cmcontinue" not in request_parameters("Category:Root", None)
    # Pointing the walk at another tree is a different query.
    assert query_sha256("Category:Root") != query_sha256("Category:Other")


def test_the_walk_follows_continuations_and_refuses_to_run_away() -> None:
    pages = [_page(["Category:A dwarf, Wrocław"], "more"), _page(["Category:B dwarf, Wrocław"])]
    seen: list[dict[str, str]] = []

    def session(parameters: dict[str, str]) -> object:
        seen.append(parameters)
        return pages[len(seen) - 1]

    assert len(fetch_pages(_config(), session)) == 2
    assert "cmcontinue" not in seen[0]
    assert seen[1]["cmcontinue"] == "more"

    with pytest.raises(CommonsDiscoveryError, match=f"exceeded {MAX_CATEGORY_PAGES} pages"):
        fetch_pages(_config(), lambda _: _page(["Category:A dwarf, Wrocław"], "endless"))


def test_enumeration_caches_and_reuses_only_a_matching_query(tmp_path: Path) -> None:
    calls = 0

    def session(parameters: dict[str, str]) -> object:
        nonlocal calls
        calls += 1
        return _page(["Category:Binio dwarf, Wrocław"])

    first = discover_commons_categories(_config(), tmp_path, session=session, write_json=_writer)
    assert first.cache_status == "fetched"
    assert calls == 1

    again = discover_commons_categories(_config(), tmp_path, session=session, write_json=_writer)
    assert again.cache_status == "hit"
    assert calls == 1
    assert again.records == first.records

    refreshed = discover_commons_categories(
        _config(), tmp_path, session=session, write_json=_writer, refresh=True
    )
    assert refreshed.cache_status == "refreshed"
    assert calls == 2

    # A cache built for one root category must not satisfy another.
    other = discover_commons_categories(
        _config(commons_root_category="Category:Something else"),
        tmp_path,
        session=session,
        write_json=_writer,
    )
    assert other.cache_status == "recovered"
    assert calls == 3


def test_a_corrupt_cache_is_refetched(tmp_path: Path) -> None:
    paths = commons_discovery_paths(tmp_path)
    paths.raw_response.parent.mkdir(parents=True, exist_ok=True)
    paths.raw_response.write_text("not json", encoding="utf-8")

    result = discover_commons_categories(
        _config(),
        tmp_path,
        session=lambda _: _page(["Category:Binio dwarf, Wrocław"]),
        write_json=_writer,
    )

    assert result.cache_status == "recovered"
    assert (tmp_path / RAW_RESPONSE_FILENAME).is_file()
    assert (tmp_path / CACHE_METADATA_FILENAME).is_file()


def _wikidata_record(dwarf_id: str, category: str) -> DwarfRecord:
    return DwarfRecord(
        dwarf_id=dwarf_id,
        display_name=dwarf_id,
        wikidata_url=f"https://www.wikidata.org/wiki/{dwarf_id}",  # type: ignore[arg-type]
        commons_category=category,
    )


def test_a_wikidata_record_wins_the_category_it_claims() -> None:
    commons, _ = normalize_categories(
        ("Category:Binio dwarf, Wrocław", "Category:Kowal dwarf, Wrocław")
    )
    wikidata = (_wikidata_record("Q1", "Binio dwarf, Wrocław"),)

    merged, audit = merge_discovery_records(wikidata, commons)

    # The QID-carrying record survives; the Commons duplicate is audited, not dropped
    # silently, because the count of these is how little of the pool Wikidata covers.
    assert [record.dwarf_id for record in merged] == ["Q1", "C-kowal-dwarf-wroclaw"]
    assert audit[0].reason is AuditReason.CLAIMED_BY_WIKIDATA
    assert audit[0].dwarf_id == "C-binio-dwarf-wroclaw"


def test_merging_refuses_to_produce_a_duplicate_identifier() -> None:
    duplicated = (_wikidata_record("Q1", "A dwarf, Wrocław"),) * 2

    with pytest.raises(CommonsDiscoveryError, match="duplicate dwarf identifiers"):
        merge_discovery_records(duplicated, ())


def test_the_recorded_hash_only_changes_when_a_source_does() -> None:
    config = _config()

    # Without the flag the artifact stays byte-identical to a Wikidata-only run.
    assert effective_query_sha256(config, include_commons=False) == QUERY_SHA256
    combined = effective_query_sha256(config, include_commons=True)
    assert combined != QUERY_SHA256
    assert combined != effective_query_sha256(
        config.model_copy(update={"commons_root_category": "Category:Other"}),
        include_commons=True,
    )


def _client(handler: Callable[[httpx.Request], httpx.Response]) -> httpx.Client:
    return httpx.Client(transport=httpx.MockTransport(handler))


def _both_sources(sparql: dict[str, Any]) -> Callable[[httpx.Request], httpx.Response]:
    def handler(request: httpx.Request) -> httpx.Response:
        if "commons" in str(request.url):
            return httpx.Response(
                200, json=_page(["Category:Binio dwarf, Wrocław", "Category:Kowal dwarf, Wrocław"])
            )
        return httpx.Response(200, json=sparql)

    return handler


@pytest.fixture
def sparql_payload() -> dict[str, Any]:
    fixture = Path(__file__).parent / "fixtures" / "wikidata-dwarfs.json"
    with fixture.open(encoding="utf-8") as handle:
        return dict(json.load(handle))


def test_discovery_merges_both_sources_into_one_artifact(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    sparql_payload: dict[str, Any],
) -> None:
    monkeypatch.setenv(USER_AGENT_ENV_VAR, "krasnal-id-tests (tests@example.com)")
    config = _config()

    wikidata_only = query_dwarfs(
        config, tmp_path / "a", client=_client(_both_sources(sparql_payload))
    )
    merged = query_dwarfs(
        config,
        tmp_path / "b",
        include_commons=True,
        client=_client(_both_sources(sparql_payload)),
    )

    commons_only = [record for record in merged.records if record.dwarf_id.startswith("C-")]
    assert len(merged.records) == len(wikidata_only.records) + len(commons_only)
    assert commons_only
    # Commons-only classes carry no coordinates, so the geographic arm cannot grow
    # with the dataset; see AGENTS.md section 5.6.
    assert all(record.coordinates is None for record in commons_only)
    assert any(record.coordinates is not None for record in merged.records)


def test_cli_reports_what_each_source_contributed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    sparql_payload: dict[str, Any],
) -> None:
    monkeypatch.setenv(USER_AGENT_ENV_VAR, "krasnal-id-tests (tests@example.com)")
    pages = _page(["Category:Binio dwarf, Wrocław"])
    discovery = tmp_path / "discovery"
    _writer(discovery / RAW_RESPONSE_FILENAME, [pages])
    _writer(
        discovery / CACHE_METADATA_FILENAME,
        {
            "schema_version": "1.0",
            "endpoint": str(_config().commons_api_endpoint),
            "query_sha256": query_sha256(_config().commons_root_category),
            "retrieved_at": "2026-09-04T00:00:00Z",
        },
    )
    _writer(discovery / "wikidata-response.json", sparql_payload)
    _writer(
        discovery / "wikidata-response.meta.json",
        {
            "schema_version": "1.0",
            "endpoint": str(_config().wikidata_endpoint),
            "query_sha256": QUERY_SHA256,
            "retrieved_at": "2026-09-04T00:00:00Z",
        },
    )

    result = runner.invoke(
        app,
        [
            "data",
            "query",
            "--include-commons",
            f"-opaths.discovery_dir={discovery}",
            "-ologging.json_output=false",
        ],
    )

    assert result.exit_code == 0, result.output
    assert "wikidata+commons" in result.output
    assert "Commons-only" in result.output
    assert "carry coordinates" in result.output


def test_a_superseded_category_stops_asking_to_be_reviewed() -> None:
    # A reviewer facing 481 categories must not be sent to look at records that
    # never reached the artifact.
    commons, commons_audit = normalize_categories(
        ("Category:Olbiniusz", "Category:Kowal dwarf, Wrocław")
    )
    assert commons_audit[0].reason is AuditReason.UNEXPECTED_CATEGORY_NAME
    assert commons_audit[0].dwarf_id == "C-olbiniusz"

    merged, audit = merge_discovery_records(
        (_wikidata_record("Q1", "Olbiniusz"),), commons, commons_audit
    )

    assert [record.dwarf_id for record in merged] == ["Q1", "C-kowal-dwarf-wroclaw"]
    reasons = {record.reason for record in audit}
    assert AuditReason.CLAIMED_BY_WIKIDATA in reasons
    assert AuditReason.UNEXPECTED_CATEGORY_NAME not in reasons


def test_an_exclusion_survives_the_merge() -> None:
    # An exclusion explains why something is absent, which stays true either way.
    commons, commons_audit = normalize_categories(
        ("Category:Abruzjusz dwarf, Wrocław", "Category:Abruzjusz dwarf, Wroclaw")
    )
    excluded = next(
        record for record in commons_audit if record.reason is AuditReason.DUPLICATE_CATEGORY_SLUG
    )

    _, audit = merge_discovery_records(
        (_wikidata_record("Q1", "Abruzjusz dwarf, Wrocław"),), commons, commons_audit
    )

    assert excluded in audit

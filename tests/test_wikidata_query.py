"""Tests for deterministic Wikidata discovery, caching, and transport behavior."""

import json
import os
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import httpx
import pytest
from typer.testing import CliRunner

from krasnal_id.cli import app
from krasnal_id.config import load_config
from krasnal_id.data_pipeline.wikidata_query import (
    CACHE_METADATA_FILENAME,
    QUERY_SHA256,
    RAW_RESPONSE_FILENAME,
    USER_AGENT_ENV_VAR,
    WikidataConfigurationError,
    WikidataQueryError,
    discovery_paths,
    normalize_query_response,
    query_dwarfs,
)
from krasnal_id.models import (
    AuditDisposition,
    AuditReason,
    DiscoveryAuditFile,
    DwarfDiscoveryFile,
)

runner = CliRunner()


@pytest.fixture
def wikidata_payload() -> dict[str, Any]:
    """Load the committed synthetic SPARQL response."""
    fixture = Path(__file__).parent / "fixtures" / "wikidata-dwarfs.json"
    with fixture.open(encoding="utf-8") as handle:
        return dict(json.load(handle))


def _contact(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(USER_AGENT_ENV_VAR, "krasnal-id-tests (tests@example.com)")


def _client(
    handler: Callable[[httpx.Request], httpx.Response],
) -> httpx.Client:
    return httpx.Client(transport=httpx.MockTransport(handler))


def _json_response(payload: object, status_code: int = 200, **headers: str) -> httpx.Response:
    return httpx.Response(status_code, json=payload, headers=headers)


def _successful_client(
    payload: object,
    requests: list[httpx.Request] | None = None,
) -> httpx.Client:
    def handler(request: httpx.Request) -> httpx.Response:
        if requests is not None:
            requests.append(request)
        return _json_response(payload)

    return _client(handler)


def test_normalizes_labels_coordinates_groups_and_audit(
    wikidata_payload: dict[str, Any],
) -> None:
    records, audit = normalize_query_response(wikidata_payload)

    assert [record.dwarf_id for record in records] == ["Q9", "Q10", "Q11", "Q14", "Q15"]
    by_id = {record.dwarf_id: record for record in records}
    assert by_id["Q9"].display_name == "English only"
    assert by_id["Q10"].display_name == "Polska nazwa"
    assert by_id["Q10"].coordinates is not None
    assert by_id["Q10"].coordinates.longitude == pytest.approx(17.04)
    assert by_id["Q11"].display_name == "Q11"
    assert by_id["Q9"].coordinates is None

    reasons = {(record.dwarf_id, record.disposition, record.reason) for record in audit}
    assert (
        "Q12",
        AuditDisposition.EXCLUDED,
        AuditReason.MISSING_COMMONS_CATEGORY,
    ) in reasons
    assert ("Q13", AuditDisposition.EXCLUDED, AuditReason.EXPLICIT_GROUP_ENTITY) in reasons
    assert (
        "Q15",
        AuditDisposition.WARNING,
        AuditReason.POSSIBLE_UNLINKED_GROUP,
    ) in reasons
    assert (
        "Q16",
        AuditDisposition.EXCLUDED,
        AuditReason.CONFLICTING_SOURCE_VALUES,
    ) in reasons
    assert ("Q17", AuditDisposition.EXCLUDED, AuditReason.INVALID_RECORD) in reasons
    assert (
        "https://example.com/entity/Q18",
        AuditDisposition.EXCLUDED,
        AuditReason.INVALID_RECORD,
    ) in reasons


def test_rejects_invalid_sparql_response_shape() -> None:
    with pytest.raises(WikidataQueryError, match="Invalid SPARQL response shape"):
        normalize_query_response({"not_results": []})


def test_fetches_writes_limits_and_reuses_cache_without_contact(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    wikidata_payload: dict[str, Any],
) -> None:
    _contact(monkeypatch)
    requests: list[httpx.Request] = []
    config = load_config().data

    with _successful_client(wikidata_payload, requests) as client:
        first = query_dwarfs(config, tmp_path, limit=3, client=client)

    assert first.cache_status == "fetched"
    assert first.eligible_total == 5
    assert [record.dwarf_id for record in first.records] == ["Q9", "Q10", "Q11"]
    assert len(requests) == 1
    assert requests[0].headers["user-agent"] == "krasnal-id-tests (tests@example.com)"
    assert "Q136276280" in requests[0].content.decode()

    paths = discovery_paths(tmp_path)
    discovery = DwarfDiscoveryFile.model_validate_json(paths.dwarfs.read_text())
    audit = DiscoveryAuditFile.model_validate_json(paths.audit.read_text())
    deterministic_records = paths.dwarfs.read_bytes()
    deterministic_audit = paths.audit.read_bytes()

    assert discovery.query_sha256 == QUERY_SHA256
    assert discovery.selection_limit == 3
    assert discovery.eligible_total == 5
    assert audit.records
    assert paths.raw_response.exists()
    assert paths.cache_metadata.exists()

    monkeypatch.delenv(USER_AGENT_ENV_VAR)
    with _client(lambda request: pytest.fail(f"unexpected request: {request.url}")) as client:
        second = query_dwarfs(config, tmp_path, limit=3, client=client)

    assert second.cache_status == "hit"
    assert paths.dwarfs.read_bytes() == deterministic_records
    assert paths.audit.read_bytes() == deterministic_audit


def test_refresh_replaces_valid_cache(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    wikidata_payload: dict[str, Any],
) -> None:
    _contact(monkeypatch)
    config = load_config().data

    with _successful_client(wikidata_payload) as client:
        query_dwarfs(config, tmp_path, client=client)
    with _successful_client(wikidata_payload) as client:
        result = query_dwarfs(config, tmp_path, refresh=True, client=client)

    assert result.cache_status == "refreshed"


@pytest.mark.parametrize("broken_file", [RAW_RESPONSE_FILENAME, CACHE_METADATA_FILENAME])
def test_recovers_from_malformed_cache(
    broken_file: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    wikidata_payload: dict[str, Any],
) -> None:
    _contact(monkeypatch)
    config = load_config().data
    with _successful_client(wikidata_payload) as client:
        query_dwarfs(config, tmp_path, client=client)

    (tmp_path / broken_file).write_text("{broken", encoding="utf-8")
    with _successful_client(wikidata_payload) as client:
        result = query_dwarfs(config, tmp_path, client=client)

    assert result.cache_status == "recovered"
    assert not list(tmp_path.glob("*.tmp"))


def test_invalidates_cache_when_query_hash_changes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    wikidata_payload: dict[str, Any],
) -> None:
    _contact(monkeypatch)
    config = load_config().data
    with _successful_client(wikidata_payload) as client:
        query_dwarfs(config, tmp_path, client=client)

    metadata_path = tmp_path / CACHE_METADATA_FILENAME
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata["query_sha256"] = "0" * 64
    metadata_path.write_text(json.dumps(metadata), encoding="utf-8")

    with _successful_client(wikidata_payload) as client:
        result = query_dwarfs(config, tmp_path, client=client)

    assert result.cache_status == "recovered"


@pytest.mark.parametrize(
    "user_agent",
    [
        "krasnal-id (owner@example.com)",
        "krasnal-id (mailto:owner@example.com)",
        "krasnal-id (https://example.com/contact)",
    ],
)
def test_accepts_contact_bearing_user_agents(
    user_agent: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    wikidata_payload: dict[str, Any],
) -> None:
    monkeypatch.setenv(USER_AGENT_ENV_VAR, user_agent)
    with _successful_client(wikidata_payload) as client:
        result = query_dwarfs(load_config().data, tmp_path, limit=1, client=client)

    assert len(result.records) == 1


@pytest.mark.parametrize("user_agent", [None, "", "krasnal-id/no-contact"])
def test_rejects_missing_or_contactless_user_agent(
    user_agent: str | None,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    if user_agent is None:
        monkeypatch.delenv(USER_AGENT_ENV_VAR, raising=False)
    else:
        monkeypatch.setenv(USER_AGENT_ENV_VAR, user_agent)

    with (
        _client(lambda request: pytest.fail(f"unexpected request: {request.url}")) as client,
        pytest.raises(WikidataConfigurationError, match=USER_AGENT_ENV_VAR),
    ):
        query_dwarfs(load_config().data, tmp_path, client=client)


def test_rejects_nonpositive_programmatic_limit(tmp_path: Path) -> None:
    with pytest.raises(WikidataConfigurationError, match="positive"):
        query_dwarfs(load_config().data, tmp_path, limit=0)


def test_retries_retryable_status_and_caps_retry_after(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    wikidata_payload: dict[str, Any],
) -> None:
    _contact(monkeypatch)
    delays: list[float] = []
    monkeypatch.setattr(time, "sleep", delays.append)
    responses = iter(
        [
            _json_response({}, status_code=503, **{"Retry-After": "999"}),
            _json_response(wikidata_payload),
        ]
    )

    with _client(lambda request: next(responses)) as client:
        result = query_dwarfs(load_config().data, tmp_path, client=client)

    assert result.cache_status == "fetched"
    assert delays == [60.0]


def test_retries_transport_error_with_configured_backoff(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    wikidata_payload: dict[str, Any],
) -> None:
    _contact(monkeypatch)
    delays: list[float] = []
    attempts = 0
    monkeypatch.setattr(time, "sleep", delays.append)

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise httpx.ConnectError("temporary", request=request)
        return _json_response(wikidata_payload)

    with _client(handler) as client:
        query_dwarfs(load_config().data, tmp_path, client=client)

    assert attempts == 2
    assert delays == [1.0]


@pytest.mark.parametrize(
    ("response", "message"),
    [
        (httpx.Response(400, text="bad request"), "HTTP 400"),
        (httpx.Response(200, content=b"not-json"), "malformed JSON"),
        (httpx.Response(200, json={"bad": "shape"}), "invalid SPARQL payload"),
    ],
)
def test_reports_permanent_or_malformed_responses(
    response: httpx.Response,
    message: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _contact(monkeypatch)
    with (
        _client(lambda request: response) as client,
        pytest.raises(WikidataQueryError, match=message),
    ):
        query_dwarfs(load_config().data, tmp_path, client=client)


def test_cli_reads_cache_prints_summary_and_writes_limited_output(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    wikidata_payload: dict[str, Any],
) -> None:
    _contact(monkeypatch)
    with _successful_client(wikidata_payload) as client:
        query_dwarfs(load_config().data, tmp_path, client=client)
    monkeypatch.delenv(USER_AGENT_ENV_VAR)

    result = runner.invoke(
        app,
        [
            "data",
            "query",
            "--limit",
            "2",
            "--override",
            f"paths.discovery_dir={tmp_path}",
            "--override",
            "logging.json_output=false",
        ],
    )

    assert result.exit_code == 0
    assert "cache=hit eligible=5 emitted=2" in result.output
    assert f"Records: {tmp_path / 'dwarfs.json'}" in result.output
    discovery = DwarfDiscoveryFile.model_validate_json((tmp_path / "dwarfs.json").read_text())
    assert len(discovery.records) == 2


def test_cli_validates_limit_and_missing_contact(tmp_path: Path) -> None:
    invalid_limit = runner.invoke(app, ["data", "query", "--limit", "0"])
    missing_contact = runner.invoke(
        app,
        ["data", "query", "--override", f"paths.discovery_dir={tmp_path}"],
        env={USER_AGENT_ENV_VAR: ""},
    )

    assert invalid_limit.exit_code == 2
    assert missing_contact.exit_code == 2
    assert "Configuration error" in missing_contact.output


def test_cli_maps_query_failure_to_exit_one(monkeypatch: pytest.MonkeyPatch) -> None:
    def fail_query(*args: object, **kwargs: object) -> None:
        raise WikidataQueryError("planned failure")

    monkeypatch.setattr("krasnal_id.cli.query_dwarfs", fail_query)
    result = runner.invoke(app, ["data", "query"])

    assert result.exit_code == 1
    assert "Query failed: planned failure" in result.output


@pytest.mark.integration
def test_live_wikidata_query_is_opt_in(tmp_path: Path) -> None:
    if os.environ.get("RUN_LIVE_WIKIDATA_TESTS") != "1" or not os.environ.get(USER_AGENT_ENV_VAR):
        pytest.skip("set RUN_LIVE_WIKIDATA_TESTS=1 and KRASNAL_ID_USER_AGENT to run")

    result = query_dwarfs(load_config().data, tmp_path, limit=3, refresh=True)

    assert result.records
    assert result.eligible_total >= len(result.records)

"""Discover Wroclaw dwarf entities and optional coordinates through Wikidata."""

import hashlib
import json
import logging
import os
import re
import time
from dataclasses import dataclass, field
from datetime import UTC, datetime
from email.utils import parsedate_to_datetime
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Final, Literal, cast

import httpx
from pydantic import BaseModel, ConfigDict, Field, HttpUrl, ValidationError

from krasnal_id.config import WikimediaDataConfig
from krasnal_id.models import (
    AuditDisposition,
    AuditReason,
    Coordinates,
    DiscoveryAuditFile,
    DiscoveryAuditRecord,
    DiscoveryResult,
    DwarfDiscoveryFile,
    DwarfRecord,
    QueryCacheMetadata,
)

LOGGER = logging.getLogger(__name__)

SCHEMA_VERSION: Final = "1.0"
USER_AGENT_ENV_VAR: Final = "KRASNAL_ID_USER_AGENT"
RAW_RESPONSE_FILENAME: Final = "wikidata-response.json"
CACHE_METADATA_FILENAME: Final = "wikidata-response.meta.json"
DWARFS_FILENAME: Final = "dwarfs.json"
AUDIT_FILENAME: Final = "audit.json"
RETRYABLE_STATUS_CODES: Final = frozenset({429, 502, 503, 504})

SPARQL_QUERY: Final = """\
PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
PREFIX wd: <http://www.wikidata.org/entity/>
PREFIX wdt: <http://www.wikidata.org/prop/direct/>

SELECT ?item ?labelPl ?labelEn ?commonsCategory ?coordinates ?member WHERE {
  ?item wdt:P31 wd:Q136276280 .
  OPTIONAL {
    ?item rdfs:label ?labelPl .
    FILTER(LANG(?labelPl) = "pl")
  }
  OPTIONAL {
    ?item rdfs:label ?labelEn .
    FILTER(LANG(?labelEn) = "en")
  }
  OPTIONAL { ?item wdt:P373 ?commonsCategory . }
  OPTIONAL { ?item wdt:P625 ?coordinates . }
  OPTIONAL { ?item wdt:P527 ?member . }
}
ORDER BY ?item ?member
"""
QUERY_SHA256: Final = hashlib.sha256(SPARQL_QUERY.encode("utf-8")).hexdigest()

_QID_URI_PATTERN = re.compile(r"^https?://www\.wikidata\.org/entity/(Q\d+)$")
_WKT_POINT_PATTERN = re.compile(
    r"^Point\(\s*"
    r"([-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?)"
    r"\s+"
    r"([-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?)"
    r"\s*\)$"
)
_CONTACT_PATTERN = re.compile(
    r"(?:https?://\S+|mailto:\S+|[^\s()<>]+@[^\s()<>]+\.[^\s()<>]+)",
    re.IGNORECASE,
)
_GROUP_PATTERN = re.compile(r"\b(?:dwarves|krasnale|krasnali)\b|\s(?:and|i)\s", re.IGNORECASE)


class WikidataQueryError(RuntimeError):
    """A live request, cache, or response could not be processed."""


class WikidataConfigurationError(WikidataQueryError):
    """Required runtime configuration is missing or invalid."""


class _SparqlValue(BaseModel):
    """One SPARQL JSON binding value."""

    model_config = ConfigDict(extra="ignore", frozen=True)

    value: str


class _SparqlBinding(BaseModel):
    """Fields selected by the dwarf discovery query."""

    model_config = ConfigDict(extra="ignore", frozen=True, populate_by_name=True)

    item: _SparqlValue | None = None
    label_pl: _SparqlValue | None = Field(default=None, alias="labelPl")
    label_en: _SparqlValue | None = Field(default=None, alias="labelEn")
    commons_category: _SparqlValue | None = Field(default=None, alias="commonsCategory")
    coordinates: _SparqlValue | None = None
    member: _SparqlValue | None = None


class _SparqlResults(BaseModel):
    """SPARQL result container."""

    model_config = ConfigDict(extra="ignore", frozen=True)

    bindings: tuple[_SparqlBinding, ...]


class _SparqlResponse(BaseModel):
    """Minimum valid shape of a Wikidata SPARQL response."""

    model_config = ConfigDict(extra="ignore", frozen=True)

    results: _SparqlResults


@dataclass(slots=True)
class _Candidate:
    """Merged values from every SPARQL row for one Wikidata item."""

    dwarf_id: str
    labels_pl: set[str] = field(default_factory=set)
    labels_en: set[str] = field(default_factory=set)
    commons_categories: set[str] = field(default_factory=set)
    coordinates: set[str] = field(default_factory=set)
    members: set[str] = field(default_factory=set)


@dataclass(frozen=True, slots=True)
class DiscoveryPaths:
    """All generated paths owned by the discovery stage."""

    raw_response: Path
    cache_metadata: Path
    dwarfs: Path
    audit: Path


def discovery_paths(discovery_dir: Path) -> DiscoveryPaths:
    """Return stable artifact paths below a configured discovery directory."""
    return DiscoveryPaths(
        raw_response=discovery_dir / RAW_RESPONSE_FILENAME,
        cache_metadata=discovery_dir / CACHE_METADATA_FILENAME,
        dwarfs=discovery_dir / DWARFS_FILENAME,
        audit=discovery_dir / AUDIT_FILENAME,
    )


def _qid_from_uri(uri: str) -> str | None:
    match = _QID_URI_PATTERN.fullmatch(uri.strip())
    return match.group(1) if match else None


def _qid_sort_key(dwarf_id: str) -> tuple[int, int | str]:
    if dwarf_id.startswith("Q") and dwarf_id[1:].isdigit():
        return (0, int(dwarf_id[1:]))
    return (1, dwarf_id)


def _audit_sort_key(record: DiscoveryAuditRecord) -> tuple[int, int | str, str, str]:
    qid_group, qid_value = _qid_sort_key(record.dwarf_id)
    return (qid_group, qid_value, record.disposition.value, record.reason.value)


def _parse_coordinates(value: str) -> Coordinates:
    match = _WKT_POINT_PATTERN.fullmatch(value.strip())
    if match is None:
        raise ValueError(f"unsupported coordinate value: {value!r}")
    longitude, latitude = (float(component) for component in match.groups())
    return Coordinates(latitude=latitude, longitude=longitude)


def _looks_like_group(record: DwarfRecord) -> bool:
    searchable = f"{record.display_name} {record.commons_category}"
    return _GROUP_PATTERN.search(searchable) is not None


def _audit(
    dwarf_id: str,
    disposition: AuditDisposition,
    reason: AuditReason,
    details: str,
) -> DiscoveryAuditRecord:
    return DiscoveryAuditRecord(
        dwarf_id=dwarf_id,
        disposition=disposition,
        reason=reason,
        details=details,
    )


def _merge_bindings(
    response: _SparqlResponse,
) -> tuple[dict[str, _Candidate], list[DiscoveryAuditRecord]]:
    candidates: dict[str, _Candidate] = {}
    audit: list[DiscoveryAuditRecord] = []

    for binding in response.results.bindings:
        if binding.item is None:
            audit.append(
                _audit(
                    "<unknown>",
                    AuditDisposition.EXCLUDED,
                    AuditReason.INVALID_RECORD,
                    "SPARQL row is missing its item binding.",
                )
            )
            continue

        dwarf_id = _qid_from_uri(binding.item.value)
        if dwarf_id is None:
            audit.append(
                _audit(
                    binding.item.value,
                    AuditDisposition.EXCLUDED,
                    AuditReason.INVALID_RECORD,
                    "Item URI is not a canonical Wikidata entity URI.",
                )
            )
            continue

        candidate = candidates.setdefault(dwarf_id, _Candidate(dwarf_id=dwarf_id))
        if binding.label_pl is not None and binding.label_pl.value.strip():
            candidate.labels_pl.add(binding.label_pl.value.strip())
        if binding.label_en is not None and binding.label_en.value.strip():
            candidate.labels_en.add(binding.label_en.value.strip())
        if binding.commons_category is not None and binding.commons_category.value.strip():
            candidate.commons_categories.add(binding.commons_category.value.strip())
        if binding.coordinates is not None and binding.coordinates.value.strip():
            candidate.coordinates.add(binding.coordinates.value.strip())
        if binding.member is not None:
            member_id = _qid_from_uri(binding.member.value)
            if member_id is not None:
                candidate.members.add(member_id)

    return candidates, audit


def _conflicting_fields(candidate: _Candidate) -> tuple[str, ...]:
    fields: list[str] = []
    if len(candidate.labels_pl) > 1:
        fields.append("Polish labels")
    if len(candidate.labels_en) > 1:
        fields.append("English labels")
    if len(candidate.commons_categories) > 1:
        fields.append("Commons categories")
    if len(candidate.coordinates) > 1:
        fields.append("coordinates")
    return tuple(fields)


def _candidate_record(
    candidate: _Candidate,
) -> tuple[DwarfRecord | None, DiscoveryAuditRecord | None]:
    if not candidate.commons_categories:
        return None, _audit(
            candidate.dwarf_id,
            AuditDisposition.EXCLUDED,
            AuditReason.MISSING_COMMONS_CATEGORY,
            "No P373 Wikimedia Commons category is present.",
        )

    conflicts = _conflicting_fields(candidate)
    if conflicts:
        return None, _audit(
            candidate.dwarf_id,
            AuditDisposition.EXCLUDED,
            AuditReason.CONFLICTING_SOURCE_VALUES,
            f"Multiple truthy values found for: {', '.join(conflicts)}.",
        )

    try:
        coordinates = (
            _parse_coordinates(next(iter(candidate.coordinates))) if candidate.coordinates else None
        )
        display_name = (
            next(iter(candidate.labels_pl))
            if candidate.labels_pl
            else next(iter(candidate.labels_en), candidate.dwarf_id)
        )
        return (
            DwarfRecord(
                dwarf_id=candidate.dwarf_id,
                display_name=display_name,
                wikidata_url=HttpUrl(f"https://www.wikidata.org/wiki/{candidate.dwarf_id}"),
                commons_category=next(iter(candidate.commons_categories)),
                coordinates=coordinates,
            ),
            None,
        )
    except (ValidationError, ValueError) as error:
        return None, _audit(
            candidate.dwarf_id,
            AuditDisposition.EXCLUDED,
            AuditReason.INVALID_RECORD,
            f"Record validation failed: {error}",
        )


def normalize_query_response(
    payload: object,
) -> tuple[tuple[DwarfRecord, ...], tuple[DiscoveryAuditRecord, ...]]:
    """Normalize a raw SPARQL payload into eligible records and audit decisions."""
    try:
        response = _SparqlResponse.model_validate(payload)
    except ValidationError as error:
        raise WikidataQueryError(f"Invalid SPARQL response shape: {error}") from error

    candidates, audit = _merge_bindings(response)
    preliminary: dict[str, DwarfRecord] = {}
    for dwarf_id in sorted(candidates, key=_qid_sort_key):
        record, exclusion = _candidate_record(candidates[dwarf_id])
        if exclusion is not None:
            audit.append(exclusion)
        elif record is not None:
            preliminary[dwarf_id] = record

    accepted: list[DwarfRecord] = []
    eligible_ids = set(preliminary)
    for dwarf_id in sorted(preliminary, key=_qid_sort_key):
        record = preliminary[dwarf_id]
        candidate = candidates[dwarf_id]
        eligible_members = candidate.members.intersection(eligible_ids)
        if eligible_members:
            members = ", ".join(sorted(eligible_members, key=_qid_sort_key))
            audit.append(
                _audit(
                    dwarf_id,
                    AuditDisposition.EXCLUDED,
                    AuditReason.EXPLICIT_GROUP_ENTITY,
                    f"Explicit P527 members are independently eligible: {members}.",
                )
            )
            continue

        accepted.append(record)
        if candidate.members or _looks_like_group(record):
            audit.append(
                _audit(
                    dwarf_id,
                    AuditDisposition.WARNING,
                    AuditReason.POSSIBLE_UNLINKED_GROUP,
                    "Record looks like a group but has no independently eligible linked members.",
                )
            )

    return tuple(accepted), tuple(sorted(audit, key=_audit_sort_key))


def _contact_user_agent() -> str:
    user_agent = os.environ.get(USER_AGENT_ENV_VAR, "").strip()
    if not user_agent or _CONTACT_PATTERN.search(user_agent) is None:
        raise WikidataConfigurationError(
            f"{USER_AGENT_ENV_VAR} must contain a contact email, mailto URI, or HTTP(S) URL."
        )
    return user_agent


def _retry_after_seconds(response: httpx.Response, maximum: float) -> float | None:
    value = response.headers.get("Retry-After")
    if value is None:
        return None
    try:
        delay = float(value)
    except ValueError:
        try:
            retry_at = parsedate_to_datetime(value)
            if retry_at.tzinfo is None:
                retry_at = retry_at.replace(tzinfo=UTC)
            delay = (retry_at - datetime.now(UTC)).total_seconds()
        except (TypeError, ValueError, OverflowError):
            return None
    return min(max(delay, 0.0), maximum)


def _request_payload(
    config: WikimediaDataConfig,
    user_agent: str,
    client: httpx.Client,
) -> object:
    for attempt in range(config.max_attempts):
        try:
            response = client.post(
                str(config.wikidata_endpoint),
                data={"query": SPARQL_QUERY, "format": "json"},
                headers={
                    "Accept": "application/sparql-results+json",
                    "User-Agent": user_agent,
                },
                timeout=config.request_timeout_seconds,
            )
        except (httpx.TimeoutException, httpx.TransportError) as error:
            if attempt + 1 >= config.max_attempts:
                raise WikidataQueryError(
                    f"Wikidata request failed after {config.max_attempts} attempts."
                ) from error
            time.sleep(config.retry_backoff_seconds[attempt])
            continue

        if response.status_code in RETRYABLE_STATUS_CODES and attempt + 1 < config.max_attempts:
            delay = _retry_after_seconds(response, config.max_retry_after_seconds)
            time.sleep(delay if delay is not None else config.retry_backoff_seconds[attempt])
            continue

        try:
            response.raise_for_status()
        except httpx.HTTPStatusError as error:
            raise WikidataQueryError(f"Wikidata returned HTTP {response.status_code}.") from error

        try:
            payload = cast(object, response.json())
        except ValueError as error:
            raise WikidataQueryError("Wikidata returned malformed JSON.") from error

        try:
            _SparqlResponse.model_validate(payload)
        except ValidationError as error:
            raise WikidataQueryError(
                f"Wikidata returned an invalid SPARQL payload: {error}"
            ) from error
        return payload

    raise WikidataQueryError("Wikidata request exhausted its retry policy.")


def _fetch_payload(
    config: WikimediaDataConfig,
    user_agent: str,
    client: httpx.Client | None,
) -> object:
    if client is not None:
        return _request_payload(config, user_agent, client)
    with httpx.Client() as owned_client:
        return _request_payload(config, user_agent, owned_client)


def _read_json(path: Path) -> object:
    with path.open(encoding="utf-8") as handle:
        return cast(object, json.load(handle))


def _load_cached_payload(
    paths: DiscoveryPaths,
    config: WikimediaDataConfig,
) -> object | None:
    try:
        metadata = QueryCacheMetadata.model_validate(_read_json(paths.cache_metadata))
        payload = _read_json(paths.raw_response)
        _SparqlResponse.model_validate(payload)
    except (OSError, json.JSONDecodeError, ValidationError):
        return None

    if metadata.query_sha256 != QUERY_SHA256:
        return None
    if str(metadata.endpoint) != str(config.wikidata_endpoint):
        return None
    return payload


def _atomic_write_json(path: Path, payload: object) -> None:
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
            json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        temporary_path.replace(path)
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


def _write_cache(
    paths: DiscoveryPaths,
    config: WikimediaDataConfig,
    payload: object,
) -> None:
    metadata = QueryCacheMetadata(
        schema_version=SCHEMA_VERSION,
        endpoint=config.wikidata_endpoint,
        query_sha256=QUERY_SHA256,
        retrieved_at=datetime.now(UTC),
    )
    _atomic_write_json(paths.raw_response, payload)
    _atomic_write_json(paths.cache_metadata, metadata.model_dump(mode="json"))


def _write_normalized_outputs(
    paths: DiscoveryPaths,
    records: tuple[DwarfRecord, ...],
    audit: tuple[DiscoveryAuditRecord, ...],
    eligible_total: int,
    limit: int | None,
) -> None:
    discovery = DwarfDiscoveryFile(
        schema_version=SCHEMA_VERSION,
        query_sha256=QUERY_SHA256,
        selection_limit=limit,
        eligible_total=eligible_total,
        records=records,
    )
    audit_file = DiscoveryAuditFile(
        schema_version=SCHEMA_VERSION,
        query_sha256=QUERY_SHA256,
        records=audit,
    )
    _atomic_write_json(paths.dwarfs, discovery.model_dump(mode="json"))
    _atomic_write_json(paths.audit, audit_file.model_dump(mode="json"))


def query_dwarfs(
    config: WikimediaDataConfig,
    discovery_dir: Path,
    *,
    limit: int | None = None,
    refresh: bool = False,
    client: httpx.Client | None = None,
) -> DiscoveryResult:
    """Discover, validate, cache, and persist normalized Wikidata dwarf records."""
    if limit is not None and limit <= 0:
        raise WikidataConfigurationError("limit must be a positive integer")

    paths = discovery_paths(discovery_dir)
    cache_artifacts_exist = paths.raw_response.exists() or paths.cache_metadata.exists()
    cached_payload = None if refresh else _load_cached_payload(paths, config)
    cache_status: Literal["hit", "fetched", "refreshed", "recovered"]

    if cached_payload is not None:
        payload = cached_payload
        cache_status = "hit"
    else:
        user_agent = _contact_user_agent()
        payload = _fetch_payload(config, user_agent, client)
        _write_cache(paths, config, payload)
        if refresh:
            cache_status = "refreshed"
        elif cache_artifacts_exist:
            cache_status = "recovered"
        else:
            cache_status = "fetched"

    eligible_records, audit = normalize_query_response(payload)
    eligible_total = len(eligible_records)
    selected_records = eligible_records if limit is None else eligible_records[:limit]
    _write_normalized_outputs(
        paths,
        selected_records,
        audit,
        eligible_total,
        limit,
    )

    LOGGER.info(
        "wikidata discovery completed",
        extra={
            "cache_status": cache_status,
            "eligible_total": eligible_total,
            "emitted_total": len(selected_records),
            "audit_total": len(audit),
        },
    )
    return DiscoveryResult(
        records=selected_records,
        audit=audit,
        eligible_total=eligible_total,
        cache_status=cache_status,
    )

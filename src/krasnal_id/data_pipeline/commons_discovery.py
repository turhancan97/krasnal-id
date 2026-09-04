"""Commons-first discovery of per-dwarf categories Wikidata has no item for.

Wikidata is the binding constraint on dataset size, not Commons: as recorded in
`AGENTS.md` section 5.6, only 44 Wikidata items exist for the 481 per-dwarf
categories Commons carries. This stage enumerates that category tree so a class
can enter the dataset without a Wikidata item, at the cost of having no
coordinates.

The enumeration is a second *source* for the one discovery artifact rather than a
second artifact. Every downstream stage validates against a single
`query_sha256`, so a parallel file with its own hash would fork the provenance
chain that section 5.5 depends on.
"""

import hashlib
import json
import logging
import re
import unicodedata
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Final, Literal, cast

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from krasnal_id.config import WikimediaDataConfig
from krasnal_id.models import (
    AuditDisposition,
    AuditReason,
    DiscoveryAuditRecord,
    DwarfRecord,
    QueryCacheMetadata,
)

LOGGER = logging.getLogger(__name__)

SCHEMA_VERSION: Final = "1.0"
RAW_RESPONSE_FILENAME: Final = "commons-categories.json"
CACHE_METADATA_FILENAME: Final = "commons-categories.meta.json"
# The page size the API allows an anonymous client; the walk pages until exhausted.
CATEGORY_PAGE_LIMIT: Final = "500"
# Guards against an unbounded walk if the API keeps offering continuations.
MAX_CATEGORY_PAGES: Final = 20

# Commons uses every one of "dwarf", "dwarfs" and "dwarves", plus the
# undiacriticked "Wroclaw". Both plurals mark a group installation rather than a
# naming mistake, and missing "dwarves" costs 76 of the 481 categories their name.
_TITLE_PATTERN: Final = re.compile(
    r"^(?P<name>.+?)\s+dwar(?:f|fs|ves),\s*Wroc[lł]aw$",
    re.IGNORECASE,
)
# 'ł' has no NFKD decomposition, so folding it needs an explicit rule or the slug
# for "Wrocław" silently loses the letter instead of transliterating it.
_ASCII_FOLD: Final = str.maketrans({"ł": "l", "Ł": "L", "ø": "o", "Ø": "O", "đ": "d", "Đ": "D"})
_SLUG_SEPARATOR: Final = re.compile(r"[^a-z0-9]+")


# The caller owns transport, retries and headers, and owns how an artifact is
# written, so this stage stays offline-testable with no HTTP client of its own.
PageRequest = Callable[[dict[str, str]], object]
JsonWriter = Callable[[Path, object], None]


class CommonsDiscoveryError(RuntimeError):
    """Raised when Commons category enumeration fails or returns an unusable shape."""


class _CategoryMember(BaseModel):
    """One subcategory of the configured root category."""

    model_config = ConfigDict(extra="ignore", frozen=True)

    title: str = Field(min_length=1)


class _CategoryQuery(BaseModel):
    model_config = ConfigDict(extra="ignore", frozen=True)

    categorymembers: tuple[_CategoryMember, ...] = ()


class _CategoryResponse(BaseModel):
    """Minimum valid shape of one `list=categorymembers` response page."""

    model_config = ConfigDict(extra="ignore", frozen=True)

    query: _CategoryQuery = _CategoryQuery()
    cmcontinue: str | None = None

    @classmethod
    def from_payload(cls, payload: object) -> "_CategoryResponse":
        """Parse one page, lifting the continuation token out of its envelope."""
        if not isinstance(payload, dict):
            raise CommonsDiscoveryError("Commons category response was not a JSON object")
        if "error" in payload:
            raise CommonsDiscoveryError(f"Commons API error: {payload['error']}")
        envelope = payload.get("continue")
        token = envelope.get("cmcontinue") if isinstance(envelope, dict) else None
        try:
            return cls.model_validate({**payload, "cmcontinue": token})
        except ValidationError as error:
            raise CommonsDiscoveryError(f"invalid Commons category response: {error}") from error


@dataclass(frozen=True, slots=True)
class CommonsDiscoveryPaths:
    """Generated cache artifacts owned by Commons enumeration."""

    raw_response: Path
    cache_metadata: Path


class CommonsDiscoveryResult(BaseModel):
    """In-memory result of one Commons category enumeration."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    records: tuple[DwarfRecord, ...]
    audit: tuple[DiscoveryAuditRecord, ...]
    cache_status: Literal["hit", "fetched", "refreshed", "recovered"]


def commons_discovery_paths(discovery_dir: Path) -> CommonsDiscoveryPaths:
    """Return the stable cache paths for Commons enumeration."""
    return CommonsDiscoveryPaths(
        raw_response=discovery_dir / RAW_RESPONSE_FILENAME,
        cache_metadata=discovery_dir / CACHE_METADATA_FILENAME,
    )


def request_parameters(root_category: str, continuation: str | None) -> dict[str, str]:
    """Build the query parameters for one page of subcategories."""
    parameters = {
        "action": "query",
        "format": "json",
        "formatversion": "1",
        "list": "categorymembers",
        "cmtitle": root_category,
        "cmtype": "subcat",
        "cmlimit": CATEGORY_PAGE_LIMIT,
    }
    if continuation is not None:
        parameters["cmcontinue"] = continuation
    return parameters


def query_sha256(root_category: str) -> str:
    """Hash the enumeration's identity so a cached page set can be validated.

    The root category is part of the identity: pointing the walk at a different
    tree is a different query, and a cache built for one must not satisfy the other.
    """
    identity = json.dumps(
        request_parameters(root_category, None), ensure_ascii=False, sort_keys=True
    )
    return hashlib.sha256(identity.encode("utf-8")).hexdigest()


def category_slug(title: str) -> str:
    """Derive a stable, filesystem-safe identifier suffix from a category title."""
    folded = unicodedata.normalize("NFKD", title.translate(_ASCII_FOLD))
    ascii_only = "".join(character for character in folded if not unicodedata.combining(character))
    slug = _SLUG_SEPARATOR.sub("-", ascii_only.encode("ascii", "ignore").decode().lower()).strip(
        "-"
    )
    if not slug:
        raise CommonsDiscoveryError(f"category {title!r} has no ASCII characters to slug")
    return slug


def display_name_for(title: str) -> str | None:
    """Return the statue's name, or None when the title does not follow the pattern."""
    matched = _TITLE_PATTERN.match(title)
    if matched is None:
        return None
    name = matched.group("name").strip()
    return name or None


def normalize_categories(
    titles: tuple[str, ...],
) -> tuple[tuple[DwarfRecord, ...], tuple[DiscoveryAuditRecord, ...]]:
    """Turn category titles into records, auditing anything a human should see.

    A title that does not follow the `<Name> dwarf, Wrocław` pattern still becomes
    a record, because the category-review step is where a mapping is accepted or
    rejected; it is only flagged, so the reviewer knows to look. A slug collision
    is different: two statues would share one identifier and one image directory,
    so the later title is excluded rather than silently merged.
    """
    records: list[DwarfRecord] = []
    audit: list[DiscoveryAuditRecord] = []
    claimed: dict[str, str] = {}

    for title in sorted(set(titles)):
        category = title.removeprefix("Category:").replace("_", " ").strip()
        if not category:
            continue
        name = display_name_for(category)
        slug = category_slug(category)
        dwarf_id = f"C-{slug}"

        if slug in claimed:
            audit.append(
                DiscoveryAuditRecord(
                    dwarf_id=dwarf_id,
                    disposition=AuditDisposition.EXCLUDED,
                    reason=AuditReason.DUPLICATE_CATEGORY_SLUG,
                    details=(
                        f"category {category!r} slugs to {slug!r}, already taken by "
                        f"{claimed[slug]!r}; resolve the collision before including either"
                    ),
                )
            )
            continue
        claimed[slug] = category

        if name is None:
            audit.append(
                DiscoveryAuditRecord(
                    dwarf_id=dwarf_id,
                    disposition=AuditDisposition.WARNING,
                    reason=AuditReason.UNEXPECTED_CATEGORY_NAME,
                    details=(
                        f"category {category!r} does not follow '<Name> dwarf, Wroclaw'; "
                        "its display name is the full title until review corrects it"
                    ),
                )
            )
        records.append(
            DwarfRecord(
                dwarf_id=dwarf_id,
                display_name=name or category,
                commons_category=category,
            )
        )

    return tuple(records), tuple(audit)


def _read_json(path: Path) -> object:
    with path.open(encoding="utf-8") as handle:
        return cast(object, json.load(handle))


def _load_cached_pages(
    paths: CommonsDiscoveryPaths,
    config: WikimediaDataConfig,
) -> tuple[object, ...] | None:
    """Return cached response pages only when they describe this exact query."""
    try:
        metadata = QueryCacheMetadata.model_validate(_read_json(paths.cache_metadata))
        payload = _read_json(paths.raw_response)
    except (OSError, json.JSONDecodeError, ValidationError):
        return None
    if metadata.query_sha256 != query_sha256(config.commons_root_category):
        return None
    if str(metadata.endpoint) != str(config.commons_api_endpoint):
        return None
    if not isinstance(payload, list) or not payload:
        return None
    try:
        for page in payload:
            _CategoryResponse.from_payload(page)
    except CommonsDiscoveryError:
        return None
    return tuple(cast(list[object], payload))


def _write_cache(
    paths: CommonsDiscoveryPaths,
    config: WikimediaDataConfig,
    pages: tuple[object, ...],
    write_json: JsonWriter,
) -> None:
    metadata = QueryCacheMetadata(
        schema_version=SCHEMA_VERSION,
        endpoint=config.commons_api_endpoint,
        query_sha256=query_sha256(config.commons_root_category),
        retrieved_at=datetime.now(UTC),
    )
    write_json(paths.raw_response, list(pages))
    write_json(paths.cache_metadata, metadata.model_dump(mode="json"))


def titles_from_pages(pages: tuple[object, ...]) -> tuple[str, ...]:
    """Collect every subcategory title across a cached or fetched page set."""
    titles: list[str] = []
    for page in pages:
        parsed = _CategoryResponse.from_payload(page)
        titles.extend(member.title for member in parsed.query.categorymembers)
    return tuple(titles)


def fetch_pages(
    config: WikimediaDataConfig,
    session: PageRequest,
) -> tuple[object, ...]:
    """Walk the root category's subcategories, following every continuation."""
    pages: list[object] = []
    continuation: str | None = None
    for _ in range(MAX_CATEGORY_PAGES):
        payload = session(request_parameters(config.commons_root_category, continuation))
        pages.append(payload)
        continuation = _CategoryResponse.from_payload(payload).cmcontinue
        if continuation is None:
            return tuple(pages)
    raise CommonsDiscoveryError(
        f"Commons category enumeration exceeded {MAX_CATEGORY_PAGES} pages; "
        f"is {config.commons_root_category!r} the intended root?"
    )


def discover_commons_categories(
    config: WikimediaDataConfig,
    discovery_dir: Path,
    *,
    session: PageRequest,
    write_json: JsonWriter,
    refresh: bool = False,
) -> CommonsDiscoveryResult:
    """Enumerate, cache, and normalize the Commons per-dwarf category tree."""
    paths = commons_discovery_paths(discovery_dir)
    cache_artifacts_exist = paths.raw_response.exists() or paths.cache_metadata.exists()
    cached = None if refresh else _load_cached_pages(paths, config)
    cache_status: Literal["hit", "fetched", "refreshed", "recovered"]

    if cached is not None:
        pages = cached
        cache_status = "hit"
    else:
        pages = fetch_pages(config, session)
        _write_cache(paths, config, pages, write_json)
        if refresh:
            cache_status = "refreshed"
        elif cache_artifacts_exist:
            cache_status = "recovered"
        else:
            cache_status = "fetched"

    records, audit = normalize_categories(titles_from_pages(pages))
    LOGGER.info(
        "commons discovery completed",
        extra={
            "cache_status": cache_status,
            "pages": len(pages),
            "emitted_total": len(records),
            "audit_total": len(audit),
        },
    )
    return CommonsDiscoveryResult(records=records, audit=audit, cache_status=cache_status)


def merge_discovery_records(
    wikidata: tuple[DwarfRecord, ...],
    commons: tuple[DwarfRecord, ...],
    commons_audit: tuple[DiscoveryAuditRecord, ...] = (),
) -> tuple[tuple[DwarfRecord, ...], tuple[DiscoveryAuditRecord, ...]]:
    """Combine both sources, letting a Wikidata record win any category it claims.

    A Wikidata record carries a QID and P625 coordinates, so it is strictly more
    informative than the Commons category describing the same statue. Every
    superseded category is audited rather than dropped quietly, because the count
    of them is how a reader sees how little of the pool Wikidata covers.

    The match is exact category equality, which is all it can honestly be: a statue
    filed under a Commons category that Wikidata's `P373` does not name appears
    twice, once from each source, and only category review can catch it. *Papa
    Krasnal* is a live example, whose Wikidata item points at its sculptor's
    category instead.

    Normalization warnings about a category the merge then supersedes are dropped,
    because a reviewer reading 481 categories should not be sent to look at records
    that never reached the artifact.
    """
    claimed = {record.commons_category.replace("_", " ").strip() for record in wikidata}
    kept: list[DwarfRecord] = list(wikidata)
    superseded: set[str] = set()
    audit: list[DiscoveryAuditRecord] = []

    for record in commons:
        if record.commons_category in claimed:
            superseded.add(record.dwarf_id)
            audit.append(
                DiscoveryAuditRecord(
                    dwarf_id=record.dwarf_id,
                    disposition=AuditDisposition.WARNING,
                    reason=AuditReason.CLAIMED_BY_WIKIDATA,
                    details=(
                        f"category {record.commons_category!r} already belongs to a Wikidata "
                        "record, which is preferred because it carries a QID and coordinates"
                    ),
                )
            )
            continue
        kept.append(record)

    identifiers = [record.dwarf_id for record in kept]
    if len(identifiers) != len(set(identifiers)):
        raise CommonsDiscoveryError("merged discovery produced duplicate dwarf identifiers")

    # Exclusions survive regardless: they explain why something is absent, which
    # stays relevant however the merge went.
    surviving = tuple(
        record
        for record in commons_audit
        if record.disposition is not AuditDisposition.WARNING or record.dwarf_id not in superseded
    )
    return tuple(kept), surviving + tuple(audit)

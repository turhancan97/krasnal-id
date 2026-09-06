"""The Commons licence template behind each public-domain file.

The fetcher records a Commons `LicenseShortName`, so 52 files arrive labelled
"Public domain" with the CC Public Domain Mark as their licence URL. PDM is a
*label*, not a licence: it asserts that something is free of known copyright
without saying why. The actual basis — the uploader's own release, an expired
term, a Polish statutory exemption — lives in a template on the file page and was
discarded.

That matters at the point of redistribution. Every other row in the dataset can
point at a licence that grants what it grants; these rows would point at a label.
Fifty-two files is small enough that "we did not check" is a choice rather than a
constraint, so this asks Commons what the basis actually is.

Like the camera metadata of `AGENTS.md` section 5.9, this sits **outside the
staging chain**: it is read by one consumer, it does not build the dataset, and
adding it to `fetched-images.json` would invalidate the manifest, the split,
twelve result artifacts and the published demo.
"""

import json
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field, HttpUrl, ValidationError

from krasnal_id.config import WikimediaDataConfig
from krasnal_id.models import DatasetManifest

LICENSE_TEMPLATE_FILENAME = "license-templates.json"
SCHEMA_VERSION = "1.0"
# The API accepts fifty page IDs per request.
PAGE_ID_BATCH = 50
# Commons short names whose basis is a template rather than a licence grant. CC0
# is a real dedication and needs no lookup; "Public domain" is the bare label.
UNGRANTED_LICENSES = frozenset({"Public domain"})
# Template titles arrive namespaced. Anything under these prefixes states a
# public-domain basis; everything else on a file page is layout and maintenance.
BASIS_PREFIXES = ("PD-", "Public domain", "CC-zero", "Copyrighted free use")


class LicenseTemplateError(RuntimeError):
    """Raised when licence templates cannot be retrieved or are unusable."""


class LicenseTemplateFile(BaseModel):
    """The public-domain basis per Commons page, with staleness provenance."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: str = Field(pattern=r"^\d+\.\d+$")
    endpoint: HttpUrl
    retrieved_at: datetime
    # Page IDs with no basis template map to an empty tuple, so a missing key
    # means "never asked" while an empty list means "asked, and Commons has none".
    templates: dict[str, tuple[str, ...]]


def license_template_path(discovery_dir: Path) -> Path:
    """Return the artifact path for the licence templates."""
    return discovery_dir / LICENSE_TEMPLATE_FILENAME


def ungranted_page_ids(manifest: DatasetManifest) -> tuple[int, ...]:
    """Return the page IDs whose recorded licence is a label, not a grant."""
    return tuple(
        sorted(
            {
                image.commons_page_id
                for image in manifest.images
                if image.commons_page_id and image.license in UNGRANTED_LICENSES
            }
        )
    )


def request_parameters(page_ids: tuple[int, ...]) -> dict[str, str]:
    """Build the parameters for one batch of page IDs.

    `tlnamespace=10` restricts the reply to the Template namespace, and the limit
    is raised because a Commons file page carries dozens of them.
    """
    return {
        "action": "query",
        "format": "json",
        "formatversion": "2",
        "prop": "templates",
        "tlnamespace": "10",
        "tllimit": "500",
        "pageids": "|".join(str(page_id) for page_id in page_ids),
    }


def basis_templates(page: object) -> tuple[int, tuple[str, ...]] | None:
    """Extract the public-domain basis templates named on one file page."""
    if not isinstance(page, dict) or "pageid" not in page:
        return None
    page_id = page["pageid"]
    if not isinstance(page_id, int):
        return None

    found: list[str] = []
    for template in page.get("templates") or []:
        if not isinstance(template, dict):
            continue
        title = str(template.get("title") or "")
        # "Template:PD-self" -> "PD-self"; a namespaceless title is left alone.
        _, _, name = title.partition(":")
        name = name or title
        if name.startswith(BASIS_PREFIXES):
            found.append(name)
    return page_id, tuple(sorted(set(found)))


def collect_templates(payloads: tuple[object, ...]) -> dict[str, tuple[str, ...]]:
    """Fold every response batch into one page-to-basis mapping."""
    templates: dict[str, tuple[str, ...]] = {}
    for payload in payloads:
        if not isinstance(payload, dict):
            raise LicenseTemplateError("Commons returned a response that was not an object")
        if "error" in payload:
            raise LicenseTemplateError(f"Commons API error: {payload['error']}")
        query = payload.get("query")
        pages = query.get("pages", []) if isinstance(query, dict) else []
        for page in pages:
            extracted = basis_templates(page)
            if extracted is not None:
                templates[str(extracted[0])] = extracted[1]
    return templates


def fetch_license_templates(
    manifest: DatasetManifest,
    config: WikimediaDataConfig,
    session: Callable[[dict[str, str]], object],
) -> LicenseTemplateFile:
    """Retrieve the basis templates for every label-licensed manifest image."""
    page_ids = ungranted_page_ids(manifest)
    if not page_ids:
        raise LicenseTemplateError(
            "no manifest image carries a label-only licence, so there is no basis to look up"
        )

    payloads = tuple(
        session(request_parameters(page_ids[start : start + PAGE_ID_BATCH]))
        for start in range(0, len(page_ids), PAGE_ID_BATCH)
    )
    return LicenseTemplateFile(
        schema_version=SCHEMA_VERSION,
        endpoint=config.commons_api_endpoint,
        retrieved_at=datetime.now(UTC),
        templates=collect_templates(payloads),
    )


def load_license_templates(path: Path) -> LicenseTemplateFile:
    """Read a licence template artifact, refusing anything malformed."""
    try:
        return LicenseTemplateFile.model_validate(json.loads(path.read_text(encoding="utf-8")))
    except (OSError, json.JSONDecodeError, ValidationError) as error:
        raise LicenseTemplateError(
            f"invalid licence templates {path}: {error}; rebuild it with "
            "krasnal-id data license-templates"
        ) from error

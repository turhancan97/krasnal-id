"""The public-domain basis behind the files Commons labels rather than licenses."""

import json
from datetime import UTC, datetime
from pathlib import Path

import pytest
from typer.testing import CliRunner

from helpers import synthetic_manifest
from krasnal_id.cli import app
from krasnal_id.config import load_config
from krasnal_id.data_pipeline.license_templates import (
    LicenseTemplateError,
    LicenseTemplateFile,
    basis_templates,
    collect_templates,
    fetch_license_templates,
    license_template_path,
    load_license_templates,
    request_parameters,
    ungranted_page_ids,
)
from krasnal_id.models import DatasetManifest

runner = CliRunner()


def _labelled(page_ids: tuple[int, ...], license_name: str = "Public domain") -> DatasetManifest:
    """A manifest whose images carry a licence label rather than a grant."""
    manifest = synthetic_manifest(dwarf_count=3, per_dwarf=3)
    images = tuple(
        image.model_copy(
            update={
                "commons_page_id": page_ids[index] if index < len(page_ids) else None,
                "license": license_name if index < len(page_ids) else "CC BY-SA 4.0",
            }
        )
        for index, image in enumerate(manifest.images)
    )
    return manifest.model_copy(update={"images": images})


def test_only_label_licensed_pages_are_looked_up() -> None:
    """CC BY-SA grants what it grants; only the bare label needs a basis."""
    manifest = _labelled((11, 12))

    assert ungranted_page_ids(manifest) == (11, 12)
    assert ungranted_page_ids(_labelled((11,), license_name="CC BY-SA 4.0")) == ()


def test_a_page_yields_only_its_public_domain_templates() -> None:
    page = {
        "pageid": 7,
        "templates": [
            {"title": "Template:PD-self"},
            {"title": "Template:Information"},
            {"title": "Template:Wrocław"},
        ],
    }

    assert basis_templates(page) == (7, ("PD-self",))
    # A page Commons has no basis template for is recorded as known-absent.
    assert basis_templates({"pageid": 8, "templates": []}) == (8, ())
    assert basis_templates({"pageid": 9}) == (9, ())
    assert basis_templates({"no": "pageid"}) is None
    assert basis_templates(["not a page"]) is None


def test_batches_are_folded_and_api_errors_surface() -> None:
    templates = collect_templates(
        (
            {"query": {"pages": [{"pageid": 1, "templates": [{"title": "Template:PD-old-70"}]}]}},
            {"query": {"pages": [{"pageid": 2, "templates": []}]}},
        )
    )

    assert templates == {"1": ("PD-old-70",), "2": ()}
    with pytest.raises(LicenseTemplateError, match="Commons API error"):
        collect_templates(({"error": {"code": "badpageid"}},))
    with pytest.raises(LicenseTemplateError, match="not an object"):
        collect_templates(([],))


def test_requests_ask_only_for_the_template_namespace() -> None:
    manifest = _labelled((3, 4))
    seen: list[dict[str, str]] = []

    def session(parameters: dict[str, str]) -> object:
        seen.append(parameters)
        return {
            "query": {
                "pages": [
                    {"pageid": int(i), "templates": [{"title": "Template:PD-Polish"}]}
                    for i in parameters["pageids"].split("|")
                ]
            }
        }

    result = fetch_license_templates(manifest, load_config().data, session)

    assert result.templates == {"3": ("PD-Polish",), "4": ("PD-Polish",)}
    assert seen[0]["tlnamespace"] == "10"
    assert set(request_parameters((1,))) >= {"action", "prop", "pageids", "tlnamespace"}
    with pytest.raises(LicenseTemplateError, match="no manifest image carries"):
        fetch_license_templates(
            _labelled((), license_name="CC BY-SA 4.0"), load_config().data, session
        )


def test_reading_rejects_a_malformed_artifact(tmp_path: Path) -> None:
    with pytest.raises(LicenseTemplateError, match="invalid licence templates"):
        load_license_templates(tmp_path / "absent.json")

    broken = tmp_path / "broken.json"
    broken.write_text("{}", encoding="utf-8")
    with pytest.raises(LicenseTemplateError, match="invalid licence templates"):
        load_license_templates(broken)

    good = tmp_path / "good.json"
    artifact = LicenseTemplateFile(
        schema_version="1.0",
        endpoint="https://commons.wikimedia.org/w/api.php",  # type: ignore[arg-type]
        retrieved_at=datetime.now(UTC),
        templates={"1": ("PD-self",)},
    )
    good.write_text(json.dumps(artifact.model_dump(mode="json")), encoding="utf-8")
    assert load_license_templates(good).templates == {"1": ("PD-self",)}


def test_the_path_sits_beside_the_other_discovery_artifacts(tmp_path: Path) -> None:
    assert license_template_path(tmp_path).name == "license-templates.json"


def test_cli_reports_a_missing_manifest(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("KRASNAL_ID_USER_AGENT", "krasnal-id/test (mailto:test@example.com)")

    result = runner.invoke(
        app,
        [
            "data",
            "license-templates",
            f"-opaths.manifest_path={tmp_path / 'absent.json'}",
            "-ologging.json_output=false",
        ],
    )

    assert result.exit_code == 2
    assert "Licence template error" in result.output

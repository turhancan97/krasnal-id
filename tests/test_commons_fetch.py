"""Tests for reviewed, cached Wikimedia Commons acquisition."""

import io
import json
import os
from collections.abc import Callable
from pathlib import Path
from typing import Any

import httpx
import pytest
from PIL import Image
from typer.testing import CliRunner

import krasnal_id.cli as cli_module
import krasnal_id.data_pipeline.commons_fetch as fetch_module
from krasnal_id.cli import app
from krasnal_id.config import load_config
from krasnal_id.data_pipeline.commons_fetch import (
    USER_AGENT_ENV_VAR,
    CommonsConfigurationError,
    commons_cache_paths,
    fetch_images,
    fetch_paths,
    prepare_category_review,
)
from krasnal_id.models import (
    AuditDisposition,
    AuditReason,
    CategoryReviewFile,
    CategoryReviewRecord,
    CategoryReviewStatus,
    DiscoveryAuditFile,
    DiscoveryAuditRecord,
    DwarfDiscoveryFile,
    DwarfRecord,
    FetchAuditDisposition,
    FetchAuditReason,
    FetchResult,
)

runner = CliRunner()
QUERY_HASH = "1" * 64


def _dump(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, sort_keys=True) + "\n", encoding="utf-8")


def _dwarf(qid: str, category: str | None = None) -> DwarfRecord:
    return DwarfRecord(
        dwarf_id=qid,
        display_name=f"Dwarf {qid}",
        wikidata_url=f"https://www.wikidata.org/wiki/{qid}",
        commons_category=category or f"Category {qid}",
    )


def _discovery(
    directory: Path,
    dwarfs: tuple[DwarfRecord, ...],
    warnings: tuple[str, ...] = (),
) -> None:
    records = DwarfDiscoveryFile(
        schema_version="1.0",
        query_sha256=QUERY_HASH,
        eligible_total=len(dwarfs),
        records=dwarfs,
    )
    audit = DiscoveryAuditFile(
        schema_version="1.0",
        query_sha256=QUERY_HASH,
        records=tuple(
            DiscoveryAuditRecord(
                dwarf_id=qid,
                disposition=AuditDisposition.WARNING,
                reason=AuditReason.POSSIBLE_UNLINKED_GROUP,
                details="Review possible group.",
            )
            for qid in warnings
        ),
    )
    _dump(directory / "dwarfs.json", records.model_dump(mode="json"))
    _dump(directory / "audit.json", audit.model_dump(mode="json"))


def _decision(
    dwarf: DwarfRecord,
    status: CategoryReviewStatus = CategoryReviewStatus.APPROVED,
    corrected: str | None = None,
) -> CategoryReviewRecord:
    return CategoryReviewRecord(
        dwarf_id=dwarf.dwarf_id,
        display_name=dwarf.display_name,
        discovered_category=dwarf.commons_category,
        status=status,
        corrected_category=corrected,
    )


def _review(path: Path, *records: CategoryReviewRecord) -> None:
    model = CategoryReviewFile(schema_version="1.0", records=records)
    _dump(path, model.model_dump(mode="json"))


def _value(value: str | int | float | bool | None) -> dict[str, object]:
    return {"value": value}


def _page(
    page_id: int,
    *,
    mime: str = "image/png",
    width: int = 900,
    height: int = 700,
    author: str | None = "<a>Alice &amp; Bob</a>",
    license_token: str = "cc-by-sa-4.0",
    license_name: str = "CC BY-SA 4.0",
    license_url: str | None = "https://creativecommons.org/licenses/by-sa/4.0/",
    sha: str = "a",
) -> dict[str, Any]:
    metadata: dict[str, object] = {
        "AuthorCount": _value(2),
        "CommonsMetadataExtension": _value(1.2),
        "License": _value(license_token),
        "LicenseShortName": _value(license_name),
    }
    if author is not None:
        metadata["Artist"] = _value(author)
    if license_url is not None:
        metadata["LicenseUrl"] = _value(license_url)
    return {
        "pageid": page_id,
        "ns": 6,
        "title": f"File:Image {page_id}.png",
        "imageinfo": [
            {
                "timestamp": "2026-08-19T12:00:00Z",
                "sha1": sha * 31,
                "url": f"https://upload.wikimedia.org/file-{page_id}.png",
                "descriptionurl": f"https://commons.wikimedia.org/wiki/File:Image_{page_id}.png",
                "thumburl": f"https://upload.wikimedia.org/thumb-{page_id}.png",
                "width": width,
                "height": height,
                "mime": mime,
                "extmetadata": metadata,
            }
        ],
    }


def _payload(
    pages: list[dict[str, Any]],
    continuation: dict[str, str] | None = None,
) -> dict[str, Any]:
    result: dict[str, Any] = {"query": {"pages": pages}}
    if continuation:
        result["continue"] = continuation
    return result


def _image(color: str = "red", size: tuple[int, int] = (600, 600)) -> bytes:
    buffer = io.BytesIO()
    Image.new("RGB", size, color).save(buffer, format="PNG")
    return buffer.getvalue()


def _animated() -> bytes:
    buffer = io.BytesIO()
    first = Image.new("RGB", (600, 600), "red")
    second = Image.new("RGB", (600, 600), "blue")
    first.save(buffer, format="GIF", save_all=True, append_images=[second])
    return buffer.getvalue()


def _client(handler: Callable[[httpx.Request], httpx.Response]) -> httpx.Client:
    return httpx.Client(transport=httpx.MockTransport(handler))


def _contact(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(USER_AGENT_ENV_VAR, "krasnal-id-tests (tests@example.com)")


def test_prepare_review_preserves_and_resets_decisions(tmp_path: Path) -> None:
    directory = tmp_path / "discovery"
    path = tmp_path / "review.json"
    q2, q10 = _dwarf("Q2", "Two"), _dwarf("Q10", "Ten")
    _discovery(directory, (q10, q2), ("Q10",))

    initial = prepare_category_review(directory, path)
    assert [record.dwarf_id for record in initial.records] == ["Q2", "Q10"]
    assert initial.records[1].discovery_warning
    _review(
        path,
        initial.records[0].model_copy(
            update={
                "status": CategoryReviewStatus.APPROVED,
                "corrected_category": "Corrected",
                "notes": "checked",
            }
        ),
        initial.records[1].model_copy(update={"status": CategoryReviewStatus.REJECTED}),
        CategoryReviewRecord(
            dwarf_id="Q99",
            display_name="Stale",
            discovered_category="Stale",
            status=CategoryReviewStatus.APPROVED,
        ),
    )

    preserved = prepare_category_review(directory, path)
    assert preserved.records[0].selected_category == "Corrected"
    assert preserved.records[-1].dwarf_id == "Q99"

    _discovery(directory, (_dwarf("Q2", "Changed"), q10))
    reset = prepare_category_review(directory, path)
    changed = next(record for record in reset.records if record.dwarf_id == "Q2")
    assert changed.status is CategoryReviewStatus.PENDING
    assert changed.corrected_category is None
    assert changed.notes is None


@pytest.mark.parametrize("case", ["missing", "malformed", "pending", "changed"])
def test_fetch_rejects_invalid_inputs(tmp_path: Path, case: str) -> None:
    directory, path = tmp_path / "discovery", tmp_path / "review.json"
    dwarf = _dwarf("Q2")
    if case != "missing":
        _discovery(directory, (dwarf,))
    if case == "malformed":
        path.write_text("{bad", encoding="utf-8")
    elif case == "pending":
        _review(path, _decision(dwarf, CategoryReviewStatus.PENDING))
    elif case == "changed":
        wrong = _decision(dwarf).model_copy(update={"discovered_category": "Changed"})
        _review(path, wrong)

    with pytest.raises(CommonsConfigurationError):
        fetch_images(directory, path, tmp_path / "images", load_config().data)


def test_rejects_nonpositive_programmatic_cap(tmp_path: Path) -> None:
    with pytest.raises(CommonsConfigurationError, match="positive"):
        fetch_images(
            tmp_path,
            tmp_path / "review.json",
            tmp_path / "images",
            load_config().data,
            max_images_per_dwarf=0,
        )


def test_fetch_paginates_caps_caches_reuses_and_refreshes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _contact(monkeypatch)
    directory, path, images = (
        tmp_path / "discovery",
        tmp_path / "review.json",
        tmp_path / "images",
    )
    dwarf = _dwarf("Q2", "Wrong")
    _discovery(directory, (dwarf,))
    _review(path, _decision(dwarf, corrected="Reviewed"))

    pd_page = _page(
        10,
        license_token="pd",
        license_name="Public domain",
        license_url=None,
    )
    missing_author = _page(30, author=None, sha="c")
    api_calls = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal api_calls
        if request.url.host == "commons.wikimedia.org":
            assert request.url.params["gcmtitle"] == "Category:Reviewed"
            assert request.url.params["gcmtype"] == "file"
            api_calls += 1
            if api_calls == 1:
                return httpx.Response(
                    200,
                    json=_payload(
                        [_page(20, sha="b")],
                        {
                            "continue": "gcmcontinue||",
                            "gcmcontinue": "next",
                        },
                    ),
                )
            return httpx.Response(200, json=_payload([pd_page, missing_author]))
        return httpx.Response(200, content=_image())

    with _client(handler) as client:
        first = fetch_images(
            directory,
            path,
            images,
            load_config().data,
            max_images_per_dwarf=1,
            client=client,
        )

    assert api_calls == 2
    assert (first.discovered_images, first.eligible_images, first.downloaded_images) == (3, 2, 1)
    assert first.images[0].image_id == "commons-10"
    assert first.images[0].author == "Alice & Bob"
    assert "publicdomain/mark" in str(first.images[0].license_url)
    assert commons_cache_paths(directory, "Q2").response.is_file()
    assert fetch_paths(directory).fetched_images.is_file()

    monkeypatch.delenv(USER_AGENT_ENV_VAR)
    with _client(lambda request: pytest.fail(f"network used: {request.url}")) as client:
        cached = fetch_images(
            directory,
            path,
            images,
            load_config().data,
            max_images_per_dwarf=1,
            client=client,
        )
    assert cached.reused_images == 1
    assert cached.images[0].acquired_at == first.images[0].acquired_at

    _contact(monkeypatch)
    calls = 0

    def refresh(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        if request.url.host != "commons.wikimedia.org":
            pytest.fail("unchanged revision was redownloaded")
        return httpx.Response(200, json=_payload([pd_page, _page(20), missing_author]))

    with _client(refresh) as client:
        result = fetch_images(
            directory,
            path,
            images,
            load_config().data,
            max_images_per_dwarf=1,
            refresh=True,
            client=client,
        )
    assert calls == 1
    assert result.reused_images == 1

    commons_cache_paths(directory, "Q2").response.write_text("{broken", encoding="utf-8")
    with _client(refresh) as client:
        recovered = fetch_images(
            directory,
            path,
            images,
            load_config().data,
            max_images_per_dwarf=1,
            client=client,
        )
    assert recovered.reused_images == 1
    assert not list(tmp_path.rglob("*.tmp"))


def test_filters_metadata_and_invalid_downloads(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _contact(monkeypatch)
    directory, path = tmp_path / "discovery", tmp_path / "review.json"
    dwarf = _dwarf("Q2")
    _discovery(directory, (dwarf,))
    _review(path, _decision(dwarf))
    pages = [
        _page(1, mime="image/svg+xml"),
        _page(2, width=399),
        _page(3, author=None),
        _page(
            4,
            license_token="cc-by-nc-4.0",
            license_name="CC BY-NC 4.0",
            license_url="https://creativecommons.org/licenses/by-nc/4.0/",
        ),
        {"pageid": 5, "ns": 6, "title": "File:Missing.png", "imageinfo": []},
        _page(6, sha="f"),
        _page(7, mime="image/gif", sha="g"),
        _page(8, width=2000, height=1000, sha="h"),
    ]

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.host == "commons.wikimedia.org":
            return httpx.Response(200, json=_payload(pages))
        page_id = int(request.url.path.split("-")[-1].split(".")[0])
        if page_id == 7:
            return httpx.Response(200, content=_animated())
        if page_id == 8:
            return httpx.Response(200, content=_image(size=(1600, 399)))
        return httpx.Response(200, content=b"bad")

    with _client(handler) as client:
        result = fetch_images(
            directory,
            path,
            tmp_path / "images",
            load_config().data,
            client=client,
        )

    reasons = {record.reason for record in result.audit}
    assert result.images == ()
    assert result.operational_failures == 3
    assert {
        FetchAuditReason.UNSUPPORTED_MEDIA,
        FetchAuditReason.IMAGE_TOO_SMALL,
        FetchAuditReason.MISSING_ATTRIBUTION,
        FetchAuditReason.UNSUPPORTED_LICENSE,
        FetchAuditReason.INVALID_METADATA,
        FetchAuditReason.DOWNLOAD_FAILURE,
    }.issubset(reasons)


def test_deduplicates_same_and_cross_label_content(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _contact(monkeypatch)
    directory, path = tmp_path / "discovery", tmp_path / "review.json"
    q2, q10 = _dwarf("Q2"), _dwarf("Q10")
    _discovery(directory, (q2, q10))
    _review(path, _decision(q2), _decision(q10))
    pages = {
        "Category:Category Q2": [
            _page(1),
            _page(2, sha="b"),
            _page(4, sha="d"),
            _page(5, sha="e"),
            _page(6, sha="f"),
        ],
        "Category:Category Q10": [_page(3, sha="c")],
    }

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.host == "commons.wikimedia.org":
            return httpx.Response(200, json=_payload(pages[request.url.params["gcmtitle"]]))
        page_id = int(request.url.path.split("-")[-1].split(".")[0])
        color = "red" if page_id in {1, 2, 3} else "green" if page_id in {5, 6} else "blue"
        return httpx.Response(200, content=_image(color))

    with _client(handler) as client:
        result = fetch_images(
            directory,
            path,
            tmp_path / "images",
            load_config().data,
            client=client,
        )

    assert [record.image_id for record in result.images] == ["commons-4", "commons-5"]
    cross = [r for r in result.audit if r.reason is FetchAuditReason.CROSS_LABEL_DUPLICATE]
    same = [r for r in result.audit if r.reason is FetchAuditReason.SAME_LABEL_DUPLICATE]
    assert {record.commons_page_id for record in cross} == {1, 2, 3}
    assert {record.commons_page_id for record in same} == {6}
    assert (tmp_path / "images" / "Q10" / "commons-3.png").is_file()


def test_retries_and_continues_after_failures(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _contact(monkeypatch)
    delays: list[float] = []
    monkeypatch.setattr(fetch_module.time, "sleep", delays.append)
    directory, path = tmp_path / "discovery", tmp_path / "review.json"
    q2, q10 = _dwarf("Q2"), _dwarf("Q10")
    _discovery(directory, (q2, q10))
    _review(path, _decision(q2), _decision(q10))
    q2_attempts = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal q2_attempts
        if request.url.host == "commons.wikimedia.org":
            if request.url.params["gcmtitle"].endswith("Q2"):
                q2_attempts += 1
                if q2_attempts == 1:
                    return httpx.Response(503, headers={"Retry-After": "999"})
                return httpx.Response(500)
            return httpx.Response(200, json=_payload([_page(10)]))
        return httpx.Response(503)

    with _client(handler) as client:
        result = fetch_images(
            directory,
            path,
            tmp_path / "images",
            load_config().data,
            client=client,
        )

    assert delays == [60.0, 1.0, 2.0]
    assert result.operational_failures == 2
    assert {
        record.reason
        for record in result.audit
        if record.disposition is FetchAuditDisposition.ERROR
    } == {FetchAuditReason.API_FAILURE, FetchAuditReason.DOWNLOAD_FAILURE}


def test_retries_transport_and_audits_malformed_json(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _contact(monkeypatch)
    delays: list[float] = []
    monkeypatch.setattr(fetch_module.time, "sleep", delays.append)
    directory, path = tmp_path / "discovery", tmp_path / "review.json"
    q2, q10 = _dwarf("Q2"), _dwarf("Q10")
    _discovery(directory, (q2, q10))
    _review(path, _decision(q2), _decision(q10))
    attempts = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal attempts
        if request.url.params["gcmtitle"].endswith("Q2"):
            attempts += 1
            if attempts == 1:
                raise httpx.ConnectError("offline", request=request)
            return httpx.Response(200, json=_payload([]))
        return httpx.Response(200, content=b"{broken")

    with _client(handler) as client:
        result = fetch_images(
            directory,
            path,
            tmp_path / "images",
            load_config().data,
            client=client,
        )

    assert attempts == 2
    assert delays == [1.0]
    assert result.operational_failures == 1
    assert result.audit[0].reason is FetchAuditReason.API_FAILURE


def test_user_agent_rejected_but_rejected_categories_need_no_network(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv(USER_AGENT_ENV_VAR, raising=False)
    directory, path = tmp_path / "discovery", tmp_path / "review.json"
    dwarf = _dwarf("Q2")
    _discovery(directory, (dwarf,))
    _review(path, _decision(dwarf))
    with (
        _client(lambda request: pytest.fail(f"network used: {request.url}")) as client,
        pytest.raises(CommonsConfigurationError, match=USER_AGENT_ENV_VAR),
    ):
        fetch_images(directory, path, tmp_path / "images", load_config().data, client=client)

    _review(path, _decision(dwarf, CategoryReviewStatus.REJECTED))
    with _client(lambda request: pytest.fail(f"network used: {request.url}")) as client:
        result = fetch_images(
            directory,
            path,
            tmp_path / "images",
            load_config().data,
            client=client,
        )
    assert result.rejected_categories == 1
    assert result.audit[0].reason is FetchAuditReason.REJECTED_CATEGORY


def test_fetch_cli_workflow(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    help_result = runner.invoke(app, ["data", "fetch", "--help"])
    assert help_result.exit_code == 0
    assert "--prepare-review" in help_result.stdout
    assert "--max-images-per-dwarf" in help_result.stdout

    directory, path = tmp_path / "discovery", tmp_path / "review.json"
    dwarf = _dwarf("Q2")
    _discovery(directory, (dwarf,))
    overrides = [
        "--override",
        f"paths.discovery_dir={directory}",
        "--override",
        f"paths.category_review_path={path}",
        "--override",
        f"paths.images_dir={tmp_path / 'images'}",
        "--override",
        "logging.json_output=false",
    ]
    prepared = runner.invoke(app, ["data", "fetch", "--prepare-review", *overrides])
    assert prepared.exit_code == 0
    assert "pending=1" in prepared.output
    incompatible = runner.invoke(
        app,
        ["data", "fetch", "--prepare-review", "--refresh", *overrides],
    )
    assert incompatible.exit_code == 2

    _review(path, _decision(dwarf, CategoryReviewStatus.REJECTED))
    completed = runner.invoke(app, ["data", "fetch", *overrides])
    assert completed.exit_code == 0
    assert "approved=0 rejected=1" in completed.output

    monkeypatch.setattr(
        cli_module,
        "fetch_images",
        lambda *args, **kwargs: FetchResult(
            images=(),
            audit=(),
            approved_categories=1,
            rejected_categories=0,
            pending_categories=0,
            discovered_images=0,
            eligible_images=0,
            downloaded_images=0,
            reused_images=0,
            operational_failures=1,
        ),
    )
    failed = runner.invoke(app, ["data", "fetch", *overrides])
    assert failed.exit_code == 1
    assert "errors=1" in failed.output


@pytest.mark.integration
def test_live_commons_fetch_is_opt_in(tmp_path: Path) -> None:
    if os.environ.get("RUN_LIVE_COMMONS_TESTS") != "1":
        pytest.skip("set RUN_LIVE_COMMONS_TESTS=1 to run")
    if not os.environ.get(USER_AGENT_ENV_VAR):
        pytest.skip(f"set {USER_AGENT_ENV_VAR} to run")
    directory, path = tmp_path / "discovery", tmp_path / "review.json"
    dwarf = _dwarf("Q2", "Wroclaw blacksmith dwarf")
    _discovery(directory, (dwarf,))
    _review(path, _decision(dwarf))
    result = fetch_images(
        directory,
        path,
        tmp_path / "images",
        load_config().data,
        max_images_per_dwarf=1,
        refresh=True,
    )
    assert result.operational_failures == 0
    assert len(result.images) <= 1

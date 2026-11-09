"""The family study's numbers, table and figure command (AGENTS.md section 7.17)."""

import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from krasnal_id.cli import app
from krasnal_id.viz.family_plot import FAMILIES, PANELS
from krasnal_id.viz.workshop_tables import (
    _FAMILY_RELEASED,
    _FAMILY_VALIDATION,
    WorkshopTableError,
    family_metrics,
    macro_file,
)

MODELS = tuple(model for _, members in FAMILIES for model in members)


def _artifact(path: Path, metrics: dict[str, tuple[float, float | None, float | None]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "metrics": [
                    {"name": n, "value": v, "lower_bound": lo, "upper_bound": hi}
                    for n, (v, lo, hi) in metrics.items()
                ]
            }
        ),
        encoding="utf-8",
    )


def _study(root: Path, skip: str | None = None) -> Path:
    suffixes = set(_FAMILY_VALIDATION) | {panel.suffix for panel in PANELS}
    validated: dict[str, tuple[float, float | None, float | None]] = {
        f"{model}_{suffix}": (0.5, 0.4, 0.6)
        for model in MODELS
        for suffix in suffixes
        if f"{model}_{suffix}" != skip
    }
    validation = root / "validation" / "workshop_validation-dinov2.json"
    _artifact(validation, validated)
    for model in MODELS:
        # One file per experiment, holding every metric the table reads from it.
        released: dict[str, dict[str, tuple[float, float | None, float | None]]] = {}
        for experiment, metric, _ in _FAMILY_RELEASED:
            released.setdefault(experiment, {})[metric] = (0.8, 0.7, 0.9)
        for experiment, metrics in released.items():
            _artifact(root / model / f"{experiment}-{model}.json", metrics)
        # The supplement's curves read each backbone's pool-size artifact too.
        _artifact(
            root / model / f"pool_size_ablation-{model}.json",
            {f"top_1_pool_{n}": (0.9 - n / 1000, 0.88, 0.92) for n in (2, 10, 306)},
        )
    return validation


def test_family_numbers_are_prefixed_and_released_points_lose_their_interval(
    tmp_path: Path,
) -> None:
    validation = _study(tmp_path)
    metrics = family_metrics(validation, tmp_path, MODELS)
    assert metrics["fam_dino_top_1"] == {"value": 0.5, "lower": 0.4, "upper": 0.6}
    # Released numbers are points: the table must not print a Wilson interval as a statue one.
    assert metrics["fam_radio_unseen_top_1"] == {"value": 0.8, "lower": None, "upper": None}
    text = macro_file(metrics, with_commands=False)
    assert r"\newcommand" not in text
    assert r"\@namedef{wv@fam_dino_top_1@ci}{40.0 to 60.0}" in text


def test_a_missing_family_number_is_refused(tmp_path: Path) -> None:
    validation = _study(tmp_path, skip="aimv2_top_1")
    with pytest.raises(WorkshopTableError, match="no aimv2_top_1"):
        family_metrics(validation, tmp_path, MODELS)


def test_the_family_command_writes_numbers_table_and_figure(tmp_path: Path) -> None:
    pytest.importorskip("matplotlib")
    _study(tmp_path / "family")
    result = CliRunner().invoke(
        app,
        [
            "visualize",
            "family",
            "--root",
            str(tmp_path / "family"),
            "--figure",
            str(tmp_path / "fig.png"),
            "--curves",
            str(tmp_path / "curves.png"),
            "--output-dir",
            str(tmp_path / "gen"),
            "--dpi",
            "72",
        ],
    )
    assert result.exit_code == 0, result.output
    table = (tmp_path / "gen" / "family-table.tex").read_text(encoding="utf-8")
    assert table.count(r"\quad ") == len(MODELS)
    assert r"\textit{Agglomerative}" in table
    assert (tmp_path / "fig.png").stat().st_size > 0
    assert (tmp_path / "curves.png").stat().st_size > 0


def test_extra_numbers_are_prefixed_and_prefixes_cannot_collide(tmp_path: Path) -> None:
    from krasnal_id.viz.workshop_tables import write_extra_numbers

    artifact = tmp_path / "a.json"
    _artifact(artifact, {"cv_cosine_top_1": (0.852, 0.83, 0.87)})
    out = write_extra_numbers([f"v3={artifact}"], tmp_path)
    text = out.read_text(encoding="utf-8")
    assert r"\@namedef{wv@v3_cv_cosine_top_1@v}{85.2}" in text
    assert r"\newcommand" not in text
    for bad in ([f"fam={artifact}"], [f"v3={artifact}", f"v3={artifact}"], ["v3"]):
        with pytest.raises(WorkshopTableError):
            write_extra_numbers(bad, tmp_path)

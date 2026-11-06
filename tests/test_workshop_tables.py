"""The CVWW manuscript's numbers are generated from artifacts, never typed."""

import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from krasnal_id.cli import app
from krasnal_id.viz.workshop_tables import (
    WorkshopTableError,
    fold_weight_table,
    formatted,
    load_metrics,
    macro_file,
)

runner = CliRunner()


def _artifact(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "metrics": [
                    {
                        "name": "dinov2_top_1",
                        "value": 0.9314,
                        "lower_bound": 0.9183,
                        "upper_bound": 0.9425,
                    },
                    {
                        "name": "dinov2_points_per_doubling",
                        "value": -0.79,
                        "lower_bound": -0.9,
                        "upper_bound": -0.7,
                    },
                    {"name": "cv_sift_vs_cosine_top_1_wins", "value": 8.0},
                    {"name": "cv_sift_vs_cosine_top_1_p_value", "value": 0.000015},
                ],
                "fold_weights": [
                    {
                        "matcher": "sift",
                        "fold": 0,
                        "weight": 0.02,
                        "selection_top_1": 0.8209,
                        "held_out_queries": 219,
                        "held_out_statues": 37,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    return path


def test_each_metric_is_printed_in_the_papers_units() -> None:
    assert formatted("dinov2_top_1", 0.9314) == "93.1"
    assert formatted("geo_minus_random", -0.0089) == "$-$0.9"
    assert formatted("dinov2_points_per_doubling", -0.79) == "$-$0.8"
    assert formatted("x_wins", 37.0) == "37"
    assert formatted("x_p_value", 0.58) == "0.58"
    assert formatted("x_p_value", 0.000015) == r"$1.5\times10^{-5}$"


def test_the_macro_file_defines_values_and_intervals(tmp_path: Path) -> None:
    text = macro_file(load_metrics(_artifact(tmp_path / "v.json")))

    assert r"\@namedef{wv@dinov2_top_1@v}{93.1}" in text
    assert r"\@namedef{wv@dinov2_top_1@ci}{91.8 to 94.2}" in text
    assert r"\@namedef{wv@dinov2_points_per_doubling@ci}{$-$0.9 to $-$0.7}" in text
    # A count has no interval, and an unknown name renders visibly, not silently.
    assert "wv@cv_sift_vs_cosine_top_1_wins@ci" not in text
    assert "MISSING" in text


def test_fold_weights_become_a_table_and_absent_ones_are_refused(tmp_path: Path) -> None:
    path = _artifact(tmp_path / "v.json")
    assert "sift & 0 & 0.02 & 82.1 & 219 & 37" in fold_weight_table(path)

    empty = tmp_path / "empty.json"
    empty.write_text(json.dumps({"metrics": [], "fold_weights": []}), encoding="utf-8")
    with pytest.raises(WorkshopTableError, match="no fold weights"):
        fold_weight_table(empty)
    with pytest.raises(WorkshopTableError, match="cannot read"):
        load_metrics(tmp_path / "absent.json")


def test_the_cli_writes_both_fragments(tmp_path: Path) -> None:
    _artifact(tmp_path / "results" / "cvww_revision2" / "workshop_validation-dinov2.json")

    result = runner.invoke(
        app,
        [
            "visualize",
            "workshop-tables",
            "--output-dir",
            str(tmp_path / "generated"),
            f"-opaths.results_dir={tmp_path / 'results'}",
            "-ologging.json_output=false",
        ],
    )

    assert result.exit_code == 0, result.output
    assert (tmp_path / "generated" / "numbers.tex").is_file()
    assert (tmp_path / "generated" / "fold-weights.tex").is_file()
    missing = runner.invoke(
        app,
        ["visualize", "workshop-tables", f"-opaths.results_dir={tmp_path / 'absent'}"],
    )
    assert missing.exit_code == 2

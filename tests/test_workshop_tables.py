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
    reviewer_tables,
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
    # Section 7.15-7.16 metrics: AUROCs as probabilities, counts and weights verbatim.
    assert formatted("dinov2_reject_margin_auroc", 0.89591) == "0.896"
    assert formatted("dinov2_reject_knn_minus_cosine_auroc", -0.0123) == "$-$0.012"
    assert formatted("dinov2_families_count", 37.0) == "37"
    assert formatted("dinov2_families_errors", 116.0) == "116"
    assert formatted("dinov2_soft_prior_gps25_fold_0_weight", 0.05) == "0.05"
    assert formatted("dinov2_families_error_share", 0.4321) == "43.2"
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
    # Each end on its own, for a sentence that bounds an effect from one side.
    assert r"\@namedef{wv@dinov2_top_1@lo}{91.8}" in text
    assert r"\@namedef{wv@dinov2_top_1@hi}{94.2}" in text
    # A count has no interval, and an unknown name renders visibly, not silently.
    assert "wv@cv_sift_vs_cosine_top_1_wins@ci" not in text
    assert "MISSING" in text
    # Metric names carry underscores, which are subscripts outside maths mode.
    assert r"MISSING:\detokenize{#1}" in text


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
    _artifact(tmp_path / "results" / "cvww_revision3" / "workshop_validation-dinov2.json")

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


def _m(
    value: float, lower: float | None = None, upper: float | None = None
) -> dict[str, float | None]:
    return {"value": value, "lower": lower, "upper": upper}


def test_a_value_that_rounds_to_zero_carries_no_sign() -> None:
    assert formatted("dinov2_geo_minus_random_pool_5", -0.0004) == "0.0"
    assert formatted("dinov2_geo_minus_random_pool_5", -0.0123) == "$-$1.2"


def test_reviewer_tables_are_built_from_whatever_parts_ran() -> None:
    metrics = {
        "dinov2_reject_cosine_auroc": _m(0.896, 0.883, 0.909),
        "dinov2_reject_margin_auroc": _m(0.927, 0.914, 0.940),
        "dinov2_reject_knn_auroc": _m(0.637, 0.611, 0.665),
        "dinov2_reject_cosine_false_acceptance_90": _m(0.383),
        "dinov2_reject_margin_false_acceptance_90": _m(0.196),
        "dinov2_reject_knn_false_acceptance_90": _m(0.822),
        "dinov2_geo_minus_random_pool_5": _m(-0.009, -0.015, -0.003),
        "dinov2_geo_permutation_pool_5_null_mean": _m(0.001, -0.005, 0.004),
        "dinov2_geo_permutation_mean_p_value": _m(1 / 201),
    }
    tables = reviewer_tables(metrics)

    # Only the parts present become tables; the rest are left out, not faked.
    assert set(tables) == {"reviewer-rejection.tex", "reviewer-permutation.tex"}
    assert "DINOv2 & 0.896 [0.883, 0.909] & 38.3 & 0.927" in tables["reviewer-rejection.tex"]
    # The p-value is printed as a p-value, not scaled like an accuracy.
    assert tables["reviewer-permutation.tex"].rstrip().splitlines()[-3].endswith("0.005 \\\\")

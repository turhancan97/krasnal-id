"""Supplementary tables, and the transcription they exist to prevent."""

import json
from pathlib import Path

import pytest

from krasnal_id.viz.paper_figures import PaperFigureError, Sources
from krasnal_id.viz.paper_tables import TABLES, _p_value, _pct, _signed, write_all


def _artifact(directory: Path, stem: str, metrics: dict[str, float]) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    (directory / f"{stem}.json").write_text(
        json.dumps(
            {
                "experiment": stem,
                "backbone": "dinov2",
                "created_at": "2026-09-14T00:00:00Z",
                "seed": 42,
                "metrics": [{"name": k, "value": v} for k, v in metrics.items()],
            }
        ),
        encoding="utf-8",
    )


def test_every_supplementary_section_the_main_text_cites_has_a_table() -> None:
    """The manuscript forward-references S3 to S13 by number.

    Renumbering the Supplementary Information breaks live pointers in prose that
    is already written, so the names are pinned here rather than in a comment.
    """
    assert set(TABLES) == {
        "s3-pool-size",
        "s4-geography",
        "s5-classifiers",
        "s6-confusion",
        "s6-separation",
        "s7-camera",
        "s8-photographer",
        "s9-rerank",
        "s9-matcher",
        "s10-recall",
        "s11-capacity",
        "s12-geometry-first",
        "s13-open-set",
        "s13-open-set-geometry",
    }


def test_a_small_p_value_is_not_rounded_away() -> None:
    """Two decimal places turn 0.0026 into 0.00, which reads as certainty."""
    assert _p_value(0.0026) == "$p = 0.003$"
    assert _p_value(0.0000001) == "$p < 0.001$"
    assert _p_value(0.878) == "$p = 0.878$"


def test_a_difference_is_printed_with_a_real_minus_sign() -> None:
    """These are set in a typeset table, where a hyphen is visibly shorter."""
    assert _signed(-1.3) == "\N{MINUS SIGN}1.30"
    assert _signed(2.59) == "+2.59"


def test_an_absent_metric_becomes_a_dash_rather_than_a_zero() -> None:
    """A missing measurement must not typeset as a measured zero."""
    assert _pct(None) == "---"
    assert _pct(0.9314) == "93.14"


def test_a_missing_artifact_names_itself_rather_than_emitting_a_blank_table(
    tmp_path: Path,
) -> None:
    sources = Sources(results_dir=tmp_path, manifest=tmp_path / "manifest.json")

    with pytest.raises(PaperFigureError, match="missing artifact"):
        write_all(sources, tmp_path / "tables")


def test_a_written_fragment_is_a_complete_table_environment(tmp_path: Path) -> None:
    """si.tex inputs these directly, so a fragment must stand on its own."""
    results = tmp_path / "results"
    for backbone in ("dinov2", "clip"):
        _artifact(
            results,
            f"probe-{backbone}",
            {
                "retrieval_top_1": 0.93,
                "retrieval_top_5": 0.96,
                "retrieval_mrr": 0.94,
                "linear_probe_top_1": 0.93,
                "linear_probe_top_5": 0.96,
                "linear_probe_mrr": 0.94,
                "prototype_top_1": 0.85,
                "prototype_top_5": 0.90,
                "prototype_mrr": 0.88,
            },
        )
    sources = Sources(results_dir=results, manifest=tmp_path / "manifest.json")

    fragment = TABLES["s5-classifiers"](sources)

    assert fragment.startswith(r"\begin{table}[ht]")
    assert fragment.rstrip().endswith(r"\end{table}")
    assert fragment.count(r"\toprule") == fragment.count(r"\botrule") == 1
    # Cosine retrieval, linear probe, class prototypes: one row each.
    assert fragment.count(r"\\") == 4


# The minimum each builder needs to run: every stem it loads, and the metrics it
# demands rather than tolerates. Metrics reached through `maybe` are left out on
# purpose, so this fixture also proves an incomplete artifact still typesets.
_REQUIRED: dict[str, dict[str, float]] = {
    "pool_size_ablation-dinov2": {"top_1_points_per_doubling": -0.79, "top_1_pool_2": 0.99},
    "pool_size_ablation-clip": {"top_1_points_per_doubling": -2.06, "top_1_pool_2": 0.98},
    "geo_ablation-dinov2": {"geo_radius_metres_pool_5": 171.1, "geo_top_1_pool_5": 0.97},
    "geo_ablation-clip": {"geo_top_1_pool_5": 0.94, "random_top_1_pool_5": 0.96},
    "probe-dinov2": {"retrieval_top_1": 0.93},
    "probe-clip": {"retrieval_top_1": 0.83},
    "confusion-dinov2": {"proximity_predicts_confusion_auroc": 0.518},
    "confusion-clip": {"proximity_predicts_confusion_auroc": 0.551},
    "camera_gap-dinov2": {"phone_top_1": 0.88, "phone_queries": 51.0},
    "camera_gap-clip": {"phone_top_1": 0.69},
    "photographer_gap-dinov2": {"attributable_top_1_gap": 0.0261, "attributable_top_5_gap": 0.0112},
    "photographer_gap-clip": {"attributable_top_1_gap": 0.1322, "attributable_top_5_gap": 0.1004},
    "rerank_ablation-dinov2": {"weight_0_top_1": 0.93},
    "rerank_ablation-clip": {"weight_0_top_1": 0.83},
    "rerank_ablation-disjoint-dinov2": {"disjoint_weight_0_top_1": 0.82},
    "rerank_ablation-disjoint-clip": {"disjoint_weight_0_top_1": 0.54},
    "matcher_rerank-disjoint-dinov2": {
        "disk-lightglue_vs_sift_top_1_wins": 34.0,
        "disk-lightglue_vs_sift_top_1_losses": 7.0,
        "disk-lightglue_vs_sift_top_1_p_value": 2.5e-05,
        "sift_w0_top_1": 0.8185,
    },
    "recall_curve-dinov2": {"full_r_at_1": 0.93, "disjoint_r_at_10": 0.9075},
    "recall_curve-clip": {"full_r_at_1": 0.83},
    "geometry_first-dinov2": {"disjoint_geometry_r_at_10": 0.4347},
    "open_set-dinov2": {"auroc": 0.8959},
    "open_set-clip": {"auroc": 0.8059},
    "open_set_geometry-dinov2": {"standard_cosine_auroc": 0.8959},
}


@pytest.fixture
def complete_results(tmp_path: Path) -> Sources:
    """Every artifact the fourteen builders load, carrying what they demand."""
    results = tmp_path / "results"
    for stem, metrics in _REQUIRED.items():
        _artifact(results, stem, metrics)
    payload = json.loads((results / "confusion-dinov2.json").read_text(encoding="utf-8"))
    payload["pairs"] = [
        {
            "true_display_name": "Słupniki Solne",
            "confused_display_name": "Słupniki Oławskie",
            "misidentifications": 2,
            "queries": 10,
            "mean_margin": 0.0865,
            "separation_metres": 429.0,
        }
    ]
    (results / "confusion-dinov2.json").write_text(json.dumps(payload), encoding="utf-8")
    return Sources(results_dir=results, manifest=tmp_path / "manifest.json")


def test_every_table_typesets_from_an_incomplete_artifact(complete_results: Sources) -> None:
    """A metric a builder merely hopes for must become a dash, not an exception.

    The artifacts here carry only what each builder demands. Everything else is
    absent, which is the shape a partially re-run experiment directory has.
    """
    for name, build in TABLES.items():
        fragment = build(complete_results)

        assert fragment.startswith(r"\begin{table}[ht]"), name
        assert r"\caption{" in fragment, name
        assert fragment.count(r"\toprule") == 1, name
        assert fragment.count(r"\botrule") == 1, name
        # A row whose columns do not match its header silently shifts every
        # number one cell to the left in the typeset table.
        lines = [ln for ln in fragment.splitlines() if ln.endswith(r"\\")]
        widths = {ln.count("&") for ln in lines}
        assert len(widths) == 1, f"{name} has ragged rows: {widths}"


def test_write_all_names_each_fragment_for_its_supplementary_section(
    complete_results: Sources, tmp_path: Path
) -> None:
    written = write_all(complete_results, tmp_path / "tables")

    assert {path.stem for path in written} == set(TABLES)
    assert all(path.read_text(encoding="utf-8").strip() for path in written)

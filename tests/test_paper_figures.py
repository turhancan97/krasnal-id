"""Publication figures, and the artifact contract they depend on."""

import json
from pathlib import Path

import pytest

from krasnal_id.viz.paper_figures import (
    FIGURES,
    PUBLICATION_DPI,
    SEPARATION_BANDS,
    PaperFigureError,
    load,
)


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


def test_a_missing_metric_names_the_artifact_it_was_wanted_from(tmp_path: Path) -> None:
    """A figure that silently plots nothing is worse than one that fails.

    These read dozens of metric names across nine artifacts; a typo in any of them
    must say which file was short, not draw an empty axis.
    """
    _artifact(tmp_path, "baseline-dinov2", {"top_1": 0.93})
    art = load(tmp_path, "baseline-dinov2")

    assert art.get("top_1") == pytest.approx(0.93)
    assert art.maybe("absent") is None
    with pytest.raises(PaperFigureError, match="baseline-dinov2 has no metric 'absent'"):
        art.get("absent")


def test_a_missing_artifact_is_refused_by_path(tmp_path: Path) -> None:
    with pytest.raises(PaperFigureError, match="missing artifact"):
        load(tmp_path, "never-run")


def test_the_publication_density_meets_the_journal_requirement() -> None:
    """Nature asks for 300 dpi minimum; the repository's own figures are lower."""
    assert PUBLICATION_DPI >= 300


def test_the_separation_bands_show_a_decay_rather_than_a_point() -> None:
    """Two bands state the effect; the claim is that it decays, which needs more.

    Section 3a argues co-location acts at the scale of a shared plinth and is gone
    by a neighbourhood. That is only visible with bands spanning both.
    """
    assert len(SEPARATION_BANDS) >= 5
    assert min(SEPARATION_BANDS) <= 100
    assert max(SEPARATION_BANDS) >= 1000
    assert list(SEPARATION_BANDS) == sorted(SEPARATION_BANDS)


def test_every_figure_is_named_for_its_position_in_the_manuscript() -> None:
    """The filenames are what main.tex includes, so they cannot drift silently."""
    assert set(FIGURES) == {
        "fig3-geography",
        "fig5-photographer",
        "fig6-matcher",
        "fig7-rejection",
    }
    for name in FIGURES:
        assert name.startswith("fig")

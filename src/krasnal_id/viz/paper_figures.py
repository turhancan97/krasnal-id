"""Publication figures for the manuscript, built from the result artifacts.

These exist because the venue allows eight display items and `RESULTS.md` holds
twenty tables. Turning tables into figures is what makes the material fit, so
each function here consolidates several source tables into one panel set.

The code is tracked even though the manuscript is not. A figure that cannot be
regenerated from a committed command is a figure nobody can check, and §6.4's
rule -- every published number traceable to a `results/*.json` artifact -- applies
to the paper at least as strongly as it applies to the repository.

Every figure is written at `PUBLICATION_DPI` in a sans-serif face, which is what
Nature asks for.
"""

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

# Nature asks for 300 DPI minimum for publication figures. The repository's own
# figures are drawn at a lower density for the web, so these are separate files
# rather than the same ones reused.
PUBLICATION_DPI = 300

# Both backbones, everywhere. Colour-blind safe and distinguishable in greyscale,
# which a print reviewer may be reading.
BACKBONE_COLOUR = {"dinov2": "#1b6ca8", "clip": "#c05640"}
BACKBONE_LABEL = {"dinov2": "DINOv2", "clip": "CLIP"}

SEPARATION_BANDS = (50, 100, 200, 300, 500, 1000)


class PaperFigureError(ValueError):
    """Raised when an artifact a figure needs is missing or lacks a metric."""


@dataclass(frozen=True, slots=True)
class Artifact:
    """One experiment's metrics, addressed by name."""

    name: str
    values: dict[str, float]

    def get(self, key: str) -> float:
        """Return one metric, naming the artifact if it is absent."""
        if key not in self.values:
            raise PaperFigureError(f"{self.name} has no metric {key!r}")
        return self.values[key]

    def maybe(self, key: str) -> float | None:
        """Return one metric, or None where its absence is meaningful."""
        return self.values.get(key)


def load(results_dir: Path, stem: str) -> Artifact:
    """Read one result artifact's metrics."""
    path = results_dir / f"{stem}.json"
    if not path.is_file():
        raise PaperFigureError(f"missing artifact: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    return Artifact(stem, {m["name"]: float(m["value"]) for m in payload["metrics"]})


def _pyplot() -> Any:
    """Import matplotlib configured for print, only when a figure is drawn."""
    try:
        import matplotlib
    except ImportError as error:  # pragma: no cover - exercised by the extra's absence
        raise PaperFigureError(
            "drawing needs the analysis extra; run uv sync --extra analysis"
        ) from error
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # Sans-serif throughout, as the journal requires. DejaVu Sans ships with
    # matplotlib, so this does not depend on host fonts.
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["DejaVu Sans"],
            "font.size": 8,
            "axes.titlesize": 9,
            "axes.labelsize": 8,
            "legend.fontsize": 7.5,
            "xtick.labelsize": 7.5,
            "ytick.labelsize": 7.5,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )
    return plt


def _save(figure: Any, path: Path) -> Path:
    """Write one figure at publication density."""
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=PUBLICATION_DPI, bbox_inches="tight")
    return path


def geography_figure(results_dir: Path, output: Path) -> Path:
    """Fig. 3 — location narrowing hurts, and the mechanism decays with distance.

    Two panels replacing thirteen table rows: the penalty against pool size, and
    the co-location enrichment that explains it. Panel (b) is what makes the
    explanation falsifiable rather than a story -- the enrichment has to decay,
    and it does.
    """
    plt = _pyplot()
    figure, (left, right) = plt.subplots(1, 2, figsize=(7.2, 2.9))

    pools = (2, 3, 5, 8, 10, 15, 20, 50, 100, 200)
    for backbone in ("dinov2", "clip"):
        art = load(results_dir, f"geo_ablation-{backbone}")
        diffs, seen = [], []
        for pool in pools:
            geo = art.maybe(f"geo_top_1_pool_{pool}")
            rnd = art.maybe(f"random_top_1_pool_{pool}")
            if geo is None or rnd is None:
                continue
            diffs.append((geo - rnd) * 100)
            seen.append(pool)
        left.plot(
            seen,
            diffs,
            "o-",
            color=BACKBONE_COLOUR[backbone],
            label=BACKBONE_LABEL[backbone],
            linewidth=1.4,
            markersize=3.5,
        )
    left.axhline(0, color="#555555", linewidth=0.8, linestyle="--")
    left.set_xscale("log")
    left.set_xlabel("Candidate pool size")
    # A true minus sign rather than a hyphen: this is a printed axis label, and
    # the journal sets figures in a sans face where the two are visibly different.
    left.set_ylabel("Geographic − random (points)")  # noqa: RUF001
    left.set_title("(a) Narrowing by location costs accuracy", loc="left")
    left.legend(frameon=False)

    width = 0.38
    positions = np.arange(len(SEPARATION_BANDS), dtype=float)
    for offset, backbone in zip((-width / 2, width / 2), ("dinov2", "clip"), strict=True):
        art = load(results_dir, f"confusion-{backbone}")
        ratios = []
        for band in SEPARATION_BANDS:
            confused = art.maybe(f"confused_pairs_within_{band}m")
            competing = art.maybe(f"competing_pairs_within_{band}m")
            ratios.append(confused / competing if confused and competing else np.nan)
        right.bar(
            positions + offset,
            ratios,
            width,
            color=BACKBONE_COLOUR[backbone],
            label=BACKBONE_LABEL[backbone],
        )
    right.axhline(1.0, color="#555555", linewidth=0.8, linestyle="--")
    right.set_xticks(positions)
    right.set_xticklabels([f"{b} m" for b in SEPARATION_BANDS])
    right.set_xlabel("Statues separated by at most")
    right.set_ylabel("Confused ÷ competing")
    right.set_title("(b) Confusion is enriched only at very short range", loc="left")
    right.legend(frameon=False)

    figure.tight_layout()
    written = _save(figure, output)
    plt.close(figure)
    return written


def photographer_figure(results_dir: Path, output: Path) -> Path:
    """Fig. 5 — what a change of photographer costs, and what a phone costs.

    Panel (a) is the decomposition, not the raw drop: the size-matched control is
    what separates "fewer references" from "a different photographer", and
    without it the 12-point fall reads as a 12-point photographer effect.
    """
    plt = _pyplot()
    figure, (left, right) = plt.subplots(1, 2, figsize=(7.2, 2.9))

    arms = ("standard_top_1", "control_top_1", "disjoint_top_1")
    labels = ("Standard\nleave-one-out", "Size-matched\nrandom control", "Photographer\nwithheld")
    width = 0.38
    positions = np.arange(len(arms), dtype=float)
    for offset, backbone in zip((-width / 2, width / 2), ("dinov2", "clip"), strict=True):
        art = load(results_dir, f"photographer_gap-{backbone}")
        values = [art.get(a) * 100 for a in arms]
        bars = left.bar(
            positions + offset,
            values,
            width,
            color=BACKBONE_COLOUR[backbone],
            label=BACKBONE_LABEL[backbone],
        )
        left.bar_label(bars, fmt="%.1f", padding=1.5, fontsize=6.5)
    left.set_xticks(positions)
    left.set_xticklabels(labels)
    left.set_ylabel("Top-1 accuracy (%)")
    left.set_ylim(0, 105)
    left.set_title("(a) Only part of the drop is the photographer", loc="left")
    left.legend(frameon=False, loc="upper center", bbox_to_anchor=(0.5, 1.20), ncol=2)

    conditions = ("camera_top_1", "phone_top_1")
    cond_labels = ("Camera", "Phone")
    positions = np.arange(len(conditions), dtype=float)
    for offset, backbone in zip((-width / 2, width / 2), ("dinov2", "clip"), strict=True):
        art = load(results_dir, f"camera_gap-{backbone}")
        values = [art.get(c) * 100 for c in conditions]
        bars = right.bar(
            positions + offset,
            values,
            width,
            color=BACKBONE_COLOUR[backbone],
            label=BACKBONE_LABEL[backbone],
        )
        right.bar_label(bars, fmt="%.1f", padding=1.5, fontsize=6.5)
    right.set_xticks(positions)
    right.set_xticklabels(cond_labels)
    right.set_ylabel("Top-1 accuracy (%)")
    right.set_ylim(0, 105)
    right.set_title("(b) Phone-shot queries are harder", loc="left")
    right.legend(frameon=False, loc="upper center", bbox_to_anchor=(0.5, 1.20), ncol=2)

    figure.tight_layout()
    written = _save(figure, output)
    plt.close(figure)
    return written


def matcher_figure(results_dir: Path, output: Path) -> Path:
    """Fig. 6 — geometric re-ranking, the matcher that widens it, and the ceiling.

    The ceiling line is the point of the figure: both curves rise, and neither can
    pass the first stage's recall, because re-ranking only reorders what was
    already proposed.
    """
    plt = _pyplot()
    figure, axes = plt.subplots(figsize=(4.6, 3.2))

    art = load(results_dir, "matcher_rerank-disjoint-dinov2")
    weights = (0.0, 0.01, 0.02, 0.05, 0.1, 0.2, 0.3, 0.5, 0.75, 1.0, 1.5, 2.0)
    style = {
        "sift": ("#7a7a7a", "o", "SIFT"),
        "disk-lightglue": ("#1b6ca8", "s", "DISK + LightGlue"),
    }
    for matcher, (colour, marker, label) in style.items():
        values = []
        for w in weights:
            v = art.maybe(f"{matcher}_w{w:g}_top_1")
            values.append(v * 100 if v is not None else np.nan)
        axes.plot(
            weights, values, marker=marker, color=colour, label=label, linewidth=1.4, markersize=3.5
        )

    ceiling = load(results_dir, "recall_curve-dinov2").get("disjoint_r_at_10") * 100
    axes.axhline(ceiling, color="#c05640", linewidth=1.0, linestyle="--")
    axes.text(
        0.012,
        ceiling + 0.35,
        f"first stage's recall@10 ({ceiling:.1f}%)",
        color="#c05640",
        fontsize=7,
    )

    axes.set_xscale("symlog", linthresh=0.01)
    # Weights are non-negative; symlog would otherwise draw a mirrored negative
    # decade and invite the reader to wonder what a negative weight means.
    axes.set_xlim(left=0.0, right=2.4)
    axes.set_xlabel("Weight given to geometry in the blended score")
    axes.set_ylabel("Top-1 accuracy, photographer-disjoint (%)")
    axes.set_title("Verifying geometry, and the ceiling it runs into", loc="left")
    axes.legend(frameon=False, loc="lower left")

    figure.tight_layout()
    written = _save(figure, output)
    plt.close(figure)
    return written


def rejection_figure(results_dir: Path, output: Path) -> Path:
    """Fig. 7 — the system cannot decline to answer, on either signal.

    Panel (b) is why the negative result is credible: appearance and geometry are
    both measured, on both regimes, and neither separates known from unknown.
    """
    plt = _pyplot()
    figure, (left, right) = plt.subplots(1, 2, figsize=(7.2, 2.9))

    targets = (90, 95, 99)
    for backbone in ("dinov2", "clip"):
        art = load(results_dir, f"open_set-{backbone}")
        known = [art.get(f"target_{t}_known_acceptance") * 100 for t in targets]
        false = [art.get(f"target_{t}_false_acceptance") * 100 for t in targets]
        left.plot(
            false,
            known,
            "o-",
            color=BACKBONE_COLOUR[backbone],
            label=f"{BACKBONE_LABEL[backbone]} (AUROC {art.get('auroc'):.3f})",
            linewidth=1.4,
            markersize=4,
        )
        # The backbones converge at the 99% target, so one series is labelled to the
        # right of its markers and the other to the left. Offsetting vertically is
        # not enough when the two points nearly coincide.
        dx = 4 if backbone == "dinov2" else -24
        for t, f, k in zip(targets, false, known, strict=True):
            left.annotate(
                f"{t}%",
                (f, k),
                textcoords="offset points",
                xytext=(dx, -7),
                fontsize=6.5,
                color="#444444",
            )
    left.set_xlabel("Unknown statues wrongly accepted (%)")
    left.set_ylabel("Known statues accepted (%)")
    left.set_title("(a) No operating point is usable", loc="left")
    left.legend(frameon=False, loc="lower right")

    signals = ("cosine", "inliers_best", "blended")
    signal_labels = ("Appearance", "Geometry", "Blended")
    width = 0.38
    positions = np.arange(len(signals), dtype=float)
    for offset, regime, colour, label in (
        (-width / 2, "standard", "#1b6ca8", "All references"),
        (width / 2, "photographer_disjoint", "#c05640", "Photographer withheld"),
    ):
        art = load(results_dir, "open_set_geometry-dinov2")
        values = [art.get(f"{regime}_{s}_auroc") for s in signals]
        bars = right.bar(positions + offset, values, width, color=colour, label=label)
        right.bar_label(bars, fmt="%.3f", padding=1.5, fontsize=6.5)
    right.axhline(0.5, color="#555555", linewidth=0.8, linestyle="--")
    right.set_xticks(positions)
    right.set_xticklabels(signal_labels)
    right.set_ylabel("AUROC, known vs unknown")
    right.set_ylim(0.4, 1.0)
    right.set_title("(b) Geometry adds almost nothing", loc="left")
    right.legend(frameon=False, loc="upper center", bbox_to_anchor=(0.5, 1.20), ncol=2)

    figure.tight_layout()
    written = _save(figure, output)
    plt.close(figure)
    return written


FIGURES = {
    "fig3-geography": geography_figure,
    "fig5-photographer": photographer_figure,
    "fig6-matcher": matcher_figure,
    "fig7-rejection": rejection_figure,
}


def draw_all(results_dir: Path, output_dir: Path) -> tuple[Path, ...]:
    """Draw every data-derived paper figure."""
    return tuple(draw(results_dir, output_dir / f"{name}.png") for name, draw in FIGURES.items())

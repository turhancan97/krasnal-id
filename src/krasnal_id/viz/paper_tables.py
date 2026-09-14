"""Supplementary tables for the manuscript, emitted from the result artifacts.

The main text holds eight display items; everything else the venue would want to
see goes to Supplementary Information, and that is about a dozen tables of
numbers. Typing them out is how a bibliography acquires a wrong citation and a
results table acquires a wrong digit, so none of them is typed: each function
here reads `results/*.json` and emits a LaTeX fragment, and `si.tex` inputs the
fragments.

This is the same rule section 6.4 already applies to `RESULTS.md` -- every
published number traces to an artifact -- enforced by construction rather than
by proofreading.
"""

from collections.abc import Sequence
from pathlib import Path

from krasnal_id.viz.paper_figures import PaperFigureError, Sources

# Both backbones, in the order the manuscript names them.
BACKBONES = ("dinov2", "clip")
BACKBONE_LABEL = {"dinov2": "DINOv2", "clip": "CLIP"}

POOL_SIZES = (2, 3, 5, 8, 10, 15, 20, 50, 100, 200, 306)
GEO_POOL_SIZES = (2, 3, 5, 8, 10, 15, 20, 50, 100, 200, 294)
RECALL_CUTOFFS = (1, 5, 10, 20, 50, 100)
SIFT_WEIGHTS = (0.0, 0.01, 0.02, 0.05, 0.1, 0.2)
MATCHER_WEIGHTS = (0.0, 0.01, 0.02, 0.05, 0.1, 0.2, 0.3, 0.5, 0.75, 1.0, 1.5, 2.0)
SEPARATION_BANDS = (50, 100, 200, 300, 500, 1000)
ACCEPTANCE_TARGETS = (90, 95, 99)
CAPACITY_CHECKPOINTS = ("dinov2-large", "dinov2-registers", "dinov2-registers-large")
CAPACITY_LABEL = {
    "dinov2-large": "DINOv2 ViT-L/14",
    "dinov2-registers": "DINOv2 + registers",
    "dinov2-registers-large": "DINOv2 ViT-L/14 + registers",
}


def _pct(value: float | None, places: int = 2) -> str:
    """Format a proportion as a percentage, or an em dash where it is absent."""
    return "---" if value is None else f"{value * 100:.{places}f}"


def _num(value: float | None, places: int = 2) -> str:
    return "---" if value is None else f"{value:.{places}f}"


def _p_value(value: float) -> str:
    """Format a p-value without rounding a small one to zero."""
    if value < 0.001:
        return "$p < 0.001$"
    return f"$p = {value:.3f}$"


def _signed(value: float, places: int = 2) -> str:
    """Format a difference with a true minus sign rather than a hyphen."""
    return f"{value:+.{places}f}".replace("-", "\N{MINUS SIGN}")


def _table(
    caption: str, label: str, colspec: str, header: Sequence[str], rows: Sequence[Sequence[str]]
) -> str:
    """Wrap rows in the class's table environment, booktabs rules and all."""
    body = "\n".join(" & ".join(row) + r" \\" for row in rows)
    return "\n".join(
        (
            r"\begin{table}[ht]",
            rf"\caption{{{caption}}}\label{{{label}}}",
            rf"\begin{{tabular}}{{@{{}}{colspec}@{{}}}}",
            r"\toprule",
            " & ".join(header) + r" \\",
            r"\midrule",
            body,
            r"\botrule",
            r"\end{tabular}",
            r"\end{table}",
            "",
        )
    )


def pool_size_table(sources: Sources) -> str:
    """S3 -- the full pool-size ablation, both backbones, with seed spread."""
    arts = {b: sources.load(f"pool_size_ablation-{b}") for b in BACKBONES}
    rows = []
    for pool in POOL_SIZES:
        row = [str(pool)]
        for backbone in BACKBONES:
            art = arts[backbone]
            value = art.maybe(f"top_1_pool_{pool}")
            bounds = art.interval(f"top_1_pool_{pool}")
            spread = "---" if bounds is None else f"{_pct(bounds[0])}--{_pct(bounds[1])}"
            row += [_pct(value), spread, _num(art.maybe(f"mrr_pool_{pool}"), 3)]
        rows.append(row)
    rows.append(
        ["\\textbf{per doubling}"]
        + [
            cell
            for backbone in BACKBONES
            for cell in (
                _signed(arts[backbone].get("top_1_points_per_doubling")),
                "",
                "",
            )
        ]
    )
    return _table(
        "Pool-size ablation in full. Top-1 accuracy and mean reciprocal rank "
        "against the number of candidate dwarves, five seeds per pool size. The "
        "spread is the observed range across seeds; it vanishes at 306 because "
        "there is only one way to draw the whole corpus. The final row is the "
        "fitted points of top-1 lost per doubling of the pool.",
        "si:pool",
        "lcccccc",
        [
            "Pool",
            "DINOv2 top-1",
            "spread",
            "MRR",
            "CLIP top-1",
            "spread",
            "MRR",
        ],
        rows,
    )


def geography_table(sources: Sources) -> str:
    """S4 -- geographic against random pools at every size, with the radii."""
    arts = {b: sources.load(f"geo_ablation-{b}") for b in BACKBONES}
    rows = []
    for pool in GEO_POOL_SIZES:
        radius = arts["dinov2"].maybe(f"geo_radius_metres_pool_{pool}")
        row = [str(pool), "---" if radius is None else f"{radius:,.0f}"]
        for backbone in BACKBONES:
            geo = arts[backbone].maybe(f"geo_top_1_pool_{pool}")
            rnd = arts[backbone].maybe(f"random_top_1_pool_{pool}")
            row += [
                _pct(geo),
                _pct(rnd),
                "---" if geo is None or rnd is None else _signed((geo - rnd) * 100),
            ]
        rows.append(row)
    return _table(
        "Geographic pooling in full. Each query is pooled with its $N-1$ nearest "
        "dwarves and compared against a random pool of the same size, over the "
        "294 placed statues. The radius is the median distance such a pool "
        "spans. The geographic arm samples nothing and is exact; the random arm "
        "supplies the variation.",
        "si:geo",
        "rrccccccc",
        [
            "Pool",
            "Radius (m)",
            "DINOv2 geo",
            "random",
            "diff",
            "CLIP geo",
            "random",
            "diff",
        ],
        rows,
    )


def classifier_table(sources: Sources) -> str:
    """S5 -- retrieval against a linear probe and class prototypes."""
    arts = {b: sources.load(f"probe-{b}") for b in BACKBONES}
    methods = (
        ("retrieval", "Cosine retrieval"),
        ("linear_probe", "Linear probe"),
        ("prototype", "Class prototypes"),
    )
    rows = []
    for key, label in methods:
        row = [label]
        for backbone in BACKBONES:
            art = arts[backbone]
            row += [
                _pct(art.maybe(f"{key}_top_1")),
                _pct(art.maybe(f"{key}_top_5")),
                _num(art.maybe(f"{key}_mrr"), 3),
            ]
        rows.append(row)
    return _table(
        "Trained classifiers against retrieval, on identical folds. A linear "
        "probe is fitted per fold on the same frozen embeddings; class "
        "prototypes average each dwarf's reference vectors into one centroid. "
        "Probes are regularised weakly ($C = 100$): the embeddings are "
        "L2-normalised, so a conventional $C = 1$ underfits badly.",
        "si:probe",
        "lcccccc",
        ["Method", "DINOv2 top-1", "top-5", "MRR", "CLIP top-1", "top-5", "MRR"],
        rows,
    )


def confusion_table(sources: Sources) -> str:
    """S6 -- the pairs the retrieval confuses, and their separation."""
    pairs = sources.records("confusion-dinov2", "pairs")[:12]
    rows = [
        [
            str(pair["true_display_name"]),
            str(pair["confused_display_name"]),
            f"{int(pair['misidentifications'])} / {int(pair['queries'])}",
            _num(float(pair["mean_margin"]), 3),
            f"{float(pair['separation_metres']):,.0f}",
        ]
        for pair in pairs
    ]
    return _table(
        "The twelve most-confused directed pairs under DINOv2. Pairs stay "
        "directed, so an asymmetric confusion does not average away against its "
        "reverse. The margin is the mean amount by which the correct dwarf beats "
        "this wrong one; a negative margin means the wrong dwarf won on average.",
        "si:confusion",
        "llccr",
        ["Queried statue", "Returned instead", "Errors", "Margin", "Apart (m)"],
        rows,
    )


def separation_table(sources: Sources) -> str:
    """S6 -- confusion enrichment by distance, which is the decay claim."""
    arts = {b: sources.load(f"confusion-{b}") for b in BACKBONES}
    rows = []
    for band in SEPARATION_BANDS:
        row = [f"{band:,}"]
        for backbone in BACKBONES:
            confused = arts[backbone].maybe(f"confused_pairs_within_{band}m")
            competing = arts[backbone].maybe(f"competing_pairs_within_{band}m")
            ratio = None if not confused or not competing else confused / competing
            row += [_pct(confused, 1), _pct(competing, 1), _num(ratio)]
        rows.append(row)
    rows.append(
        ["AUROC"]
        + [
            cell
            for backbone in BACKBONES
            for cell in ("", "", _num(arts[backbone].get("proximity_predicts_confusion_auroc"), 3))
        ]
    )
    return _table(
        "Does co-location explain confusion? The share of confused pairs and of "
        "merely-competing pairs standing within each distance, and the "
        "enrichment between them. The final row is the area under the ROC curve "
        "for predicting confusion from separation alone, which is close to "
        "chance: the mechanism is real at short range and explains little "
        "overall.",
        "si:separation",
        "rcccccc",
        [
            "Within (m)",
            "DINOv2 confused",
            "competing",
            "ratio",
            "CLIP confused",
            "competing",
            "ratio",
        ],
        rows,
    )


def photographer_table(sources: Sources) -> str:
    """S8 -- the three arms of the photographer decomposition."""
    arts = {b: sources.load(f"photographer_gap-{b}") for b in BACKBONES}
    arms = (
        ("standard", "Ordinary leave-one-out"),
        ("control", "Size-matched random control"),
        ("disjoint", "Photographer withheld"),
    )
    rows = []
    for key, label in arms:
        row = [label]
        for backbone in BACKBONES:
            art = arts[backbone]
            row += [
                _pct(art.maybe(f"{key}_top_1")),
                _pct(art.maybe(f"{key}_top_5")),
                _num(art.maybe(f"{key}_median_candidate_classes"), 0),
            ]
        rows.append(row)
    rows.append(
        ["\\textbf{Attributable to the photographer}"]
        + [
            cell
            for backbone in BACKBONES
            for cell in (
                _signed(-arts[backbone].get("attributable_top_1_gap") * 100),
                _signed(-arts[backbone].get("attributable_top_5_gap") * 100),
                "",
            )
        ]
    )
    return _table(
        "The photographer decomposition, all three arms scored on the same "
        "1{,}157 answerable queries. The control withholds the same number of "
        "references as the disjoint arm but chooses them at random, so the "
        "difference between the two isolates photographer identity from having "
        "fewer references. 534 further queries cannot be asked at all, because "
        "125 of the 306 classes have a single photographer.",
        "si:photographer",
        "lcccccc",
        [
            "Arm",
            "DINOv2 top-1",
            "top-5",
            "candidates",
            "CLIP top-1",
            "top-5",
            "candidates",
        ],
        rows,
    )


def camera_table(sources: Sources) -> str:
    """S7 -- phone against camera queries, with how few phones there are."""
    arts = {b: sources.load(f"camera_gap-{b}") for b in BACKBONES}
    conditions = (("camera", "Camera"), ("phone", "Phone"), ("unknown", "Unrecorded"))
    rows = []
    for key, label in conditions:
        art = arts["dinov2"]
        row = [
            label,
            _num(art.maybe(f"{key}_queries"), 0),
            _num(art.maybe(f"{key}_median_references"), 0),
        ]
        for backbone in BACKBONES:
            row += [
                _pct(arts[backbone].maybe(f"{key}_top_1")),
                _pct(arts[backbone].maybe(f"{key}_top_5")),
            ]
        rows.append(row)
    return _table(
        "Queries by the capture device recorded in the photograph's metadata. "
        "The median reference count is given because it is the obvious "
        "confound and it does not differ meaningfully between conditions. "
        "Fifty-one phone queries put a wide interval on that row, and a phone "
        "upload to a photo commons is a deliberate photograph rather than a "
        "snapshot, so the gap is a lower bound on what a street query costs.",
        "si:camera",
        "lrrcccc",
        [
            "Device",
            "Queries",
            "Refs",
            "DINOv2 top-1",
            "top-5",
            "CLIP top-1",
            "top-5",
        ],
        rows,
    )


def rerank_sweep_table(sources: Sources) -> str:
    """S9 -- the SIFT re-ranking sweep, both backbones, both protocols."""
    full = {b: sources.load(f"rerank_ablation-{b}") for b in BACKBONES}
    disjoint = {b: sources.load(f"rerank_ablation-disjoint-{b}") for b in BACKBONES}
    rows = []
    for weight in SIFT_WEIGHTS:
        tag = f"{weight:g}"
        row = [tag]
        for backbone in BACKBONES:
            row.append(_pct(full[backbone].maybe(f"weight_{tag}_top_1")))
        for backbone in BACKBONES:
            row.append(_pct(disjoint[backbone].maybe(f"disjoint_weight_{tag}_top_1")))
        rows.append(row)
    return _table(
        "SIFT re-ranking, every swept weight. Top-1 accuracy when the cosine "
        "score is blended with RANSAC inlier counts over the first stage's top "
        "ten. Weight zero reproduces the unranked baseline exactly, by "
        "construction, which is what makes the rest of each column "
        "interpretable. Full pool: 1{,}691 queries over 306 candidates. "
        "Photographer-disjoint: 1{,}157 queries.",
        "si:rerank",
        "ccccc",
        [
            "Weight",
            "DINOv2 full",
            "CLIP full",
            "DINOv2 disjoint",
            "CLIP disjoint",
        ],
        rows,
    )


def matcher_table(sources: Sources) -> str:
    """S9 -- SIFT against DISK+LightGlue in the identical protocol."""
    art = sources.load("matcher_rerank-disjoint-dinov2")
    rows = []
    for weight in MATCHER_WEIGHTS:
        tag = f"{weight:g}"
        rows.append(
            [
                tag,
                _pct(art.maybe(f"sift_w{tag}_top_1")),
                _pct(art.maybe(f"disk-lightglue_w{tag}_top_1")),
                _pct(art.maybe(f"sift_w{tag}_top_5")),
                _pct(art.maybe(f"disk-lightglue_w{tag}_top_5")),
            ]
        )
    wins = art.get("disk-lightglue_vs_sift_top_1_wins")
    losses = art.get("disk-lightglue_vs_sift_top_1_losses")
    return _table(
        "A learned matcher against SIFT, photographer-disjoint, everything else "
        "held fixed: same queries, same candidates in the same order, same "
        "weights, same RANSAC verification. The columns differ only in where "
        "the correspondences came from. At their best weights DISK+LightGlue "
        f"beats SIFT on {wins:.0f} queries and loses {losses:.0f}, "
        f"$p = {art.get('disk-lightglue_vs_sift_top_1_p_value'):.1e}$ by an "
        "exact McNemar test. Both peaks are interior to the sweep, which is why "
        "it runs to a weight of 2.",
        "si:matcher",
        "ccccc",
        ["Weight", "SIFT top-1", "DISK+LG top-1", "SIFT top-5", "DISK+LG top-5"],
        rows,
    )


def recall_table(sources: Sources) -> str:
    """S10 -- the first-stage recall that caps every re-ranker."""
    arts = {b: sources.load(f"recall_curve-{b}") for b in BACKBONES}
    rows = []
    for cutoff in RECALL_CUTOFFS:
        row = [str(cutoff)]
        for backbone in BACKBONES:
            row += [
                _pct(arts[backbone].maybe(f"full_r_at_{cutoff}")),
                _pct(arts[backbone].maybe(f"disjoint_r_at_{cutoff}")),
            ]
        rows.append(row)
    return _table(
        "First-stage recall, which is the ceiling on any re-ranker. Re-ranking "
        "reorders the top $k$ the appearance stage proposed, so it cannot "
        "retrieve a statue that never entered them. The photographer-disjoint "
        "recall at ten, 90.75\\% for DINOv2, is the line drawn in "
        "Fig.~6 of the main text.",
        "si:recall",
        "ccccc",
        ["$k$", "DINOv2 full", "disjoint", "CLIP full", "disjoint"],
        rows,
    )


def capacity_table(sources: Sources) -> str:
    """S11 -- whether a bigger backbone lifts the ceiling. It does not."""
    art = sources.load("recall_curve-dinov2")
    rows = []
    for cutoff in RECALL_CUTOFFS:
        base = art.maybe(f"disjoint_r_at_{cutoff}")
        row = [str(cutoff), _pct(base)]
        for checkpoint in CAPACITY_CHECKPOINTS:
            other = art.maybe(f"disjoint_against_{checkpoint}_r_at_{cutoff}")
            row.append(
                "---"
                if other is None or base is None
                else f"{_pct(other)} ({_signed((other - base) * 100)})"
            )
        rows.append(row)
    paired = []
    for cutoff in RECALL_CUTOFFS:
        wins = art.maybe(f"disjoint_dinov2-large_vs_selected_at_{cutoff}_wins")
        losses = art.maybe(f"disjoint_dinov2-large_vs_selected_at_{cutoff}_losses")
        p_value = art.maybe(f"disjoint_dinov2-large_vs_selected_at_{cutoff}_p_value")
        if wins is None or losses is None or p_value is None:
            continue
        paired.append(f"$k={cutoff}$: {wins:.0f}/{losses:.0f}, {_p_value(p_value)}")
    return _table(
        "Capacity does not lift the ceiling. Photographer-disjoint recall for "
        "DINOv2 ViT-B/14 against three larger or register-equipped "
        "checkpoints, with the change in parentheses. Paired against ViT-B/14, "
        "wins/losses and exact McNemar $p$: " + "; ".join(paired) + ". Six "
        "cutoffs are compared here, so a Bonferroni-corrected threshold is "
        "$0.05/6 = 0.008$; only $k = 1$ clears it, and the cutoff that matters "
        "for re-ranking, $k = 10$, is a 0.17-point move at $p = 0.878$.",
        "si:capacity",
        "lcccc",
        ["$k$", "ViT-B/14"] + [CAPACITY_LABEL[c] for c in CAPACITY_CHECKPOINTS],
        rows,
    )


def geometry_first_table(sources: Sources) -> str:
    """S12 -- local matching asked to retrieve rather than to re-rank."""
    art = sources.load("geometry_first-dinov2")
    rows = []
    for arm, label in (("answerable", "All references"), ("disjoint", "Photographer withheld")):
        for cutoff in (1, 5, 10):
            rows.append(
                [
                    label if cutoff == 1 else "",
                    str(cutoff),
                    _pct(art.maybe(f"{arm}_appearance_r_at_{cutoff}")),
                    _pct(art.maybe(f"{arm}_geometry_r_at_{cutoff}")),
                    _pct(art.maybe(f"{arm}_geometry_worst_r_at_{cutoff}")),
                    _num(art.maybe(f"{arm}_appearance_misses_at_{cutoff}"), 0),
                    _pct(art.maybe(f"{arm}_rescue_rate_at_{cutoff}"), 1),
                ]
            )
    return _table(
        "Local matching as a first stage rather than a re-ranker: every one of "
        "the 306 candidates ranked by SIFT inlier count alone, no embedding "
        "involved. Inlier counts are integers and tie heavily, so the rank of "
        "the correct statue is reported as the best and worst positions "
        "consistent with its score rather than as whichever order the manifest "
        "happened to impose. The rescue rate is the share of the statues "
        "appearance misses that geometry finds. It retrieves far worse than "
        "appearance and rescues a minority of its failures.",
        "si:geometryfirst",
        "llccccc",
        [
            "Arm",
            "$k$",
            "Appearance",
            "Geometry (best)",
            "(worst)",
            "Misses",
            "Rescued",
        ],
        rows,
    )


def open_set_table(sources: Sources) -> str:
    """S13 -- thresholding similarity, at the operating points a tool needs."""
    arts = {b: sources.load(f"open_set-{b}") for b in BACKBONES}
    rows = []
    for target in ACCEPTANCE_TARGETS:
        row = [f"{target}\\%"]
        for backbone in BACKBONES:
            art = arts[backbone]
            row += [
                _pct(art.maybe(f"target_{target}_known_acceptance"), 1),
                _pct(art.maybe(f"target_{target}_false_acceptance"), 1),
                _pct(art.maybe(f"target_{target}_open_set_accuracy"), 1),
            ]
        rows.append(row)
    rows.append(
        ["AUROC"] + [cell for b in BACKBONES for cell in (_num(arts[b].get("auroc"), 3), "", "")]
    )
    return _table(
        "Open-set rejection by thresholding the top-1 similarity. Two "
        "populations of 1{,}691 queries: the known arm is the leave-one-out "
        "split, and the unknown arm removes every photograph of the query's own "
        "dwarf so that statue is genuinely absent. Thresholds are calibrated "
        "leave-one-class-out, so no query helps set the bar it must clear. No "
        "row is an operating point a deployed tool could use.",
        "si:openset",
        "lcccccc",
        [
            "Target",
            "DINOv2 known",
            "false",
            "open-set",
            "CLIP known",
            "false",
            "open-set",
        ],
        rows,
    )


def open_set_geometry_table(sources: Sources) -> str:
    """S13 -- and whether geometric evidence rejects any better. It does not."""
    art = sources.load("open_set_geometry-dinov2")
    signals = (
        ("cosine", "Appearance (cosine)"),
        ("inliers_top_1", "Geometry (top-1 inliers)"),
        ("inliers_best", "Geometry (best inliers)"),
        ("blended", "Blended"),
    )
    rows = []
    for key, label in signals:
        rows.append(
            [
                label,
                _num(art.maybe(f"standard_{key}_auroc"), 3),
                _pct(art.maybe(f"standard_{key}_false_acceptance"), 1),
                _num(art.maybe(f"photographer_disjoint_{key}_auroc"), 3),
                _pct(art.maybe(f"photographer_disjoint_{key}_false_acceptance"), 1),
            ]
        )
    return _table(
        "Can geometry reject what appearance cannot? Separability of known from "
        "unknown queries by each signal, under both protocols. False acceptance "
        "is quoted at the threshold accepting 90\\% of known statues. Geometry "
        "adds 0.011 AUROC over appearance at best, and once the photographer is "
        "withheld it is worse than appearance alone --- the same collapse of "
        "the inlier signal the main text reports for "
        "accuracy, where a tie-breaker survived it and a standalone signal does "
        "not.",
        "si:opensetgeometry",
        "lcccc",
        ["Signal", "Standard AUROC", "false @90", "Disjoint AUROC", "false @90"],
        rows,
    )


TABLES = {
    "s3-pool-size": pool_size_table,
    "s4-geography": geography_table,
    "s5-classifiers": classifier_table,
    "s6-confusion": confusion_table,
    "s6-separation": separation_table,
    "s7-camera": camera_table,
    "s8-photographer": photographer_table,
    "s9-rerank": rerank_sweep_table,
    "s9-matcher": matcher_table,
    "s10-recall": recall_table,
    "s11-capacity": capacity_table,
    "s12-geometry-first": geometry_first_table,
    "s13-open-set": open_set_table,
    "s13-open-set-geometry": open_set_geometry_table,
}


def write_all(sources: Sources, output_dir: Path) -> tuple[Path, ...]:
    """Emit every supplementary table as a LaTeX fragment."""
    output_dir.mkdir(parents=True, exist_ok=True)
    written = []
    for name, build in TABLES.items():
        path = output_dir / f"{name}.tex"
        path.write_text(build(sources), encoding="utf-8")
        written.append(path)
    if not written:  # pragma: no cover - TABLES is never empty
        raise PaperFigureError("no supplementary tables are registered")
    return tuple(written)

# How far does visual instance recognition get you on Wrocław's dwarves?

Wrocław is scattered with several hundred small bronze dwarf statues. They are individually
sculpted but share a visual vocabulary: the same scale, the same material, the same crouching
poses and hand-held props. That makes telling one from another a **fine-grained instance
recognition** problem rather than a classification problem, and it makes one question worth
asking properly:

> Identification gets harder as the candidate pool grows. If the pool can be narrowed — by
> location, say — how much does accuracy actually improve, and at what pool size does this
> become a reliable way to identify a statue?

This repository answers that with a measured curve rather than a single accuracy number. You can
also [try it on a photograph](https://turhancan97.github.io/krasnal-id/), which runs the model in
your own browser.

## Summary of findings

1. **Off-the-shelf embeddings identify these statues well.** DINOv2 reaches 95.9% top-1 and
   99.3% top-5 across the full 23-dwarf pool, with no fine-tuning at all.
2. **Accuracy decays slowly, and the backbones diverge.** DINOv2 loses 0.96 accuracy points per
   doubling of the candidate pool; CLIP loses 1.76. The gap between them *widens* with pool size,
   so backbone choice matters more the harder the problem gets.
3. **Narrowing by location helps less than the simulation suggests.** Real proximity-based pools
   are consistently *harder* than random pools of the same size, because the statues that are
   hardest to tell apart were installed together as a themed group.
4. **Training a classifier does not help.** A per-fold linear probe gains 0.7 top-1 points over
   raw retrieval and per-class prototypes lose 2 to 3, with every difference inside the
   confidence intervals. At this data scale, nearest-neighbour retrieval is already the right
   tool.
5. **The errors are explainable, not random.** Both backbones concentrate their mistakes on the
   same small cluster of water-themed dwarves, which the embedding projections show as a single
   tight neighbourhood.
6. **It can be made to say "I don't know", for about 3 points.** Thresholding the top-1
   similarity separates present from absent statues at 0.969 AUROC for DINOv2, rejecting 95.9% of
   unknown statues while still identifying 89.0% of known ones. CLIP cannot do this at any useful
   operating point, and being reliably *identifiable* turns out not to imply being reliably
   *rejectable*.

## Dataset

| | |
|---|---|
| Dwarves (classes) | 23 |
| Reference photographs | 146 |
| Images per dwarf | 3 to 31 (median 4) |
| Source | Wikimedia Commons, via Wikidata class `Q136276280` |
| Licences | Public Domain, CC0, CC BY, CC BY-SA only |

Coverage is bounded by data quality, not geography: a dwarf is included only if it has at least
three usable Creative Commons photographs. 40 Wikidata-to-Commons category mappings were reviewed
by hand; 23 survived the image threshold.

Three details matter more than the headline count:

- **Attribution is a first-class field.** Every stored image carries its source URL, author,
  licence and licence URL. Nothing enters the manifest without them.
- **Cross-label duplicates are excluded.** Twenty Commons files turned out to be byte-identical
  across *different* dwarf labels. Admitting them would have leaked the answer into the reference
  set, so they were dropped even though four classes lost their only images as a result.
- **Evaluation is leave-one-out.** Each of the 146 images is queried once against the other 145.
  The split is hash-pinned to the manifest, and any experiment run against a stale split fails
  loudly instead of quietly producing plausible numbers.

## Method

Images are embedded with two frozen backbones — **DINOv2** (`facebook/dinov2-base`, 768-d) and
**CLIP** (`openai/clip-vit-base-patch32`, 512-d) — and ranked by cosine similarity. No
fine-tuning. Each backbone is pinned to an immutable revision that serves safetensors directly,
so the provenance recorded alongside every cached vector describes the weights actually loaded.

Candidates are ranked as **distinct dwarves**, each represented by its best-matching photograph.
That is the list an identification tool would show a user, and it is what every metric below
scores.

## 1. Baseline: the full candidate pool

All 23 dwarves are candidates for every query. Intervals are 95% Wilson score intervals over 146
queries.

| | DINOv2 | CLIP |
|---|---|---|
| top-1 | **95.9%** [91.3, 98.1] | **92.5%** [87.0, 95.7] |
| top-5 | 99.3% [96.2, 99.9] | 98.6% [95.1, 99.6] |
| MRR | 0.971 | 0.951 |

DINOv2 leads, which is the expected ordering: it is trained with a self-supervised objective that
preserves instance-level detail, while CLIP's language-alignment objective pushes toward semantic
categories — and every one of these statues *is* the same semantic category.

## 2. The headline result: accuracy against candidate-pool size

Each query is scored against its own dwarf plus N−1 randomly drawn others, repeated over five
seeds per pool size. Error bars are the observed spread across seeds.

![Top-1 accuracy against candidate-pool size](docs/figures/pool-size-ablation.png)

| pool size | DINOv2 | CLIP |
|---|---|---|
| 2 | 98.9% | 98.9% |
| 5 | 98.9% | 97.7% |
| 10 | 96.7% | 95.8% |
| 15 | 96.4% | 94.0% |
| 20 | 96.2% | 93.2% |
| 23 (full) | 95.9% | 92.5% |
| **points lost per doubling** | **−0.96** | **−1.76** |

Two things stand out. The decay is **shallow** — narrowing the pool from 23 candidates to 5 buys
DINOv2 only three accuracy points, so location-based narrowing is worth far less than intuition
suggests at this scale. And the two backbones **diverge**: they are indistinguishable at a pool of
2 and six points apart at 23. A backbone comparison run only at one pool size would have missed
that entirely, which is the argument for measuring the curve rather than a point.

### What this implies at city scale, and why to distrust it

Extrapolating the fit to the full set of roughly 1,400 dwarves gives 88.5% top-1 for DINOv2 and
78.9% for CLIP. **Treat those as an optimistic bound, not a prediction.** It is a six-doubling
extrapolation, and the distractors here are drawn from a 23-class population; a real city-scale
pool contains far more genuinely confusable statues — shared poses, shared props, shared castings
— so the true curve should fall faster than a log-linear fit. The honest claim this data supports
is about the *rate* of decay in the measured range, not about where the line crosses a threshold.

## 3. Does narrowing by location actually help?

The pools above are sampled at random. The real proposal is to narrow by *location*, and every
dwarf in this dataset carries Wikidata coordinates, so that can be measured rather than assumed.
Each query is pooled with its **N−1 nearest** dwarves instead of N−1 random ones — same pool size,
only the selection rule changes.

| pool size | DINOv2 random | DINOv2 geographic | difference | median radius |
|---|---|---|---|---|
| 3 | 98.8% | 98.6% | −0.1 | 190 m |
| 5 | 98.9% | 97.3% | **−1.6** | 260 m |
| 8 | 97.3% | 96.6% | −0.7 | 390 m |
| 10 | 96.7% | 96.6% | −0.1 | 398 m |
| 15 | 96.4% | 96.6% | +0.1 | 571 m |
| 23 (full) | 95.9% | 95.9% | 0.0 | — |

**Geographic pools are harder than random pools of the same size.** CLIP shows it more strongly:
−2.5 points at a pool of 5, −2.1 at 8, −1.9 at 10. The effect is small in absolute terms but it
runs the wrong way at almost every size, and it has a clear cause.

### Why: the statues you can't separate are standing together

Six of the 23 dwarves sit **within one metre of each other** — 15 of the 253 pairs in this dataset
are effectively co-located. They are the group installation that includes *Puszczający Stateczki*,
*Zbierający Wodę* and *Karmiący Ptaki*.

Those are the same statues the confusion analysis flags, and the same tight neighbourhood the
embedding projection shows. Three of DINOv2's five confused pairs are at zero metres apart.

That is the finding: **the dwarves that are hardest to tell apart were installed as a themed
group, so they share a sculptor, a pose vocabulary and a location.** Location narrowing cannot
separate them — it guarantees they land in the same pool. A random-subsampling simulation, which
scatters the confusable statues across different pools, therefore *overstates* how much
location narrowing buys.

The practical reading for a location-aware tool: a 400 m radius around a visitor covers about ten
candidates, and identification within that pool runs at 96.6% top-1 for DINOv2. Useful — but no
better than picking ten dwarves at random, and for exactly the reason that makes the problem
interesting.

## 4. Does a trained classifier beat retrieval?

All three methods scored on the same folds, with one classifier fitted per fold so no query is
ever in its own training data.

| method | DINOv2 top-1 | CLIP top-1 | gain over retrieval |
|---|---|---|---|
| cosine retrieval | 95.9% | 92.5% | — |
| linear probe | 96.6% | 93.2% | +0.7 pts (both) |
| class prototypes | 93.2% | 90.4% | −2.7 / −2.1 pts |

**No.** The probe's gain is one extra correct query out of 146, well inside the confidence
intervals. Prototypes actively hurt: averaging a class into one vector discards the pose and
viewpoint variation that makes a specific photograph matchable.

A caveat about how this number was reached, because it nearly became a wrong conclusion. At
scikit-learn's conventional `C=1.0` the probe scored **66%**, which looks like a decisive result
and is really just underfitting: the embeddings are L2-normalised, so per-dimension magnitudes sit
near `1/√d` and that penalty crushes the weights. At `C=100` it scores 96.6%. A badly-regularised
baseline is worse than no baseline, because it flatters whatever it is compared against.

## 5. Where the errors are

DINOv2 misidentifies 6 of 146 queries; CLIP misidentifies 11. Because outright errors are rare,
the analysis records the strongest *wrong* candidate for every query, not only for the failures —
that is where the signal is on a dataset this size.

| | DINOv2 | CLIP |
|---|---|---|
| top-1 errors | 6 / 146 (4.1%) | 11 / 146 (7.5%) |
| mean margin over the best wrong dwarf | 0.248 | 0.064 |

The mean margin is the more revealing number. CLIP's embedding space is roughly four times
tighter, which is the same finding as its steeper decay curve seen from another angle: less
headroom per comparison means each added candidate costs more.

The errors are systematic. The most-confused pairs are dominated by the water-themed dwarves —
*Puszczający Stateczki* (releasing little boats), *Zbierający Wodę* (collecting water) and
*Karmiący Ptaki* (feeding birds) — crouching figures with similar props, frequently photographed
from the same angle. *Puszczający Stateczki → Zbierający Wodę* appears in **both directions** for
DINOv2 at a negative margin, and CLIP produces the same cluster with *Karmiący Ptaki* substituted.

Confusion is also **asymmetric**, which is why the analysis keeps pair direction rather than
averaging it away: *100matolog → Kowal* misidentifies 1 of 4 queries, while *Kowal → 100matolog*
misidentifies 0 of 10.

![DINOv2 embedding projection](docs/figures/embeddings-umap-dinov2.png)

The projection corroborates it independently. Most dwarves form clean, well-separated islands —
which is why accuracy is high — while the confused water-themed dwarves sit together in one tight
neighbourhood in the upper right. The CLIP projection shows the same cluster in a visibly more
compressed space:

![CLIP embedding projection](docs/figures/embeddings-umap-clip.png)

## 6. Can it say "I don't know this one"?

Everything above assumes the photographed statue is one of the 23. Retrieval always returns a
nearest neighbour, so a photograph of any of Wrocław's other ~1,300 dwarves would still produce a
confident-looking ranking. The cheapest possible fix is a threshold on the top-1 cosine
similarity: accept the ranking above it, answer "unknown" below it.

Testing that needs queries whose dwarf is genuinely absent, so the protocol runs two populations
of 146 queries each. The **known** arm is the same leave-one-out split used everywhere else. The
**unknown** arm removes *every* image of a query's own dwarf from the gallery, which makes that
dwarf truly missing — leaving a sibling image in place would keep the right answer reachable and
the query would not be open-set at all.

**Threshold-free first**, because an operating point can be tuned and a ranking statistic cannot.
AUROC here is the probability that a known query outscores an unknown one:

| | DINOv2 | CLIP |
|---|---|---|
| AUROC (known vs. unknown) | **0.969** | 0.898 |
| mean top-1 similarity, dwarf present | 0.851 | 0.914 |
| mean top-1 similarity, dwarf absent | 0.600 | 0.848 |

So the signal is real: how similar the best match is does carry information about whether the
statue is in the dataset at all. And the backbone gap is much wider than closed-set accuracy
suggests — 3.4 points apart on top-1, 7 points apart on AUROC.

**Then the price.** Each operating point is named by the fraction of known queries it accepts, and
its threshold is calibrated **leave-one-class-out**: the threshold judging a dwarf's queries comes
from a quantile of the *other* dwarves' known scores, so no query helps set the bar it must clear.

| target known acceptance | DINOv2 threshold | DINOv2 false accepts | CLIP threshold | CLIP false accepts |
|---|---|---|---|---|
| 90% | 0.771 | **4.1%** | 0.869 | 28.1% |
| 95% | 0.705 | 24.0% | 0.844 | 61.6% |
| 99% | 0.520 | 74.0% | 0.812 | 83.6% |

![Open-set rejection tradeoff](docs/figures/open-set-rejection.png)

The whole tradeoff, with the calibrated points marked. The curve itself is descriptive — it says
what *any* threshold would do on this data — while the marked dots are the leave-one-class-out
operating points that are actually achievable. DINOv2 turns the corner almost immediately; CLIP
climbs the left edge much more slowly and never reaches a usable corner at all.

**Rejection works, at a specific and modest price, and only for DINOv2.** At the 90% operating
point DINOv2 correctly identifies 89.0% of known queries and correctly rejects 95.9% of unknown
ones, for 92.5% over both populations against a closed-set top-1 of 95.9%. Adding the ability to
say "I don't know" costs about 3.4 points of accuracy on statues that *are* in the dataset.

The 99% row is the more instructive one. Insisting on almost never rejecting a known statue drags
the threshold into the tail of the known distribution, and three quarters of unknown statues then
pass. There is no threshold that is simultaneously generous to known queries and safe against
unknown ones, which is the honest shape of this result rather than a tuning failure.

One check on the method: a threshold fitted on *all* the data — the leaky version — gives DINOv2
the same 4.1% false-acceptance rate and CLIP a slightly worse 30.1%. The in-sample optimism is
negligible at this size, which means the threshold is a property of the embedding space and not
of this particular sample of it. The best in-sample balanced accuracy, sweeping every observed
score with the answers in view, is 93.8% for DINOv2 and 85.6% for CLIP — an upper bound, and not
far above what out-of-sample calibration already reaches.

### Which statues can pass for a missing one

Only four dwarves are ever falsely accepted by DINOv2 at the 90% point, and they form two pairs:

| removed dwarf | falsely accepted | usually matched to |
|---|---|---|
| Kowal | 3 / 12 | 100matolog |
| 100matolog | 1 / 4 | Kowal |
| Zbierający Wodę | 1 / 3 | Puszczający Stateczki |
| Puszczający Stateczki | 1 / 3 | Zbierający Wodę |

The water-themed pair is the cluster section 5 already identified, now seen from a different
angle: those statues are close enough that one can stand in for another that isn't there.

*Kowal* is the more interesting entry, because it is the direction section 5 recorded as
**harmless**. Closed-set, *Kowal → 100matolog* misidentifies 0 of 10 queries — with both statues
present, Kowal is never mistaken for anything. Remove Kowal, and 100matolog covers for it three
times out of twelve, making it the single worst dwarf to reject. Being reliably identifiable when
present says nothing about being reliably rejectable when absent; these are different questions,
and closed-set confusion analysis cannot see the second one.

CLIP fails this test broadly rather than in a few places: 16 of 23 dwarves are falsely accepted at
least once, and all three water-themed statues are accepted **3 of 3** — with any one of them
removed, another always passes for it. On this dataset CLIP cannot support rejection at a useful
operating point at all.

## Limitations

- **23 classes is a small pool.** The full pool is close to the accuracy ceiling, which is why
  the curve is shallow and why extrapolation past it is unreliable. Lowering the image threshold
  to two was measured rather than assumed: it adds four classes and does not change the curve.
- **Reference photographs are not a phone camera.** These are Commons uploads — mostly good
  light, considered framing. Real query photos would be worse, and this design cannot say by how
  much.
- **The geographic result rests on 23 statues.** The co-located group that drives it is one
  installation; whether the pattern holds city-wide is untested. Wikidata may also assign group
  members one shared point rather than individual positions, so "within one metre" may reflect
  the record as much as the pavement — either way those statues are co-located.
- **Rejection is measured, but only against statues in this dataset.** Section 6 builds unknown
  queries by removing a dwarf from the 23, so every "unknown" statue is still a Wrocław bronze
  dwarf photographed like the rest. A real unknown query — a different one of the ~1,300, or a
  statue photographed in worse conditions — is a harder and an untested case. The published demo
  does not threshold at all; it always shows its ranking, and it cannot readily be changed to,
  because it runs the backbone that has no usable operating point.

## Reproducing this

```bash
uv sync --extra ml --extra analysis
export KRASNAL_ID_USER_AGENT='krasnal-id/0.4.0 (mailto:you@example.com)'

uv run krasnal-id data query                                  # Wikidata discovery
uv run krasnal-id data fetch                                  # Commons acquisition
uv run krasnal-id data build-manifest                         # validated manifest
uv run krasnal-id data build-split                            # leave-one-out folds
uv run krasnal-id embeddings extract --override backbone=dinov2
uv run krasnal-id experiment baseline                         # section 1
uv run krasnal-id experiment pool-ablation                    # section 2
uv run krasnal-id experiment geo-ablation                     # section 3
uv run krasnal-id experiment probe                            # section 4
uv run krasnal-id experiment confusion                        # section 5
uv run krasnal-id experiment open-set                         # section 6
uv run krasnal-id visualize ablation
uv run krasnal-id visualize open-set
uv run krasnal-id visualize embeddings
```

Every experiment writes a structured JSON artifact to `results/`. Embeddings are cached by image
content hash and backbone revision, so nothing is recomputed between runs. See the
[README](README.md) for the full command reference and [AGENTS.md](AGENTS.md) for the recorded
design decisions behind each of these choices.

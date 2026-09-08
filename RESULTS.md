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

1. **Off-the-shelf embeddings identify these statues well, at real scale.** DINOv2 reaches 93.1%
   top-1 across 306 dwarves with no fine-tuning at all. CLIP reaches 82.9%.
2. **The decay is shallow for DINOv2 and accelerating for CLIP.** DINOv2 loses 0.79 accuracy
   points per doubling of the candidate pool and holds that rate across the whole measured range;
   CLIP loses 2.06 and gets steadily worse as the pool grows. The gap between them widens from
   under a point at a pool of two to **ten points** at 306.
3. **Narrowing by location makes identification slightly *harder*, not easier.** Real
   proximity-based pools lose to random pools of the same size at every pool size measured, and the
   penalty peaks at a 171–291 m radius — exactly the range a phone-based tool would use. Sculptors
   install related pieces near each other, so a small radius selects for a statue's own lookalikes.
4. **Training a classifier helps only the weaker backbone.** A per-fold linear probe is worth
   nothing to DINOv2 (−0.1 points) but gains CLIP 3.1. Class prototypes cost both about 8 points.
   Where retrieval is already strong, it is already the right tool.
5. **The errors are explainable, not random.** Both backbones concentrate their mistakes on
   near-identical statues installed as families — above all the three *Słupniki* pillar dwarves —
   and the embedding projection picks the same clusters out unprompted.
6. **Phone photographs are measurably harder queries.** Using the 51 references that were
   themselves shot on phones, DINOv2 loses 5.3 top-1 points and CLIP 15.6 against camera-shot
   queries — and not because they belong to easier or harder classes. It is a lower bound, since
   these are still Commons uploads.
7. **Open-set rejection does not survive the larger pool.** Thresholding similarity looked usable
   at 23 classes and does not work at 306: DINOv2's false-acceptance rate rises from 4% to 38% at
   the same operating point. The earlier positive result was an artefact of a small pool.

Findings 2, 4 and 7 all revise conclusions this project previously published from a 23-class
dataset. Section 7 is about which of them the small pool got wrong, and why.

## Dataset

| | |
|---|---|
| Dwarves (classes) | 306 |
| Reference photographs | 1,691 |
| Images per dwarf | 3 to 31 (median 4) |
| Source | Wikimedia Commons, via Wikidata and the Commons category tree |
| Licences | Public Domain, CC0, CC BY, CC BY-SA only |

Coverage is bounded by data quality, not geography: a dwarf is included only if it has at least
three usable Creative Commons photographs. 482 category mappings were reviewed, 469 approved, and
306 classes survived the image threshold.

**Wikidata is not the limit — Commons is the source.** Only 44 Wikidata items exist for the 481
per-dwarf categories Commons holds, and the same 44 are the only ones linked to any of them. So
283 of these 306 classes have no Wikidata item at all and are identified by their Commons
category. That has one consequence worth carrying into section 3: coordinates come from Wikidata's
`P625`, which places only 23 of them. The rest are placed from their own photographs' camera
positions instead, which section 3 validates at a median of 9 m — bringing coverage to **294 of
306**.

Three details matter more than the headline count:

- **Attribution is a first-class field.** Every stored image carries its source URL, author,
  licence and licence URL. Nothing enters the manifest without them.
- **Cross-label duplicates are excluded.** 131 Commons files were byte-identical across
  *different* dwarf labels, mostly where a group category and its member categories hold the same
  photographs. Admitting them would leak the answer into the reference set, so they are dropped —
  even where that empties a class, as it does for three robbers photographed in a single scene.
- **Evaluation is leave-one-out.** Each of the 1,691 images is queried once against the other
  1,690. The split is hash-pinned to the manifest, and any experiment run against a stale split
  fails loudly instead of quietly producing plausible numbers.

## Method

Images are embedded with two frozen backbones — **DINOv2** (`facebook/dinov2-base`, 768-d) and
**CLIP** (`openai/clip-vit-base-patch32`, 512-d) — and ranked by cosine similarity. No
fine-tuning. Each backbone is pinned to an immutable revision that serves safetensors directly,
so the provenance recorded alongside every cached vector describes the weights actually loaded.

Candidates are ranked as **distinct dwarves**, each represented by its best-matching photograph.
That is the list an identification tool would show a user, and it is what every metric below
scores.

## 1. Baseline: the full candidate pool

All 306 dwarves are candidates for every query. Intervals are 95% Wilson score intervals over
1,691 queries.

| | DINOv2 | CLIP |
|---|---|---|
| top-1 | **93.1%** [91.8, 94.3] | **82.9%** [81.0, 84.6] |
| top-5 | 95.7% [94.6, 96.6] | 90.7% [89.2, 92.0] |
| MRR | 0.943 | 0.866 |

DINOv2 leads, which is the expected ordering: it is trained with a self-supervised objective that
preserves instance-level detail, while CLIP's language-alignment objective pushes toward semantic
categories — and every one of these statues *is* the same semantic category. The gap is now ten
points, where a 23-class pool put it at three.

## 2. The headline result: accuracy against candidate-pool size

Each query is scored against its own dwarf plus N−1 randomly drawn others, repeated over five
seeds per pool size. Error bars are the observed spread across seeds.

![Top-1 accuracy against candidate-pool size](docs/figures/pool-size-ablation.png)

| pool size | DINOv2 | CLIP |
|---|---|---|
| 2 | 98.9% | 98.0% |
| 5 | 97.7% | 95.6% |
| 10 | 97.1% | 94.0% |
| 20 | 96.2% | 92.0% |
| 50 | 95.2% | 89.5% |
| 100 | 94.4% | 87.2% |
| 200 | 93.6% | 84.5% |
| 306 (full) | 93.1% | 82.9% |
| **points lost per doubling** | **−0.79** | **−2.06** |

The two curves have different *shapes*, which is the result that a single fitted slope hides.
Measured from a pool of 2 to 20, and then from 50 to 306, DINOv2 loses 0.82 points per
doubling early and 0.77 late — its decay is log-linear across the whole range, and if anything
eases. CLIP loses 1.79 early and **2.53** late: its decay accelerates.

So narrowing the pool is worth much more to a weak backbone than a strong one. Going from 306
candidates to 10 buys DINOv2 3.9 points and CLIP 11.1. If you cannot narrow the pool, the
backbone choice is most of the battle; if you can, it matters much less.

### What this implies at city scale

Extrapolating to the roughly 1,400 dwarves in the city gives 91.4% top-1 for DINOv2 and 78.4% for
CLIP. The DINOv2 figure is now a reasonably supported extrapolation rather than a guess: it is
two doublings past measured data, over a range where the rate has been constant for seven
doublings. **The CLIP figure remains an optimistic bound**, because its rate is still worsening at
the edge of the measured range, so a log-linear fit understates the loss.

## 3. Does narrowing by location actually help?

The pools above are sampled at random. The real proposal is to narrow by *location*. Each query is
pooled with its **N−1 nearest** dwarves instead of N−1 random ones — same pool size, only the
selection rule changes.

**This covers 294 of 306 classes (96%)**, which took an extra step to achieve. Wikidata's `P625`
places only 23 of them, so the rest are placed from their own photographs: 73.5% of Commons files
here carry a coordinate, almost always the *camera* position. That is not the statue's position,
so it was validated rather than assumed — across the 21 classes that have both, the median of a
class's camera positions falls a **median of 9 m** from its Wikidata point, against a smallest
pool radius of 171 m. Every coordinate records which kind it is, and one class was dropped for a
one-degree latitude typo that put it 111 km away.

| pool size | DINOv2 random | DINOv2 geographic | difference | median radius |
|---|---|---|---|---|
| 2 | 99.1% | 98.6% | −0.46 | 52 m |
| 5 | 97.9% | 97.0% | **−0.89** | 171 m |
| 10 | 96.9% | 96.2% | −0.68 | 291 m |
| 20 | 96.3% | 95.6% | −0.76 | 467 m |
| 50 | 95.4% | 94.7% | −0.67 | 712 m |
| 100 | 94.7% | 94.2% | −0.48 | 909 m |
| 200 | 94.0% | 93.6% | −0.35 | 2,223 m |
| 294 (full) | 93.4% | 93.4% | 0.00 | — |

**Geographic pools are harder than random pools of the same size, at every single pool size, for
both backbones.** CLIP shows it about twice as strongly, peaking at −2.12 points at a pool of ten.
Nothing here is sampled — a geographic pool is exact — so these are not seed noise.

At 23 classes this result rested on six statues in one themed installation, and could fairly have
been dismissed as a quirk of that installation. It now holds across 294 statues spanning the city.

### Why: co-location causes confusion, but only at very short range

The obvious explanation is that sculptors install related pieces near each other, so a small pool
is disproportionately made of a statue's own lookalikes. With 294 statues placed, that can be
measured rather than assumed — every competing pair has a ground distance, and pairs that produced
an outright misidentification can be compared against pairs that merely competed.

| | DINOv2 | CLIP |
|---|---|---|
| median separation, confused pairs | 1,606 m | 1,285 m |
| median separation, competing pairs | 1,893 m | 1,877 m |
| P(confused pair closer than a competing one) | 0.518 | 0.551 |
| share of confused pairs within 100 m | **11.9%** | 9.0% |
| share of competing pairs within 100 m | 5.5% | 5.9% |

**The effect is real, sharply local, and smaller than the story wants it to be.** Confused pairs
are about **twice as likely** to stand within 100 m of each other as merely-competing pairs (2.1×
for DINOv2, 1.5× for CLIP). But that enrichment decays fast — 1.2× by 300 m, 1.1× by a kilometre —
and the rank statistic over the whole population is only 0.518, barely above the 0.5 that would
mean distance says nothing at all.

So co-location acts at the scale of a shared plinth, not a shared neighbourhood. And it explains
only a minority of confusion: **88% of DINOv2's confused pairs are more than 100 m apart.** Most
statues that look alike here do not stand together; they just look alike.

That is still enough to explain the curve. A pool drawn at a 171 m radius only needs a modest
enrichment of lookalikes to come out worse than a random pool of the same size — and the
enrichment is concentrated exactly in that range, which is why the penalty peaks there and fades
by 2.2 km. **Location narrowing does select for the confusions it needs to avoid**, and a
random-subsampling simulation therefore overstates what narrowing buys, worst at the small radii a
phone-based tool would use. The mechanism is correct; its magnitude is modest.

The practical reading for a location-aware tool: a 291 m radius around a visitor covers about ten
candidates and identifies at 96.2% top-1 with DINOv2. That is useful. It is also slightly *worse*
than showing that visitor ten dwarves picked at random from the whole city, which is the part worth
sitting with.

## 4. Does a trained classifier beat retrieval?

All three methods scored on the same folds, with one classifier fitted per fold so no query is
ever in its own training data — 1,691 classifiers per backbone.

| method | DINOv2 top-1 | CLIP top-1 |
|---|---|---|
| cosine retrieval | **93.1%** | 82.9% |
| linear probe | 93.0% (−0.1) | **86.0%** (+3.1) |
| class prototypes | 85.1% (−8.0) | 74.6% (−8.3) |

**For DINOv2, no; for CLIP, yes — and that asymmetry is the finding.** The probe changes nothing
for DINOv2, two queries out of 1,691. For CLIP it gains 3.1 points, and while the confidence
intervals still overlap slightly ([84.3, 87.6] against [81.0, 84.6]) the direction is consistent
and the size is not trivial. A supervised layer can partly repair an embedding space that is not
laid out for instance-level discrimination; it has nothing to add to one that already is.

At 23 classes both backbones showed the same +0.7 non-effect. The larger pool separates them.

Prototypes are now clearly harmful — about 8 points for both, against 2 to 3 at 23 classes.
Averaging a class into one vector discards the pose and viewpoint variation that makes a specific
photograph matchable, and the more classes there are, the more that costs.

A caveat about how these numbers were reached, because it nearly became a wrong conclusion. At
scikit-learn's conventional `C=1.0` the probe scored **66%** on the small dataset, which looks
decisive and is really just underfitting: the embeddings are L2-normalised, so per-dimension
magnitudes sit near `1/√d` and that penalty crushes the weights. At `C=100` it scores as above. A
badly-regularised baseline is worse than no baseline, because it flatters whatever it is compared
against.

## 5. Where the errors are

DINOv2 misidentifies 116 of 1,691 queries (6.9%); CLIP misidentifies 289 (17.1%). Because outright
errors are still relatively rare, the analysis records the strongest *wrong* candidate for every
query, not only for the failures.

| | DINOv2 | CLIP |
|---|---|---|
| top-1 errors | 116 / 1691 (6.9%) | 289 / 1691 (17.1%) |
| mean margin over the best wrong dwarf | 0.174 | 0.040 |

The mean margin is the more revealing number. CLIP's embedding space is roughly four times
tighter, which is the same finding as its steeper decay curve seen from another angle: less
headroom per comparison means each added candidate costs more.

The errors are systematic, and the dominant cluster has changed with the larger dataset. It is now
the **Słupniki** family — *Słupniki Solne*, *Słupniki Oławskie* and *Słupniki Świdnickie*, near
identical pillar dwarves installed on different streets. They are the top confusion for both
backbones, in both directions. The water-themed trio that dominated at 23 classes is still there
(*Puszczający Stateczki → Zbierający Wodę*, 2 of 3 queries) but is no longer the largest effect.

![DINOv2 embedding projection](docs/figures/embeddings-umap-dinov2.png)

The projection corroborates it independently. With 306 classes it names only the 24 sitting
closest to another class and greys the rest — and that automatic selection, which uses nothing but
centroid distances, picks out the Słupniki pair, the water trio and *Ślepak*/*Głuchak* (the blind
and deaf dwarves, a sculpted pair). The clusters the error analysis finds are the clusters the
geometry shows.

## 6. Can it say "I don't know this one"?

Retrieval always returns a nearest neighbour, so a photograph of a statue outside the reference set
still produces a confident-looking ranking. The test: threshold the top-1 cosine similarity, and
measure it on two populations of 1,691 — the known arm is the leave-one-out split, and the unknown
arm removes *every* image of a query's own dwarf so that dwarf is genuinely absent. Operating
points are calibrated leave-one-class-out, so no query helps set the bar it must clear.

| | 23 classes | **306 classes** |
|---|---|---|
| DINOv2 AUROC | 0.969 | **0.896** |
| DINOv2 false accepts @ 90% known acceptance | 4.1% | **38.3%** |
| CLIP AUROC | 0.898 | **0.806** |
| CLIP false accepts @ 90% known acceptance | 28.1% | **67.4%** |

**Rejection does not survive the larger pool.** At 23 classes DINOv2 rejected 96% of unknown
statues while identifying 89% of known ones, and that looked like a working feature. At 306 the
same operating point lets 38% of unknown statues through. Overall open-set accuracy is 75.1%
against a closed-set 93.1%.

The mechanism is not subtle. An absent statue's nearest neighbour is drawn from 305 candidates
instead of 22, so the chance that *something* in the gallery resembles it closely is far higher.
Mean top-1 similarity for an absent dwarf rises from 0.60 to 0.66 while the present case barely
moves. The two distributions slide together.

This is a **correction to a previously published result of this project**, not a new caveat on it.
The 0.4.0 release reported open-set rejection as working at a modest cost. It worked on 23 classes
and does not work on 306, and there is no reason to expect it to improve at 1,400.

## 7. What the small pool got wrong

Three conclusions changed when the dataset grew 13×, and the pattern in which ones changed is
itself the lesson.

| conclusion at 23 classes | at 306 classes |
|---|---|
| DINOv2 loses 0.96 pts/doubling | 0.79 — the small pool was **pessimistic** |
| CLIP loses 1.76 pts/doubling | 2.06 and accelerating — the small pool was **optimistic** |
| A trained classifier does not help either backbone | True for DINOv2, false for CLIP (+3.1) |
| Similarity thresholding gives usable rejection | It does not; 4% false accepts became 38% |
| Errors concentrate on the water-themed trio | They concentrate on the Słupniki family |
| Geographic pools are harder than random ones | Unchanged, and now at every pool size over 294 classes rather than 23 |

The previous writeup warned that extrapolating past 23 classes would be optimistic, because a
small pool holds too few genuinely confusable distractors. That warning was **half right**: true
for CLIP, wrong for DINOv2, which did better than its own small-pool curve predicted. A blanket
"small datasets are optimistic" heuristic would have mispredicted one of the two backbones.

What actually distinguishes them is that the small pool could not measure *variance in
confusability*. Adding 283 classes added a few near-identical families — the Słupniki pillars,
Ślepak and Głuchak — and a long tail of statues that are easy to tell apart. A strong backbone
absorbs the tail and only pays for the families; a weak one pays for both. That is invisible until
the tail exists.

## 8. Are phone photographs harder queries?

Every reference here is a Wikimedia Commons upload, and the limitations have always said the gap
to a casual phone photograph is the largest untested thing in the project. Fieldwork is the clean
way to close it. This is what can be measured without it: Commons records each file's EXIF camera,
so the existing references split into phone-originated and camera-originated queries, scored
against the same reference set.

| | phone (51 queries) | camera (1,565) | gap |
|---|---|---|---|
| DINOv2 top-1 | 88.2% [76.6, 94.5] | 93.5% [92.2, 94.6] | **−5.3** |
| CLIP top-1 | 68.6% [55.0, 79.7] | 84.2% [82.3, 85.9] | **−15.6** |

**Phone-originated photographs are harder queries, and dramatically so for CLIP.** Its intervals do
not overlap. DINOv2's 5.3-point gap points the same way but with 51 queries it is not
statistically distinguishable from noise — the honest reading is "consistent with a real gap,
too few queries to prove one".

The obvious objection is that phone photographs might simply belong to harder classes. They do
not: the median phone query's dwarf has **seven** reference photographs against six for camera
queries, so if anything the phone group had the easier task. They are also not lower resolution
(3.0 MP median against 2.7) and no more likely to belong to a class the confusion analysis already
flags (25.5% against 22.3%).

A third group is worth noting. 75 queries carry no EXIF at all, usually because it was stripped on
upload, and they score 89.3% and 65.3% — much closer to the phone group than the camera group.
Stripped metadata correlates with the same kind of photograph.

**This is a lower bound, and it does not close the question.** These are still Commons uploads:
chosen, often composed, taken by someone who meant to document the statue. A real snapshot — bad
angle, passers-by, whatever light was available — is a harder query than anything measured here.
What this establishes is that the domain gap is real and that CLIP suffers it far worse, which
also means the published demo, which runs CLIP, is the version most exposed to it.

## 9. Is it recognising the statue, or the photographer?

122 people took these 1,691 photographs, but **two of them took 67.4%** — Pnapora 715, Fallaner
424. Byte-identical cross-label duplicates were removed; near-duplicates from one photographer's
single visit to a statue were not. So a leave-one-out query is often only near-duplicate-distant
from one of its own references, and a method could score by recognising a camera, a distance and a
processing style rather than a sculpture.

`experiment photographer-gap` withholds every image by the query's own photographer, so the
correct statue can only be matched through somebody else's photograph.

**Two thirds of the pool cannot be asked.** 125 of the 306 classes have a single photographer, and
under this protocol their 534 queries are not hard but *unanswerable*: no correct reference exists
at all. They are excluded and counted rather than scored as failures, which would measure the
dataset's coverage and call it the model's weakness. The remaining **1,157 queries** are what this
section reports, and they score slightly higher than the whole set under the ordinary protocol
(94.2% against 93.1% for DINOv2) because a class with two photographers tends to be a
better-photographed class.

**The gap has to be decomposed, or it means nothing.** Withholding a photographer also removes
distractors — and section 2's whole finding is that accuracy *rises* as the pool shrinks, so the
disjoint arm gets an unearned boost that would mask the very penalty being measured. Each query is
therefore also scored against a **size-matched random control**: the same number of references,
including the same number of correct ones, drawn at random over five seeds.

| | DINOv2 | CLIP |
|---|---|---|
| Standard leave-one-out | 94.2% [92.7, 95.4] | 82.0% [79.7, 84.1] |
| Size-matched random control | 84.5% [83.5, 85.4] | 67.3% [66.1, 68.5] |
| Photographer-disjoint | 81.8% [79.5, 84.0] | 54.1% [51.2, 57.0] |
| **Total drop** | **−12.4** | **−27.9** |
| **Attributable to the photographer** | **−2.6** | **−13.2** |

**DINOv2's headline is mostly not photographer leakage.** Of its 12.4-point drop, 9.8 points are
the cost of simply having fewer references — the random control pays that too — leaving **2.6
points** attributable to the photographer's identity. So 93.1% is inflated, but by a couple of
points, not by the double digits the concentration of the corpus might suggest.

**CLIP leans on the photographer five times as hard**: 13.2 points against DINOv2's 2.6. That fits
the ordering section 1 found and sharpens it. DINOv2's self-supervised objective preserves
instance-level detail, so it is looking at the statue; CLIP's language alignment pulls toward
appearance and style, which is exactly what covaries with who held the camera.

Two things make these figures a **lower** bound on the attributable gap rather than an estimate of
it. The disjoint arm ends up with a median of 252 candidate classes against the control's 305,
because withholding a photographer removes whole classes from contention — by section 2's logic a
54-class-smaller pool is *easier*, and the disjoint arm still lost. And `author` is Commons' own
free-text field, compared exactly: one photographer using two spellings is counted as two people,
which leaves some of their own work in the reference set.

**What this means for the demo.** The published page runs CLIP, and CLIP identifies a statue
correctly just **54.1%** of the time when it cannot lean on the same photographer's other
photographs. A visitor photographing a statue that Commons documents through one contributor is
much closer to that number than to the 82.9% the page reports.

## 10. Does verifying geometry fix the lookalikes?

Section 5 puts the errors on families of near-identical sculptures, and section 9 shows part of the
ranking signal is the photographer's style rather than the statue's shape. Both say the same thing:
a whole-image embedding compares *appearance*, and appearance is what these families share.
Matching keypoints and fitting a homography compares *geometry*, which they do not.

`experiment rerank` takes the global top-10 candidates, counts RANSAC inliers between the query and
each candidate's best-matching photograph, and blends that count into the cosine similarity. The
blend weight is swept, and **weight zero is the control**: it must reproduce the unranked baseline
exactly, and it does, to the digit — 93.14% for DINOv2 either way. Without that check the rest of
the sweep would not be readable as a difference.

| Blend weight | DINOv2 top-1 | promoted / demoted | CLIP top-1 | promoted / demoted |
|---|---|---|---|---|
| 0 (control) | 93.14% | — | 82.91% | — |
| 0.01 | 93.32% | +3 / −0 | 84.68% | +32 / −2 |
| 0.02 | 93.55% | +7 / −0 | 85.39% | +47 / −5 |
| 0.05 | 93.67% | +13 / −4 | **86.34%** | +70 / −12 |
| 0.1 | **94.03%** | +19 / −4 | 86.16% | +79 / −24 |
| 0.2 | 93.79% | +20 / −9 | 85.39% | +87 / −45 |

**Geometry helps, and it helps the weaker backbone four times as much.** DINOv2 gains **0.9 top-1
points** at its best weight, CLIP **3.4**. That is the same asymmetry section 9 found from the
other side: CLIP leans on appearance, so supplying it with shape evidence is worth more. MRR moves
with it — 0.9419 to 0.9478 for DINOv2, 0.8640 to 0.8860 for CLIP.

The sweep is reported with promotions and demotions rather than only the net, because they are
different results. At its best weight DINOv2 fixes 19 queries and breaks 4; push the weight to 0.2
and it fixes 20 but breaks 9. The net barely moves while the churn doubles, which is what a blend
weight past its useful range looks like.

**Re-ranking cannot fix what the ranking never proposed.** With k=10, the correct statue is absent
from the candidate list for 64 DINOv2 queries and 123 CLIP queries, so the ceiling here is 96.2%
and 92.7% respectively. CLIP reaches 86.3% of a possible 92.7%; most of its remaining error is now
recall, not verification.

### The separation is near-duplicate confirmation. The gain mostly is not.

The inlier separation above is enormous — a median of 119 for the correct statue against 4 for a
wrong one — and the obvious worry is that it is section 9's leakage in another guise: a candidate's
*best-matching* photograph is often one the same photographer took on the same visit, and two
frames from one visit verify trivially. Running the sweep with each query's own photographer
withheld separates the two. Over the 1,157 answerable queries:

| | baseline | best blend | gain | median inliers, correct / wrong |
|---|---|---|---|---|
| DINOv2, all references | 94.21% | 94.81% | +0.61 | 58 / 4 |
| DINOv2, photographer-disjoint | 81.85% | 82.28% | +0.43 | **6 / 4** |
| CLIP, all references | 82.02% | 85.74% | +3.72 | 64 / 4 |
| CLIP, photographer-disjoint | 54.11% | **57.04%** | **+2.94** | **6 / 4** |

**The worry was half right.** The evidence really was mostly near-duplicate confirmation: the
separation collapses from 58–64 inliers against 4 down to **6 against 4** once the photographer is
gone. Almost all of that spectacular verification was two frames from one visit.

**But the accuracy gain survives it.** DINOv2 keeps +0.43 of +0.61, CLIP +2.94 of +3.72 — **79% of
the gain in both cases**, which is a suspiciously tidy agreement between two backbones that differ
in everything else. A median of 6 inliers against 4 is a weak signal, and it turns out a weak
signal is enough, because the blend is a tie-breaker on top of cosine rather than a replacement
for it. Geometry does not have to be decisive to be useful; it has to be uncorrelated with the
mistake the embedding is making.

So re-ranking is worth having for a real user, whose photograph has no near-duplicate in the
reference set by construction. **CLIP cross-photographer goes from 54.1% to 57.0%** — still poor,
but the improvement is not an artifact.

**Recall becomes the binding limit.** In the disjoint arm the correct statue never enters the top
10 for 107 DINOv2 queries and 288 CLIP ones, capping them at 90.8% and 75.1%. CLIP reaches 57.0%
of a possible 75.1%: raising k, or a better first stage, now matters more than better verification.

## 11. Why re-ranking stops there: the first stage's recall

Section 10 ends against a wall. Re-ranking can only reorder what the first stage proposed, so its
ceiling is recall: with the query's own photographer withheld, the correct statue never enters
CLIP's top 10 for 288 of the 1,157 queries, and no verification can rescue them.
`experiment recall` measures that ceiling and tests the standard ways of raising it. All three
fail, and how they fail is more informative than the ceiling itself.

**The curve.** Recall at k, photographer-disjoint, over the 1,157 answerable queries:

| k | DINOv2 | CLIP |
|---:|---|---|
| 1 | 81.8% | 54.1% |
| 5 | 89.0% | 69.3% |
| 10 | 90.8% | 75.1% |
| 20 | 92.0% | 79.7% |
| 50 | 94.2% | 85.7% |

There is real headroom — CLIP's ceiling rises 10.6 points between k=10 and k=50 — so the obvious
move is to verify more candidates.

**Verifying 50 candidates instead of 10 buys 0.18 points.** Re-running section 10's sweep at k=50
takes CLIP's best from 57.04% to 57.22%, and at higher blend weights it is *worse* than k=10: at
weight 0.2 it demotes 105 queries and lands 1.9 points below its own baseline. The reason the
headroom is not convertible is that the two failures are correlated. CLIP pushes the right statue
past rank 10 exactly on the hard queries — awkward viewpoint, poor light, a lookalike family — and
those are the same queries where geometry is weakest, at 6 inliers against 4. The candidates a
larger k newly admits are the ones verification is least able to adjudicate, while every one of
them is a fresh chance to fluke a homography.

**Fusing the two backbones does not beat the better one.** Summing both models' similarities gives
79.6% at r@1, against DINOv2's 81.8% alone. It is a large gain over CLIP's 54.1%, but that is not
an independent view being added: the fused score is essentially DINOv2 doing the work with CLIP
dragging slightly. Two models that fail on the same lookalike families do not decorrelate by being
averaged.

**Query expansion actively hurts, and the reason is this dataset's error structure.** Re-querying
with the query averaged into its own top results is the classical recall fix. Here:

| variant | DINOv2 r@10 | CLIP r@10 |
|---|---|---|
| no expansion | 90.8% | 75.1% |
| expansion, 2 neighbours | 89.5% | 68.2% |
| expansion, 3 neighbours | 89.1% | 68.3% |
| expansion, 5 neighbours | 88.9% | 69.0% |

Expansion assumes the top results are mostly correct. At 54% precision they are not, and section 5
says what the wrong ones are: near-identical statues. So the expanded query moves toward its own
confuser and the mistake is reinforced rather than corrected. The damage scales with how many
neighbours are folded in, which is the signature of exactly that mechanism — CLIP's r@1 falls from
54.1% to 43.2% at five neighbours.

**What this leaves.** The bottleneck is the representation, not the amount of it that gets
searched, and it is specific to one backbone: DINOv2's *first* guess cross-photographer (81.8%) is
better than CLIP's *tenth* (75.1%). CLIP is in this project because the browser demo needs a model
small enough to ship, so this is a deployment constraint rather than a research one — and the
honest fixes are a smaller strong model or a page that says it is weaker than the pipeline, which
section 10's limitation now does.

## Limitations

- **Reference photographs are not a phone camera.** These are Commons uploads — mostly good light,
  considered framing. Section 8 puts a lower bound on what that costs, using the 51 references that
  were themselves shot on phones, but a real snapshot is harder than any photograph in this
  dataset. This remains the largest untested gap in the work, and closing it needs fieldwork.
- **Most coordinates are derived, not stated.** 271 of the 294 placed statues are located from
  where photographers stood rather than from a `P625` statement. Validation against the 21 that
  have both puts the median error at 9 m, an order of magnitude below the smallest pool radius the
  experiment uses — but it is an inference, and a statue photographed only from across a square
  would be placed across that square. Twelve classes remain unplaced.
- **Two classes overlap with a group class.** *Grajek i Meloman* is a class, and so are *Grajek*
  and *Meloman*; likewise *Ogrodnik i Kierownik* and *Ogrodnik*. Their photographs are not
  byte-identical, so the duplicate guard does not catch them, and they duly appear in the confusion
  pairs. Two of 306 is a small effect, but it is a real one and it is not the model's fault.
- **Two photographers took two-thirds of the photographs.** 122 people contributed, but Pnapora
  uploaded 715 images (42.3%) and Fallaner 424 (25.1%) — 67.4% between them, and near-duplicates
  from a single visit were never removed. Section 9 measures what that costs: on the 1,157 queries
  the protocol can ask about, **2.6 top-1 points of DINOv2's accuracy are attributable to the
  photographer** rather than the statue, and 13.2 points of CLIP's. So the headline is inflated,
  but modestly for the backbone the headline uses. The concentration still bites elsewhere: 125 of
  306 classes have a single photographer, so a third of the dataset cannot be evaluated
  cross-photographer at all.
- **Class sizes are uneven.** The median class has 4 images and the largest 31, so a handful of
  well-photographed statues carry disproportionate weight in the query set.
- **Open-set rejection is measured against statues inside this dataset.** Every "unknown" query is
  still a Wrocław bronze dwarf photographed like the rest. A genuinely out-of-distribution query is
  a harder and untested case. The published demo does not threshold at all, and could not usefully:
  it runs CLIP, which has the weaker rejection of the two.

## Reproducing this

The photographs, both backbones' embeddings and the evaluation folds are published at
[turhancan97/wroclaw-dwarves](https://huggingface.co/datasets/turhancan97/wroclaw-dwarves), so the
baselines can be re-derived without re-crawling Commons: load the `embeddings_dinov2` and
`leave_one_out` configs and score them. The pipeline below is what produced them.

```bash
uv sync --extra ml --extra analysis
export KRASNAL_ID_USER_AGENT='krasnal-id/0.12.0 (mailto:you@example.com)'

uv run krasnal-id data query --include-commons                # Wikidata + Commons discovery
uv run krasnal-id data fetch --prepare-review                 # then review the mappings
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
uv run krasnal-id data camera-metadata                        # EXIF for section 8
uv run krasnal-id experiment camera-gap                       # section 8
uv run krasnal-id visualize ablation
uv run krasnal-id visualize embeddings
uv run krasnal-id visualize open-set
```

Every experiment writes a structured JSON artifact to `results/`. Embeddings are cached by image
content hash and backbone revision, so nothing is recomputed between runs. The probe fits 1,691
classifiers per backbone and takes about 1.5 hours; everything else runs in minutes. See the
[README](README.md) for the full command reference and [AGENTS.md](AGENTS.md) for the recorded
design decisions behind each of these choices.

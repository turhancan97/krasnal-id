<picture>
  <source media="(prefers-color-scheme: dark)" srcset="docs/brand/krasnal-lockup-horizontal-dark.svg">
  <img src="docs/brand/krasnal-lockup-horizontal-light.svg" alt="Krasnal-ID" width="260">
</picture>

[![DOI](https://zenodo.org/badge/1339843718.svg)](https://doi.org/10.5281/zenodo.22548023)

Krasnal-ID is a research prototype for fine-grained visual instance retrieval of
Wrocław's bronze dwarf statues. Its central experiment asks how identification accuracy
changes as the candidate pool grows, and how much a simulated location-based narrowing
would help.

## Project status

Current version **0.17.0**. Every stage runs: Wikidata and Commons discovery, reviewed Commons
acquisition, audited manifest construction, deterministic leave-one-out splits, resumable
DINOv2/CLIP embedding extraction, cosine k-NN retrieval, and thirteen experiments — the full-pool
baseline, the candidate-pool-size ablation, the geographic ablation, the trained-classifier
comparison, confusion analysis, open-set rejection, geometric rejection, the camera-origin query
gap, the photographer gap, geometric re-ranking, the first stage's recall, geometry as a first
stage, and the field-query gap that waits on photographs — plus embedding visualization,
single-image retrieval, a local demo and a published in-browser one.

The v0.1-v0.3 build order finished at `0.3.0`; every release since closes one research question.
`0.4.0` added open-set rejection. `0.5.0` rebuilt the dataset Commons-first at **306 classes and
1,691 images**, which overturned three published conclusions — `RESULTS.md` section 7 records which
and why. `0.6.0` placed 294 of those classes by deriving positions from their own photographs,
taking the geographic result from six statues in one installation to city-wide. `0.7.0` put a lower
bound on the query-domain gap without fieldwork, using the 51 references that were themselves shot
on phones. `0.8.0` built the path that measures it properly, so the only thing the question still
waits on is photographs. `0.9.0` published the dataset itself, as a fine-grained instance-retrieval
benchmark anyone can load. `0.10.0` asked whether the headline was recognising statues or
photographers, and answered it: mostly statues for DINOv2, much less so for CLIP. `0.11.0` added
geometric verification on top of the ranking — the first accuracy gain here from method rather
than data. `0.12.0` measured why it stops there: the first stage's recall, and three standard ways
of raising it that all fail. `0.13.0` put the pipeline's own DINOv2 in the browser on a *smaller*
download than the CLIP it replaced, taking the published demo from 82.4% to 93.2% top-1 and
retiring a size constraint the project had been designing around that turned out not to exist.
`0.14.0` asked whether geometry can reject what similarity cannot, and answered no — while finding
that geometry leans on the photographer harder than appearance does. `0.15.0` is the first release
about the site rather than the research: the findings move to their own page, the identifier
offers both backbones so the gap between them can be seen on a visitor's own photograph, and three
defects that only a real browser could show turn up in the process. `0.16.0` publishes the dataset
on Kaggle too, closing §5.11's last open question — and finds three more defects that only an
external platform could reject. `0.17.0` closes both branches section 11 left
open, and both are negative: a DINOv2 at 3.5x the parameters moves the re-ranking ceiling by 0.17
points, and local features promoted from re-ranker to *first* stage find the right statue in the
top ten for 43.5% of cross-photographer queries where cosine similarity manages 90.8%. What scale
buys is robustness to the photographer rather than recall.

The dataset is published on
[Hugging Face](https://huggingface.co/datasets/turhancan97/wroclaw-dwarves) and
[Kaggle](https://www.kaggle.com/datasets/turhancankargin/wroclaw-dwarves): 1,691 attributed
photographs, both backbones' embeddings and the evaluation folds, from which the headline result
re-derives without this repository.

One question stays open, and it is the one the numbers above cannot answer: **how much accuracy a
real phone photograph taken in the street costs.** The protocol, the route, the cohorts and the
whole measuring path are built and tested; what is missing is a day in Wrocław with a phone. See
`data/field-guide.md`, and `AGENTS.md` section 8 for the full open-question list.

## Findings

DINOv2 reaches 93.1% top-1 across a 306-dwarf pool with no fine-tuning, losing 0.79 accuracy
points per doubling of the candidate pool where CLIP loses 2.06 and accelerates. Narrowing by real
location helps *less* than random subsampling suggests, a trained classifier helps only the weaker
backbone, and the errors concentrate on families of near-identical statues. Thresholding
similarity to answer "I don't know this one" worked at 23 classes and **does not** at 306 — one of
three conclusions the larger dataset overturned. Queries shot on phones rather than cameras cost
DINOv2 5.3 top-1 points and CLIP 15.6, which is a lower bound on what a real street photograph
would cost. And identifying a statue from *someone else's* photograph costs DINOv2 2.6 points but
CLIP 13.2 — CLIP leans on the photographer five times as hard. Verifying geometry on the top
candidates recovers some of that: **94.0% for DINOv2 and 86.3% for CLIP**, and 79% of the gain
survives when the photographer's own photographs are withheld. It goes no further because of the
first stage's recall, and a larger candidate list, backbone fusion and query expansion were all
measured and all failed. Geometry does not rescue rejection either — 0.65 AUROC points for DINOv2
and 2.3 for CLIP, with four-fifths of unknown statues still accepted — because a known query's 144
average inliers collapse to 17 once its own photographer is withheld, making geometry *more*
photographer-dependent than appearance rather than less. Neither of the two ways out works. A
DINOv2 at 3.5x the parameters moves that first-stage ceiling by **0.17 points**, winning 22
queries and losing 20 — what it does buy is 2.59 points of robustness to a change of photographer.
And SIFT promoted to the first stage, ranking all 1,690 references by inlier count with no
embedding involved, reaches **43.5%** in the top ten where cosine reaches 90.8%, rescuing 12% of
what cosine loses for seven seconds a photograph — measured for SIFT, and not yet for the learned
matchers built for the regime where it fails.

![One query photograph and the five dwarves each backbone ranks highest, for four queries: one both backbones identify, one only DINOv2 identifies, one only CLIP identifies, and one neither identifies. Correct statues are outlined in green and wrong ones in red.](docs/figures/retrieval-examples.jpg)

*What the accuracy numbers are an average over. Each query is followed by the five dwarves each
backbone ranks highest, green for the correct statue. The four rows are not hand-picked: folds are
grouped by which backbones ranked the right statue first and the first query in image-ID order
represents its group, so this is what the ordinary case, the gap between the backbones, and their
shared failures actually look like. Regenerate with `krasnal-id visualize retrieval-examples`.*

- [**Identify a photograph**](https://turhancan97.github.io/krasnal-id/) — the findings, plus a
  working identifier that runs the model in your browser. Nothing is uploaded.
- [**RESULTS.md**](RESULTS.md) — the complete written record: dataset construction, all eight
  result sections, limitations, and how to reproduce them.
- [**The dataset on Hugging Face**](https://huggingface.co/datasets/turhancan97/wroclaw-dwarves) —
  1,691 attributed photographs of 306 statues, both backbones' embeddings, and the evaluation
  folds, as a fine-grained instance-retrieval benchmark. Parquet, with the images embedded.
- [**The dataset on Kaggle**](https://www.kaggle.com/datasets/turhancankargin/wroclaw-dwarves) —
  the same corpus as images on disk beside CSV tables, which is the shape Kaggle's explorer
  previews and its notebooks expect.

## Setup

Python 3.12 or newer and [`uv`](https://docs.astral.sh/uv/) are required.

```bash
uv sync
uv run krasnal-id --help
```

Install optional capabilities only when needed:

```bash
uv sync --extra ml
uv sync --extra analysis
uv sync --extra demo
```

## Discover dwarf records

Live Wikidata requests require a contact-bearing user agent supplied outside Git:

```bash
export KRASNAL_ID_USER_AGENT='krasnal-id/0.17.0 (mailto:you@example.com)'
uv run krasnal-id data query
```

Use a deterministic pilot subset or bypass a valid cache:

```bash
uv run krasnal-id data query --limit 5
uv run krasnal-id data query --refresh
```

Wikidata alone caps the dataset at 41 classes: only 44 items carry the dwarf-statue type, and the
same 44 are the only ones pointing at any of the 481 per-dwarf categories Commons holds. Add those
categories as a second source:

```bash
uv run krasnal-id data query --include-commons
```

That discovers 482 classes — 41 from Wikidata, 441 from Commons alone. A Wikidata record wins any
category it claims, because it carries a QID and the coordinates the geographic ablation needs;
Commons-only classes have neither, so the geographic arm does not grow with the dataset. Both
sources share one artifact and one recorded query hash, and without the flag the command
reproduces the Wikidata-only artifact exactly. Every new class still needs a category-review
decision, and a rebuild invalidates every published result: see
[AGENTS.md](AGENTS.md) section 5.6.

The command writes ignored raw-cache, normalized-record, and audit files below
`data/discovery/`. Cached results can be normalized again without the environment variable.

## Review categories and fetch images

Generate or update the tracked review file without making network requests:

```bash
uv run krasnal-id data fetch --prepare-review
```

Open `data/category-review.json`, inspect each Wikidata-to-Commons mapping, and change its
`status` from `pending` to either `approved` or `rejected`. Use
`corrected_category` when Wikidata points to a broad or incorrect category. A changed source
mapping is reset to `pending` the next time review preparation runs.

After every emitted mapping has a decision, fetch the approved categories:

```bash
export KRASNAL_ID_USER_AGENT='krasnal-id/0.17.0 (mailto:you@example.com)'
uv run krasnal-id data fetch
```

Pilot runs and metadata refreshes are deterministic:

```bash
uv run krasnal-id data fetch --max-images-per-dwarf 5
uv run krasnal-id data fetch --refresh
```

The fetcher reads exactly the current `dwarfs.json`, visits only direct category files,
accepts known Public Domain/CC0/CC BY/CC BY-SA static rasters, verifies attribution and image
content, requires a 400-pixel minimum short side, and stores at most 2,000-pixel research copies
under `data/images/<QID>/`. Oversized Commons derivatives are downscaled locally without
upscaling smaller images. It caches complete paginated API responses and reuses verified local
files when their Commons revision
has not changed. Generated image records and a detailed audit are written atomically below
`data/discovery/`.

Exit code 2 indicates invalid input or unfinished review. Exit code 1 indicates that processing
continued but at least one API or download operation still failed.

Image-level review decisions are tracked in `data/image-review.json`. It is keyed by dwarf ID
and Commons page ID, records explicit retain/exclude reasons for known staging exceptions, and
is tied to the current authoritative staging hash. Display-name corrections are stored as
`display_name_override` values in `data/category-review.json`; generated discovery files are
not edited by hand.

Build the validated manifest offline from the cached artifacts:

````bash
uv run krasnal-id data build-manifest
````

The command applies both review files, rejects stale or inconsistent staging inputs, filters
classes below the configured threshold, records discovery/staging/review provenance hashes,
and writes `data/manifest.json` atomically.

## Evaluation split and embeddings

Create the shared deterministic leave-one-out split from the generated manifest:

    uv run krasnal-id data build-split

The split creates one query fold per admitted image and is written to the ignored
data/splits/leave-one-out.json artifact. It is invalidated when the manifest changes.

Install the optional ML dependencies before extracting embeddings:

    uv sync --extra ml
    uv run krasnal-id embeddings extract --override backbone=dinov2
    uv run krasnal-id embeddings extract --override backbone=clip

Every Hydra override is passed with `--override` (`-o`), repeated per value:

    uv run krasnal-id embeddings extract -o backbone.device=cuda -o backbone.batch_size=8

Extraction validates every manifest image, reuses valid vectors, and stores normalized .npy
vectors under the ignored data/embeddings/ directory. CI uses deterministic fake backbones and
does not download model weights.

Both backbones have been run over the current 146-image manifest, caching 768-dimensional
DINOv2 and 512-dimensional CLIP vectors. Each backbone is pinned to a revision that serves
safetensors directly, because `transformers` otherwise falls back to a mutable conversion
reference and the revision recorded in the cache key would no longer describe the loaded
weights.

## Baseline evaluation

Measure full-pool retrieval quality once embeddings are cached:

    uv run krasnal-id experiment baseline
    uv run krasnal-id experiment baseline --override backbone=clip

Each run writes `results/baseline-<backbone>.json` atomically and prints every metric. Headline
`top_k` and `mrr` rank distinct dwarves by their best-matching image, which is the candidate list
an identification tool would present; `image_top_k` and `image_mrr` rank individual reference
images for comparison. Accuracy proportions carry 95% Wilson score intervals. The baseline is
exhaustive, so its seed is recorded for provenance rather than used to sample. A split whose
recorded manifest hash no longer matches the manifest is refused rather than scored.

Exit code 2 indicates missing embeddings, a stale split, or an unreadable artifact.

## Candidate-pool-size ablation

Measure the headline curve, accuracy against candidate-pool size:

    uv run krasnal-id experiment pool-ablation
    uv run krasnal-id experiment pool-ablation --override backbone=clip

Each query is scored against its own dwarf plus a sampled set of others, which simulates the
candidate narrowing a location-aware tool would perform. Every pool size is measured once per
configured seed and reported with the observed spread across seeds as its error bars. Configured
pool sizes larger than the available class count are skipped with a warning, and the full pool is
always measured. Results are written to `results/pool_size_ablation-<backbone>.json`.

![Top-1 accuracy against candidate-pool size on a log axis. DINOv2 falls from 98.9% at a pool of two to 93.1% at 306; CLIP falls from 98.0% to 82.9% and its decline steepens as the pool grows.](docs/figures/pool-size-ablation.png)

*The headline curve. Error bars are the observed spread across seeds. Draw it from saved results
with `krasnal-id visualize ablation`.*

The reported `top_1_points_per_doubling` is a least-squares fit of top-1 accuracy against
log2 pool size over every measured size. Small pools sit near the accuracy ceiling, so the fit is
a conservative estimate of degradation in the larger-pool regime. See `AGENTS.md` §7.1 for the
dataset-scale decision this result is framed by, including why extrapolating past the current
class count is optimistic.

## Confusion analysis and embedding visualization

Find the dwarf pairs that systematically compete for the same queries:

    uv run krasnal-id experiment confusion
    uv run krasnal-id experiment confusion --override backbone=clip

Every query contributes its strongest wrong candidate, not only the queries that were
misidentified, because near-misses carry most of the signal on a dataset with few outright
errors. Pairs are directed: a mutual confusion appears once per direction rather than being
averaged into one entry. The margin is the correct dwarf's best similarity minus the
competitor's, so it is negative exactly when the query was misidentified. Results are written
to `results/confusion-<backbone>.json`.

Project the cached vectors into a labeled two-dimensional figure:

    uv sync --extra analysis
    uv run krasnal-id visualize embeddings
    uv run krasnal-id visualize embeddings --override experiment.method=tsne

![UMAP projection of 1,691 DINOv2 embeddings. The 24 classes sitting closest to another class are coloured and named; the remaining 282 are grey.](docs/figures/embeddings-umap-dinov2.png)

*With 306 classes the figure names only the 24 sitting closest to another class and greys the
rest. That selection uses nothing but centroid distance, and it picks out the same Słupniki pair
and water-themed trio the error analysis finds.*

The figure is written to `results/embeddings-<method>-<backbone>.png`. Classes are separated by
color and, beyond the twenty-color palette, by marker shape, and each class is named at its own
centroid with overlapping labels nudged apart and connected by leader lines. Projections are
seeded and reproducible.

Draw the retrieval itself rather than an average of it:

    uv run krasnal-id visualize retrieval-examples

One query photograph beside the five dwarves each backbone ranks highest, written to
`results/retrieval-examples.jpg`. The queries are chosen by rule, not by eye: every fold is scored
under both backbones, folds are grouped by which backbones ranked the right statue first, and the
first query in image-ID order represents its group. `experiment.backbones` names the arms to draw.

## Identify a single photograph

Rank the most likely dwarves for one image:

    uv run krasnal-id retrieve path/to/photo.jpg
    uv run krasnal-id retrieve path/to/photo.jpg --top-k 3 --override backbone=clip

The command reports each candidate dwarf with the cosine similarity and the reference image it
matched. A query whose file content already has a cached vector reuses it, so querying a dataset
image needs neither a model load nor the `ml` extra; any other image is embedded with the
configured backbone, which does require it. Every reference sharing the query's content hash is
withheld, so a dataset image cannot simply match itself.

Exit code 2 indicates a missing or undecodable image, missing embeddings, or an unreadable
manifest.

## Does a trained classifier beat retrieval?

Compare per-class prototypes and a per-fold linear probe against raw cosine retrieval:

    uv sync --extra analysis
    uv run krasnal-id experiment probe
    uv run krasnal-id experiment probe --override backbone=clip

All three methods are scored on the same leave-one-out folds, so the comparison is readable from
one artifact at `results/probe-<backbone>.json`, including each trained method's explicit top-1
gain over retrieval. One classifier is fitted per fold, so a query is never part of the data its
own classifier was trained on.

The linear probe is regularized weakly by default (`C=100`). Embeddings are L2-normalized, so a
conventional `C=1.0` underfits badly: it scored 66% top-1 on this dataset against 96% at the
default. BLAS is held to one thread while fitting, because each per-fold classifier is small
enough that thread oversubscription dominates the runtime.

## Interactive demonstration

The published demo at <https://turhancan97.github.io/krasnal-id/> is a static page: it embeds an
uploaded photograph with a quantised ONNX backbone in the visitor's own browser and ranks it
against reference vectors built by the same code, so nothing is uploaded and there is no backend.
The written findings live on a second page, [`findings.html`](docs/findings.html), so the
identifier itself stays short.

![The published identifier after running one of its example photographs: a DINOv2/CLIP model switch showing 56 MB at 93.2% and 64 MB at 82.4%, the query photograph, the five closest dwarves with similarity scores and photographer credits, and a warning that this statue stands with six others installed as one themed group.](docs/figures/demo-ui.png)

*The identifier, mid-result. Everything shown was computed in the browser.*

**The page runs either backbone, and the switch is a comparison rather than a size tier.** DINOv2
is the default — it is the pipeline's own model, 56 MB at `q4`, and 93.2% top-1 on the vectors it
ships. CLIP is offered beside it at 64 MB and 82.4%, which makes it the *larger* download and the
less accurate one; it is there because the gap between the two is what most of this project's
findings are about, and switching re-ranks the photograph already on screen so a visitor can see
that gap on their own photograph instead of reading it off a chart. CLIP's weights and its
vectors are fetched only if it is selected, so the default page load is unchanged.

Rebuild the data after the manifest changes:

    cd docs/demo && npm install && node build.mjs

The build embeds every reference photograph with both backbones, writes one
`assets/references-<backbone>.bin` per backbone beside the shared `assets/references.json`, and
re-scores the leave-one-out protocol on exactly the vectors it just wrote — the accuracy on each
button is that measurement, never a figure copied from the research pipeline. It also records
probes so the page can verify itself: `?selftest=1` reports cosine agreement with the build for
the selected backbone, and `?selftest=full` re-embeds all 1,691 reference photographs in the
browser and scores the protocol there.

Both sides read `docs/backbones.mjs`, which is the only place the two models are described. They
differ in three ways that must agree between the build and the page or every cosine is
meaningless: the model class, the pooling (DINOv2's CLS token against CLIP's `image_embeds`), and
the shortest edge (256 against 224).

The build sets `intraOpNumThreads` explicitly. onnxruntime sizes its thread pool from the host's
core count rather than the cpuset it is confined to, and on a shared machine that difference cost
**7502 ms per image against 426 ms** — seven hours against twenty-four minutes for a full build.
Set it from `nproc` and re-measure on the machine that runs the build.

There is also a local Gradio version, which uses the research pipeline rather than the browser one:

    uv sync --extra demo
    uv run krasnal-id demo
    uv run krasnal-id demo --top-k 3 --port 7860 --override backbone=clip

Upload a photograph to see the ranked candidate dwarves, their similarity scores, and the
reference photographs each one matched. The manifest and cached vectors load once per session
rather than per query. Uploading a photograph that is already in the dataset withholds its own
reference copies and says so, so it cannot simply match itself.

The demo serves locally and reports nothing to any external service. It writes its scratch files
to a per-user directory rather than the shared `/tmp/gradio`, which fails on a multi-user machine
where another account created that path first; set `GRADIO_TEMP_DIR` to override.

## Does location narrowing help?

Build candidate pools from the real Wikidata coordinates instead of sampling them:

    uv run krasnal-id experiment geo-ablation
    uv run krasnal-id experiment geo-ablation --override backbone=clip

Each query is pooled with its N-1 nearest dwarves, so pool size matches the random ablation and
only the selection rule changes. Both arms are reported side by side with the advantage proximity
buys at each size, along with the median and maximum radius each pool spans so the result reads in
metres. Geographic pools are exact rather than sampled, so they carry no seed spread; the random
arm supplies the error bars. Results are written to `results/geo_ablation-<backbone>.json`.

Every dwarf must have coordinates, and the command says which ones are missing if any are.

## Can it reject a dwarf it has never seen?

Measure whether a similarity threshold can answer "unknown" instead of always naming a nearest
neighbour:

    uv run krasnal-id experiment open-set
    uv run krasnal-id experiment open-set --override backbone=clip

Two populations of equal size are scored. The known arm is the leave-one-out split. The unknown
arm removes every image of a query's own dwarf, so that dwarf is genuinely absent and the correct
answer is rejection. Results are written to `results/open_set-<backbone>.json`.

![Rejection tradeoff curves for both backbones: the fraction of unknown statues wrongly accepted against the fraction of known statues accepted.](docs/figures/open-set-rejection.png)

*Why there is no "I don't know" button on the demo. Draw it from saved results with
`krasnal-id visualize open-set`.*

The headline is AUROC, which is threshold-free and so cannot be tuned. Operating points are named
by the fraction of known queries they accept, and each one's threshold is calibrated
leave-one-class-out: the bar a dwarf's queries must clear comes from the other dwarves' scores
alone. A threshold fitted on everything is reported too, labelled `in_sample`, as an optimistic
reference. Per-dwarf rows record which statues slip through when removed and which statue covered
for them.

Change the operating points, or how many per-dwarf rows are kept:

    uv run krasnal-id experiment open-set --override experiment.target_known_acceptance=[0.8,0.9]
    uv run krasnal-id experiment open-set --override thresholds.open_set_top_rejections=5

The first configured target is the one the per-dwarf rows describe.

Draw the tradeoff from the saved artifacts:

    uv run krasnal-id visualize open-set

Every saved backbone is drawn on one axis, with its calibrated operating points marked on the
descriptive curve they sit on. The figure lands at `results/open-set-rejection.png`.

### Does geometry reject where similarity cannot?

A similarity threshold fails here because every statue is the same semantic category, so a missing
statue's nearest neighbour scores much like a present one's. Geometry is different evidence: two
photographs of one physical object admit a consistent homography and two photographs of
similar-but-different objects do not. Ask whether that rejects:

    uv run krasnal-id experiment open-set-geometry
    uv run krasnal-id experiment open-set-geometry --override backbone=clip

Four signals are computed for every query in one pass — `cosine` (the control), `inliers_top_1`,
`inliers_best` over the checked candidates, and `blended` — so they are compared on an identical
population rather than across runs. Each is reported as a threshold-free AUROC, a false-acceptance
rate at a leave-one-class-out calibrated operating point, and an in-sample balanced accuracy that
is an upper bound. Results are written to `results/open_set_geometry-<backbone>.json`.

Every signal is measured twice. The second condition withholds each query's own photographer from
both arms, because [AGENTS.md](AGENTS.md) section 7.6 found the inlier separation inflated by
same-visit near-duplicates:
a geometric signal that works only when the same person shot the reference has not been shown to
work at all. It also drops the queries whose statue one person documented, which are unanswerable
rather than hard once that person is removed.

    uv run krasnal-id experiment open-set-geometry --override experiment.top_k=10
    uv run krasnal-id experiment open-set-geometry --override experiment.photographer_disjoint=false

Widening `top_k` gives a present statue more chances to verify — and an absent one more chances to
fluke a homography, which is the same tradeoff [RESULTS.md](RESULTS.md) section 11 measured for
accuracy. It reads pixels,
so it needs the images on disk and takes tens of minutes rather than seconds.

## Are phone photographs harder queries?

Commons records each file's EXIF camera, so the references split into phone-shot and camera-shot
queries and the two can be compared against the same reference set:

    uv run krasnal-id data camera-metadata
    uv run krasnal-id experiment camera-gap
    uv run krasnal-id experiment camera-gap --override backbone=clip

The metadata step needs the contact-bearing user agent and writes
`data/discovery/camera-metadata.json`; the experiment is offline and writes
`results/camera_gap-<backbone>.json`. Results are reported per group with the confounds beside
them — median references per class, and query counts — because "phone photographs belong to harder
classes" is the first thing to rule out. Images whose EXIF was stripped form their own `unknown`
group rather than being folded into either side.

This is a lower bound on the real query-domain gap: these are still Commons uploads. See
`data/field-guide.md` for the fieldwork that measures it properly, and the next section for the
path that scores it.

## Measure the query-domain gap with real photographs

Photographs taken in the street are **queries, never references**: the manifest is not rebuilt to
include them, because admitting them as references would destroy the comparison they exist to make.
Drop three to five phone photographs of a statue into `data/field-queries/<dwarf_id>/`, following
`data/field-guide.md`, then:

    uv run krasnal-id data field-queries                    # stage them into a query manifest
    uv run krasnal-id embeddings extract --field-queries    # embed them with the pinned backbone
    uv run krasnal-id experiment field-gap                  # score them against the references

Staging is offline and needs no ML extra. It refuses a directory that names no statue in the
dataset, a statue absent from the reviewed route, a file that will not decode, and any photograph
byte-identical to a reference — that last one would be a Commons upload copied into the query set,
which measures the protocol rather than the domain gap.

The finding is a difference, not an absolute. The experiment compares the field photographs with
the **leave-one-out folds of the same statues**, so pool size and class difficulty are held fixed
and only the query's origin varies, and it writes overall, per-cohort and per-statue rows to
`results/field_gap-<backbone>.json`. Which statues are members of a confused family and which are
controls is fixed in tracked `data/field-route.json` before anything is scored, so the split cannot
be chosen after seeing the result. One asymmetry is reported rather than hidden: a leave-one-out
query is withheld from its own class and so sees one fewer reference of the right statue than a
field query does, which favours the field queries and understates the gap.

## Use the published dataset

The dataset is published twice, so you do not need this repository, a Commons crawl or a GPU to
work with it. On the Hub as
[turhancan97/wroclaw-dwarves](https://huggingface.co/datasets/turhancan97/wroclaw-dwarves), in
parquet with the images embedded and one config per piece, so a comparison of the two backbones
costs about 8 MB rather than the full 676 MB. On
[Kaggle](https://www.kaggle.com/datasets/turhancankargin/wroclaw-dwarves) as images on disk
beside CSV tables, which is what its explorer previews and its notebooks expect —
[`docs/kaggle-starter.ipynb`](docs/kaggle-starter.ipynb) reproduces the headline there in about
twenty lines. The Hub copy is the one this section shows:

```python
from datasets import load_dataset

images = load_dataset("turhancan97/wroclaw-dwarves", "default", split="reference")
vectors = load_dataset("turhancan97/wroclaw-dwarves", "embeddings_dinov2", split="reference")
folds = load_dataset("turhancan97/wroclaw-dwarves", "leave_one_out", split="test")
```

Six configs are published: `default` (the photographs and their attribution), `metadata` (the same
rows without the pixels), `classes` (one row per statue), `embeddings_dinov2` and
`embeddings_clip` (precomputed vectors), and `leave_one_out` (the 1,691 evaluation folds). Only
the config you ask for is downloaded, so scoring from vectors alone costs about 8 MB rather than
the full 676 MB of photographs. The split is called `reference`, not `train`, because nothing here
is trained.

Scoring `embeddings_dinov2` under `leave_one_out` reproduces the headline 93.1% top-1 from the
published files alone.

## Publish the dataset

Only needed to release a new version. The whole dataset — photographs, metadata, embeddings and
the leave-one-out folds — builds into a Hugging Face repository with one command:

```bash
uv run krasnal-id data license-templates   # once: the basis behind the public-domain files
uv run krasnal-id data export-hf           # builds data/export/huggingface
```

That writes six configs (`default`, `metadata`, `classes`, `embeddings_dinov2`,
`embeddings_clip`, `leave_one_out`), a dataset card, a licence inventory, credits grouped by
photographer, a machine-readable `credits.csv`, and a `provenance.json` digesting every emitted
file. It reads the manifest, the split and the embedding cache, and writes to none of them, so
building an export invalidates no published result.

Nothing is published until you say so:

```bash
uv run krasnal-id data export-hf --push                # creates a *private* repository
uv run krasnal-id data export-hf --push --public       # …or a public one
```

The token is never passed or logged — `huggingface_hub` resolves `HF_TOKEN` or your stored login
— and `--push` creates the repository private unless `--public` is given, because a mistyped
repository id becoming world-readable is not recoverable.

Every photograph keeps the licence it arrived with; the export never relicenses one. Each row
carries its photographer, licence, SPDX identifier, source URL and a ready-to-paste
`attribution_text`, plus a `modified` flag derived by comparing the stored bytes against the
Commons digest — 1,538 of the 1,691 files are downscaled adaptations and 153 are byte-identical
to their originals.

### Kaggle

Kaggle is a second writer rather than a second target, because file shape is a platform
convention: the Hub wants parquet whose image column is a `{bytes, path}` struct, and a Kaggle
user opening an image dataset expects a folder of images beside a table describing them.

```bash
uv run krasnal-id data export-kaggle                       # builds data/export/kaggle
uv run krasnal-id data export-kaggle --no-images           # metadata and vectors only, ~10 MB
uv run krasnal-id data export-kaggle --dataset-id you/wroclaw-dwarves
```

That writes `images/<dwarf_id>/`, `images.csv`, `classes.csv`, `folds.csv`, one
`embeddings_<backbone>.npy` per backbone, Kaggle's `dataset-metadata.json`, and the same
`LICENSES.md`, `ATTRIBUTION.md`, `credits.csv` and `provenance.json` the Hugging Face export
publishes — deliberately the same generated artifacts, so the two platforms cannot disagree about
a photographer.

Three things it refuses rather than discovers late. Kaggle rejects an over-long title or slug
instead of trimming it, so the limits are checked before 676 MB is copied. `folds.csv` stores
only the query, since every fold's reference set is every other image — a rule the export
verifies fold by fold and refuses if it stops holding. And a `.npy` has no keys, so the vector
order is compared against `images.csv` before writing, because a mismatch would make every
downstream number wrong without failing.

The licence field says `other` — Kaggle's "Other (specified in description)" — because Kaggle
takes one licence and this corpus has ten. The generated description carries the family
breakdown, the modified/unmodified split, the freedom-of-panorama disclosure and the removal
path, which is what makes that field honest rather than vague.

It also writes `data/export/kaggle-cover.jpg`, a 1200x600 grid of 32 statues sampled by even
stride — *beside* the upload directory, not inside it, because Kaggle's cover image is set in
the web UI and a file in the folder would be published as data. `docs/kaggle-starter.ipynb`
reproduces the headline from the published files in about twenty lines, and is worth publishing
as a notebook on the dataset.

Two things Kaggle's renderer and validator will not tell you. The description is rendered as
markdown **with HTML parsing**, so an angle bracket anywhere in it — `embeddings_<backbone>.npy`,
say, even inside backticks — is read as an unknown opening tag and silently swallows everything
after it. And `resources` may only name files that exist in the folder: a directory entry fails
the upload with "does not exist", and so would `images.zip`, which `--dir-mode` creates during
the upload rather than in the folder. Tests pin both.

Publishing is left to you, because a Kaggle dataset slug cannot be renamed once created:

```bash
kaggle datasets create -p data/export/kaggle --dir-mode zip
kaggle datasets version -p data/export/kaggle --dir-mode zip -m "manifest <hash>"
```

Scoring leave-one-out from `embeddings_dinov2.npy` and `images.csv` alone — no project import —
reproduces 93.1% top-1 and 95.7% top-5, and CLIP's 82.9%, which are the published numbers exactly.

## Development checks

```bash
uv lock --check
uv run ruff format --check .
uv run ruff check .
uv run mypy
uv run pytest
```

`mypy` runs in strict mode over `src/krasnal_id` and `tests`, so `uv run mypy src tests` is clean
too. The asset generator under `docs/brand/` is deliberately outside that scope: it is a one-off
script whose only errors come from untyped plotting dependencies.

See [CONTRIBUTING.md](CONTRIBUTING.md) for the development workflow and
[AGENTS.md](AGENTS.md) for the research scope, architecture, and build order.

## Citing this work

Archived on Zenodo with a DOI that always resolves to the newest release:

```bibtex
@software{kargin2026krasnalid,
  author    = {Kargın, Turhan Can},
  title     = {Krasnal-ID: fine-grained visual instance retrieval of
               Wrocław's dwarf statues},
  year      = {2026},
  publisher = {Zenodo},
  doi       = {10.5281/zenodo.22548023},
  url       = {https://doi.org/10.5281/zenodo.22548023}
}
```

That DOI is the *concept* DOI — cite it, and citations accumulate across versions rather than
splitting between them. Each release also gets its own version DOI, listed on the Zenodo record,
for when you need to pin exactly what you ran.

The record archives the source code, the documentation and the published demo assets. The
photographs themselves live on the
[Hugging Face dataset](https://huggingface.co/datasets/turhancan97/wroclaw-dwarves) and its
[Kaggle copy](https://www.kaggle.com/datasets/turhancankargin/wroclaw-dwarves), where each one
carries its own photographer and licence; cite that alongside the DOI if the data is what you
used, and credit the photographers as `credits.csv` there sets out.

## Data and licensing

The source code is licensed under the MIT License. **It does not apply to the photographs.**

The 1,691 reference images come from Wikimedia Commons and each keeps its own licence: 1,577
CC BY-SA (2.0 through 4.0), 55 CC BY, 52 public domain and 7 CC0. Every manifest record retains
its source URL, author, licence and licence URL as a schema requirement, and the published dataset
carries all four per row so the attribution obligation travels with the data. Nothing in this
project relicenses a photograph or applies one licence across the collection.

1,538 of the stored files are downscaled to at most 2,000 px and re-encoded, which makes them
adaptations; the other 153 are byte-identical to the Commons original. The `modified` column says
which, derived from the bytes rather than asserted.

The statues themselves are contemporary sculptures under copyright. Commons hosts photographs of
them under Poland's freedom-of-panorama provision; the Creative Commons licences here cover the
**photographs**, granted by the photographers, and grant no rights in the sculptures depicted. If
you are a rights-holder and want an image removed, open an issue: exclusions are recorded in
`data/image-review.json`, and a rebuild propagates the removal through the manifest, the folds,
the embeddings and the published dataset.

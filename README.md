<picture>
  <source media="(prefers-color-scheme: dark)" srcset="docs/brand/krasnal-lockup-horizontal-dark.svg">
  <img src="docs/brand/krasnal-lockup-horizontal-light.svg" alt="Krasnal-ID" width="260">
</picture>


Krasnal-ID is a research prototype for fine-grained visual instance retrieval of
Wrocław's bronze dwarf statues. Its central experiment asks how identification accuracy
changes as the candidate pool grows, and how much a simulated location-based narrowing
would help.

## Project status

The repository contains the complete typed scaffold for versions 0.1-0.3 plus implemented,
cached Wikidata discovery, reviewed Wikimedia Commons acquisition, audited manifest construction,
deterministic evaluation splits, resumable DINOv2/CLIP embedding extraction, cosine k-NN
retrieval, the full-pool accuracy baseline, the candidate-pool-size ablation, confusion
analysis, embedding visualization, single-image retrieval, the trained-classifier comparison,
and the interactive demonstration. The v0.1-v0.3 build order is complete and released as
`0.3.0`. Open-set rejection was added and released as `0.4.0`, and `AGENTS.md` section 8 records
the research questions that remain open.

## Findings

DINOv2 reaches 95.9% top-1 across the full 23-dwarf pool, accuracy decays by about one point per
doubling of the candidate pool against CLIP's 1.8, narrowing by real location helps *less* than
random subsampling suggests, a trained classifier does not beat raw retrieval, and the errors
concentrate on one explainable cluster of water-themed statues that turns out to be a single
co-located installation. Thresholding similarity does let it answer "I don't know this one",
for about three points of accuracy on the statues it does know.

- [**Identify a photograph**](https://turhancan97.github.io/krasnal-id/) — the findings, plus a
  working identifier that runs the model in your browser. Nothing is uploaded.
- [**RESULTS.md**](RESULTS.md) — the complete written record: dataset construction, all four
  experiments, limitations, and how to reproduce them.

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
export KRASNAL_ID_USER_AGENT='krasnal-id/0.4.0 (mailto:you@example.com)'
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
export KRASNAL_ID_USER_AGENT='krasnal-id/0.4.0 (mailto:you@example.com)'
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

The figure is written to `results/embeddings-<method>-<backbone>.png`. Classes are separated by
color and, beyond the twenty-color palette, by marker shape, and each class is named at its own
centroid with overlapping labels nudged apart and connected by leader lines. Projections are
seeded and reproducible.

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
uploaded photograph with a quantised ONNX CLIP in the visitor's own browser and ranks it against
reference vectors built by the same pipeline, so nothing is uploaded and there is no backend.
Rebuild its data after the manifest changes:

    cd docs/demo && npm install && node build.mjs

The build re-scores the leave-one-out protocol on the vectors it ships (91.8% top-1, 98.6% top-5)
and records probes so the page can verify itself: `?selftest=1` reports cosine agreement with the
build, and `?selftest=full` re-embeds all 146 reference photographs in the browser and scores the
protocol there.

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

## Data and licensing

The source code is licensed under the MIT License. Images and metadata obtained from
Wikimedia Commons remain subject to their individual licenses. Every image manifest record
must retain its source URL, author, license, and license URL. The MIT License does not apply
to downloaded third-party images.

# Agents.md — Krasnal-ID (working title)
Research prototype: fine-grained visual instance recognition on Wrocław's bronze dwarf statues. Built as a CV/portfolio piece, not a shipping product.

## 1. Project overview
Given a photo of a Wrocław dwarf statue, retrieve which specific one it is from a reference set built from public CC-licensed photos. The interesting part isn't "call a pretrained model" — it's quantifying **how retrieval accuracy degrades as the candidate pool grows**, which is the real research question a location-narrowed mobile version would depend on.

## 2. Goals
- Produce a clean, well-documented research repo suitable for a CV/portfolio link.
- Answer one real research question rigorously (see §3), not just report a single accuracy number.
- Reuse solid engineering habits: pinned dependencies, structured logging, cached embeddings, reproducible experiments.

## 3. Non-goals (explicit, to keep scope tight)
- No mobile app, no production backend, no city/tourism-board partnership.
- No requirement to cover all ~1,000–1,400 dwarves — coverage is capped by what has usable reference photos, not by geography.
- No live GPS pipeline — location-narrowing is *simulated*, not built as a real service (see §5.2).

## 4. Research question (the core hook)
> Naive global instance recognition across N similar-looking classes gets harder as N grows. If the true candidate pool can be narrowed (e.g. by location) to a small local set, how much does accuracy actually improve, and at what pool size does it become a genuinely reliable identification method?

This ablation — **accuracy vs. candidate-pool size** — is the headline result. Everything else in this repo supports producing that one curve credibly.

## 5. Data

### 5.1 Sourcing and licensing
- Primary source: Wikimedia Commons ("Dwarves in Wrocław" category and per-dwarf subcategories) + Wikidata (structured entries with coordinates for some dwarves).
- Most images are **CC-BY-SA**: attribution and share-alike required. Every stored image record must carry `source_url`, `author`, `license`, `license_url` as first-class metadata — not an afterthought bolted on at the end. This isn't legal advice; confirm exact terms hold before any public-facing use of the dataset itself.
- Selection criterion for inclusion: **≥3 usable reference photos per dwarf** (threshold tunable). This replaces the earlier "geographic cluster" scoping from the mobile-app version — for a research prototype, data quality is the constraint, not tourist zones.

### 5.2 Location metadata (for the ablation)
- Where Wikidata provides coordinates, use them for a **real** geo-based candidate-pool experiment (secondary/bonus — coverage will be partial).
- The **primary** ablation should not depend on coordinate coverage: simulate candidate pools by repeatedly sampling random subsets of size N from the full class set and measuring retrieval accuracy at each N. This guarantees the headline experiment works regardless of how much real geo-data ends up available.

### 5.3 Pipeline
1. Query Wikidata for Wrocław dwarf-statue entities (name, coordinates if present, linked Commons category).
2. Pull images from each linked Commons category via the Commons API.
3. Filter to dwarves meeting the ≥3-image threshold.
4. Cache bounded research image copies + metadata to disk in a structured format (one JSON manifest + image files, not a database).

### 5.4 Wikidata discovery decisions

- Discover only items explicitly typed as Wikidata class `Q136276280`. Require a `P373` Commons category, keep `P625` coordinates optional, and prefer Polish labels, then English labels, then QID.
- Exclude group entities only when `P527` links to independently eligible member records. Naming heuristics may create manual-review warnings but must not exclude records.
- Cache the complete raw SPARQL response with endpoint/query-hash provenance. A deterministic `--limit` is applied only after full normalization; `--refresh` bypasses a valid cache.
- Live requests require a contact-bearing `KRASNAL_ID_USER_AGENT` environment variable. Cached normalization does not.
- Write ignored, atomic artifacts below `data/discovery/`: raw response, cache metadata, normalized dwarf records, and an exclusion/warning audit.

### 5.5 Commons acquisition decisions

- Treat `data/discovery/dwarfs.json` as the exact handoff into fetching. Every emitted
  Wikidata-to-Commons mapping must be approved or rejected in tracked
  `data/category-review.json`; corrections belong there rather than in generated discovery
  files. New or changed mappings return to `pending`.
- Query direct category files only, follow pagination, and cache complete API responses.
  Accept known Public Domain, CC0, CC BY, and CC BY-SA static rasters with complete attribution.
- Store Pillow-verified research copies with a 400-pixel minimum short side and 2,000-pixel
  maximum long side. Downscale oversized Commons derivatives locally with Pillow using
  aspect-ratio-preserving Lanczos resampling; never upscale undersized images. Preserve Commons
  page/revision provenance and reuse verified unchanged files.
- Treat a fixed multi-dwarf sculpture installation as one class only when its direct files
  consistently depict that installation. Reject umbrella categories whose files represent many
  distinct statues. Component categories may remain approved, but shared files are quarantined.
- Retain the lowest Commons page ID for same-label byte duplicates. Exclude byte-identical
  content spanning different labels to prevent evaluation leakage. Never delete orphan files
  automatically.

### 5.6 Commons-first discovery decision (2026-09-04)

Accepted as the second post-0.3.0 scope change, answering the "larger pool" question §8 lists.
Wikidata is the binding constraint and cannot be widened into one: measured on 2026-09-04, only
**44** items carry `P31 = Q136276280`, and a reverse check found the same 44 to be the only
Wikidata items pointing at *any* of the **481** per-dwarf categories under
`Category:Dwarves in Wrocław by name`. Relaxing the type filter therefore gains nothing; Wikidata
simply has no item for the other ~437 statues.

Commons holds the photographs. A 150-category sample of those 481 found 65% with three or more
files and a mean of 4.14, projecting ~314 categories over the current threshold and ~2,000 files
in total, against 23 classes and 146 images today.

Decisions:

- **`dwarfs.json` stays the single discovery artifact with a single `query_sha256`.** Every
  downstream stage validates against that hash, so a second discovery file with a second hash
  would fork the provenance chain that §5.5 depends on. Commons enumeration is therefore a second
  source *inside* `data query`, behind an explicit `--include-commons`, and the recorded hash
  covers both queries. Without the flag the command reproduces the Wikidata-only artifact exactly.
- **A Wikidata record always wins over a Commons category describing the same statue.** The 44
  typed items carry `P625` coordinates and a stable QID; the merge prefers them and keeps the
  Commons-only records only for categories no Wikidata item claims.
- **Commons-only classes take a `C-<slug>` identifier** derived from the category title, which is
  unique on Commons by construction. The slug is ASCII-folded and filesystem-safe because the
  identifier names an image directory. A slug collision is an error to review, never a silent
  merge of two statues.
- **`wikidata_url` becomes optional and the dwarf-ID pattern widens** in the manifest and both
  review contracts. A Commons-only statue has no Wikidata item, so requiring one would be a
  schema lie.
- **The geographic ablation does not grow with the dataset.** Coordinates come from `P625`, so
  they exist for the 44 and for none of the rest. §7.1's geographic finding stays scoped to the
  statues that have coordinates, and the arm must report how many classes it actually covers
  rather than implying it covers the pool. Do not report a geo result over a Commons-first
  manifest without saying what fraction of it carried coordinates.
- **Review burden scales with discovery, and that is accepted rather than automated away.** 481
  category decisions instead of 41 is the cost of the larger pool. Mechanical mappings may be
  pre-approved by pattern in a later change, but a human decision remains the contract.
- **The merge matches on exact category equality, which is all it can honestly do.** A statue
  filed under a Commons category that Wikidata's `P373` does not name appears twice, once per
  source, and only category review catches it. *Papa Krasnal* is the live example: its Wikidata
  item points at its sculptor's category (`Olaf Brzeski`) rather than at `Papa Krasnal`, so both
  records survive discovery. Never assume the merge deduplicates statues; it deduplicates
  category strings.
- **A normalization warning about a category the merge then supersedes is dropped.** A reviewer
  reading 481 categories must not be sent to look at records that never reached the artifact.
  Exclusions survive regardless, because they explain an absence.
- **Pattern pre-approval is a human decision applied by pattern, taken on 2026-09-04.** Of 441
  new mappings, 367 matched a singular `<Name> dwarf, Wrocław` title with no discovery warning
  and were approved in bulk. The reasoning is that a Commons-only class has no Wikidata mapping
  to verify — the category *is* the identity — and every image still passes the license, size,
  attribution, prominence and cross-label duplicate gates that §5.5 defines. A further 13 were
  approved because Commons holds fewer than three files for them: the mapping is correct, and
  the documented image threshold is what decides whether a class exists. Rejection stays
  reserved for a mapping that is *wrong*, which a thin category is not.
- **Plural and irregular titles are reviewed by hand, always.** A plural title may be one fixed
  installation or an umbrella over many distinct statues, and only looking at the files can tell
  them apart. Two plural categories are known to be unusable in advance: `Troszka i Adoratorek`
  and `Śpiewak Operowy i Tancerka Balerina` are the sources of every `cross_label_duplicate`
  quarantine in §12.1, so their files would be rejected again under a new label.
- **Re-acquisition invalidates every published number.** Changing the discovered set changes
  `query_sha256`, which invalidates staging, the manifest, the split, the caches' relevance and
  all five experiments plus both figures. Treat a Commons-first rebuild as one deliberate
  re-run of §6.4's whole reproduction sequence, never as an incremental data top-up.

Measured on 2026-09-04, running `data query --include-commons` against both live sources:

- **482 records: 41 from Wikidata, 441 Commons-only.** The Wikidata arm emits exactly the 41 it
  always did, so the flag adds classes without disturbing the existing ones. 41 of the 482 carry
  coordinates, which is the geographic ceiling this decision accepts.
- 56 audit entries: 40 categories superseded by a Wikidata record, 12 titles too irregular to
  parse a name from, 4 possible unlinked groups and 3 excluded group entities. Twelve titles to
  read by hand out of 481 is the review surface, not 481 — but every one of the 441 new records
  still needs a category-review decision before `data fetch` will touch it.
- Commons uses `dwarf`, `dwarfs` **and** `dwarves`, plus an undiacriticked `Wroclaw`. Missing
  `dwarves` initially cost 76 categories their display name, which is why the title pattern
  accepts all three and why anything it cannot parse is flagged rather than guessed.

### 5.7 Derived coordinates decision (2026-09-05)

§5.2 took location from Wikidata's `P625` alone, which capped the geographic experiment at 23 of
306 classes — 7.5% of the pool. Commons file metadata lifts that cap. Measured on 2026-09-05:

- **1,135 of 1,545 photographs (73.5%) of otherwise-unlocated classes carry a coordinate**, almost
  all of type `camera` — where the photographer stood, not where the statue is. That gives at least
  one coordinate for **293 classes that had none**, for 295 of 306 in total.
- Commons *category* pages are not a source: 1 of 283 carries a coordinate.
- **Validated against the 21 classes that have both**: the median of a class's camera positions
  falls a **median of 9 m** from its Wikidata point, with all 21 within 147 m and 14 within 50 m.
  The geographic arm's smallest median pool radius is 190 m, so the noise is an order of magnitude
  below the signal it is used to measure.

Decisions:

- **A derived coordinate is stored as such and never passed off as authoritative.**
  `DwarfRecord.coordinate_source` records `wikidata` or `commons_camera`, and a record carrying
  coordinates must name where they came from. Wikidata always wins when both exist.
- **The per-class position is the median of its images' coordinates, not the mean.** One
  mis-tagged photograph should not drag a class across the city, and the median of an even
  handful of points is robust to that; the mean is not.
- **Coordinates are image metadata and belong on `ImageRecord`**, fetched in the same Commons
  request that already retrieves image info rather than through a separate enrichment pass. One
  source of truth for what is known about an image, and one provenance chain, as §5.5 requires.
- **The three worst validation cases are the water installation** — *Zbierający Wodę*, *Karmiący
  Ptaki* and *Wierzbownik*, all 147 m out. §12.1 already suspected Wikidata assigns group members
  one shared point, so there the camera positions are plausibly the *more* accurate figure. Do not
  treat `P625` as ground truth when reporting the disagreement; report it as disagreement.
- **This makes the §7.1 co-location finding testable at scale.** That result currently rests on six
  statues in one themed installation. Whether proximity is genuinely unhelpful city-wide, or only
  around that installation, is answerable at 295 classes and was not at 23.

## 6. Technical architecture

### 6.1 Embedding backbone
- DINOv2 and CLIP, used for zero-shot/near-zero-training feature extraction (no fine-tuning needed for the core experiment).

- **Design for a swappable backbone interface** (`get_embedding(image) -> vector`) rather than hard-coding a single library call. This keeps the option open to plug in a shared embedding-extraction module later without this repo depending on it existing or being finished first. Until then, load DINOv2/CLIP directly (e.g. via `transformers`).
- Cache embeddings to disk — never recompute per experiment run.
- Pin every backbone to an immutable revision that actually serves `model.safetensors` at that
  revision. `transformers` 5 refuses `pytorch_model.bin` and silently falls back to a mutable
  community safetensors-conversion ref, which would make the revision recorded in the embedding
  cache key untrue. CLIP is therefore pinned to the conversion commit
  `c237dc49a33fc61debc9276459120b7eac67e7ef` rather than to `main`.
- Treat backbone output shape as version-specific: `CLIPModel.get_image_features` returns a
  vision-output object whose `pooler_output` holds the projected features, not a bare tensor.
- `torchvision` belongs in the `ml` extra; `AutoImageProcessor` requires it.

### 6.2 Retrieval
- Cosine similarity k-NN over cached reference embeddings is the primary method.
- Report headline `top_k`/`mrr` over distinct **dwarves**, ranked by their best-matching image,
  because that is the candidate list an identification tool presents. Image-level `image_top_k`
  and `image_mrr` are reported alongside for comparison, never as the headline.
- Every evaluation must refuse a split whose recorded manifest hash does not match the manifest
  it is run against, rather than silently scoring a stale protocol.
- Accuracy proportions carry 95% Wilson score intervals as error bars. Rank averages such as MRR
  are not proportions and carry no interval.
- The baseline is exhaustive and deterministic, so its configured seed is recorded for provenance
  only. Nothing in it samples.
- The ablation samples a pool per query from one generator per (pool size, seed), so a run is
  reproducible no matter which pool sizes were requested. Configured pool sizes above the class
  count are skipped with a warning rather than clamped, and the full pool is always measured as a
  comparable right-hand anchor.
- Per-pool error bars are the observed spread across seeds, not a distributional assumption.
- The geographic ablation pools each query with its **N-1 nearest** dwarves rather than by a fixed
  radius, so pool size is matched to the random arm and only the selection rule varies. The median
  and maximum radius each pool spans are reported alongside, which keeps the result interpretable
  in metres. Nothing samples there, so a geographic measurement is exact and carries no seed
  spread; the random comparison arm supplies the error bars.
- Confusion analysis records the strongest wrong dwarf for **every** query, not only for failures.
  On a dataset with few outright errors the near-misses are where the signal is. Pairs stay
  directed, so an asymmetric confusion does not average away against its reverse.
- Strict `mypy` covers `src/krasnal_id` and `tests`. Test fixtures construct validated models with
  real `HttpUrl`, `datetime` and `Path` values rather than leaning on Pydantic's runtime coercion,
  so a fixture that drifts from a contract fails type checking rather than silently coercing.
- Analysis dependencies load lazily behind `import_optional_analysis`, mirroring the ML
  backbones, so the package imports without the `analysis` extra installed.
- Single-image retrieval reuses a cached vector whenever the query file's content hash already
  has one, so querying a dataset image needs no model load and no `ml` extra. The cache is only
  read there; populating it stays the job of `embeddings extract`.
- A query is compared against references by content hash, and every byte-identical copy of it is
  withheld. Otherwise a dataset image would match itself at similarity 1.0 and report nothing.
- The probe comparison always evaluates a `retrieval` arm on the same folds as the trained
  methods, so the answer to "does training beat retrieval" is readable from one artifact rather
  than assembled across two.
- Regularize the linear probe **weakly**. Embeddings are L2-normalized, so per-dimension
  magnitudes are near `1/sqrt(d)` and a conventional `C=1.0` underfits badly: on the current
  dataset it scored 66% top-1 against 96% at `C=100`. The default is 100.
### 6.3 Browser demo decisions (2026-09-02)

The published demo at `turhancan97.github.io/krasnal-id` is static: GitHub Pages cannot run the
Gradio app, so the model runs in the visitor's browser instead. Four decisions there were measured
rather than assumed, and each cost accuracy when guessed wrong.

- **Anything that embeds a query must embed the references.** Reference vectors built by the Python
  pipeline scored 89.7% top-1 against browser-built queries, where the build reported 93.2%.
  `docs/demo/build.mjs` therefore runs the same library, model and dtype the browser runs.
- **Never ship `vision_model_quantized.onnx`.** Of the exports offered it is the only one that
  degrades: 87.7% top-1 against 93.2% for `q4`, `uint8` and full precision, which are
  indistinguishable from each other here. `q4` is used, at 64 MB.
- **Resize through `docs/resize.mjs`, which both sides import.** sharp's lanczos3 and a browser
  canvas disagree by enough to matter, and canvas quality varies between browsers, so neither
  platform's built-in resampler can provide the guarantee. Letting transformers.js do the
  downscale itself is worse still: it resizes a 2000-pixel photograph in one aliasing step and
  loses about two points outright.
- **Decode drift is not eliminable and does not matter.** References are decoded by sharp and a
  visitor's photograph by their browser; measured cosine agreement is 0.986. Scored in a real
  browser over all 146 references, that costs nothing: 91.8% top-1 either way.
- The site reports the accuracy of the vectors it actually ships, re-scored at build time, and
  `?selftest=full` lets any visitor reproduce it in their own browser. Never quote the research
  numbers as the demo's.
- Rebuilt at 306 classes on 2026-09-04: the shipped vectors score 82.4% top-1 against 82.9% for the
  research CLIP pipeline, so the browser/pipeline drift is 0.5 points at this scale. Assets are
  29 MB, of which 22 MB is 1,691 thumbnails; that size was accepted deliberately so every match
  shows its own reference photograph rather than a stand-in for its class.
- The page's chart data in `docs/chart.js` is hardcoded and must be regenerated from
  `results/pool_size_ablation-*.json` whenever the ablation is re-run. Its gridlines and axis
  bounds are hardcoded too: check they still span the series, or a backbone's whole curve can land
  in an unlabelled void, as happened when CLIP's floor fell from 92.5% to 82.9%.

### 6.4 Result publication policy (2026-09-01)

- `results/` stays ignored: it is regenerated output. Figures selected for publication are copied
  to tracked `docs/figures/` and referenced from `RESULTS.md`, which is the written record of what
  the experiments found. Regenerate a published figure with its `visualize` command and copy it
  across rather than editing it by hand.
- Every number in `RESULTS.md` must be traceable to a committed command and a `results/` artifact.
  Extrapolations beyond the measured range are labeled as such, together with why they are
  optimistic.

- The demo loads the manifest and cached vectors once per session, not per query, and its
  callback returns an explanatory status string instead of raising, because a Gradio callback
  that raises shows the visitor a stack trace.
- Point Gradio at a per-user temporary directory unless `GRADIO_TEMP_DIR` is already set. Its
  default `/tmp/gradio` fails outright on a multi-user machine where another account created it
  first. Gradio analytics are disabled: launching a local research demo must not report to an
  external service.
- Hold BLAS to one thread while fitting per-fold classifiers. Each fit is tiny, so thread
  oversubscription dominates: the leave-one-out sweep went from over six minutes to twelve
  seconds. `threadpoolctl` is declared in the `analysis` extra but its absence is not an error.
- Optional stretch baseline: a simple linear probe or per-class prototype (mean embedding) comparison, to see whether a trained classifier beats raw retrieval — useful discussion material for the writeup, not required for the headline result.

## 7. Experiments
1. **Baseline accuracy**: top-1, top-5, and mean reciprocal rank, DINOv2 vs. CLIP, full candidate pool.
2. **Headline experiment**: accuracy vs. candidate-pool size N (synthetic random subsampling, repeated with multiple seeds per N for error bars; real geo-based pools as a secondary comparison if coordinate coverage allows).

### 7.1 Dataset-scale decision (2026-09-01)

The dataset stays at its ≥3-image threshold, and the headline result is reported as the
**rate of degradation per doubling of pool size** rather than as a pool size at which
identification becomes unreliable. Measured evidence behind that choice:

- The current 23-class pool already sits near ceiling (DINOv2 top-1 95.9%, CLIP 92.5%), but the
  curve is not flat: DINOv2 loses about 1 accuracy point per doubling and CLIP about 1.8, with a
  seed spread near one point. The backbone gap widening with N is itself a reportable result.
- Lowering the threshold to 2 was measured, not estimated: it adds 4 classes and 8 images and
  leaves the curve unchanged (DINOv2 full-pool top-1 96.1% against 95.9%). Not worth the weaker
  per-class reference sets.
- Recovering the excluded images is not available locally. Of 23 fetch-audit exclusions, 20 are
  `cross_label_duplicate` files on 4 single-image classes, and admitting them would create the
  evaluation leakage §5.5 forbids. Growing the class count needs new Commons acquisition with
  unknown yield.
- Any extrapolation past N=23 must be labeled as such and called optimistic: distractors are
  drawn from a small class population, while a real 1,000-plus dwarf pool holds far more
  genuinely confusable statues, so the true curve should fall faster than a log-linear fit.
- The 2026-09-01 geographic ablation measured what §5.2 left open. Real proximity pools are
  *harder* than random pools of the same size, because six of the 23 dwarves stand within one
  metre of each other as one themed installation and are exactly the statues the confusion
  analysis flags. Random subsampling scatters them across pools and therefore overstates what
  location narrowing buys. Do not describe the simulated curve as a lower bound on a
  location-aware system.
### 7.2 Open-set rejection decision (2026-09-03)

Accepted as the first post-0.3.0 scope change, under the option §8 records as the nearest one.
`RESULTS.md` lists it as a limitation: a query of a statue outside the reference set still
returns neighbours, because the system cannot say "I don't know this one." Closing it needs no
new data, so it is measured on the existing 23-class manifest and cached vectors.

The mechanism under test is the cheapest one that could work: threshold the top-1 cosine
similarity. Accept the ranking when the best match scores at or above a threshold, reject the
query as unknown below it. No new model, no training.

Protocol — two query populations of 146 each, both derived from the manifest with no sampling
and therefore no seed:

- **Known queries** are the existing leave-one-out folds. The correct dwarf is in the gallery, so
  the right behavior is to accept *and* rank that dwarf first. Accepting a query but naming the
  wrong dwarf is not counted as a success; that is what makes the metric mean what it says.
- **Unknown queries** remove every image of the query's own dwarf from the gallery, which makes
  that dwarf genuinely absent. The right behavior is to reject.

Reporting rules, so the result cannot be read as better than it is:

- **AUROC over the two populations is the headline**, because it is threshold-free and cannot be
  tuned. The operating points are secondary.
- **A threshold calibrated on all the data is in-sample and is labeled `in_sample`.** It is
  reported as an optimistic reference only. Every headline operating point instead calibrates
  leave-one-class-out: the threshold for a held-out dwarf is a quantile of the known-query scores
  of the *other* dwarves, so no query helps set the threshold that judges it.
- Operating points are named by their target known-acceptance rate, and the achieved rate is
  recorded next to the target rather than assumed to equal it. A quantile of 146 discrete scores
  does not land exactly on a requested rate.
- Per-dwarf rejection rows record which dwarves survive removal, and which dwarf their orphaned
  queries fall through to. A cluster of visually similar statues should be able to cover for a
  removed member, so this is where the §7.1 water-themed installation is expected to reappear.

Measured on 2026-09-03, recorded here because two of these constrain future work:

- Rejection works for DINOv2 and not for CLIP, and the gap is far wider than closed-set accuracy
  implies: 0.969 against 0.898 AUROC, where top-1 differs by only 3.4 points. At a 90% known
  acceptance target DINOv2 falsely accepts 4.1% of unknown queries and CLIP 28.1%. **Do not offer
  a rejection threshold on CLIP embeddings**; on this dataset there is no useful operating point.
- The in-sample and leave-one-class-out false-acceptance rates agree (DINOv2 4.1% both, CLIP 30.1%
  against 28.1%), so the threshold is a property of the embedding space rather than of this
  sample. Leave-one-class-out calibration stays the reported default regardless, because that
  agreement is a finding about this dataset and not a licence to fit on the answers.
- Identifiable does not imply rejectable, and this is the result that generalizes. *Kowal* is
  never confused for anything while present (§5 confusion: 0 of 10) yet is the worst dwarf to
  reject once removed, with *100matolog* covering for it 3 times in 12. Closed-set confusion
  analysis cannot surface this, so neither analysis substitutes for the other.
- The water-themed installation does reappear as predicted, and under CLIP it is absolute: all
  three statues are falsely accepted 3 of 3.
- The artifact's `curve` is **descriptive, not calibrated**: it is swept in-sample over every
  observed score so that any threshold's cost is readable, and `visualize open-set` draws it with
  the calibrated leave-one-class-out points marked on top. Do not quote a point off the curve as
  an achievable operating point; quote the metrics.
- The published demo ships no rejection, and **it cannot simply be given one**: section 6.3's
  browser build runs CLIP, which is the backbone with no usable operating point. Giving the demo
  an "I don't know" answer means porting DINOv2 to the browser — a section 6.3-scale change of
  model, reference vectors, self-test baselines and download size — not adding a threshold to
  what is there.
- A DINOv2 port would still have to re-measure its own threshold on the vectors it ships. Section
  6.3 measured `q4` quantization as indistinguishable from full precision, but it measured that
  for *ranking*, and rejection depends on absolute similarity rather than on order. That
  equivalence therefore does not transfer, and section 6.3's rule that the site reports the
  accuracy of the vectors it actually ships applies to any threshold it might adopt.

3. **Error analysis**: confusion matrix for most-confused pairs — which dwarves get mixed up, and why (visually similar poses/props is the expected story).
4. **Embedding-space visualization**: t-SNE or UMAP plot of the reference set, colored by class, to make the "why confusion happens" argument visually.
5. **Open-set rejection**: can a top-1 similarity threshold answer "unknown" for a statue outside the reference set, and what does that cost on the statues inside it (see §7.2).

### 7.3 Dataset-scale result (2026-09-04)

The Commons-first rebuild of §5.6 took the dataset from 23 classes and 146 images to **306 and
1,691**, so §7.1's extrapolation caveat is now a measured quantity rather than a warning. What it
found, recorded here because three of these revise published conclusions:

- **§7.1's warning was half right, and the half it got wrong matters.** It said extrapolating past
  N=23 would be optimistic because a small pool holds too few genuinely confusable distractors.
  True for CLIP, which decays 2.06 points per doubling against the 1.76 the small pool predicted
  and is still worsening at N=306. False for DINOv2, which decays 0.79 against a predicted 0.96 —
  the small pool was *pessimistic* about it. Do not apply "small datasets flatter the result" as a
  blanket rule; it mispredicted one of the two backbones.
- **The curves differ in shape, not just slope.** DINOv2 loses 0.82 points per doubling from a
  pool of 2 to 20 and 0.77 from 50 to 306; CLIP loses 1.79 then 2.53. A single fitted slope hides
  that, so report the early and late rates whenever the range spans more than a few doublings.
- **Open-set rejection does not survive the larger pool**, which retires the §7.2 headline. DINOv2
  falls from 0.969 to 0.896 AUROC, and false acceptance at the 90% target from 4.1% to 38.3%,
  because an absent statue now has 305 chances to find a lookalike instead of 22. The mechanism
  predicts no recovery at city scale. Do not describe rejection as a working feature.
- **A linear probe now helps CLIP (+3.1 points) and still does nothing for DINOv2 (−0.1).** At 23
  classes both showed the same +0.7 non-effect. A supervised layer can partly repair an embedding
  space that is not laid out for instance discrimination and has nothing to add to one that is.
  The confidence intervals still overlap slightly, so report the direction, not a decisive win.
- **The dominant confusion cluster changed** from the water-themed trio to the three *Słupniki*
  pillar dwarves, which are near-identical statues installed on different streets. The embedding
  projection selects them without being told, using centroid distance alone.
- **Two group classes overlap their own members** (*Grajek i Meloman* with *Grajek* and *Meloman*;
  *Ogrodnik i Kierownik* with *Ogrodnik*). Their files are not byte-identical, so §5.5's duplicate
  guard does not catch them and they appear in the confusion pairs. Two of 306 is tolerable, but a
  future review pass should decide whether a group class may coexist with its members at all.
- **One BLAS thread per fit remains right, and by more than before.** Re-measured at 1,690 samples
  and 306 classes on the assumption the old §6.4 finding might have inverted: 1 thread is 3.1
  s/fold, 4 threads 7.1 s, 16 threads 20.8 s. Sixteen threads is 6.7x *slower*. Parallelise the
  probe across processes, never across BLAS threads.

### 7.4 Geographic result at scale (2026-09-05)

The §5.7 derived coordinates took the geographic arm from 23 classes to 294, and the §7.1 finding
survives the move intact — which it might not have, since at 23 it rested on six statues in one
themed installation.

- **Geographic pools lose to random pools at every measured pool size, for both backbones.** The
  arm is exact rather than sampled, so this is not seed noise. DINOv2 peaks at −0.89 points at a
  pool of five; CLIP at −2.12 at a pool of ten, roughly double throughout.
- **The penalty has a shape, and the shape is the mechanism.** It is largest at a 171–291 m radius
  and fades to −0.35 by 2.2 km. Sculptors install related pieces near each other, so a small radius
  is disproportionately made of a statue's own lookalikes; a wide one dilutes them with unrelated
  statues. Report the radius alongside the pool size — the radius is what carries the argument.
- **The overstatement is worst where a real tool would operate.** Random subsampling scatters
  clustered lookalikes across pools, so it flatters location narrowing most at exactly the small
  radii a phone-based tool would use. Never quote the random curve as a proxy for a location-aware
  system at small pool sizes.

## 8. Build order (strict, versioned)
- **v0.1**: data pipeline (Wikidata query → Commons pull → filtered manifest) + embedding extraction + basic k-NN retrieval + baseline top-1/top-5/MRR metrics.
- **v0.2**: candidate-pool-size ablation (the headline experiment) + confusion matrix + embedding visualization.
- **v0.3 (stretch)**: linear-probe/prototype baseline comparison + a small Gradio demo (upload a photo → top-5 candidates with similarity scores).

Agent should scaffold the full directory structure with stubs and docstrings before writing any real logic, in build order.

**Status: this build order is finished and released as `0.3.0` (2026-09-03).** Every stage above
has real behavior, plus two additions not in the original plan: the geographic ablation of section
5.2 and a static in-browser demo published from `docs/`. There is therefore no "next stage" to
pick up. Further work is a new research direction, and the limitations recorded in `RESULTS.md`
are the open questions:

- ~~**Open-set rejection**~~ — done on 2026-09-03 as `experiment open-set`; see §7.2 for the
  protocol and what it measured. What it leaves open is a *product* question rather than a
  research one: the published demo still shows a ranking unconditionally, and giving it a
  threshold means choosing an operating point on a visitor's behalf.
- **Real query photographs** — the reference set is Commons uploads, so the domain gap to a phone
  camera is unmeasured. Needs fieldwork in Wroclaw.
- **A larger pool** — 16 represented classes sit below the three-image threshold, and the full
  23-class pool is close to the accuracy ceiling, which is why the ablation curve is shallow and
  why extrapolating past it is unreliable.

Any of these is a scope change. Record the decision here before implementing it.

## 9. Repository structure

```
krasnal-id/
├── .github/workflows/ci.yml   # Python 3.12 quality gate
├── AGENTS.md
├── CHANGELOG.md
├── CONTRIBUTING.md
├── LICENSE                    # MIT applies to original source code only
├── README.md
├── pyproject.toml             # exact direct dependency pins and tool configuration
├── uv.lock                    # locked transitive dependency graph
├── data/
│   ├── category-review.json   # tracked human review of Commons mappings
│   ├── discovery/             # ignored Wikidata/Commons caches, staging, and audits
│   ├── images/                # ignored cached research copies
│   ├── embeddings/            # ignored embedding cache
│   └── manifest.json          # ignored generated manifest
├── src/krasnal_id/
│   ├── cli.py                 # unified Typer CLI
│   ├── config.py              # Hydra composition + Pydantic validation
│   ├── models.py              # manifest and attribution schema
│   ├── configs/               # packaged Hydra configuration groups
│   ├── data_pipeline/
│   │   ├── wikidata_query.py
│   │   ├── commons_fetch.py   # reviewed, cached Commons acquisition
│   │   └── build_manifest.py
│   ├── embeddings/
│   │   ├── backbone.py
│   │   └── cache.py
│   ├── retrieval/
│   │   └── knn.py
│   ├── experiments/
│   │   ├── baseline_accuracy.py
│   │   ├── pool_size_ablation.py
│   │   ├── confusion_analysis.py
│   │   └── open_set.py         # unknown-query rejection
│   └── viz/
│       └── embedding_plot.py
├── results/                   # ignored generated results
└── tests/                     # schemas, configs, CLI, and interface contracts
```

## 10. Engineering conventions

- Require Python 3.12 or newer. Use `uv` for environment management and commit `uv.lock`.
- Keep all importable code under the single `src/krasnal_id/` package. Use Hatchling as the build backend.
- Use Pydantic v2 for manifest and application contracts, packaged Hydra configuration groups for composition, and one Typer CLI (`krasnal-id`) for all pipeline stages.
- Pinned dependencies, structured (JSON) logging for experiment runs, cached embeddings — don't recompute what's already on disk.
- Keep PyTorch/Transformers, analysis libraries, and Gradio in separate optional dependency groups; core imports and CI must work without them.
- Pin backbone configurations to immutable upstream model revisions before inference.
- Every stored image record keeps license/attribution metadata; treat this as a schema requirement, not optional.
- Config-driven experiment parameters (pool sizes, seeds, thresholds) — no magic numbers buried in scripts.
- Enforce Ruff formatting/linting, strict mypy checks, pytest coverage of at least 85%, and the same checks in GitHub Actions on Python 3.12.
- Do not introduce DVC. Ignore downloaded data, manifests, embeddings, and generated results by default; selectively tracking final portfolio artifacts requires a later documented decision.
- Original source code uses the MIT License. It does not relicense downloaded Wikimedia assets.

## 11. Deliverables
- Clean GitHub repo with a README that tells the story: the problem, why naive classification is hard, the pool-size ablation as the key result, and what it implies for a real location-aware version.
- `results/` folder with the accuracy-vs-pool-size plot, confusion matrix, and embedding visualization as saved figures.
- Optional: a small live demo (Gradio) for anyone reviewing the portfolio to try it themselves.

## 12. Living documentation and handoffs
- Treat this file as the authoritative project brief and decision record. Update it in the same change whenever implementation work introduces or changes architecture, scope, schemas, conventions, build order, or other decisions that future contributors must follow.
- Keep `CHANGELOG.md` current throughout implementation. Every material addition, change, fix, removal, or documentation milestone must be recorded under an `Unreleased` section as part of the same change.
- Changelog entries must describe what is actually present and working in the repository, not planned work. This gives future AI agents an accurate handoff point.
- When cutting a version, move the relevant `Unreleased` entries into a dated version section and recreate an empty `Unreleased` section.
- Before completing an implementation task, verify whether both this file and `CHANGELOG.md` need corresponding updates. Documentation-only wording fixes do not require a new architectural decision, but should still be logged when material.

### 12.1 Current dataset-audit and implementation handoff (updated 2026-09-04)

**The manifest is 306 classes and 1,691 images**, rebuilt Commons-first on 2026-09-04 per §5.6.
482 category mappings carry a decision: 478 approved, 4 rejected. 1,958 images staged across 462
categories, of which 306 clear the three-image threshold. 23 classes come from Wikidata and carry
`P625` coordinates; the other 283 are Commons-only and carry none.

Three approvals were reversed after acquisition measured what they cost, and the reason
generalizes: a category byte-identical to another empties *both* sides through §5.5's cross-label
quarantine. `Papa Krasnal` duplicated `Q11823412` and wiped out a class that was in the published
results; the `Detektyw Magda i Rabusie` and `Doktor Basia i Krasnalątko` umbrellas emptied their
own members. Rejecting the three restored `Q11823412`. Where members still collide with each other
— three robbers photographed in one scene — the quarantine is correct and those classes stay
absent.

The paragraphs below record the 2026-08-20 audit of the *previous* 23-class dataset. They are kept
because the image-level decisions in `data/image-review.json` are still the live ones, all seven
still apply, and the reasoning behind each is not repeated anywhere else.

- The latest canonical staging output contains 173 images across 40 represented classes; 24
  classes meet the current three-image threshold. All staged files decode, match their recorded
  checksums and dimensions, stay within the 400–2,000-pixel bounds, and have complete attribution
  and recognized licenses.
- `data/discovery/fetched-images.json` is authoritative. There are currently 199 files under
  `data/images/`, so 26 files are quarantined or orphaned and must not enter a manifest through
  directory scanning. Do not delete them automatically.
- Image-level audit decisions are recorded in tracked `data/image-review.json`:
  - retain Papa Krasnal page `166491` as the canonical lower-page-ID duplicate winner;
  - exclude Papa Krasnal page `22381955` as its differently encoded duplicate;
  - exclude pages `22398133` (Papa Krasnal), `52890654` and `52890655` (Pralinka), and
    `89462414` (Binio) for low subject prominence;
  - exclude Capgeminiusz Programista page `134103757` as a heavily posterized,
    non-photographic reference.
- Tracked `data/category-review.json` stores durable display-name overrides for
  `Q136001294` -> `Abruzjusz`, `Q136001318` -> `Ossolinek`, and `Q136001344` -> `Demokracja`.
  Generated discovery files remain unchanged; downstream manifest construction must use the
  reviewed override when present.
- Cross-label protection is working: 14 Troszka/Adoratorek records and six Śpiewak
  Operowy/Tancerka Balerina records were quarantined as exact cross-label duplicates. Perceptual
  hashing found no remaining cross-class near-duplicate candidate in staging.
- Sixteen represented classes remain below threshold: Ołbiniusz, Adwokatka, and Rowerzysta have
  two images each; Abruzjusz, Szpitalnik, Troszka, Adoratorek, Komisia i Euruś, Tancerka
  Balerina, Śpiewak Operowy, Bankierek, Ditek, Glamour, Gryfosław, Solidariusz Walczący, and
  Unicefuś have one each.
- The deterministic image-level exclusion/override contract is recorded in tracked
  `data/image-review.json`, tied to the current `fetched-images.json` staging hash.
- `data build-manifest` is implemented. It consumes `fetched-images.json` plus both tracked
  review files, applies the three-image threshold, records discovery/staging/review provenance
  hashes, and writes the manifest atomically. It never scans the filesystem; the current
  audited artifacts produce 23 classes and 146 images.
- data build-split is implemented. It consumes only the validated manifest, creates one
  deterministic leave-one-out fold per admitted image, records the canonical manifest hash, and
  writes the ignored split artifact atomically.
- embeddings extract is implemented for the pinned DINOv2 and CLIP configurations. It validates
  manifest image checksums and dimensions, supports CPU/automatic CUDA selection and configured
  batching, reuses valid normalized .npy vectors, and keeps model loading lazy so CI remains
  offline. CI uses deterministic fake backbones; real weights are downloaded only on local ML runs.
- Retrieval, baseline metrics, the candidate-pool and geographic ablations, confusion analysis,
  visualization, the trained-classifier comparison, open-set rejection, and both demos are
  implemented and have been run end to end on this dataset. No module raises `NotImplementedError`. See section 8 for what
  is open beyond this point, and `CHANGELOG.md` for the per-stage record.

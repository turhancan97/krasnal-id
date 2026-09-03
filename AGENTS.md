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
- The published demo deliberately does **not** threshold. Section 6.3's browser build ships no
  rejection, and adding one would need the operating point chosen for a visitor rather than for a
  research artifact. That is a product decision, not a measured one, so it stays open.

3. **Error analysis**: confusion matrix for most-confused pairs — which dwarves get mixed up, and why (visually similar poses/props is the expected story).
4. **Embedding-space visualization**: t-SNE or UMAP plot of the reference set, colored by class, to make the "why confusion happens" argument visually.
5. **Open-set rejection**: can a top-1 similarity threshold answer "unknown" for a statue outside the reference set, and what does that cost on the statues inside it (see §7.2).

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

### 12.1 Current dataset-audit and implementation handoff (updated 2026-09-03)

The dataset facts below come from the 2026-08-20 audit and still hold: the manifest is
unchanged at 23 classes and 146 images.

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

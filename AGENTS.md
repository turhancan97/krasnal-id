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
- Store Pillow-verified research copies with a 512-pixel minimum short side and 1,600-pixel
  maximum long side. Preserve Commons page/revision provenance and reuse verified unchanged
  files.
- Retain the lowest Commons page ID for same-label byte duplicates. Exclude byte-identical
  content spanning different labels to prevent evaluation leakage. Never delete orphan files
  automatically.

## 6. Technical architecture

### 6.1 Embedding backbone
- DINOv2 and CLIP, used for zero-shot/near-zero-training feature extraction (no fine-tuning needed for the core experiment).

- **Design for a swappable backbone interface** (`get_embedding(image) -> vector`) rather than hard-coding a single library call. This keeps the option open to plug in a shared embedding-extraction module later without this repo depending on it existing or being finished first. Until then, load DINOv2/CLIP directly (e.g. via `transformers`).
- Cache embeddings to disk — never recompute per experiment run.

### 6.2 Retrieval
- Cosine similarity k-NN over cached reference embeddings is the primary method.
- Optional stretch baseline: a simple linear probe or per-class prototype (mean embedding) comparison, to see whether a trained classifier beats raw retrieval — useful discussion material for the writeup, not required for the headline result.

## 7. Experiments
1. **Baseline accuracy**: top-1, top-5, and mean reciprocal rank, DINOv2 vs. CLIP, full candidate pool.
2. **Headline experiment**: accuracy vs. candidate-pool size N (synthetic random subsampling, repeated with multiple seeds per N for error bars; real geo-based pools as a secondary comparison if coordinate coverage allows).
3. **Error analysis**: confusion matrix for most-confused pairs — which dwarves get mixed up, and why (visually similar poses/props is the expected story).
4. **Embedding-space visualization**: t-SNE or UMAP plot of the reference set, colored by class, to make the "why confusion happens" argument visually.

## 8. Build order (strict, versioned)
- **v0.1**: data pipeline (Wikidata query → Commons pull → filtered manifest) + embedding extraction + basic k-NN retrieval + baseline top-1/top-5/MRR metrics.
- **v0.2**: candidate-pool-size ablation (the headline experiment) + confusion matrix + embedding visualization.
- **v0.3 (stretch)**: linear-probe/prototype baseline comparison + a small Gradio demo (upload a photo → top-5 candidates with similarity scores).

Agent should scaffold the full directory structure with stubs and docstrings before writing any real logic, in build order.

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
│   │   └── confusion_analysis.py
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

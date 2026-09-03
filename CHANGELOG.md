# Changelog

This file records material changes that are actually present in the repository so that human contributors and future AI agents can quickly establish the current implementation state.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and
[Semantic Versioning](https://semver.org/spec/v2.0.0.html). Version numbers track the build
order recorded in `AGENTS.md` section 8, so `0.3.0` is the release that completes v0.1-v0.3.

## [Unreleased]

### Added

- Added open-set rejection as `krasnal-id experiment open-set`, answering the limitation
  `RESULTS.md` recorded: a query of a statue outside the reference set no longer has to be given a
  nearest neighbour. It scores two equal populations built from the manifest without sampling —
  the leave-one-out folds, and the same images queried against a gallery holding none of their own
  dwarf — and reports threshold-free AUROC alongside operating points whose thresholds are
  calibrated leave-one-class-out, so no query helps set the bar it must clear. A threshold fitted
  on all the data is reported too, labeled `in_sample`, as an optimistic reference. Per-dwarf rows
  record which statues slip through when removed and which statue covered for them.
- Added `OpenSetRejectionResult` and `DwarfRejection` to the experiment contracts, the
  `experiment=open_set` Hydra group with configurable acceptance targets, and the
  `thresholds.open_set_top_rejections` cap on reported per-dwarf rows.
- Added the rejection tradeoff figure as `krasnal-id visualize open-set`, drawing every saved
  backbone's curve on one axis with its calibrated operating points marked on it. The open-set
  artifact now carries that curve as `RejectionOperatingPoint` rows, swept over every observed
  score, so any threshold's cost is readable from the artifact rather than only the three
  configured targets. The curve is descriptive and the metrics stay the calibrated source of
  truth. Published to `docs/figures/open-set-rejection.png` under the section 6.4 policy.
- Added `RESULTS.md` section 6 with the measured outcome: DINOv2 separates present from absent
  statues at 0.969 AUROC and rejects 95.9% of unknown queries while still identifying 89.0% of
  known ones, costing about 3.4 points against its closed-set top-1 of 95.9%. CLIP reaches 0.898
  AUROC and has no useful operating point. Being reliably identifiable turns out not to imply
  being reliably rejectable: *Kowal* is never misidentified while present but is the worst dwarf
  to reject once removed.

## [0.3.0] - 2026-09-03

### Added

- Added the initial project brief in `AGENTS.md`, covering the research question, data and licensing requirements, proposed architecture, experiments, build order, engineering conventions, and deliverables.
- Added repository rules requiring `AGENTS.md` to remain synchronized with new project decisions and this changelog to remain synchronized with material implementation work.
- Added an installable Python 3.12+ `src/krasnal_id/` package with typed v0.1–v0.3 module stubs and a unified Typer CLI.
- Added validated Pydantic contracts for manifests, attribution metadata, configuration, retrieval results, experiment results, and embedding-cache identities.
- Added packaged Hydra groups for Wikimedia access, DINOv2 and CLIP, experiment variants, logging, paths, thresholds, and deterministic seeds.
- Added exact dependency pins and a `uv.lock`, with ML, analysis, and Gradio dependencies kept optional.
- Added Ruff, strict mypy, pytest, an 85% coverage gate, and Python 3.12 GitHub Actions CI.
- Added the MIT source-code license, setup and contribution documentation, and tracked guidance for ignored data and result directories.
- Added deterministic Wikidata dwarf discovery with typed normalization, explicit group filtering, audit records, atomic staging outputs, raw-response caching, refresh/limit controls, retry handling, and contact-bearing user-agent enforcement.
- Added synthetic fixture coverage for normalization, cache reuse and recovery, transport failures, CLI behavior, and an opt-in live Wikidata integration test.
- Added an offline, tracked category-review workflow that preserves decisions, supports
  corrected Commons categories, and resets changed mappings to pending.
- Added paginated Commons acquisition with raw-response caching, strict reusable-license and
  attribution checks, bounded Pillow-verified downloads, revision-aware reuse, deterministic
  caps, checksum deduplication, atomic staging outputs, retries, audits, and CLI exit semantics.
- Added comprehensive mocked Commons coverage and an opt-in live Commons integration test.
- Added a tracked image-level review contract with retain/exclude reasons, staging provenance,
  duplicate handling, and validation tests.
- Added audited manifest construction with strict artifact/hash validation, review application,
  threshold filtering, provenance fields, atomic output, CLI summaries, and offline integration
  coverage.

- Added dark-ground raster plates to `docs/brand/`: a 1280x640 `social-preview.png` carrying the
  tagline, a plain variant without it, and a 2x stacked lockup for slides.
- Added the project identity under `docs/brand/`: a mark and horizontal and stacked lockups, each
  in light and dark variants, with the wordmark converted to outlines so the files render without
  a font installed, plus the generator that produces them and the usage rules.
- Added a published findings summary linked from the README: an illustrated short read of the
  headline results, with `RESULTS.md` remaining the complete written record.
- Added `RESULTS.md`, the written research record: dataset construction, the four experiments,
  what the numbers mean, limitations, and a reproduction script.
- Added the accuracy-versus-pool-size figure as `krasnal-id visualize ablation`, drawing every
  saved ablation curve on one log-scaled axis with seed-spread error bars and the fitted slope.
- Added tracked `docs/figures/` holding the figures published in `RESULTS.md`, under the result
  publication policy now recorded in `AGENTS.md` section 6.4.
- Added the Gradio demonstration as `krasnal-id demo [--top-k N] [--port P] [--share]`: upload a
  photograph to see ranked candidate dwarves with similarity scores and their closest reference
  photographs, with the reference set loaded once per session.
- Published the demo at <https://turhancan97.github.io/krasnal-id/> from `docs/` on `main`, and
  linked it from the README in place of the retired findings artifact.
- Added a static browser demo under `docs/`, deployable to GitHub Pages: upload or photograph a
  statue and identify it entirely on-device, with attribution beside every match and a note when
  the top match belongs to a co-located installation.
- Added the demo's build at `docs/demo/build.mjs`, producing the reference vectors, thumbnails and
  attribution metadata. It runs the same library, model and dtype the browser uses, re-scores the
  leave-one-out protocol on the vectors it ships, and records self-test probes so the page can
  verify its own pipeline in a real browser.
- Added `docs/resize.mjs`, one Lanczos-3 resampler imported by both the browser and the build, so
  a query and a reference are preprocessed by identical arithmetic rather than by two platforms'
  differing built-in resamplers.
- Added the geographic ablation as `krasnal-id experiment geo-ablation`: candidate pools built
  from real Wikidata coordinates by nearest-neighbour proximity, scored against randomly sampled
  pools of matched size, with the median and maximum radius each pool spans.
- Added a `geo_ablation` Hydra experiment group and its validated `GeoAblationConfig`.
- Added the trained-classifier comparison as `krasnal-id experiment probe`: per-class prototype
  and per-fold linear-probe classifiers scored against a retrieval arm on the same folds, with
  Wilson intervals, mean reciprocal rank, and an explicit top-1 gain over retrieval.
- Added a `probe` Hydra experiment group and its validated `ProbeExperimentConfig`.
- Added single-image retrieval as `krasnal-id retrieve <image> [--top-k N]`, reporting ranked
  candidate dwarves with similarity scores and the reference image each matched, reusing a cached
  vector for identical file content and withholding every byte-identical copy of the query.
- Added confusion analysis reporting directed most-confused dwarf pairs with query counts,
  misidentification counts, and mean cosine margins, plus a `ConfusionPair` contract, summary
  margin metrics, and a wired `krasnal-id experiment confusion` command.
- Added UMAP and t-SNE embedding visualization with lazy analysis-dependency loading, a
  headless backend, deterministic seeding, per-class colors and markers beyond a twenty-color
  palette, decluttered centroid labels with leader lines, and a wired
  `krasnal-id visualize embeddings` command.
- Added the candidate-pool-size ablation: per-pool top-1 and MRR across seeds with the observed
  seed spread as error bars, a fitted accuracy-points-per-doubling slope, dataset-aware pool-size
  resolution, atomic result artifacts, and a wired `krasnal-id experiment pool-ablation` command.
- Added the full-pool retrieval baseline: dwarf-level and image-level top-k accuracy with 95%
  Wilson intervals, mean reciprocal rank, split/manifest hash verification, atomic result
  artifacts under `results/`, and a wired `krasnal-id experiment baseline` command.
- Added a shared embedding store that loads manifest-ordered cached vectors and holds the single
  definition of the embedding cache key used by both extraction and evaluation.
- Added cosine-similarity k-NN retrieval that rescales inputs to unit length, breaks equal
  similarities by ascending image ID for reproducible rankings, clamps similarities into the
  validated range, truncates to `top_k`, and returns a whole small pool rather than failing.
- Added deterministic leave-one-out split generation, validated split contracts, atomic split output,
  lazy DINOv2 and CLIP adapters, normalized resumable embedding caches, strict manifest-image
  validation, and offline fake-backbone extraction coverage.

### Changed

- Removed the CLI placeholder helper, which no command referenced once every stage was
  implemented.
- Set the default linear-probe regularization to `C=100`. At the conventional `C=1.0` the probe
  underfits L2-normalized embeddings badly, scoring 66% top-1 against 96%.
- Declared `threadpoolctl` in the `analysis` extra and hold BLAS to one thread while fitting
  per-fold classifiers, which took the probe sweep from over six minutes to twelve seconds.
- Extended strict `mypy` coverage to `tests`, and fixed the 35 errors that surfaced: fixtures now
  build `HttpUrl`, `datetime` and `Path` values explicitly instead of relying on Pydantic
  coercion, the CLIP output-shape test uses typed fakes rather than stacked ignores, retry tests
  patch the `time` module directly, and the CLI command tree is walked through a typed helper.
- Made the baseline's Wilson interval helpers public so confusion analysis reuses them.
- Installed the analysis extra in CI so the visualization code is exercised there.
- Recorded the dataset-scale decision in `AGENTS.md` §7.1: keep the >=3-image threshold and
  report degradation per doubling, with the measured evidence for rejecting a lower threshold
  and for treating the excluded duplicates as unrecoverable.
- Widened the default ablation pool sizes to `[2, 3, 5, 8, 10, 15, 20, 50, 100]`; sizes above the
  available class count are skipped with a warning.
- Indexed `EmbeddingMatrix` image IDs, which the ablation resolves hundreds of thousands of times.
- Moved the shared synthetic dataset builders used by evaluation tests into `tests/helpers.py`.
- Pinned CLIP to the immutable `c237dc49a33fc61debc9276459120b7eac67e7ef` safetensors-conversion
  revision so the loaded weights match the revision recorded in the embedding cache key.
- Added `torchvision` to the `ml` extra, which `transformers` requires for image processing.
- Changed the default Commons research-image bounds to a 400-pixel minimum short side and a
  2,000-pixel maximum long side.
- Re-reviewed all 41 tracked category mappings, corrected the stale `Q11823412` mapping to
  `Papa Krasnal`, rejected the visually heterogeneous Philharmonic umbrella category, and
  recorded evidence-based notes for every decision.
- Added durable display-name overrides for Abruzjusz, Ossolinek, and Demokracja to the tracked
  category review without modifying generated Wikidata discovery artifacts.
- Documented the 2026-08-20 dataset-audit handoff, including resolved image-level exclusions,
  below-threshold classes, display-name corrections, and the canonical staging boundary for
  manifest construction.

### Fixed

- Fixed CLIP feature extraction to read `pooler_output` from the vision-output object that
  `transformers` 5 returns from `get_image_features`, with regression coverage for both shapes.
- Made CLI help tests inspect generated command metadata instead of environment-dependent Rich-rendered text.
- Fixed Commons response parsing to accept scalar-valued extension metadata returned by the
  live API while continuing to require string-valued attribution and license fields.
- Fixed Commons downloads whose responsive derivative exceeds the requested bound by locally
  downscaling static images to the configured maximum while preserving their aspect ratio.

## Current state

This section is a rolling summary of what the repository does today, kept outside the version
sections above so that a new contributor or agent can read one place to orient themselves.

- Repository scaffolding, Wikidata discovery, and reviewed Commons image acquisition are
  complete, with automated checks passing.
- Audited manifest construction is implemented and produces 23 classes and 146 images from
  the current local artifacts.
- Deterministic evaluation split generation and resumable DINOv2/CLIP embedding extraction are
  implemented and have been run end to end on the local dataset, caching 146 normalized
  768-dimensional DINOv2 vectors and 146 normalized 512-dimensional CLIP vectors.
- Cosine k-NN retrieval and the full-pool baseline are implemented. On the current 23-class,
  146-image dataset the baseline reports dwarf-level top-1 of 95.9% for DINOv2 and 92.5% for
  CLIP, with top-5 at 99.3% and 98.6% and MRR at 0.9714 and 0.9506.
- The candidate-pool-size ablation is implemented and run. Top-1 falls from 98.9% at a pool of
  two to 95.9% at the full pool of 23 for DINOv2, and from 98.9% to 92.5% for CLIP, giving fitted
  slopes of -0.96 and -1.76 accuracy points per doubling.
- Confusion analysis and embedding visualization are implemented and run. DINOv2 misidentifies
  6 of 146 queries against CLIP's 11, and both backbones agree that the Puszczajacy Stateczki,
  Zbierajacy Wode, and Karmiacy Ptaki water-themed dwarves are the systematically confused
  cluster, which the projections show as a single tight neighborhood.
- The geographic ablation is implemented and run, answering the question `AGENTS.md` section 5.2
  left open. Real proximity pools are consistently harder than random pools of the same size
  (DINOv2 -1.6 points at a pool of five, CLIP -2.5), because six of the 23 dwarves stand within
  one metre of each other as one themed installation and are the same statues the confusion
  analysis flags.
- Single-image retrieval is implemented and reproduces the confusion finding interactively: a
  Puszczajacy Stateczki query ranks Zbierajacy Wode first under both backbones, with the water-
  themed cluster filling the top three candidates.
- The trained-classifier comparison is implemented and run. Neither trained method meaningfully
  beats retrieval: the linear probe gains 0.7 top-1 points over retrieval for both backbones
  (DINOv2 96.6% against 95.9%, CLIP 93.2% against 92.5%) while class prototypes lose 2.7 and 2.1
  points, and every difference sits inside the confidence intervals.
- The Gradio demonstration is implemented, completing the v0.1-v0.3 build order. Every
  scaffolded stage now has real behavior and no module raises `NotImplementedError`.
- A static browser demo is published at <https://turhancan97.github.io/krasnal-id/> from
  `docs/` on `main`, embedding a query with a quantised ONNX CLIP on the visitor's own device
  and scoring 91.8% top-1 and 98.6% top-5 on the vectors it ships.
- Everything above is released as `0.3.0`. The planned build order is finished, so further
  work is a new research direction rather than a remaining stage; the limitations listed in
  `RESULTS.md` name the open questions.
- Open-set rejection is implemented and run, as the first of those directions. DINOv2 supports a
  usable threshold and CLIP does not, and the demo deliberately still shows its ranking
  unconditionally, because choosing an operating point for a visitor is a product decision rather
  than a measured one. The remaining open questions are real query photographs and a larger pool,
  both of which need new data.

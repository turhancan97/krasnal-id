# Changelog

This file records material changes that are actually present in the repository so that human contributors and future AI agents can quickly establish the current implementation state.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/). The project does not yet use semantic version releases.

## [Unreleased]

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

### Changed

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

- Fixed Commons response parsing to accept scalar-valued extension metadata returned by the
  live API while continuing to require string-valued attribution and license fields.
- Fixed Commons downloads whose responsive derivative exceeds the requested bound by locally
  downscaling static images to the configured maximum while preserving their aspect ratio.

### Current state

- Repository scaffolding, Wikidata discovery, and reviewed Commons image acquisition are
  complete, with automated checks passing.
- Audited manifest construction is implemented and produces 23 classes and 146 images from
  the current local artifacts.
- Model inference, embedding persistence, retrieval algorithms, experiment computation, plotting,
  and the Gradio interface remain intentionally unimplemented.

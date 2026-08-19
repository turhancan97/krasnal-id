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

### Current state

- Repository scaffolding and the Wikidata `data query` stage are complete, with automated checks passing.
- Wikidata discovery, configuration composition, validation contracts, structured logging, CLI routing, deterministic caches, and discovery audits are implemented.
- Commons API acquisition, image downloading, manifest construction, model inference, embedding persistence, retrieval algorithms, experiment computation, plotting, and the Gradio interface remain intentionally unimplemented.

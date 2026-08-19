# Contributing

## Environment

Install Python 3.12 and `uv`, then run:

```bash
uv sync
```

Optional ML, analysis, and demo dependencies are deliberately excluded from the default
environment. Install only the extras needed for the task.

## Workflow

1. Follow the strict v0.1–v0.3 build order in `AGENTS.md`.
2. Add or update tests with every implemented behavior.
3. Run all checks listed in the README before submitting a change.
4. Update `AGENTS.md` when a project decision changes.
5. Record material completed work under `Unreleased` in `CHANGELOG.md`.

Generated images, embeddings, manifests, model weights, and experiment outputs must not be
committed. Final portfolio artifacts may be allow-listed later through a documented decision.

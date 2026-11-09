# Development

[Back to the README](../README.md) · [All guides](README.md)

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

See [CONTRIBUTING.md](../CONTRIBUTING.md) for the development workflow and
[AGENTS.md](../AGENTS.md) for the research scope, architecture, and build order.

## Brand assets

The Doppeldwarf-306 mark and lockups are generated, never edited by hand:

```bash
uv run --with matplotlib --with cairosvg python docs/brand/doppeldwarf.py
```

The wordmark is outlined from TeX Gyre Heros, vendored under `src/krasnal_id/viz/fonts/`, so the
assets rebuild from this repository alone. The earlier Krasnal-ID mark comes from
`docs/brand/generate.py`; `docs/brand/README.md` describes both.

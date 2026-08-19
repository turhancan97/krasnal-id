# Krasnal-ID

Krasnal-ID is a research prototype for fine-grained visual instance retrieval of
Wrocław's bronze dwarf statues. Its central experiment asks how identification accuracy
changes as the candidate pool grows, and how much a simulated location-based narrowing
would help.

## Project status

The repository currently contains the complete typed scaffold for versions 0.1–0.3. Data
downloads, model inference, retrieval algorithms, and experiments are intentionally not yet
implemented. Commands that represent those stages validate their configuration and exit with
a clear `not implemented` message.

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

## Development checks

```bash
uv lock --check
uv run ruff format --check .
uv run ruff check .
uv run mypy
uv run pytest
```

See [CONTRIBUTING.md](CONTRIBUTING.md) for the development workflow and
[AGENTS.md](AGENTS.md) for the research scope, architecture, and build order.

## Data and licensing

The source code is licensed under the MIT License. Images and metadata obtained from
Wikimedia Commons remain subject to their individual licenses. Every image manifest record
must retain its source URL, author, license, and license URL. The MIT License does not apply
to downloaded third-party images.

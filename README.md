# Krasnal-ID

Krasnal-ID is a research prototype for fine-grained visual instance retrieval of
Wrocław's bronze dwarf statues. Its central experiment asks how identification accuracy
changes as the candidate pool grows, and how much a simulated location-based narrowing
would help.

## Project status

The repository contains the complete typed scaffold for versions 0.1–0.3 plus implemented,
cached Wikidata discovery and reviewed Wikimedia Commons image acquisition. Manifest building,
model inference, retrieval algorithms, and experiments remain explicit placeholders.

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
export KRASNAL_ID_USER_AGENT='krasnal-id/0.0.0 (mailto:you@example.com)'
uv run krasnal-id data query
```

Use a deterministic pilot subset or bypass a valid cache:

```bash
uv run krasnal-id data query --limit 5
uv run krasnal-id data query --refresh
```

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
export KRASNAL_ID_USER_AGENT='krasnal-id/0.0.0 (mailto:you@example.com)'
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

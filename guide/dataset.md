# The dataset: use, publish, licensing

[Back to the README](../README.md) · [All guides](README.md)

Doppeldwarf-306 is published on Hugging Face as `turhancan97/wroclaw-dwarves` and on Kaggle as `turhancankargin/wroclaw-dwarves`.

## Use the published dataset

The dataset is published twice, so you do not need this repository, a Commons crawl or a GPU to
work with it. On the Hub as
[turhancan97/wroclaw-dwarves](https://huggingface.co/datasets/turhancan97/wroclaw-dwarves), in
parquet with the images embedded and one config per piece, so a comparison of the two backbones
costs about 8 MB rather than the full 676 MB. On
[Kaggle](https://www.kaggle.com/datasets/turhancankargin/wroclaw-dwarves) as images on disk
beside CSV tables, which is what its explorer previews and its notebooks expect —
[`docs/kaggle-starter.ipynb`](../docs/kaggle-starter.ipynb) reproduces the headline there in about
twenty lines. The Hub copy is the one this section shows:

```python
from datasets import load_dataset

images = load_dataset("turhancan97/wroclaw-dwarves", "default", split="reference")
vectors = load_dataset("turhancan97/wroclaw-dwarves", "embeddings_dinov2", split="reference")
folds = load_dataset("turhancan97/wroclaw-dwarves", "leave_one_out", split="test")
```

Six configs are published: `default` (the photographs and their attribution), `metadata` (the same
rows without the pixels), `classes` (one row per statue), `embeddings_dinov2` and
`embeddings_clip` (precomputed vectors), and `leave_one_out` (the 1,691 evaluation folds). Only
the config you ask for is downloaded, so scoring from vectors alone costs about 8 MB rather than
the full 676 MB of photographs. The split is called `reference`, not `train`, because nothing here
is trained.

Scoring `embeddings_dinov2` under `leave_one_out` reproduces the headline 93.1% top-1 from the
published files alone.

## Publish the dataset

Only needed to release a new version. The whole dataset — photographs, metadata, embeddings and
the leave-one-out folds — builds into a Hugging Face repository with one command:

```bash
uv run krasnal-id data license-templates   # once: the basis behind the public-domain files
uv run krasnal-id data export-hf           # builds data/export/huggingface
```

That writes six configs (`default`, `metadata`, `classes`, `embeddings_dinov2`,
`embeddings_clip`, `leave_one_out`), a dataset card, a licence inventory, credits grouped by
photographer, a machine-readable `credits.csv`, and a `provenance.json` digesting every emitted
file. It reads the manifest, the split and the embedding cache, and writes to none of them, so
building an export invalidates no published result.

Nothing is published until you say so:

```bash
uv run krasnal-id data export-hf --push                # creates a *private* repository
uv run krasnal-id data export-hf --push --public       # …or a public one
```

The token is never passed or logged — `huggingface_hub` resolves `HF_TOKEN` or your stored login
— and `--push` creates the repository private unless `--public` is given, because a mistyped
repository id becoming world-readable is not recoverable.

Every photograph keeps the licence it arrived with; the export never relicenses one. Each row
carries its photographer, licence, SPDX identifier, source URL and a ready-to-paste
`attribution_text`, plus a `modified` flag derived by comparing the stored bytes against the
Commons digest — 1,538 of the 1,691 files are downscaled adaptations and 153 are byte-identical
to their originals.

### Kaggle

Kaggle is a second writer rather than a second target, because file shape is a platform
convention: the Hub wants parquet whose image column is a `{bytes, path}` struct, and a Kaggle
user opening an image dataset expects a folder of images beside a table describing them.

```bash
uv run krasnal-id data export-kaggle                       # builds data/export/kaggle
uv run krasnal-id data export-kaggle --no-images           # metadata and vectors only, ~10 MB
uv run krasnal-id data export-kaggle --dataset-id you/wroclaw-dwarves
```

That writes `images/<dwarf_id>/`, `images.csv`, `classes.csv`, `folds.csv`, one
`embeddings_<backbone>.npy` per backbone, Kaggle's `dataset-metadata.json`, and the same
`LICENSES.md`, `ATTRIBUTION.md`, `credits.csv` and `provenance.json` the Hugging Face export
publishes — deliberately the same generated artifacts, so the two platforms cannot disagree about
a photographer.

Three things it refuses rather than discovers late. Kaggle rejects an over-long title or slug
instead of trimming it, so the limits are checked before 676 MB is copied. `folds.csv` stores
only the query, since every fold's reference set is every other image — a rule the export
verifies fold by fold and refuses if it stops holding. And a `.npy` has no keys, so the vector
order is compared against `images.csv` before writing, because a mismatch would make every
downstream number wrong without failing.

The licence field says `other` — Kaggle's "Other (specified in description)" — because Kaggle
takes one licence and this corpus has ten. The generated description carries the family
breakdown, the modified/unmodified split, the freedom-of-panorama disclosure and the removal
path, which is what makes that field honest rather than vague.

It also writes `data/export/kaggle-cover.jpg`, a 1200x600 grid of 32 statues sampled by even
stride — *beside* the upload directory, not inside it, because Kaggle's cover image is set in
the web UI and a file in the folder would be published as data. `docs/kaggle-starter.ipynb`
reproduces the headline from the published files in about twenty lines, and is worth publishing
as a notebook on the dataset.

Two things Kaggle's renderer and validator will not tell you. The description is rendered as
markdown **with HTML parsing**, so an angle bracket anywhere in it — `embeddings_<backbone>.npy`,
say, even inside backticks — is read as an unknown opening tag and silently swallows everything
after it. And `resources` may only name files that exist in the folder: a directory entry fails
the upload with "does not exist", and so would `images.zip`, which `--dir-mode` creates during
the upload rather than in the folder. Tests pin both.

Publishing is left to you, because a Kaggle dataset slug cannot be renamed once created:

```bash
kaggle datasets create -p data/export/kaggle --dir-mode zip
kaggle datasets version -p data/export/kaggle --dir-mode zip -m "manifest <hash>"
```

Scoring leave-one-out from `embeddings_dinov2.npy` and `images.csv` alone — no project import —
reproduces 93.1% top-1 and 95.7% top-5, and CLIP's 82.9%, which are the published numbers exactly.

## Data and licensing

The source code is licensed under the MIT License. **It does not apply to the photographs.**

The 1,691 reference images come from Wikimedia Commons and each keeps its own licence: 1,577
CC BY-SA (2.0 through 4.0), 55 CC BY, 52 public domain and 7 CC0. Every manifest record retains
its source URL, author, licence and licence URL as a schema requirement, and the published dataset
carries all four per row so the attribution obligation travels with the data. Nothing in this
project relicenses a photograph or applies one licence across the collection.

1,538 of the stored files are downscaled to at most 2,000 px and re-encoded, which makes them
adaptations; the other 153 are byte-identical to the Commons original. The `modified` column says
which, derived from the bytes rather than asserted.

The statues themselves are contemporary sculptures under copyright. Commons hosts photographs of
them under Poland's freedom-of-panorama provision; the Creative Commons licences here cover the
**photographs**, granted by the photographers, and grant no rights in the sculptures depicted. If
you are a rights-holder and want an image removed, open an issue: exclusions are recorded in
`data/image-review.json`, and a rebuild propagates the removal through the manifest, the folds,
the embeddings and the published dataset.

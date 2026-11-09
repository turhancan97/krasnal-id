<picture>
  <source media="(prefers-color-scheme: dark)" srcset="docs/brand/doppeldwarf-lockup-horizontal-dark.svg">
  <img src="docs/brand/doppeldwarf-lockup-horizontal-light.svg" alt="Doppeldwarf-306: near-identical instance retrieval" width="380">
</picture>

[![DOI](https://zenodo.org/badge/1339843718.svg)](https://doi.org/10.5281/zenodo.22548023)
[![CI](https://github.com/turhancan97/krasnal-id/actions/workflows/ci.yml/badge.svg)](https://github.com/turhancan97/krasnal-id/actions/workflows/ci.yml)
[![Code licence: MIT](https://img.shields.io/badge/code-MIT-blue.svg)](LICENSE)
[![Dataset on Hugging Face](https://img.shields.io/badge/%F0%9F%A4%97%20dataset-wroclaw--dwarves-yellow)](https://huggingface.co/datasets/turhancan97/wroclaw-dwarves)

**Which dwarf is this?** Wrocław has hundreds of small bronze dwarf statues, many made in themed
series and alike in style, scale and setting. **Krasnal-ID** identifies one from a photograph, and
measures how hard that gets as the number of candidate statues grows. Its corpus,
**Doppeldwarf-306**, holds 1,691 openly licensed photographs of 306 statues, each carrying its
source, photographer and licence, with 294 statues placed on the map.

[**Try the identifier**](https://turhancan97.github.io/krasnal-id/) (runs in your browser; nothing
is uploaded) · [**Dataset on Hugging Face**](https://huggingface.co/datasets/turhancan97/wroclaw-dwarves)
· [**Dataset on Kaggle**](https://www.kaggle.com/datasets/turhancankargin/wroclaw-dwarves) ·
[**Full results**](RESULTS.md)

## Headline results

- **Frozen features identify these statues well, with no training.** Nearest-neighbour search
  names the right statue first for 93.1% of queries with DINOv2 and up to 94.7% across thirteen
  backbones. Models trained for invariant embeddings lead; MAE, BEiT and I-JEPA reach 47–56% under
  the same fixed pooling rule.
- **Accuracy decays slowly as candidates are added**: DINOv2 loses about 0.8 points each time the
  candidate pool doubles, CLIP about 2.1, and the gap between them widens with the pool.
- **Location helps more as a weight than as a filter.** Restricting candidates to the nearest
  statues helps less than an equal-sized random restriction, because lookalikes stand close
  together. Weighting candidates by distance instead lifts DINOv2 from 93.4% to **98.7%** for a
  simulated user 25 m away.
- **A new photographer costs accuracy, mostly through scarcity.** With the query's own photographer
  withheld DINOv2 scores 81.8%; with one reference of the right statue left it scores 60%, with six
  or more 92%.
- **Learned verification helps where SIFT does not.** Re-ranking the top ten with DISK+LightGlue or
  LoFTR, weights chosen on held-out statues, lifts that 81.8% to 84.4–85.5%; SIFT adds nothing
  measurable, and verifying only the least certain 30% of queries keeps the whole gain.
- **Rejection needs a margin.** Comparing the two best statues, rather than thresholding the top
  similarity, cuts DINOv3's false acceptance of unknown statues from 21.6% to 7.7%.

![One query photograph and the five dwarves each backbone ranks highest, for four queries: one both backbones identify, one only DINOv2 identifies, one only CLIP identifies, and one neither identifies. Correct statues are outlined in green and wrong ones in red.](docs/figures/retrieval-examples.jpg)

*Four queries and the five statues each backbone ranks highest; green is correct. Rows are chosen
by rule, not by hand. The [experiments guide](guide/experiments.md) has the full findings.*

## Quick start

Python 3.12 or newer and [`uv`](https://docs.astral.sh/uv/) are required.

```bash
uv sync
uv run krasnal-id --help
```

- **Reproduce the headline without building anything**: the published dataset carries every
  backbone's embeddings and the evaluation folds. See
  [Use the published dataset](guide/dataset.md#use-the-published-dataset).
- **Rebuild everything from Wikimedia Commons**: follow the [pipeline guide](guide/pipeline.md).

## Documentation

| | |
|---|---|
| [Pipeline](guide/pipeline.md) | Discovery, review, acquisition, the split, embeddings and the baseline |
| [Experiments](guide/experiments.md) | Every experiment's command and what it found |
| [Demonstrations](guide/demo.md) | The local demo and the in-browser identifier |
| [Dataset](guide/dataset.md) | Using and publishing the dataset, and its licensing |
| [Development](guide/development.md) | Quality checks, contributing and the brand assets |
| [RESULTS.md](RESULTS.md) | The complete written record of every experiment |
| [CHANGELOG.md](CHANGELOG.md) | What each release added |

## Project status

Current version **0.20.0**. Every stage runs, from discovery to the in-browser demo, and every
release since `0.3.0` has closed one research question; [CHANGELOG.md](CHANGELOG.md) records each.
One question stays open: **how much a casual street photograph costs**. Every query here is a
Commons upload, so phone-shot references give only a lower bound (about 5 top-1 points for DINOv2),
and the measuring path for real field photographs is built and waits on a day in Wrocław
([`data/field-guide.md`](data/field-guide.md)).

## Citing this work

Archived on Zenodo with a DOI that always resolves to the newest release:

```bibtex
@software{kargin2026krasnalid,
  author    = {Kargın, Turhan Can},
  title     = {Krasnal-ID: fine-grained visual instance retrieval of
               Wrocław's dwarf statues},
  year      = {2026},
  publisher = {Zenodo},
  doi       = {10.5281/zenodo.22548023},
  url       = {https://doi.org/10.5281/zenodo.22548023}
}
```

That is the *concept* DOI, so citations accumulate across versions. Each release also has its own
version DOI on the Zenodo record, for pinning exactly what you ran. If you used the photographs,
cite the dataset too and credit the photographers as its `credits.csv` sets out.

## Data and licensing

The source code is MIT-licensed. **That licence does not cover the photographs.** Each of the
1,691 images keeps its own Wikimedia Commons licence (CC BY-SA, CC BY, CC0 or public domain), with
its source, author and licence recorded per image. The statues are contemporary sculptures; the
licences cover the photographs, not the sculptures depicted. To request a removal, open an issue.
The [dataset guide](guide/dataset.md#data-and-licensing) has the full terms.

"""The dataset card, the licence inventory and the credit ledger.

Generated, never hand-maintained. This repository's discipline is that every
number traces to a committed command and an artifact, and attribution is the last
place to abandon that: a hand-edited credit list drifts from the data the first
time the manifest is rebuilt, and the drift is invisible until a photographer
notices they are miscredited.
"""

import csv
import io
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass

from krasnal_id.export.rows import MODIFICATION_NOTE, ImageRow

# The dotless i is LATIN SMALL LETTER DOTLESS I: the correct Turkish spelling of
# the author's name, not a homoglyph slip, so the ambiguity check is suppressed.
DATASET_AUTHOR = "Turhan Can Kargın"  # noqa: RUF001
# BibTeX wants "Family, Given" so the surname sorts correctly.
DATASET_AUTHOR_BIBTEX = "Kargın, Turhan Can"  # noqa: RUF001
DATASET_AUTHOR_ORCID = "https://orcid.org/0000-0002-6751-4773"
# The concept DOI, which resolves to the newest release, so citations of the work
# accumulate on one identifier instead of splitting across versions.
PROJECT_DOI = "10.5281/zenodo.22548023"

CREDIT_COLUMNS = (
    "image_id",
    "dwarf_id",
    "commons_file",
    "source_url",
    "author",
    "license",
    "license_spdx",
    "license_url",
    "license_template",
    "modified",
    "modification",
    "commons_sha1",
    "sha256",
    "attribution_text",
)


@dataclass(frozen=True, slots=True)
class CardFacts:
    """The measured facts the card states, so no number in it is written by hand."""

    images: int
    classes: int
    modified: int
    unmodified: int
    photographers: int
    top_photographers: tuple[tuple[str, int], ...]
    licenses: tuple[tuple[str, str | None, str, int], ...]
    placed: int
    unplaced: int
    derived_positions: int
    manifest_sha256: str
    backbones: tuple[tuple[str, str, str, int], ...]


def measure(
    rows: Sequence[ImageRow],
    class_count: int,
    manifest_sha256: str,
    backbones: Sequence[tuple[str, str, str, int]],
) -> CardFacts:
    """Derive every figure the card quotes from the rows themselves."""
    authors = Counter(row.record.author for row in rows)
    licenses = Counter((row.record.license, row.license_spdx, row.license_url) for row in rows)
    placed = {row.dwarf.dwarf_id for row in rows if row.dwarf.coordinates}
    derived = {
        row.dwarf.dwarf_id
        for row in rows
        if row.dwarf.coordinates and str(row.dwarf.coordinate_source) == "commons_camera"
    }
    return CardFacts(
        images=len(rows),
        classes=class_count,
        modified=sum(1 for row in rows if row.modified),
        unmodified=sum(1 for row in rows if not row.modified),
        photographers=len(authors),
        top_photographers=tuple(authors.most_common(2)),
        licenses=tuple(
            (name, spdx, url, count)
            for (name, spdx, url), count in sorted(licenses.items(), key=lambda i: -i[1])
        ),
        placed=len(placed),
        unplaced=class_count - len(placed),
        derived_positions=len(derived),
        manifest_sha256=manifest_sha256,
        backbones=tuple(backbones),
    )


def render_credits_csv(rows: Sequence[ImageRow]) -> str:
    """Render the machine-readable credit ledger.

    Separate from the parquet on purpose: complying with an attribution licence
    should not require downloading 676 MB of photographs or owning a parquet
    reader. A photographer checking whether they were credited correctly can read
    this file.
    """
    buffer = io.StringIO()
    writer = csv.writer(buffer, lineterminator="\n")
    writer.writerow(CREDIT_COLUMNS)
    for row in rows:
        writer.writerow(
            [
                row.record.image_id,
                row.record.dwarf_id,
                row.commons_file,
                str(row.record.source_url),
                row.record.author,
                row.record.license,
                row.license_spdx or "",
                row.license_url,
                row.license_template or "",
                "true" if row.modified else "false",
                row.modification,
                row.record.commons_sha1 or "",
                row.stored_sha256,
                row.attribution_text,
            ]
        )
    return buffer.getvalue()


def render_licenses(facts: CardFacts) -> str:
    """Render the licence inventory and the modification statement."""
    lines = [
        "# Licences",
        "",
        f"This dataset is a **collection of {facts.images:,} separately licensed photographs**, ",
        "not a single relicensed work. Each row keeps the licence it arrived with, recorded in ",
        "its own `license`, `license_spdx` and `license_url` columns. Nothing here relicenses ",
        "any photograph, and the repository-level licence tag is metadata rather than a grant.",
        "",
        "## Inventory",
        "",
        "| Licence | SPDX | Files | Deed |",
        "|---|---|---:|---|",
    ]
    for name, spdx, url, count in facts.licenses:
        lines.append(f"| {name} | {spdx or '—'} | {count} | <{url}> |")
    lines += [
        "",
        "## Modification",
        "",
        f"**{facts.modified:,} of the {facts.images:,} files are modified**: they were "
        f"{MODIFICATION_NOTE}. ",
        f"The remaining **{facts.unmodified:,} are byte-identical to the Commons original**. ",
        "The `modified` column says which, and it is derived by comparing the stored bytes ",
        "against the Commons digest in `commons_sha1` rather than asserted — so you can check ",
        "it rather than trust it.",
        "",
        "## Public-domain files",
        "",
        "Files recorded as *Public domain* carry the Creative Commons Public Domain Mark, ",
        "which is a label rather than a licence: it asserts freedom from known copyright ",
        "without saying on what basis. The `license_template` column carries the Commons ",
        "template naming that basis where it could be retrieved.",
        "",
        "## What you must do",
        "",
        "- **Attribute each photograph to its own photographer.** The `attribution_text` ",
        "  column is a ready-to-paste credit line; `credits.csv` carries all of them.",
        "- **Keep the licence with the row.** If you redistribute a subset, filter ",
        "  `credits.csv` to it and ship that too.",
        "- **Say the files are modified** where `modified` is true. They are adaptations.",
        "- **ShareAlike binds your adaptations**, not this collection. If you crop, augment ",
        "  or composite a CC BY-SA photograph, release the result under that photograph's own ",
        "  licence version. Because this collection spans CC BY-SA 2.0 to 4.0, an arbitrary ",
        "  slice inherits a *mix* of obligations — filter by `license_spdx` first if you need ",
        "  a single-licence derivative.",
        "- **Add no further restrictions.** No terms are imposed here and none may be added.",
        "",
    ]
    return "\n".join(lines) + "\n"


def render_attribution(rows: Sequence[ImageRow], facts: CardFacts) -> str:
    """Render the credits grouped by photographer.

    Grouping is the point: it is the form in which the concentration of the
    corpus is visible rather than merely stated.
    """
    by_author: dict[str, list[ImageRow]] = {}
    for row in rows:
        by_author.setdefault(row.record.author, []).append(row)

    lines = [
        "# Attribution",
        "",
        f"{facts.images:,} photographs by {facts.photographers:,} photographers, who are the ",
        "copyright holders of the images and the reason this dataset can exist. Sections are ",
        "ordered by contribution.",
        "",
    ]
    for author, authored in sorted(by_author.items(), key=lambda item: (-len(item[1]), item[0])):
        share = len(authored) / facts.images * 100
        licenses = ", ".join(sorted({row.record.license for row in authored}))
        lines += [
            f"## {author}",
            "",
            f"{len(authored)} photographs ({share:.1f}%) — {licenses}",
            "",
        ]
        for row in sorted(authored, key=lambda item: item.commons_file):
            lines.append(f"- [{row.commons_file}]({row.record.source_url})")
        lines.append("")
    return "\n".join(lines) + "\n"


def license_link_uri(repo_id: str, license_link: str) -> str:
    """Return an absolute URI for the licence inventory.

    The Hub's metadata validator requires `license_link` to be an https URI, so a
    repository-relative filename is resolved against the repository it is being
    published to. Pointing at the in-repo inventory rather than at
    creativecommons.org is deliberate: the collection spans ten licences, and a
    link to any one of them would misdescribe the other nine.
    """
    if license_link.startswith(("http://", "https://")):
        return license_link
    return f"https://huggingface.co/datasets/{repo_id}/blob/main/{license_link.lstrip('/')}"


def _front_matter(
    facts: CardFacts,
    configs: Sequence[tuple[str, str, str]],
    features: dict[str, list[dict[str, object]]],
    license_name: str,
    license_link: str,
    repo_id: str,
) -> str:
    """Render the YAML the Hub reads.

    `license: other` rather than one SPDX identifier: the Hub's key is
    repository-level, and there is no honest single value across these licences.
    Tagging the whole set CC BY-SA 4.0 would be the convenient lie — it asserts
    ShareAlike over the CC BY and public-domain files, which grant no such thing.
    The mix is carried per row instead, and `license_link` points at the
    inventory rather than at one licence among many.
    """
    import yaml

    payload: dict[str, object] = {
        "pretty_name": "Wrocław Dwarves — Fine-Grained Instance Retrieval",
        "license": "other",
        "license_name": license_name,
        "license_link": license_link_uri(repo_id, license_link),
        "task_categories": ["image-feature-extraction", "image-classification"],
        "tags": [
            "instance-retrieval",
            "image-retrieval",
            "fine-grained",
            "benchmark",
            "leave-one-out",
            "cultural-heritage",
            "public-art",
            "sculpture",
            "wikimedia-commons",
            "dinov2",
            "clip",
            "wroclaw",
            "poland",
        ],
        "size_categories": ["1K<n<10K"],
        "annotations_creators": ["found", "expert-generated"],
        "source_datasets": ["original"],
        "configs": [
            {
                "config_name": name,
                "data_files": [{"split": split, "path": path}],
                **({"default": True} if name == "default" else {}),
            }
            for name, split, path in configs
        ],
        "dataset_info": [
            {"config_name": name, "features": features[name]}
            for name, _, _ in configs
            if name in features
        ],
    }
    rendered = yaml.safe_dump(payload, sort_keys=False, allow_unicode=True, width=100)
    return str(rendered)


def render_card(
    facts: CardFacts,
    configs: Sequence[tuple[str, str, str]],
    features: dict[str, list[dict[str, object]]],
    *,
    repo_id: str,
    license_name: str,
    license_link: str,
) -> str:
    """Render the dataset card."""
    first, second = (*facts.top_photographers, ("", 0), ("", 0))[:2]
    concentration = (first[1] + second[1]) / facts.images * 100 if facts.images else 0.0
    descriptions = {
        "default": ("Photographs, metadata and rights columns", facts.images),
        "metadata": ("The same rows without the pixels", facts.images),
        "classes": ("One row per statue, with coordinates", facts.classes),
        "leave_one_out": ("The evaluation folds", facts.images),
    }
    config_table = "\n".join(
        "| `{name}` | `{split}` | {rows} | {what} |".format(
            name=name,
            split=split,
            **dict(
                zip(
                    ("what", "rows"),
                    descriptions.get(
                        name,
                        (f"Precomputed {name.removeprefix('embeddings_')} vectors", facts.images),
                    ),
                    strict=True,
                )
            ),
        )
        for name, split, _ in configs
    )
    license_href = license_link_uri(repo_id, license_link)
    vector_configs = [name for name, _, _ in configs if name.startswith("embeddings_")]
    vector_example = (
        f'\nvectors = load_dataset("{repo_id}", "{vector_configs[0]}", split="reference")'
        if vector_configs
        else ""
    )
    backbones = "\n".join(
        f"| {name} | `{model_id}` | `{revision[:12]}` | {dimensions} |"
        for name, model_id, revision, dimensions in facts.backbones
    )

    return f"""---
{_front_matter(facts, configs, features, license_name, license_link, repo_id)}---

# Wrocław Dwarves — Fine-Grained Instance Retrieval

Wrocław is scattered with several hundred small bronze dwarf statues — *krasnale* — installed
across the city since 2001. They are individually sculpted, but they share a visual vocabulary:
the same scale, the same material, the same crouching poses and hand-held props. Telling one
from another is therefore a **fine-grained instance recognition** problem rather than a
classification one. Every statue here belongs to the same semantic category; what distinguishes
them is a hat, a tool, a pose.

This is a **retrieval benchmark**: {facts.images:,} reference photographs of {facts.classes:,}
distinct statues from Wikimedia Commons, with per-image attribution and licence metadata,
per-statue coordinates where they could be established, precomputed embeddings, and a
hash-pinned leave-one-out protocol. It is deliberately **not a training set**. The median class
has four photographs and the largest has thirty-one — enough to retrieve against, not enough to
train on. The point of the benchmark is that with a good enough general-purpose embedding you do
not need to train at all.

The question it was built to answer is how identification accuracy decays as the candidate pool
grows, and whether narrowing that pool by location — as a phone app would — actually helps. It
does not: proximity-based pools are consistently *harder* than random pools of the same size,
because sculptors install related pieces near each other, so a small radius selects for a
statue's own lookalikes. The full method and every result artifact are in the
[source repository](https://github.com/turhancan97/krasnal-id).

## Task and protocol

Given a query photograph of one statue, rank the {facts.classes:,} known statues by how likely
each is to be the one depicted. Candidates are ranked as *distinct statues*, each represented by
its best-matching reference photograph — the list an identification tool would actually show.

**Leave-one-out.** Each of the {facts.images:,} photographs is used once as a query and scored
against the other {facts.images - 1:,}, with the query withheld from its own class. The
{facts.images:,} folds are enumerated explicitly in the `leave_one_out` config so results are
comparable across methods rather than dependent on a re-derived split. Intervals are 95% Wilson
score intervals over the queries.

## Baselines

Off-the-shelf embeddings identify these statues well with no fine-tuning:

| | DINOv2 (`facebook/dinov2-base`) | CLIP (`openai/clip-vit-base-patch32`) |
|---|---|---|
| top-1 | **93.1%** [91.8, 94.3] | **82.9%** [81.0, 84.6] |
| top-5 | 95.7% [94.6, 96.6] | 90.7% [89.2, 92.0] |
| MRR | 0.943 | 0.866 |

Both backbones are frozen and pinned to immutable revisions; images are embedded and ranked by
cosine similarity. No fine-tuning, no probe, no metric learning. The ten-point gap is the
expected ordering — DINOv2's self-supervised objective preserves instance-level detail, while
CLIP's language alignment pulls toward semantic categories, and every statue here *is* the same
semantic category.

Adding geometric verification on top — RANSAC inlier counts between the query and its top-10
candidates, blended into the similarity — lifts DINOv2 to **94.0%** and CLIP to **86.3%**, gaining
0.9 and 3.4 points. Geometry is worth four times as much to the weaker backbone, for the same
reason the limitations below give.

Four secondary results are worth knowing before using this data. A per-fold linear probe is
worth nothing to DINOv2 and gains CLIP about three points — where retrieval is already strong,
training is not the right tool. Proximity-based candidate pools lose to random pools of the same
size at every radius measured. Thresholding similarity for open-set rejection does **not**
survive the larger pool: it worked at 23 classes and does not at {facts.classes:,}, which retracts
a positive result an earlier version of this project published. And identifying a statue from
*somebody else's* photograph is much harder than the headline suggests — see the limitations.

## Configurations

| Config | Split | Rows | What it is |
|---|---|---:|---|
{config_table}

```python
from datasets import load_dataset

images = load_dataset("{repo_id}", "default", split="reference"){vector_example}
folds = load_dataset("{repo_id}", "leave_one_out", split="test")
```

The split is called `reference`, not `train`, because nothing here is trained. Every config is
sorted by the same key, so row *i* of `default` and row *i* of either embeddings config are the
same photograph.

Only the config you ask for is downloaded, so working from vectors alone costs a few megabytes
rather than the full corpus.

| Backbone | Model | Revision | Dimensions |
|---|---|---|---:|
{backbones}

## Licensing and attribution

Every photograph keeps the licence it arrived with. See
**[{license_link}]({license_href})** for
the full inventory and **`ATTRIBUTION.md`** for the credits; `credits.csv` is the same ledger in
machine-readable form, downloadable without a parquet reader.

The obligations travel in the data. Each row carries `author`, `license`, `license_spdx`,
`license_url`, `source_url`, `commons_file`, `modified` and a ready-to-paste
`attribution_text`. **{facts.modified:,} of the {facts.images:,} files are modified** —
{MODIFICATION_NOTE} — and **{facts.unmodified:,} are byte-identical to the Commons original**;
the `modified` column is derived by comparing the stored bytes against `commons_sha1`, so it can
be checked rather than trusted.

Filtering by licence is one line, if you need a permissive subset:

```python
permissive = images.filter(lambda row: row["license_spdx"] in {{"CC0-1.0", "CC-BY-4.0"}})
```

No further terms are imposed here, and none may be added by anyone redistributing it.

## Personal and sensitive information

The dataset documents public sculpture, not people, and no photograph was selected for the
people in it. But these are photographs taken in city streets, so **passers-by may appear
incidentally** and may be identifiable. No face detection or blurring was applied, and the
frequency was not measured. Every file was already publicly hosted on Wikimedia Commons before
inclusion here, and each carries the Commons page it came from.

The EXIF of the {facts.modified:,} re-encoded files was not preserved, so the photographers'
camera serial numbers and GPS traces are absent from them; the {facts.unmodified:,} untouched
files retain whatever EXIF Commons holds. Per-photograph coordinates in the metadata are
*camera positions* recorded by Commons, already public there, and are used to locate statues
rather than photographers.

If you appear in one of these photographs and want it removed, the removal path below applies
to you as much as to a rights-holder, and no legal argument is required.

## Depicted works and freedom of panorama

The statues are contemporary sculptures by living artists, protected by copyright independently
of the photographs. Wikimedia Commons hosts photographs of them under Poland's freedom-of-panorama
provision (Art. 33(1) of the Polish Copyright Act), which permits publishing images of works
permanently displayed in public places. Poland's provision is broad; the United States has no
equivalent exemption for sculpture, and this platform is US-hosted.

**The Creative Commons licences recorded per row cover the photographs, granted by the
photographers. They do not, and cannot, grant any rights in the sculptures depicted.** A reuse
that would require the sculptor's permission in your jurisdiction still requires it here. This is
a factual disclosure rather than legal advice, and it is recorded because it is the one rights
question the per-file licences do not answer.

**Removal requests.** If you are a sculptor, photographer, rights-holder or their representative
and want a photograph removed, open a discussion on this repository. Requests are honoured
without requiring a legal argument. Removal is implemented upstream rather than patched here: the
file is added to the source project's tracked `data/image-review.json` exclusions, the manifest,
folds, embeddings and credit files are regenerated, and a new revision is pushed.

## Limitations

- **The reference photographs are not phone photographs.** They are Commons uploads: mostly good
  light, considered framing, taken by someone who meant to document the statue. Using the
  photographs that were themselves shot on phones as a proxy, DINOv2 loses 5.3 top-1 points and
  CLIP 15.6 — and that is a *lower* bound, since those are still Commons uploads. Treat 93.1% as
  a ceiling a phone-camera application will not reach.
- **The corpus is concentrated in few hands, and it costs a measurable amount.**
  {facts.photographers:,} photographers contributed, but {first[0]} took {first[1]:,} photographs
  and {second[0]} took {second[1]:,} — together {concentration:.1f}% of the dataset.
  Near-duplicates from a single visit were never removed, so a method can score partly by
  recognising a camera and a processing style rather than a statue. Withholding each query's own
  photographer and comparing against a size-matched random control puts the part attributable to
  the photographer at **2.6 top-1 points for DINOv2 and 13.2 for CLIP** — so the headline is
  inflated, modestly for DINOv2 and substantially for CLIP. Two further caveats follow from the
  same concentration: **125 of the 306 classes have a single photographer**, so a third of the
  dataset cannot be evaluated cross-photographer at all, and cross-photographer accuracy is far
  below the headline — 81.8% for DINOv2 and **54.1% for CLIP**. If your use case means matching a
  photograph against references somebody else took, those are the numbers to plan around.
- **Most coordinates are derived, not stated.** {facts.derived_positions:,} of the {facts.placed:,}
  placed statues are located from where photographers stood rather than from a Wikidata `P625`
  statement; validated against the classes that have both, the median error is 9 m.
  {facts.unplaced:,} statues are unplaced. `coordinate_source` says which is which. Do not present
  these as official locations.
- **Two classes overlap with a group class.** *Grajek i Meloman* is a class, and so are *Grajek*
  and *Meloman*; likewise *Ogrodnik i Kierownik* and *Ogrodnik*. Their photographs are not
  byte-identical so the duplicate guard does not catch them, and they duly appear among the
  confusions.
- **Class sizes are uneven** — three to thirty-one photographs, median four. Any metric averaged
  over queries rather than classes is weighted by photographic popularity.
- **Identifiers are dataset-local.** Statues without a Wikidata item are keyed by a slug of their
  Commons category title, and a renamed category changes the slug. Pin the revision and the
  manifest hash below if you need stable keys.
- **The files are not EXIF-faithful.** The {facts.modified:,} re-encoded files lost their EXIF; the
  {facts.unmodified:,} untouched ones retain theirs.

## Provenance

Built from manifest `{facts.manifest_sha256[:16]}`. `provenance.json` records every input hash,
the pinned backbone revisions and the digest of every published file.

## Citation

Curated by [{DATASET_AUTHOR}]({DATASET_AUTHOR_ORCID}).

```bibtex
@software{{kargin2026krasnalid,
  author    = {{{DATASET_AUTHOR_BIBTEX}}},
  title     = {{Krasnal-ID: fine-grained visual instance retrieval of
               Wrocław's dwarf statues}},
  year      = {{2026}},
  publisher = {{Zenodo}},
  doi       = {{{PROJECT_DOI}}},
  url       = {{https://doi.org/{PROJECT_DOI}}}
}}
```

The DOI archives the code and this dataset's build pipeline; the photographs are here. Please also
credit the photographers, which `credits.csv` makes a one-column job.
"""

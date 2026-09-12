# Agents.md — Krasnal-ID (working title)
Research prototype: fine-grained visual instance recognition on Wrocław's bronze dwarf statues. Built as a CV/portfolio piece, not a shipping product.

## 1. Project overview
Given a photo of a Wrocław dwarf statue, retrieve which specific one it is from a reference set built from public CC-licensed photos. The interesting part isn't "call a pretrained model" — it's quantifying **how retrieval accuracy degrades as the candidate pool grows**, which is the real research question a location-narrowed mobile version would depend on.

## 2. Goals
- Produce a clean, well-documented research repo suitable for a CV/portfolio link.
- Answer one real research question rigorously (see §3), not just report a single accuracy number.
- Reuse solid engineering habits: pinned dependencies, structured logging, cached embeddings, reproducible experiments.

## 3. Non-goals (explicit, to keep scope tight)
- No mobile app, no production backend, no city/tourism-board partnership.
- No requirement to cover all ~1,000–1,400 dwarves — coverage is capped by what has usable reference photos, not by geography.
- No live GPS pipeline — location-narrowing is *simulated*, not built as a real service (see §5.2).

## 4. Research question (the core hook)
> Naive global instance recognition across N similar-looking classes gets harder as N grows. If the true candidate pool can be narrowed (e.g. by location) to a small local set, how much does accuracy actually improve, and at what pool size does it become a genuinely reliable identification method?

This ablation — **accuracy vs. candidate-pool size** — is the headline result. Everything else in this repo supports producing that one curve credibly.

## 5. Data

### 5.1 Sourcing and licensing
- Primary source: Wikimedia Commons ("Dwarves in Wrocław" category and per-dwarf subcategories) + Wikidata (structured entries with coordinates for some dwarves).
- Most images are **CC-BY-SA**: attribution and share-alike required. Every stored image record must carry `source_url`, `author`, `license`, `license_url` as first-class metadata — not an afterthought bolted on at the end. This isn't legal advice; confirm exact terms hold before any public-facing use of the dataset itself.
- Selection criterion for inclusion: **≥3 usable reference photos per dwarf** (threshold tunable). This replaces the earlier "geographic cluster" scoping from the mobile-app version — for a research prototype, data quality is the constraint, not tourist zones.

### 5.2 Location metadata (for the ablation)
- Where Wikidata provides coordinates, use them for a **real** geo-based candidate-pool experiment (secondary/bonus — coverage will be partial).
- The **primary** ablation should not depend on coordinate coverage: simulate candidate pools by repeatedly sampling random subsets of size N from the full class set and measuring retrieval accuracy at each N. This guarantees the headline experiment works regardless of how much real geo-data ends up available.

### 5.3 Pipeline
1. Query Wikidata for Wrocław dwarf-statue entities (name, coordinates if present, linked Commons category).
2. Pull images from each linked Commons category via the Commons API.
3. Filter to dwarves meeting the ≥3-image threshold.
4. Cache bounded research image copies + metadata to disk in a structured format (one JSON manifest + image files, not a database).

### 5.4 Wikidata discovery decisions

- Discover only items explicitly typed as Wikidata class `Q136276280`. Require a `P373` Commons category, keep `P625` coordinates optional, and prefer Polish labels, then English labels, then QID.
- Exclude group entities only when `P527` links to independently eligible member records. Naming heuristics may create manual-review warnings but must not exclude records.
- Cache the complete raw SPARQL response with endpoint/query-hash provenance. A deterministic `--limit` is applied only after full normalization; `--refresh` bypasses a valid cache.
- Live requests require a contact-bearing `KRASNAL_ID_USER_AGENT` environment variable. Cached normalization does not.
- Write ignored, atomic artifacts below `data/discovery/`: raw response, cache metadata, normalized dwarf records, and an exclusion/warning audit.

### 5.5 Commons acquisition decisions

- Treat `data/discovery/dwarfs.json` as the exact handoff into fetching. Every emitted
  Wikidata-to-Commons mapping must be approved or rejected in tracked
  `data/category-review.json`; corrections belong there rather than in generated discovery
  files. New or changed mappings return to `pending`.
- Query direct category files only, follow pagination, and cache complete API responses.
  Accept known Public Domain, CC0, CC BY, and CC BY-SA static rasters with complete attribution.
- Store Pillow-verified research copies with a 400-pixel minimum short side and 2,000-pixel
  maximum long side. Downscale oversized Commons derivatives locally with Pillow using
  aspect-ratio-preserving Lanczos resampling; never upscale undersized images. Preserve Commons
  page/revision provenance and reuse verified unchanged files.
- Treat a fixed multi-dwarf sculpture installation as one class only when its direct files
  consistently depict that installation. Reject umbrella categories whose files represent many
  distinct statues. Component categories may remain approved, but shared files are quarantined.
- Retain the lowest Commons page ID for same-label byte duplicates. Exclude byte-identical
  content spanning different labels to prevent evaluation leakage. Never delete orphan files
  automatically.

### 5.6 Commons-first discovery decision (2026-09-04)

Accepted as the second post-0.3.0 scope change, answering the "larger pool" question §8 lists.
Wikidata is the binding constraint and cannot be widened into one: measured on 2026-09-04, only
**44** items carry `P31 = Q136276280`, and a reverse check found the same 44 to be the only
Wikidata items pointing at *any* of the **481** per-dwarf categories under
`Category:Dwarves in Wrocław by name`. Relaxing the type filter therefore gains nothing; Wikidata
simply has no item for the other ~437 statues.

Commons holds the photographs. A 150-category sample of those 481 found 65% with three or more
files and a mean of 4.14, projecting ~314 categories over the current threshold and ~2,000 files
in total, against the 23 classes and 146 images the dataset held when this was written. The
rebuild landed at 306 classes and 1,691 images.

Decisions:

- **`dwarfs.json` stays the single discovery artifact with a single `query_sha256`.** Every
  downstream stage validates against that hash, so a second discovery file with a second hash
  would fork the provenance chain that §5.5 depends on. Commons enumeration is therefore a second
  source *inside* `data query`, behind an explicit `--include-commons`, and the recorded hash
  covers both queries. Without the flag the command reproduces the Wikidata-only artifact exactly.
- **A Wikidata record always wins over a Commons category describing the same statue.** The 44
  typed items carry `P625` coordinates and a stable QID; the merge prefers them and keeps the
  Commons-only records only for categories no Wikidata item claims.
- **Commons-only classes take a `C-<slug>` identifier** derived from the category title, which is
  unique on Commons by construction. The slug is ASCII-folded and filesystem-safe because the
  identifier names an image directory. A slug collision is an error to review, never a silent
  merge of two statues.
- **`wikidata_url` becomes optional and the dwarf-ID pattern widens** in the manifest and both
  review contracts. A Commons-only statue has no Wikidata item, so requiring one would be a
  schema lie.
- **The geographic ablation does not grow with the dataset.** Coordinates come from `P625`, so
  they exist for the 44 and for none of the rest. §7.1's geographic finding stays scoped to the
  statues that have coordinates, and the arm must report how many classes it actually covers
  rather than implying it covers the pool. Do not report a geo result over a Commons-first
  manifest without saying what fraction of it carried coordinates.
- **Review burden scales with discovery, and that is accepted rather than automated away.** 481
  category decisions instead of 41 is the cost of the larger pool. Mechanical mappings may be
  pre-approved by pattern in a later change, but a human decision remains the contract.
- **The merge matches on exact category equality, which is all it can honestly do.** A statue
  filed under a Commons category that Wikidata's `P373` does not name appears twice, once per
  source, and only category review catches it. *Papa Krasnal* is the live example: its Wikidata
  item points at its sculptor's category (`Olaf Brzeski`) rather than at `Papa Krasnal`, so both
  records survive discovery. Never assume the merge deduplicates statues; it deduplicates
  category strings.
- **A normalization warning about a category the merge then supersedes is dropped.** A reviewer
  reading 481 categories must not be sent to look at records that never reached the artifact.
  Exclusions survive regardless, because they explain an absence.
- **Pattern pre-approval is a human decision applied by pattern, taken on 2026-09-04.** Of 441
  new mappings, 367 matched a singular `<Name> dwarf, Wrocław` title with no discovery warning
  and were approved in bulk. The reasoning is that a Commons-only class has no Wikidata mapping
  to verify — the category *is* the identity — and every image still passes the license, size,
  attribution, prominence and cross-label duplicate gates that §5.5 defines. A further 13 were
  approved because Commons holds fewer than three files for them: the mapping is correct, and
  the documented image threshold is what decides whether a class exists. Rejection stays
  reserved for a mapping that is *wrong*, which a thin category is not.
- **Plural and irregular titles are reviewed by hand, always.** A plural title may be one fixed
  installation or an umbrella over many distinct statues, and only looking at the files can tell
  them apart. Two plural categories are known to be unusable in advance: `Troszka i Adoratorek`
  and `Śpiewak Operowy i Tancerka Balerina` are the sources of every `cross_label_duplicate`
  quarantine in §12.1, so their files would be rejected again under a new label.
- **Re-acquisition invalidates every published number.** Changing the discovered set changes
  `query_sha256`, which invalidates staging, the manifest, the split, the caches' relevance and
  all five experiments plus both figures. Treat a Commons-first rebuild as one deliberate
  re-run of §6.4's whole reproduction sequence, never as an incremental data top-up.

Measured on 2026-09-04, running `data query --include-commons` against both live sources:

- **482 records: 41 from Wikidata, 441 Commons-only.** The Wikidata arm emits exactly the 41 it
  always did, so the flag adds classes without disturbing the existing ones. 41 of the 482 carry
  coordinates, which is the geographic ceiling this decision accepts.
- 56 audit entries: 40 categories superseded by a Wikidata record, 12 titles too irregular to
  parse a name from, 4 possible unlinked groups and 3 excluded group entities. Twelve titles to
  read by hand out of 481 is the review surface, not 481 — but every one of the 441 new records
  still needs a category-review decision before `data fetch` will touch it.
- Commons uses `dwarf`, `dwarfs` **and** `dwarves`, plus an undiacriticked `Wroclaw`. Missing
  `dwarves` initially cost 76 categories their display name, which is why the title pattern
  accepts all three and why anything it cannot parse is flagged rather than guessed.

### 5.7 Derived coordinates decision (2026-09-05)

§5.2 took location from Wikidata's `P625` alone, which capped the geographic experiment at 23 of
306 classes — 7.5% of the pool. Commons file metadata lifts that cap. Measured on 2026-09-05:

- **1,135 of 1,545 photographs (73.5%) of otherwise-unlocated classes carry a coordinate**, almost
  all of type `camera` — where the photographer stood, not where the statue is. That gives at least
  one coordinate for **293 classes that had none**, for 295 of 306 in total. The 0.6.0 drift check
  then dropped one derived position as impossible — a one-degree latitude typo, 111 km out — so the
  shipped figure is 294 placed and 12 unplaced, which is what `RESULTS.md` reports.
- Commons *category* pages are not a source: 1 of 283 carries a coordinate.
- **Validated against the 21 classes that have both**: the median of a class's camera positions
  falls a **median of 9 m** from its Wikidata point, with all 21 within 147 m and 14 within 50 m.
  The geographic arm's smallest median pool radius is 190 m, so the noise is an order of magnitude
  below the signal it is used to measure.

Decisions:

- **A derived coordinate is stored as such and never passed off as authoritative.**
  `DwarfRecord.coordinate_source` records `wikidata` or `commons_camera`, and a record carrying
  coordinates must name where they came from. Wikidata always wins when both exist.
- **The per-class position is the median of its images' coordinates, not the mean.** One
  mis-tagged photograph should not drag a class across the city, and the median of an even
  handful of points is robust to that; the mean is not.
- **Coordinates are image metadata and belong on `ImageRecord`**, fetched in the same Commons
  request that already retrieves image info rather than through a separate enrichment pass. One
  source of truth for what is known about an image, and one provenance chain, as §5.5 requires.
- **The three worst validation cases are the water installation** — *Zbierający Wodę*, *Karmiący
  Ptaki* and *Wierzbownik*, all 147 m out. §12.1 already suspected Wikidata assigns group members
  one shared point, so there the camera positions are plausibly the *more* accurate figure. Do not
  treat `P625` as ground truth when reporting the disagreement; report it as disagreement.
- **This makes the §7.1 co-location finding testable at scale.** That result currently rests on six
  statues in one themed installation. Whether proximity is genuinely unhelpful city-wide, or only
  around that installation, is answerable at 294 classes and was not at 23.

### 5.8 Field query set decision (2026-09-05)

`RESULTS.md` names the gap between a Commons upload and a phone photograph as the largest untested
thing in the project, and it is the only §8 open question left. Closing it needs photographs taken
in Wrocław, so the contract for them is fixed here before any exist.

- **Field photographs are queries, never references.** They go in `data/field-queries/<dwarf_id>/`,
  git-ignored like every other image directory, and the manifest is not rebuilt to include them.
  Admitting them as references would destroy the very comparison they exist to make.
- **The directory name is the dataset ID**, which is what ties a photograph to a statue. Filenames
  carry no meaning and keep whatever the camera assigned.
- **The measurement is the difference against the leave-one-out baseline**, not an absolute number.
  93.1% top-1 is what the existing protocol reports; whatever these queries score, the gap between
  the two is the finding.
- **Hard classes and controls are both required.** The core route deliberately includes all
  eleven statues in the four families both backbones already confuse, plus eight statues in the
  same streets that belong to no such family. A uniform drop and a drop concentrated on the
  confusable families are different findings, and only the second is visible if both are present.
  Without controls a drop is confounded by hard statues also standing somewhere awkward.
- **Cohort is family membership, and is fixed before anything is scored.** It lives in tracked,
  reviewed `data/field-route.json`, so the split cannot be chosen after seeing the result. Four of
  the eight controls — Syzyfki, Capgeminiusz Programista, Kowal, Śpioch — draw one to three top-1
  errors of their own, so "control" means "in no confused family", not "never confused". Their
  error counts are recorded on the route entries beside the label, and the contamination is
  reported rather than corrected: it makes the two cohorts look more alike, so it understates a
  concentrated drop rather than manufacturing one. Defining the cohort by error count instead
  would leave four controls against fifteen hard classes and mix an eight-error Słupnik with a
  statue that errs once in thirty-one photographs.
- **Shooting instructions are part of the protocol, not advice.** A carefully composed phone
  photograph measures nothing this dataset does not already contain. `data/field-guide.md` asks for
  deliberate variation in angle, distance and light, and explicitly asks the photographer not to
  wait for a clean frame.
- **Photo GPS is the intended sorting mechanism.** Every statue on the route is geolocated by
  §5.7, so a geotagged photograph can be matched to a proposed statue rather than sorted by hand.
  A proposal, never an assignment: the directory a photograph sits in is authoritative.

### 5.9 Camera metadata decision (2026-09-05)

The §5.8 fieldwork is postponed, so the query-domain gap needs a proxy. Commons exposes each
file's EXIF camera model, and 51 of the 1,691 references were shot on phones — enough to ask
whether phone-originated photographs are harder queries than camera-originated ones.

- **Camera metadata lives outside the staging chain, in `data/discovery/camera-metadata.json`.**
  This is a deliberate exception to §5.7's rule that what is known about an image belongs on
  `ImageRecord`, and the reason is proportionality. Anything added to `fetched-images.json` changes
  the staging hash, which changes the manifest hash, which invalidates the split, all twelve result
  artifacts and the published demo — about two hours of recompute. Coordinates earned that because
  they *build* the dataset by placing classes; a camera model is read by one analysis and nothing
  else. If a second consumer ever appears, move it onto `ImageRecord` and pay the re-run.
- **The artifact records its own provenance** — endpoint, retrieval time, and the page IDs it
  covers — so a stale or partial file is detectable rather than silently mixed with a newer
  manifest.
- **Phone detection is a documented heuristic, not a fact.** It matches manufacturer strings, and
  it will mis-file an unusual device either way. The experiment therefore reports the group sizes
  and an `unknown` bucket alongside the result, so a reader can judge the classification rather
  than trust it.
- **The result is a lower bound on the real domain gap, and must be described as one.** These are
  still Wikimedia Commons uploads: chosen, often composed, and uploaded by someone who meant to
  document the statue. A casual snapshot is a harder query than anything measured here, so this
  does **not** retire the §5.8 fieldwork question — `data/field-guide.md` stays.
- **Report the confounds with the result.** Phone photographs could be harder because of the class
  they belong to rather than the camera. Median references per class and the share falling in an
  already-confused class are reported for both groups, because they are the first thing a reader
  should check.

### 5.10 Field query implementation decision (2026-09-05)

Section 5.8 fixed the contract for field photographs before any existed; this records how it is
implemented, built ahead of the fieldwork so the walk produces a result the same day rather than
starting a build.

- **Three stages, matching the existing layering.** `data field-queries` stages the photographs on
  disk into a generated query manifest, `embeddings extract --field-queries` embeds them, and
  `experiment field-gap` scores them. Staging needs no ML extra and no network; extraction remains
  the only thing that writes vectors, so the experiment cannot quietly load a model mid-run.
- **The reference set is the manifest, untouched.** No stage writes to `dwarfs.json`,
  `fetched-images.json`, the manifest or the split, so nothing here invalidates a published result.
  The query manifest is generated and git-ignored; only `data/field-route.json` is tracked.
- **The embedding cache is shared, the datasets are not.** Cache keys are content hashes plus the
  pinned backbone identity, so a field photograph is as addressable as a reference one and needs
  no second cache. `load_embedding_matrix` still reads the manifest and nothing else, so a field
  vector can never enter the reference matrix.
- **The comparison is the same statues' leave-one-out folds, not the headline 93.1%.** Restricting
  the Commons side to the photographed classes holds pool size and class difficulty fixed so that
  only the query's origin varies. Comparing against the whole-dataset number instead would confound
  the domain gap with which nineteen statues happen to be on the route.
- **One asymmetry is unavoidable and is reported, not hidden.** A leave-one-out query is withheld
  from its own class and therefore sees one fewer reference of the right statue than a field query
  does. That favours the field queries, so it understates the gap. It cannot be removed: an
  in-dataset query that is not withheld matches itself at similarity 1.0.
- **Staging refuses what would quietly corrupt the measurement**: a directory naming no dwarf in
  the manifest, a dwarf that is not on the reviewed route, a file that will not decode, and any
  photograph byte-identical to a reference — the last being a Commons upload copied into the query
  set, which would measure the protocol rather than the domain gap.

### 5.11 Dataset redistribution decision (2026-09-06)

Published to Hugging Face as `turhancan97/wroclaw-dwarves`, tier 3 — the photographs, the
metadata and the embeddings — public and ungated. This records what the licences permit, measured
from the manifest rather than assumed. **Not legal advice**: confirm the terms before any upload.

What the 1,691 images are licensed under:

| Licence family | Images | Share |
|---|---:|---:|
| CC BY-SA (4.0, 3.0, 3.0 pl, 2.5, 2.0) | 1,577 | 93.3% |
| CC BY (4.0, 3.0, 2.0) | 55 | 3.3% |
| Public domain | 52 | 3.1% |
| CC0 | 7 | 0.4% |

Every one of the 1,691 carries `author`, `license`, `license_url` and `source_url`, with no gaps.
That matters more than the licence mix: attribution at scale is the hard part of redistributing a
CC BY-SA corpus, and §5.1 made it a schema requirement from the start, so the obligation is already
satisfiable per file rather than needing reconstruction.

**Redistribution is permitted, and the project already does it.** The published demo ships all
1,691 photographs as thumbnails from `docs/assets/thumbs/`, names each photographer and licence on
the result that uses it, and states the licence families on the page. A Kaggle or Hugging Face
upload is the same act at a different address, so the question is not *whether* but *under which
obligations*. Three tiers, in increasing order of what they require:

1. **Metadata only** — the manifest, the split, the results, and a script that refetches the images
   from Commons. Redistributes no pixels, so no image licence obligation attaches at all, and it is
   enough to reproduce every experiment given a download. The weakest form, and the safest.
2. **Metadata plus embeddings** — adds the 1,691 cached vectors per backbone. A vector is a derived
   numeric representation rather than a reproduction of the photograph, so the practical licence
   burden stays at tier 1 while the artifact becomes directly usable: every retrieval experiment
   here runs from vectors alone, without an image.
3. **The photographs as well.** Permitted, with obligations that must be met per file and not per
   dataset: attribute each photographer, link each licence, and **state which files are
   modified**. That is per file, not blanket: the fetcher only resizes what exceeds the long-side
   cap, so **1,538 of the 1,691 files are downscaled adaptations and 153 are byte-identical to
   the Commons original** — measured by comparing each stored file against its recorded
   `commons_sha1`. An earlier draft of this section asserted modification over all of them, which
   would have been a false statement in a rights field on 153 files and would have destroyed the
   one signal telling a reuser which copies are exact. ShareAlike binds the adaptations. A dataset is a *collection* rather than a
   single adapted work, so per-image licences are preserved side by side instead of collapsing to
   one; the project's own metadata files can carry whatever licence is chosen for them. Both
   platforms can express this — Kaggle has an "Other (specified in description)" licence option,
   Hugging Face takes `license: other` with `license_name` and `license_link` plus a dataset card
   that carries the per-file terms.

Two things settled before uploading:

- **The sculptures are copyrighted, and the photographs are not the only work involved.** The
  dwarves are contemporary works by living sculptors. Commons hosts photographs of them under
  Polish freedom of panorama, which permits publishing images of works permanently displayed in
  public places. Kaggle and Hugging Face are US-hosted, and US law grants no equivalent exemption
  for sculpture. **Answered by disclosure plus a named removal path**, not by declining to
  publish: the card states that the CC licences cover the photographs and cannot grant rights in
  the sculptures, and it names `data/image-review.json` as the mechanism by which a request is
  honoured — one exclusion entry, a rebuild, a re-push. The same reasoning already covers the
  1,691 thumbnails the demo publishes. A removal promise backed by a named artifact is credible
  in a way "contact us" is not.
- **A public dataset is a promise about identifiers.** `dwarf_id` values for Commons-only classes
  are slugs of category titles, and a renamed Commons category changes the slug. Anything published
  should record the manifest hash it was built from and say plainly that the identifiers are
  dataset-local, or downstream users will treat them as stable keys.

### 5.12 Dataset export decisions (2026-09-06)

How §5.11's decision is implemented, by `krasnal-id data export-hf`.

- **Hand-written parquet with `pyarrow`, not the `datasets` library.** A Hugging Face `Image`
  column at rest is a `{bytes, path}` struct plus a declared feature type, which pyarrow can write
  directly. `datasets` would add pandas, dill, multiprocess and xxhash to the runtime for an
  encoder we do not need, return `Any` through a strict-typed module, and — decisively — its
  `push_to_hub` re-encodes and re-shards from its own representation, so the published bytes would
  not be the bytes just hashed into the receipt. It is a **dev dependency instead**, used as a
  test oracle: `Features.from_dict(...)` must derive exactly the arrow schema we wrote, and
  `_to_yaml_list()` exactly the card block we emit. That test is what makes hand-writing safe, and
  it earned its place immediately by catching that `datasets` models every field as nullable.
- **Nullability is a build-time contract, not an arrow flag.** Matching `datasets` means every
  field is nullable in the file. The columns that must never be empty — the attribution and
  licence ones — are checked before writing and asserted after reading instead, which makes the
  §5.11 obligation something the writer refuses to violate rather than something the card
  promises.
- **Six configs**, because config granularity is download granularity: `default` (pixels),
  `metadata` (the same rows without them, so §5.11's tier 1 ships inside tier 3), `classes`,
  `embeddings_dinov2`, `embeddings_clip`, `leave_one_out`. A user comparing backbones must not
  have to fetch 676 MB of JPEG to do it.
- **The split is named `reference`, not `train`.** Nothing here is trained; the median class has
  four photographs. A split called `train` invites someone to fine-tune on it and report a number
  that means nothing.
- **Every derived rights column is computed at export time**, never added to `ImageRecord`: the
  staging chain stays untouched, so an export invalidates no published result. This is the same
  trade §5.9 priced for camera metadata.
- **`--push` is opt-in and creates a private repository unless `--public` is passed.** The
  intended dataset is public, but a mistyped repo id becoming instantly world-readable is not
  recoverable and flipping private to public is one click. The token is never passed, held or
  logged — `huggingface_hub` resolves it — and a push failure exits 1 while a configuration
  failure exits 2.
- **The 52 public-domain rows get their basis re-queried** by `krasnal-id data license-templates`,
  which writes `data/discovery/license-templates.json` outside the staging chain for the same
  reason §5.9 gives. The Commons Public Domain Mark is a label, not a licence: it says a file is
  free of known copyright without saying why, and the fetcher discarded the template that does.
  Fifty-two files is small enough that publishing an unverified rights claim would be a choice.

### 5.13 Kaggle export decision (2026-09-10)

§5.11 left Kaggle open on the grounds that "the same export directory would serve". Building it
showed that it would not, and the reason is the one worth recording: **config granularity and
file shape are platform conventions, not dataset properties.** The Hub wants parquet whose image
column is a `{bytes, path}` struct with a declared feature type; Kaggle's data explorer previews
CSV and serves files, and its users open an image dataset expecting a folder of images beside a
table describing them. Shipping the Hub's parquet to Kaggle would hand a Kaggle user a column
their tools cannot open. So `krasnal-id data export-kaggle` is a second writer, not a second
target.

What is deliberately *not* duplicated is everything the rights obligation rests on.
`build_image_rows` derives the same licence URL, SPDX identifier, per-file modification flag and
credit line for both, and `LICENSES.md`, `ATTRIBUTION.md` and `credits.csv` are the same generated
artifacts. §5.11's obligation is per file, so the two exports must not be able to disagree about a
photographer.

- **The licence field says `other`, and that is the honest answer rather than a lazy one.** Kaggle
  accepts exactly one licence per dataset from a fixed list. This corpus has ten across four
  families, so every specific name is false of most of it: `CC-BY-SA-4.0` would assert 4.0 over
  the 3.0, 2.5 and 2.0 files and assert a licence at all over the 52 public-domain and 7 CC0 ones.
  `other` is Kaggle's "Other (specified in description)", which only stays honest if the
  description *does* specify them — so the generated description carries the family breakdown, the
  modified/unmodified split, the freedom-of-panorama disclosure and the `data/image-review.json`
  removal path, and a test asserts each is present.
- **Kaggle's limits are checked before the upload, not by it.** Title 6-50 characters, subtitle
  20-80, slug 3-50 and alphanumeric-with-hyphens: Kaggle rejects rather than trims, and learning
  that a title is one character too long after sending 676 MB is a bad trade. The packaged title
  is 48 characters and a test pins it inside the bound.
- **`folds.csv` states a rule and verifies it.** Written out, each fold's reference set is 1,690
  image IDs and the table is 30 MB of identifiers nobody reads. Leave-one-out makes it
  unnecessary: the references are every image except the query. That is a property of the current
  split rather than a promise, so the export *checks* it fold by fold and refuses if it ever stops
  holding, instead of publishing a table whose stated rule has quietly become false.
- **Row alignment is checked, because a `.npy` has no keys.** The only thing tying a vector to a
  photograph is its position in `images.csv`. A mismatch would not fail; it would silently make
  every downstream number wrong, so the export compares the matrix's image order against the
  table's and refuses on disagreement.
- **`resources` may only name files, and that is not obvious.** Kaggle's CLI validates every
  entry with `os.path.isfile` against the source folder before it zips anything, so declaring the
  `images/` directory — the most useful thing to describe — aborts the upload with
  "does not exist", and declaring `images.zip` fails too because `--dir-mode zip` produces that
  during the upload rather than in the folder. The photographs upload regardless, through the
  folder walk. This cost one failed upload attempt to learn, and a test now pins it.
- **The description is markdown *with HTML parsing*, and that ate half of it.** The first
  published version rendered only its opening paragraphs: `embeddings_<backbone>.npy` was read as
  an unknown opening tag and swallowed everything after it, backticks included. What went missing
  was the entire licensing section — the family breakdown, the modification statement, the
  freedom-of-panorama disclosure and the removal path — which is precisely the text that makes a
  `other` licence tag honest. **Never put an angle bracket in text a platform will render**, and
  assert its absence rather than trusting a code span.
- **Describe every file and every column, because Kaggle scores it.** Its usability score checks
  for file descriptions and column descriptions among other things, and an undescribed file reads
  as "This file does not have a description yet." `resources[].schema.fields` carries a name, type
  and description per column; `COLUMN_NOTES` holds one entry per column across every table and
  `_fields_for` refuses a column missing from it, so adding a column without describing it fails
  the export rather than publishing a blank.
- **Publishing stays a human step.** `export-kaggle` writes the directory and prints the
  `kaggle datasets create` and `kaggle datasets version` commands. §5.12 made `--push` opt-in for
  Hugging Face because a mistyped repo id becoming world-readable is unrecoverable; on Kaggle it
  is worse, because a dataset slug **cannot be renamed after creation**. No `kaggle` dependency is
  added and no credential path exists in this repository.
- **Validated by re-deriving the headline from the export alone.** Scoring leave-one-out from
  `embeddings_*.npy` and `images.csv` with no project import reproduces **93.1% top-1 / 95.7%
  top-5 for DINOv2 and 82.9% for CLIP** — the published numbers exactly. That is the property that
  makes a dataset release worth making, and it is checkable in twenty lines by anyone who
  downloads it.

## 6. Technical architecture

### 6.1 Embedding backbone
- DINOv2 and CLIP, used for zero-shot/near-zero-training feature extraction (no fine-tuning needed for the core experiment).

- **Design for a swappable backbone interface** (`get_embedding(image) -> vector`) rather than hard-coding a single library call. This keeps the option open to plug in a shared embedding-extraction module later without this repo depending on it existing or being finished first. Until then, load DINOv2/CLIP directly (e.g. via `transformers`).
- Cache embeddings to disk — never recompute per experiment run.
- Pin every backbone to an immutable revision that actually serves `model.safetensors` at that
  revision. `transformers` 5 refuses `pytorch_model.bin` and silently falls back to a mutable
  community safetensors-conversion ref, which would make the revision recorded in the embedding
  cache key untrue. CLIP is therefore pinned to the conversion commit
  `c237dc49a33fc61debc9276459120b7eac67e7ef` rather than to `main`.
- Treat backbone output shape as version-specific: `CLIPModel.get_image_features` returns a
  vision-output object whose `pooler_output` holds the projected features, not a bare tensor.
- `torchvision` belongs in the `ml` extra; `AutoImageProcessor` requires it.

### 6.2 Retrieval
- Cosine similarity k-NN over cached reference embeddings is the primary method.
- Report headline `top_k`/`mrr` over distinct **dwarves**, ranked by their best-matching image,
  because that is the candidate list an identification tool presents. Image-level `image_top_k`
  and `image_mrr` are reported alongside for comparison, never as the headline.
- Every evaluation must refuse a split whose recorded manifest hash does not match the manifest
  it is run against, rather than silently scoring a stale protocol.
- Accuracy proportions carry 95% Wilson score intervals as error bars. Rank averages such as MRR
  are not proportions and carry no interval.
- The baseline is exhaustive and deterministic, so its configured seed is recorded for provenance
  only. Nothing in it samples.
- The ablation samples a pool per query from one generator per (pool size, seed), so a run is
  reproducible no matter which pool sizes were requested. Configured pool sizes above the class
  count are skipped with a warning rather than clamped, and the full pool is always measured as a
  comparable right-hand anchor.
- Per-pool error bars are the observed spread across seeds, not a distributional assumption.
- The geographic ablation pools each query with its **N-1 nearest** dwarves rather than by a fixed
  radius, so pool size is matched to the random arm and only the selection rule varies. The median
  and maximum radius each pool spans are reported alongside, which keeps the result interpretable
  in metres. Nothing samples there, so a geographic measurement is exact and carries no seed
  spread; the random comparison arm supplies the error bars.
- Confusion analysis records the strongest wrong dwarf for **every** query, not only for failures.
  On a dataset with few outright errors the near-misses are where the signal is. Pairs stay
  directed, so an asymmetric confusion does not average away against its reverse.
- Strict `mypy` covers `src/krasnal_id` and `tests`. Test fixtures construct validated models with
  real `HttpUrl`, `datetime` and `Path` values rather than leaning on Pydantic's runtime coercion,
  so a fixture that drifts from a contract fails type checking rather than silently coercing.
- Analysis dependencies load lazily behind `import_optional_analysis`, mirroring the ML
  backbones, so the package imports without the `analysis` extra installed.
- Single-image retrieval reuses a cached vector whenever the query file's content hash already
  has one, so querying a dataset image needs no model load and no `ml` extra. The cache is only
  read there; populating it stays the job of `embeddings extract`.
- A query is compared against references by content hash, and every byte-identical copy of it is
  withheld. Otherwise a dataset image would match itself at similarity 1.0 and report nothing.
- The probe comparison always evaluates a `retrieval` arm on the same folds as the trained
  methods, so the answer to "does training beat retrieval" is readable from one artifact rather
  than assembled across two.
- Regularize the linear probe **weakly**. Embeddings are L2-normalized, so per-dimension
  magnitudes are near `1/sqrt(d)` and a conventional `C=1.0` underfits badly: on the current
  dataset it scored 66% top-1 against 96% at `C=100`. The default is 100.
### 6.3 Browser demo decisions (2026-09-02)

The published demo at `turhancan97.github.io/krasnal-id` is static: GitHub Pages cannot run the
Gradio app, so the model runs in the visitor's browser instead. Four decisions there were measured
rather than assumed, and each cost accuracy when guessed wrong.

The decisions below are all still in force; the numbers in the two model-specific bullets are
CLIP's, measured before §6.5 replaced it with DINOv2 on 2026-09-08. Read them as the reasoning
that survived the swap, not as the shipped figures.

- **Anything that embeds a query must embed the references.** Reference vectors built by the Python
  pipeline scored 89.7% top-1 against browser-built queries, where the build reported 93.2%.
  `docs/demo/build.mjs` therefore runs the same library, model and dtype the browser runs.
- **Never ship `vision_model_quantized.onnx`.** Of the exports offered it is the only one that
  degrades: 87.7% top-1 against 93.2% for `q4`, `uint8` and full precision, which are
  indistinguishable from each other here. `q4` is used, at 64 MB.
- **Resize through `docs/resize.mjs`, which both sides import.** sharp's lanczos3 and a browser
  canvas disagree by enough to matter, and canvas quality varies between browsers, so neither
  platform's built-in resampler can provide the guarantee. Letting transformers.js do the
  downscale itself is worse still: it resizes a 2000-pixel photograph in one aliasing step and
  loses about two points outright.
- **Decode drift is not eliminable and does not matter.** References are decoded by sharp and a
  visitor's photograph by their browser; measured cosine agreement is 0.986. Scored in a real
  browser over all 146 references, that costs nothing: 91.8% top-1 either way.
- The site reports the accuracy of the vectors it actually ships, re-scored at build time, and
  `?selftest=full` lets any visitor reproduce it in their own browser. Never quote the research
  numbers as the demo's.
- Rebuilt at 306 classes on 2026-09-04: the shipped vectors score 82.4% top-1 against 82.9% for the
  research CLIP pipeline, so the browser/pipeline drift is 0.5 points at this scale. Assets are
  29 MB, of which 22 MB is 1,691 thumbnails; that size was accepted deliberately so every match
  shows its own reference photograph rather than a stand-in for its class.
- The page's chart data in `docs/chart.js` is hardcoded and must be regenerated from
  `results/pool_size_ablation-*.json` whenever the ablation is re-run. Its gridlines and axis
  bounds are hardcoded too: check they still span the series, or a backbone's whole curve can land
  in an unlabelled void, as happened when CLIP's floor fell from 92.5% to 82.9%.

### 6.4 Result publication policy (2026-09-01)

- `results/` stays ignored: it is regenerated output. Figures selected for publication are copied
  to tracked `docs/figures/` and referenced from `RESULTS.md`, which is the written record of what
  the experiments found. Regenerate a published figure with its `visualize` command and copy it
  across rather than editing it by hand.
- Every number in `RESULTS.md` must be traceable to a committed command and a `results/` artifact.
  Extrapolations beyond the measured range are labeled as such, together with why they are
  optimistic.
- **Every result artifact records the experiment group that produced it**, in a `configuration`
  field. Before that, an artifact could not say which pool sizes, weights or cut-offs it used, and
  two runs of one experiment under different settings were indistinguishable — the filename carries
  only the experiment and the backbone, because `visualize` globs it and expects one file per
  backbone. A `top_k=50` re-ranking sweep therefore came one command away from silently
  overwriting the `top_k=10` result section 10 cites.
- **A run whose artifact would discard a different run is refused, before it is computed.**
  `guard_result_path` runs at the top of every `experiment` command and `write_experiment_result`
  repeats the check as a backstop. Identical settings overwrite freely, which is the ordinary case
  of re-running after re-extracting embeddings; differing settings name what would be lost and say
  to point `paths.results_dir` elsewhere. The pre-flight is the one that matters: a re-ranking
  sweep takes forty minutes, and refusing at the end would waste exactly as much time as no check.
- An artifact written before configurations were recorded cannot be compared against, so it is
  replaced rather than blocking its own regeneration; the replacement records one, which arms the
  check from then on. **Do not backfill the field from the packaged defaults** — the defaults are
  not necessarily what ran, so a backfilled configuration would be an assertion rather than a
  record, which is worse than an absent one.

- The demo loads the manifest and cached vectors once per session, not per query, and its
  callback returns an explanatory status string instead of raising, because a Gradio callback
  that raises shows the visitor a stack trace.
- Point Gradio at a per-user temporary directory unless `GRADIO_TEMP_DIR` is already set. Its
  default `/tmp/gradio` fails outright on a multi-user machine where another account created it
  first. Gradio analytics are disabled: launching a local research demo must not report to an
  external service.
- Hold BLAS to one thread while fitting per-fold classifiers. Each fit is tiny, so thread
  oversubscription dominates: the leave-one-out sweep went from over six minutes to twelve
  seconds. `threadpoolctl` is declared in the `analysis` extra but its absence is not an error.
- Optional stretch baseline: a simple linear probe or per-class prototype (mean embedding) comparison, to see whether a trained classifier beats raw retrieval — useful discussion material for the writeup, not required for the headline result.

### 6.5 Browser model swap (2026-09-08)

The demo ran CLIP for six days on one unmeasured premise: that DINOv2 was too big for a browser.
That premise was never tested, and it was wrong. `Xenova/dinov2-base` at `q4` is **56 MB against
the 64 MB CLIP export it replaced**, so the page now runs the pipeline's own model on a *smaller*
download. Everything that followed from the premise — §7.2's "porting DINOv2 is a §6.3-scale
change", §7.7's "CLIP is here only because §6.3 needs a model small enough for a browser", §8's
open question — followed from something that cost one afternoon to disprove.

Measured on an identical 40-class, 200-image subset before committing to a dtype:

| | top-1 | download |
|---|---:|---:|
| Python DINOv2 (the pipeline) | 98.0% | — |
| Browser DINOv2 `fp16` | 97.5% | 173.5 MB |
| **Browser DINOv2 `q4` (shipped)** | **96.5%** | **56.4 MB** |
| Python CLIP (the old demo) | 89.0% | 63.6 MB |

- **`q4`, not `fp16`.** `fp16` buys 1.0 point for 117 MB more download. `q4` still beats the CLIP
  it replaced by 7.5 points on the same subset, which is the comparison that decides whether the
  swap is worth making at all.
- **`uint8` is broken for DINOv2 and must never be shipped.** Its cosine agreement with the Python
  pipeline is **0.111** — noise, not drift. This is §6.3's `vision_model_quantized.onnx` trap
  wearing a different name: the export loads, produces plausible-looking vectors, and retrieves
  nothing. `q4` and `q4f16` agree at 0.935, `fp16` and `fp32` at 0.991. **Measure agreement against
  the pipeline before trusting any new export**, because nothing else catches this.
- **Take the CLS token, `last_hidden_state[:, 0, :]`.** DINOv2 has no pooled output to fall back on,
  so both `build.mjs` and `app.js` slice position 0 of the sequence explicitly — exactly what
  `embeddings/extract.py` takes. Mean-pooling the patch tokens instead is a different embedding and
  would not compare with the shipped references.
- **`EMBED_SHORTEST_EDGE = 256`, not CLIP's 224.** DINOv2's processor resizes the shortest edge to
  256 and then centre-crops 224, so pre-scaling to 224 through `docs/resize.mjs` would crop away
  the border and silently change the input. The constant appears in both files and they must match.
- §6.3's four decisions carry over unchanged and were re-verified against the new model: the build
  embeds the references with the same library, model and dtype the browser runs; both sides resize
  through `docs/resize.mjs`; the site re-scores the vectors it ships rather than quoting the
  research numbers.
- **Decode drift re-measured for DINOv2 on 2026-09-09: 0.989 mean, 0.980 min** over the 8 shipped
  probes, against 0.986 for the CLIP it replaced — so §6.3's "not eliminable and does not matter"
  holds for the new model too. `?selftest=1`'s threshold was wrong and had to be recalibrated:
  it required `min > 0.99`, which the documented drift cannot clear, so it reported the expected
  outcome as a failure. The check exists to catch a broken export, and that failure is not subtle —
  `uint8` sits at 0.111 — so the bound is now 0.95, an order of magnitude clear of both regimes.
  **Set a tolerance from the two measured regimes it must separate, not from how close to 1.0 the
  number looks like it should be.**
- **Rebuilt over all 306 classes on 2026-09-08: the shipped vectors score 93.2% top-1, 95.9% top-5,
  MRR 0.945**, against the research DINOv2 pipeline's 93.1% [91.8, 94.3]. The browser/pipeline drift
  is 0.1 points *in the browser's favour* — inside the pipeline's own interval, so at 306 classes
  4-bit quantisation and browser decode together cost nothing measurable. The demo it replaced
  scored 82.4%, so the page gained 10.8 points on a smaller download. Assets are 28 MB, of which
  22 MB is the 1,691 thumbnails; `references.bin` grew from 3.5 MB to 5.2 MB with the wider
  768-dimensional vectors, which is the only size cost of the swap on this side.

### 6.6 Two backbones in the browser, and two pages (2026-09-10)

The page now offers a choice of backbone, and the site is split in two: `index.html` is the
identifier, `findings.html` is the written result, and `style.css` is shared so they stay one
design. `index.html` went from 550 lines to about 100, which was the point — the upload control
was below a fold of prose that most visitors are not there for.

**The choice is a comparison, not a size tier, and the difference matters because the obvious
label is false.** CLIP is the *larger* download of the two — 63.6 MB against DINOv2 q4's 56.4 MB,
measured in §6.5 — and 10.8 top-1 points worse in the browser. A control that offered CLIP as the
lighter option would re-assert on the front page exactly the premise §6.5 was written to kill.
What CLIP is good for here is the comparison: §7.3, §7.5, §7.6 and §7.8 are all in some part about
the gap between these two representations, and a visitor who watches CLIP miss a statue DINOv2
identifies has learned that faster than the charts teach it. So both buttons carry the measured
download size and the measured top-1, DINOv2 is the default, and switching re-ranks the photograph
already on screen rather than asking for it again.

- **The page must never quote these numbers from the research pipeline.** Each button's accuracy
  comes from `meta.backbones[id].measured`, which the build computes by scoring the leave-one-out
  protocol on exactly the vectors it just wrote. This is §6.3's rule, now applied per backbone.
- **CLIP costs nothing until it is asked for.** Its weights and its `references-clip.bin` are
  fetched on selection, so the default page load is what it was. The shared metadata is one
  `references.json`; only the vectors are per backbone.
- **The stale-cache guard from `0.13.0` now guards a case that can really happen.** It was written
  against a hypothetical 512-wide buffer read as 768-wide; with two bin files of genuinely
  different widths served from one origin, a visitor holding one fresh file and one cached file is
  an ordinary event rather than a thought experiment. The length check is per backbone.
- **One descriptor, two importers: `docs/backbones.mjs`.** The two models differ in three ways that
  must agree between the build and the page or every cosine is meaningless — the model class
  (`AutoModel` against `CLIPVisionModelWithProjection`), the pooling (DINOv2's CLS token against
  CLIP's `image_embeds`), and the shortest edge (256 against 224). These used to be constants typed
  into both files, which held only while nobody edited one of them. The class is named as a string
  rather than imported, because the browser resolves transformers.js from a CDN and the build from
  npm.
- **Set `intraOpNumThreads` explicitly in the build.** Embedding one image with DINOv2 q4 takes
  **7502 ms** at onnxruntime-node's default and **426 ms** at four threads — 17.6x, or seven hours
  against twenty-four minutes for a two-backbone build. The cause is worth knowing because it is
  invisible: onnxruntime sizes its thread pool from the *host's* core count, not from the cpuset
  the process is confined to, so on this shared machine it opened around forty threads onto the
  four CPUs the session actually had. The `pthread_setaffinity_np` errors it prints on every run
  are that mismatch, and they had been printing for as long as the demo has had a build without
  anyone reading them as a performance problem. **Set the count from `nproc`, not from
  `/proc/cpuinfo`, and re-measure on the machine that runs the build.**

## 7. Experiments
1. **Baseline accuracy**: top-1, top-5, and mean reciprocal rank, DINOv2 vs. CLIP, full candidate pool.
2. **Headline experiment**: accuracy vs. candidate-pool size N (synthetic random subsampling, repeated with multiple seeds per N for error bars; real geo-based pools as a secondary comparison if coordinate coverage allows).

### 7.1 Dataset-scale decision (2026-09-01)

The dataset stays at its ≥3-image threshold, and the headline result is reported as the
**rate of degradation per doubling of pool size** rather than as a pool size at which
identification becomes unreliable. Measured evidence behind that choice:

- The current 23-class pool already sits near ceiling (DINOv2 top-1 95.9%, CLIP 92.5%), but the
  curve is not flat: DINOv2 loses about 1 accuracy point per doubling and CLIP about 1.8, with a
  seed spread near one point. The backbone gap widening with N is itself a reportable result.
- Lowering the threshold to 2 was measured, not estimated: it adds 4 classes and 8 images and
  leaves the curve unchanged (DINOv2 full-pool top-1 96.1% against 95.9%). Not worth the weaker
  per-class reference sets.
- Recovering the excluded images is not available locally. Of 23 fetch-audit exclusions, 20 are
  `cross_label_duplicate` files on 4 single-image classes, and admitting them would create the
  evaluation leakage §5.5 forbids. Growing the class count needs new Commons acquisition with
  unknown yield.
- Any extrapolation past N=23 must be labeled as such and called optimistic: distractors are
  drawn from a small class population, while a real 1,000-plus dwarf pool holds far more
  genuinely confusable statues, so the true curve should fall faster than a log-linear fit.
- The 2026-09-01 geographic ablation measured what §5.2 left open. Real proximity pools are
  *harder* than random pools of the same size, because six of the 23 dwarves stand within one
  metre of each other as one themed installation and are exactly the statues the confusion
  analysis flags. Random subsampling scatters them across pools and therefore overstates what
  location narrowing buys. Do not describe the simulated curve as a lower bound on a
  location-aware system.
### 7.2 Open-set rejection decision (2026-09-03)

Accepted as the first post-0.3.0 scope change, under the option §8 records as the nearest one.
`RESULTS.md` lists it as a limitation: a query of a statue outside the reference set still
returns neighbours, because the system cannot say "I don't know this one." Closing it needs no
new data, so it is measured on the existing 23-class manifest and cached vectors.

The mechanism under test is the cheapest one that could work: threshold the top-1 cosine
similarity. Accept the ranking when the best match scores at or above a threshold, reject the
query as unknown below it. No new model, no training.

Protocol — two query populations of 146 each, both derived from the manifest with no sampling
and therefore no seed:

- **Known queries** are the existing leave-one-out folds. The correct dwarf is in the gallery, so
  the right behavior is to accept *and* rank that dwarf first. Accepting a query but naming the
  wrong dwarf is not counted as a success; that is what makes the metric mean what it says.
- **Unknown queries** remove every image of the query's own dwarf from the gallery, which makes
  that dwarf genuinely absent. The right behavior is to reject.

Reporting rules, so the result cannot be read as better than it is:

- **AUROC over the two populations is the headline**, because it is threshold-free and cannot be
  tuned. The operating points are secondary.
- **A threshold calibrated on all the data is in-sample and is labeled `in_sample`.** It is
  reported as an optimistic reference only. Every headline operating point instead calibrates
  leave-one-class-out: the threshold for a held-out dwarf is a quantile of the known-query scores
  of the *other* dwarves, so no query helps set the threshold that judges it.
- Operating points are named by their target known-acceptance rate, and the achieved rate is
  recorded next to the target rather than assumed to equal it. A quantile of 146 discrete scores
  does not land exactly on a requested rate.
- Per-dwarf rejection rows record which dwarves survive removal, and which dwarf their orphaned
  queries fall through to. A cluster of visually similar statues should be able to cover for a
  removed member, so this is where the §7.1 water-themed installation is expected to reappear.

Measured on 2026-09-03, recorded here because two of these constrain future work:

- Rejection works for DINOv2 and not for CLIP, and the gap is far wider than closed-set accuracy
  implies: 0.969 against 0.898 AUROC, where top-1 differs by only 3.4 points. At a 90% known
  acceptance target DINOv2 falsely accepts 4.1% of unknown queries and CLIP 28.1%. **Do not offer
  a rejection threshold on CLIP embeddings**; on this dataset there is no useful operating point.
- The in-sample and leave-one-class-out false-acceptance rates agree (DINOv2 4.1% both, CLIP 30.1%
  against 28.1%), so the threshold is a property of the embedding space rather than of this
  sample. Leave-one-class-out calibration stays the reported default regardless, because that
  agreement is a finding about this dataset and not a licence to fit on the answers.
- Identifiable does not imply rejectable, and this is the result that generalizes. *Kowal* is
  never confused for anything while present (§5 confusion: 0 of 10) yet is the worst dwarf to
  reject once removed, with *100matolog* covering for it 3 times in 12. Closed-set confusion
  analysis cannot surface this, so neither analysis substitutes for the other.
- The water-themed installation does reappear as predicted, and under CLIP it is absolute: all
  three statues are falsely accepted 3 of 3.
- The artifact's `curve` is **descriptive, not calibrated**: it is swept in-sample over every
  observed score so that any threshold's cost is readable, and `visualize open-set` draws it with
  the calibrated leave-one-class-out points marked on top. Do not quote a point off the curve as
  an achievable operating point; quote the metrics.
- The published demo ships no rejection. When this was written that was doubly blocked: the
  browser ran CLIP, the backbone with no usable operating point, so an "I don't know" answer meant
  porting DINOv2 to the browser first. **That port happened on 2026-09-08 (§6.5), so the model is
  no longer the obstacle** — but the threshold still is. §7.3 measures DINOv2's false-acceptance
  rising from 4% at 23 classes to 38% at 306 at the same operating point, so there is no operating
  point worth shipping at this scale for either backbone. The demo's silence is now a measured
  conclusion rather than a limitation of its model — and section 7.8 closes the obvious escape
  route: geometric evidence, which discriminates where similarity fails, does not reject either.
- A DINOv2 port would still have to re-measure its own threshold on the vectors it ships. Section
  6.3 measured `q4` quantization as indistinguishable from full precision, but it measured that
  for *ranking*, and rejection depends on absolute similarity rather than on order. That
  equivalence therefore does not transfer, and section 6.3's rule that the site reports the
  accuracy of the vectors it actually ships applies to any threshold it might adopt.

3. **Error analysis**: confusion matrix for most-confused pairs — which dwarves get mixed up, and why (visually similar poses/props is the expected story).
4. **Embedding-space visualization**: t-SNE or UMAP plot of the reference set, colored by class, to make the "why confusion happens" argument visually.
5. **Open-set rejection**: can a top-1 similarity threshold answer "unknown" for a statue outside the reference set, and what does that cost on the statues inside it (see §7.2).

### 7.3 Dataset-scale result (2026-09-04)

The Commons-first rebuild of §5.6 took the dataset from 23 classes and 146 images to **306 and
1,691**, so §7.1's extrapolation caveat is now a measured quantity rather than a warning. What it
found, recorded here because three of these revise published conclusions:

- **§7.1's warning was half right, and the half it got wrong matters.** It said extrapolating past
  N=23 would be optimistic because a small pool holds too few genuinely confusable distractors.
  True for CLIP, which decays 2.06 points per doubling against the 1.76 the small pool predicted
  and is still worsening at N=306. False for DINOv2, which decays 0.79 against a predicted 0.96 —
  the small pool was *pessimistic* about it. Do not apply "small datasets flatter the result" as a
  blanket rule; it mispredicted one of the two backbones.
- **The curves differ in shape, not just slope.** DINOv2 loses 0.82 points per doubling from a
  pool of 2 to 20 and 0.77 from 50 to 306; CLIP loses 1.79 then 2.53. A single fitted slope hides
  that, so report the early and late rates whenever the range spans more than a few doublings.
- **Open-set rejection does not survive the larger pool**, which retires the §7.2 headline. DINOv2
  falls from 0.969 to 0.896 AUROC, and false acceptance at the 90% target from 4.1% to 38.3%,
  because an absent statue now has 305 chances to find a lookalike instead of 22. The mechanism
  predicts no recovery at city scale. Do not describe rejection as a working feature.
- **A linear probe now helps CLIP (+3.1 points) and still does nothing for DINOv2 (−0.1).** At 23
  classes both showed the same +0.7 non-effect. A supervised layer can partly repair an embedding
  space that is not laid out for instance discrimination and has nothing to add to one that is.
  The confidence intervals still overlap slightly, so report the direction, not a decisive win.
- **The dominant confusion cluster changed** from the water-themed trio to the three *Słupniki*
  pillar dwarves, which are near-identical statues installed on different streets. The embedding
  projection selects them without being told, using centroid distance alone.
- **Two group classes overlap their own members** (*Grajek i Meloman* with *Grajek* and *Meloman*;
  *Ogrodnik i Kierownik* with *Ogrodnik*). Their files are not byte-identical, so §5.5's duplicate
  guard does not catch them and they appear in the confusion pairs. Two of 306 is tolerable, but a
  future review pass should decide whether a group class may coexist with its members at all.
- **One BLAS thread per fit remains right, and by more than before.** Re-measured at 1,690 samples
  and 306 classes on the assumption the old §6.4 finding might have inverted: 1 thread is 3.1
  s/fold, 4 threads 7.1 s, 16 threads 20.8 s. Sixteen threads is 6.7x *slower*. Parallelise the
  probe across processes, never across BLAS threads.

### 7.4 Geographic result at scale (2026-09-05)

The §5.7 derived coordinates took the geographic arm from 23 classes to 294, and the §7.1 finding
survives the move intact — which it might not have, since at 23 it rested on six statues in one
themed installation.

- **Geographic pools lose to random pools at every measured pool size, for both backbones.** The
  arm is exact rather than sampled, so this is not seed noise. DINOv2 peaks at −0.89 points at a
  pool of five; CLIP at −2.12 at a pool of ten, roughly double throughout.
- **The penalty has a shape**: largest at a 171–291 m radius, fading to −0.35 by 2.2 km. Report
  the radius alongside the pool size — the radius is what carries the argument.
- **The co-location mechanism is measured, not inferred, and it is weaker than it sounds.** Every
  competing pair now carries a ground distance. Confused pairs are ~2x more likely than
  merely-competing pairs to stand within 100 m (11.9% against 5.5% for DINOv2), but the enrichment
  decays to 1.2x by 300 m and the whole-population rank statistic is only 0.518. Co-location acts
  at the scale of a shared plinth, not a shared neighbourhood, and **88% of confused pairs are more
  than 100 m apart**. State the mechanism as correct in direction and modest in magnitude; do not
  write as though proximity explains most confusion, because it does not.
- **Report the distance bands, not just the rank statistic.** The aggregate AUROC of 0.518 reads as
  "almost nothing" and would have hidden a real 2x effect concentrated below 100 m. A single
  summary number over a population whose effect is confined to one tail is the wrong summary.
- **The overstatement is worst where a real tool would operate.** Random subsampling scatters
  clustered lookalikes across pools, so it flatters location narrowing most at exactly the small
  radii a phone-based tool would use. Never quote the random curve as a proxy for a location-aware
  system at small pool sizes.

### 7.5 Photographer-disjoint result (2026-09-07)

`RESULTS.md` said the headline was "inflated by an unmeasured amount". It is inflated by 2.6
points for DINOv2 and 13.2 for CLIP. Three decisions made that measurable rather than merely
suggestive, and each is the difference between a number and a headline.

- **The gap is decomposed against a size-matched random control, never reported whole.**
  Withholding a photographer removes distractors too, and §7.3's finding is that accuracy *rises*
  as the pool shrinks — so the disjoint arm gets an unearned boost that would mask the penalty.
  Each query is also scored against the same number of references, including the same number of
  correct ones, drawn at random over five seeds. The raw drop is 12.4 points for DINOv2; the
  control pays 9.8 of them. Reporting 12.4 as photographer leakage would have overstated it
  fivefold. The geographic arm answers its question the same way, for the same reason.
- **Unanswerable queries are excluded and counted, not scored as failures.** 125 of 306 classes
  have one photographer, so 534 queries have no correct reference at all under this protocol.
  Counting them wrong would measure the dataset's coverage and report it as the model's weakness.
  The reported rates cover the 1,157 answerable queries, and both counts are in the artifact.
- **The result is stated as a lower bound on the attributable gap.** The disjoint arm ends with a
  median 252 candidate classes against the control's 305, because withholding a photographer
  removes whole classes — by §7.3 an easier pool, and it still lost. And `author` is Commons'
  free-text field compared exactly, so one contributor spelling their name two ways is two people
  and some of their own work stays in the reference set.

The finding worth carrying forward is the *ratio*, not either number: **CLIP leans on the
photographer five times as hard as DINOv2**. That sharpens §7.3's ordering from "DINOv2 is better
at instances" into something mechanical — CLIP's language alignment pulls toward appearance and
style, which is what covaries with who held the camera. It also has a product consequence, since
the published demo runs CLIP: 54.1% cross-photographer against the 82.9% the page reports.

### 7.6 Geometric re-ranking result (2026-09-07)

The first accuracy improvement in the project that comes from method rather than data. DINOv2 gains
0.9 top-1 points and CLIP 3.4, by counting RANSAC inliers between a query and each of the global
top-10 candidates and blending that into the cosine similarity.

- **SIFT, not a learned detector.** SuperPoint or LoFTR would match better under viewpoint change,
  but both download weights, and §10's CI rule keeps the `ml` extra out for exactly that reason —
  the re-ranking code would then be the only pipeline stage never exercised in CI. `opencv-python-headless`
  in a new `rerank` extra downloads nothing, so CI runs it. If a learned matcher is ever tried, it
  belongs behind the same interface and in `ml`.
- **Geometry is blended into the similarity, never substituted for it.** A pilot over sampled pairs
  gave a median of 10 inliers for the same statue against 4 for a different one — real separation,
  overlapping distributions, and some correct pairs verifying at zero. Sorting by inliers alone
  would demote correct answers that photograph badly. The blend also caps the inlier count at 30,
  so one spectacular match cannot dominate.
- **Weight zero is a control, and it is checked rather than assumed.** It reproduces the unranked
  baseline to the digit (93.14%). Ties in the blended score break by the global order, which is
  what makes that hold; without it the control would shuffle equal scores and every other column
  would be unreadable.
- **Promotions and demotions are reported, not just the net.** DINOv2's best weight fixes 19 and
  breaks 4; at twice that weight it fixes 20 and breaks 9. The net barely moves while the churn
  doubles, and only the decomposition shows it.

Two things bound the result. Re-ranking cannot rescue a statue the ranking never proposed, and at
k=10 that is 64 DINOv2 queries and 123 CLIP ones, so the ceilings are 96.2% and 92.7%.

And the inlier separation is inflated by §7.5's leakage — **measured, not suspected**, by running
the sweep with `experiment.photographer_disjoint=true`. The separation collapses from 58–64 inliers
against 4 down to **6 against 4**, so the spectacular verification was largely two frames from one
photographer's visit. **The accuracy gain nonetheless survives: 79% of it, for both backbones**
(DINOv2 +0.43 of +0.61, CLIP +2.94 of +3.72). The lesson is that the blend is a tie-breaker rather
than a replacement, so geometry does not need to be decisive to be useful — it needs to be
uncorrelated with the mistake the embedding is making. The two-arm form is the reporting decision
that makes this readable: each arm keeps its own weight-zero control, because the disjoint arm's
baseline is not the ordinary one.

In the disjoint arm recall becomes the binding limit rather than verification: the correct statue
never enters the top 10 for 107 DINOv2 and 288 CLIP queries, capping them at 90.8% and 75.1%.

### 7.7 First-stage recall result (2026-09-08)

Three standard ways of raising re-ranking's ceiling, all measured and all rejected. The value is in
*why* each fails, since each failure says something about this dataset that the successes did not.

- **Verifying 50 candidates instead of 10 buys 0.18 points** (CLIP disjoint, 57.04% to 57.22%), and
  at higher blend weights it is worse than k=10. The headroom is real — the ceiling rises 10.6
  points from k=10 to k=50 — but not convertible, because the two failures are correlated: the
  ranking loses the right statue on hard queries, and geometry is weak on exactly those queries.
  Every extra candidate is another chance to fluke a homography and nothing more.
- **Fusing the backbones does not beat the better one**: 79.6% at r@1 against DINOv2's 81.8%. The
  fused score is DINOv2 with CLIP dragging. Two models that fail on the same lookalike families do
  not decorrelate by being averaged, which is the assumption fusion needs.
- **Query expansion hurts, by 7 points for CLIP at r@10**, and the damage scales with the number of
  neighbours folded in. It assumes the top results are mostly right; at 54% precision they are not,
  and §5 says the wrong ones are near-identical statues, so the expanded query moves onto its own
  confuser. A technique that is standard elsewhere is actively harmful on a fine-grained set whose
  errors are lookalikes.

Reporting decisions worth keeping. The experiment reads cached vectors and no photographs, so it
runs in seconds and can be consulted *before* committing to a verification sweep that takes forty
minutes — which is the order these two should be run in. Three arms are reported (`full`,
`answerable`, `disjoint`) so the columns line up with §7.3's whole-dataset figures and §7.5's
answerable subset rather than needing a reader to reconcile them. And the fused row is identical in
both backbones' artifacts by construction, because the sum is symmetric; the config documents that
rather than leaving it as a puzzle.

The conclusion is that the bottleneck is the representation rather than the amount of it searched,
and that it belongs to one backbone: DINOv2's first guess cross-photographer beats CLIP's tenth.
CLIP was here only because §6.3 needed a model small enough for a browser. That constraint turned
out to be false: §6.5 replaced it with DINOv2 at q4, which is *smaller* than the CLIP it replaced,
so the demo now runs the same model as the pipeline. **Nothing in this project should use CLIP as
a first stage.**

### 7.8 Geometric rejection result (2026-09-09)

**Geometry does not fix open-set rejection, and the reason it looked like it would is the
photographer.** Section 7.6 showed geometry discriminating where similarity does not, so the
obvious next question was whether inliers give the "I don't know" answer section 7.3 could not
calibrate. They do not. All four signals over one query population, DINOv2:

| Signal | DINOv2 AUROC | disjoint | CLIP AUROC | disjoint |
|---|---:|---:|---:|---:|
| `cosine` (control) | 0.8959 | 0.7981 | 0.8059 | 0.6527 |
| `inliers_top_1` | 0.8880 | 0.7149 | 0.8532 | 0.6464 |
| `inliers_best` | 0.9015 | 0.7069 | **0.8798** | 0.6635 |
| `blended` | **0.9071** | **0.8046** | 0.8583 | **0.6754** |

False acceptance at a 90% known-acceptance target, best signal against the control: DINOv2 38.3% to
35.1% standard and 74.2% to 73.7% disjoint; CLIP 67.4% to 60.9% and 85.1% to 84.4%.

- **Both controls reproduce section 7.3 exactly**, which is what licenses reading the rest:
  identical false acceptance, identical known acceptance, identical in-sample balanced accuracy,
  AUROC within 3.5e-7 for DINOv2 and 1.8e-7 for CLIP. Any difference in the geometric rows is the
  signal and not the harness.
- **The honest gain is 0.65 AUROC points and half a point of false acceptance** — 74.2% to 73.7%
  with the photographer withheld. Three-quarters of unknown statues are still accepted. There is no
  operating point worth shipping, so the demo's silence stands as a measured conclusion for
  geometry too, not only for similarity.
- **Geometry alone is *worse* than similarity once the photographer is withheld**, 0.707 and 0.715
  against 0.798. It is only ever useful blended, which is also what section 7.6 found for accuracy.
- **The mechanism, and the reason this experiment needed the disjoint condition: a known query
  averages 144 inliers in the standard condition and 17 with its own photographer withheld.** An
  eightfold collapse. 144 inliers is not two photographs of one statue, it is the same frame from
  the same visit; the unknown arm barely moves (4.15 to 3.78) because it never had a near-duplicate
  to find. Geometry's apparent edge at rejection was largely re-identifying the photographer's own
  shot, and the standard condition alone would have reported that as a finding. **Geometry is more
  photographer-dependent than appearance, not less** — its AUROC falls 17 to 19 points between
  conditions where cosine falls 9.8.
- **Do not compare the pure-inlier false-acceptance rates against cosine's at face value.** Inlier
  counts are small integers with heavy mass at zero, so leave-one-class-out calibration cannot land
  on the 90% target: it achieves 95.6%, 91.8% and 99.3%. `inliers_best`'s 97.0% is that artefact,
  not a 97% failure at the requested operating point. **AUROC is the comparison that holds here**,
  because it is threshold-free; a signal on a coarse integer scale cannot be calibrated to an
  arbitrary acceptance rate at all, which is itself a reason not to ship it.
- **Geometry helps CLIP five times as much as DINOv2, and that is the same pattern a third time.**
  Disjoint, the best geometric signal gains CLIP 2.3 AUROC points against DINOv2's 0.65; standard,
  7.4 against 1.1. Section 4's linear probe was worth 3.1 points to CLIP and nothing to DINOv2, and
  section 10's re-ranking gained CLIP 3.4 top-1 points against DINOv2's 0.9. **Every add-on this
  project has measured helps only where the representation is weak** — they substitute for a poor
  backbone rather than extending a good one. It is not evidence that geometry rejects; it is
  evidence that CLIP's similarity is bad enough to be worth replacing with almost anything.
- **The inlier distributions are nearly identical across backbones, which is the internal check
  that the mechanism is real.** SIFT reads pixels and knows nothing about the embedding, so only
  the candidate sets differ between the two runs — and the known-arm means come out 156.4 against
  152.5 standard and 21.5 against 19.5 disjoint, with the unknown arms at 7.28 against 7.16 and
  6.51 against 6.53. The eightfold collapse is a property of *this dataset's photographers*, not of
  either backbone, and it reproduces independently in both runs.
- **CLIP disjoint is where the integer-tie problem becomes total: `inliers_top_1` records 100.0%
  false acceptance at 100.0% known acceptance.** More than a tenth of known queries have zero
  inliers against their top-1 candidate, so the 90% quantile *is* zero and every query clears it.
  The signal did not fail to reject; it could not express a threshold at all. Treat that row as a
  demonstration of the calibration limit rather than a measurement.

### 7.9 Capacity result (2026-09-11)

The stronger-embedding branch of §8, answered: **the first stage's limit is not capacity.** Four
DINOv2 checkpoints crossing size against the register fix, every one of them scoring the same 1,157
photographer-disjoint queries, and `dinov2-large` at 3.5x the parameters moves r@10 by **0.17
points on 22 wins against 20 losses, p = 0.88**. r@10 is §7.6's re-ranking ceiling, so the statues
the first stage loses are not found by a bigger model of the same family.

- **The gain that is real is about the photographer, not retrieval.** `dinov2-large` takes r@1 up
  **2.59 points (62 to 32, p = 0.0026)** disjoint and 0.77 points (p = 0.09) on the full arm. A
  larger model is better at a statue shot by *someone else* and barely different on one shot by the
  same person, which files this under §7.5's photographer gap rather than under recall. The
  headline moves 93.1% to 93.9% and that difference is not established.
- **Registers are aimed somewhere else.** Both register checkpoints lose to their plain
  counterparts at every disjoint cut-off but r@50, by 1.30 points at r@10 for the base (p = 0.036)
  and 1.82 for the large (p = 0.019). Artifact tokens spoil dense feature maps; this pipeline reads
  the CLS token, which they evidently were not spoiling. A fix for a real defect is not a fix for
  *this* defect.
- **State the family, because it decides the answer.** The pre-specified family is the nine
  disjoint comparisons the question was posed about (threshold 0.0056) and the r@1 gain clears it
  at p = 0.0026; over the eighteen §13 tabulates it still clears (0.0028); over all sixty-three
  p-values the artifact computes it does not (0.0008). `RESULTS.md` says all three rather than
  picking the one that flatters the result. Nothing else approaches any threshold, so the register
  penalties stay directions. The r@10 null needs no correction either way.
- **Paired, because unpaired would have been the wrong instrument.** Every checkpoint answers every
  query, so the evidence lives entirely in the queries where exactly one succeeds. Reading two
  overlapping confidence intervals instead would have called the r@1 gain undecided and the r@10
  null indistinguishable from it — and §4 is the precedent for how that goes wrong. `summarize`
  emits the discordant counts and an exact McNemar p-value per comparison, and
  `exact_mcnemar_p_value` computes the two-sided binomial directly rather than adding a runtime
  dependency for four lines of `math.comb`.
- **`compare_backbones` had to be declared empty in the packaged config, not omitted.** Hydra
  refuses to override a key the composed config does not carry, so leaving it out turned
  `experiment.compare_backbones=[...]` into a composition error. It cannot carry a real default
  either: a fresh clone has only the two extracted backbones and naming a third would fail the
  default run on a missing vector.
- **What this does not test, and must not be read as testing.** One pretraining recipe and one
  corpus across all four cells, so this is capacity *within* DINOv2. A different pretraining is
  untested and DINOv3 stays out while it is gated. What is retired is the cheap version of §8's
  question — scaling the model already in the pipeline — and with it most of the case for
  fine-tuning the same family, since scaling its pretraining did not move the ceiling.

### 7.10 Geometry-first result (2026-09-12)

§7.7's other branch, answered: **local features cannot be this pipeline's first stage, and the
margin is not close.** Every reference ranked by RANSAC inlier count with no appearance involved,
over the same 1,157 photographer-disjoint queries — geometry reaches **43.5% at r@10 against
appearance's 90.8%**, winning 13 queries and losing 560 (p ≈ 7 x 10^-147). Its recall at fifty
candidates, 63.1%, is below appearance's at one.

- **The rescue rate is the number that decided it, and it is 12.1%.** Of the 107 disjoint queries
  whose statue falls outside appearance's top 10, geometry retrieves 13 — 9 under the pessimistic
  tie-break. Union-ing both candidate lists would lift recall@10 by about one point, from 90.8% to
  91.9%, for 1,690 homographies a query. That is the entire case for building a real local-feature
  index here and it does not pay for one.
- **The failure is wrong evidence, not absent evidence.** The correct statue has no inliers at all
  for only 2.7% of disjoint queries, and conditioning the curve on having evidence moves r@10 from
  43.5% to 44.7%. Some *other* statue admits the better homography, which is §5's lookalike
  families arriving through a second channel.
- **Geometry is three times as photographer-dependent as appearance, now measured on retrieval.**
  Withholding the photographer costs appearance 12.4 points at r@1 and geometry **42.4** (70.7% to
  28.3%). §7.8 found this from the rejection side as the 144-to-17 inlier collapse; this is the
  same fact seen as retrieval, and it is the mechanism behind the headline. With a co-visit
  near-duplicate in the reference set geometry is strong; without one it is largely matching
  incidental background.
- **Ties had to be handled before the run, not after.** Inlier counts are small integers and most
  of the corpus scores zero, so reading a rank off a sorted array would have resolved hundreds of
  ties by manifest order — and could have put a statue inside k=50 on nothing but its position in
  the file. Every figure is therefore published as a range, `geometry` against `geometry_worst`,
  and every conclusion above holds at both ends. This was caught by reading the code while the
  sweep warmed up; it would otherwise have cost two hours and produced a partly manufactured
  number.
- **The sweep journals and resumes.** The first attempt was killed at five minutes by a memory
  watchdog on a machine with 1.8 TB free, so the run has to survive that rather than avoid it.
  Each query's ranks are fsynced before the next begins and the journal's digest covers the
  manifest, the backbone, `max_keypoints` **and a schema version** — without the last, changing
  what a row means would let a resume reinterpret old rows, which is silent corruption.

## 8. Build order (strict, versioned)
- **v0.1**: data pipeline (Wikidata query → Commons pull → filtered manifest) + embedding extraction + basic k-NN retrieval + baseline top-1/top-5/MRR metrics.
- **v0.2**: candidate-pool-size ablation (the headline experiment) + confusion matrix + embedding visualization.
- **v0.3 (stretch)**: linear-probe/prototype baseline comparison + a small Gradio demo (upload a photo → top-5 candidates with similarity scores).

Agent should scaffold the full directory structure with stubs and docstrings before writing any real logic, in build order.

**Status: this build order is finished and released as `0.3.0` (2026-09-03).** Every stage above
has real behavior, plus two additions not in the original plan: the geographic ablation of section
5.2 and a static in-browser demo published from `docs/`. There is therefore no "next stage" to
pick up. Further work is a new research direction, and the limitations recorded in `RESULTS.md`
are the open questions. Releases after `0.3.0` each close one of them: `0.4.0` open-set rejection,
`0.5.0` the Commons-first rebuild, `0.6.0` derived coordinates, `0.7.0` the camera-origin gap.
`0.8.0` builds the field-query path of §5.10, which does not close its question: it leaves it
waiting on photographs rather than on code. `0.9.0` publishes the dataset per §5.11 and §5.12,
which is a distribution milestone rather than a research one. `0.10.0` closes the
photographer-disjoint question of §7.5, and `0.11.0` adds the geometric re-ranking of §7.6 —
the first accuracy improvement from method rather than data. `0.12.0` closes the recall question of
§7.7 by measuring that its three obvious answers do not work. `0.13.0` closes the last of §8's
open questions by putting DINOv2 in the browser (§6.5) — the only release so far whose finding is
that a constraint the project had been designing around did not exist. `0.14.0` measures geometry
as a rejection signal (§7.8) and finds it does not reject, and that geometry is more
photographer-dependent than appearance rather than less. `0.15.0` is the first release about the
site rather than the research (§6.6): two pages, a backbone switch offered as a comparison, and
three defects that only loading the page in a browser could surface. `0.16.0` publishes the
dataset to Kaggle (§5.13), which closes the last of §5.11's open questions and surfaces three
more defects that only an external platform could reject.

- ~~**Open-set rejection**~~ — done on 2026-09-03 as `experiment open-set`; see §7.2 for the
  protocol and what it measured. What it leaves open is a *product* question rather than a
  research one: the published demo still shows a ranking unconditionally, and giving it a
  threshold means choosing an operating point on a visitor's behalf.
- **Real query photographs** — the reference set is Commons uploads, so the domain gap to a phone
  camera is unmeasured. `experiment camera-gap` (§5.9) puts a lower bound on it from the 51
  references shot on phones, and does not retire the question. The protocol (§5.8), the route and
  cohorts (`data/field-route.json`), and the whole measuring path (§5.10) are built and tested;
  what is missing is the photographs, which need a day in Wrocław.
- ~~**Publishing the dataset to Hugging Face**~~ — done on 2026-09-06 as
  `krasnal-id data export-hf`; §5.11 records the decision and §5.12 the implementation.
- ~~**Publishing the dataset to Kaggle**~~ — done on 2026-09-10 as `krasnal-id data
  export-kaggle`; see §5.13. The premise that "the same export directory would serve" was wrong:
  file shape and config granularity are platform conventions, so Kaggle gets images-on-disk plus
  CSV while the Hub keeps parquet, and only the rights artifacts are shared. The dataset is live at
  `turhancankargin/wroclaw-dwarves`; publishing remains a human step, because a Kaggle slug cannot
  be renamed, so `export-kaggle` still only writes the directory and prints the
  `kaggle datasets create` and `kaggle datasets version` commands. Nothing is left open: three of
  §5.13's defects were invisible until the platform rejected them, and each is pinned by a test.
- ~~**Re-ranking under the photographer-disjoint protocol**~~ — done on 2026-09-07 as
  `experiment rerank -oexperiment.photographer_disjoint=true`; see §7.6. The separation was mostly
  near-duplicate confirmation, the gain mostly was not.
- ~~**Raising the first stage's recall**~~ — done on 2026-09-08 as `experiment recall`; see §7.7.
  A larger `top_k`, backbone fusion and query expansion were all measured and all rejected, so the
  question is closed in the sense that the obvious answers are gone. What remains open is the
  *representation*: a stronger or fine-tuned embedding, or local features used as a first stage
  rather than a re-ranker. Neither has been tried, and the second would be a different pipeline
  rather than a parameter.
- ~~**Does geometric evidence reject where similarity cannot?**~~ — answered on 2026-09-09, no;
  see §7.8. What it leaves is sharper than what it closed: **rejection here needs a signal neither
  appearance nor geometry provides.** Both scores this project computes have now been measured for
  it and both fail at 306 classes, so the next candidate is not another threshold on the same
  evidence — it is a model trained to abstain, or a second view of the same statue.
- ~~**A browser-sized model that is not CLIP**~~ — done on 2026-09-08; see §6.5. No distillation
  was needed: `Xenova/dinov2-base` at q4 is 56 MB against the 64 MB CLIP export it replaced, so the
  demo runs the pipeline's own model on a *smaller* download. The premise that CLIP was there for
  size was simply wrong.
- ~~**Photographer-disjoint evaluation**~~ — done on 2026-09-07 as
  `experiment photographer-gap`; see §7.5 and `RESULTS.md` section 9. Of DINOv2's 12.4-point drop
  when its own photographer is withheld, a size-matched random control pays 9.8, leaving **2.6
  points attributable to the photographer**; CLIP's is 13.2. What it leaves open is *coverage*
  rather than leakage: 125 of 306 classes have a single photographer, so 534 of the 1,691 queries
  cannot be asked cross-photographer at all. Shrinking that fraction needs more contributors per
  statue, which is a data question, and the field photographs of §5.8 would each add one.
- ~~**A larger pool**~~ — done on 2026-09-04 as the Commons-first rebuild of §5.6, which took the
  pool from 23 classes to 306 and overturned three conclusions the small pool had supported; see
  §7.3 and `RESULTS.md` section 7. What it leaves is a *data* question rather than a research one:
  152 staged classes still sit below the three-image threshold, 43 of them with a single
  photograph, and Wrocław has several hundred statues Commons documents thinly or not at all. More
  images per class would admit them; more classes would extend the ablation curve past 306.

- ~~**Is the first stage's bottleneck capacity?**~~ — answered on 2026-09-11, **no**; see §7.9.
  3.5x the parameters moves the re-ranking ceiling by 0.17 points on 22 wins against 20 losses
  (p = 0.88), so the statues the first stage loses are not found by a bigger model of the same
  family. The one real gain is 2.59 points at disjoint r@1 (p = 0.0026), which is about surviving
  a change of photographer rather than about recall, and registers hurt. What it leaves open is
  narrower than what it closed: a *different pretraining* is still untested, and local features as
  a first stage — §7.7's other branch — is now the only cheap idea left standing.
  - **Four cells, two variables, one of them not size.** `facebook/dinov2-base` is the cell the
    project already has. Adding `dinov2-large`, `dinov2-with-registers-base` and
    `dinov2-with-registers-large` crosses capacity against the register fix, so a gain can be
    attributed to one or the other rather than to "a bigger model". Registers are in the design
    because DINOv2's feature maps carry high-norm artifact tokens that the CLS token sees, which is
    a plausible defect for instance retrieval and is not a capacity story. Every checkpoint is
    pinned by revision like the existing two.
  - **DINOv3 is deliberately excluded.** `facebook/dinov3-*` is `gated=manual` on the Hub, so a
    run of this repository would need a human to accept a licence and a token to exist, and §5.13
    keeps credential paths out of this repository. If it is ever added it is a separate decision.
  - **The new backbones are experiment-local until one of them wins, and none did.**
    `export/huggingface.yaml` lists the backbones an export writes and the visualization config
    lists the ones it draws, so a backbone absent from those lists costs the published datasets,
    the browser demo and the other eleven experiments nothing. Promoting one means re-cutting both
    published datasets and re-running everything, which is a release rather than an experiment —
    and §7.9 gives no reason to: `dinov2-large` buys 0.8 headline points that are not
    statistically established, for 3.5x the parameters and a 1024-wide vector in every published
    file. `dinov2` stays the pipeline's backbone.
  - **Fine-tuning is not the first branch, and the honest reason is that the leak-free data is not
    there.** §12.1's sub-threshold pool is the only image set disjoint from the benchmark, and it
    is 261 admissible images over 152 classes — 262 staged, since Binio `Q136343586` stages three
    and loses page `89462414` to image review — of which 43 classes have one photograph and 109
    have two. That is about **109 positive pairs** from 29 photographers, which overfits a ViT
    rather than adapting one. Fine-tuning therefore needs a class-disjoint split of the 306, paying
    comparability with every published number. §4 is also prior evidence against it: a linear probe
    on frozen DINOv2 features moved two queries of 1,691.

- ~~**Can local features retrieve what appearance loses?**~~ — answered on 2026-09-12, **no**;
  see §7.10. Geometry alone reaches 43.5% at disjoint r@10 against appearance's 90.8%, wins 13
  queries and loses 560, and rescues 12.1% of what appearance misses — about one point of recall
  for 1,690 homographies a query. §7.7's two branches are now both closed, and with them the cheap
  ideas: what remains is a genuinely different pretraining, a local-feature index that would have
  to beat this by an order of magnitude, or more photographs.
  - **Scored over every answerable query, not a sample.** SIFT matching costs 4.0 ms a pair with
    OpenCV's own threading, so all 1,157 answerable queries against all 1,690 references is 2.2
    hours rather than the ten it was assumed to be. A sample was the plan until it was measured;
    the full sweep makes the curve directly comparable to §7.7's table instead of nearly so.
  - **One pair matrix, two arms.** Inliers are computed once per query against every reference and
    the arms are masks over it, so `answerable` and `disjoint` cost the same as either alone. The
    `full` arm is deliberately absent: its extra 534 queries are the single-photographer classes,
    which cannot be asked cross-photographer at all, and including them would buy a third of the
    runtime for a column §7.7 already reports appearance-only.
  - **Appearance is re-scored here rather than read from `recall_curve-dinov2.json`.** The
    comparison is paired per query, which needs both rankings over the identical candidate set in
    the same run. Reading one from another artifact would pair them by assumption.
  - **The headline is the rescue rate, not geometry's own recall.** A first stage does not have to
    beat appearance everywhere; it has to find what appearance misses. So the reported number is
    how many of the queries appearance loses at k geometry retrieves at k, and geometry's own
    recall curve is the context for it.
  - **This is a feasibility probe and must not be described as a retrieval system.** 1,690
    homographies per query is seven seconds a photograph; nothing about it is deployable. It
    measures whether an actual local-feature index — ASMK, VLAD, a learned detector — could be
    worth building, and §7.8 predicts it is not: a known query's 144 inliers collapse to 17 once
    its own photographer is withheld, which is the regime this arm runs in.

Any of these is a scope change. Record the decision here before implementing it.

## 9. Repository structure

```
krasnal-id/
├── .github/workflows/ci.yml   # Python 3.12 quality gate
├── .zenodo.json               # metadata for the DOI minted on each GitHub release
├── AGENTS.md
├── CHANGELOG.md
├── CITATION.cff               # how to cite; its licence field covers the code only
├── CONTRIBUTING.md
├── LICENSE                    # MIT applies to original source code only
├── README.md
├── RESULTS.md                 # the complete written record of every experiment
├── pyproject.toml             # exact direct dependency pins and tool configuration
├── uv.lock                    # locked transitive dependency graph
├── data/
│   ├── category-review.json   # tracked human review of Commons mappings
│   ├── image-review.json      # tracked image-level exclusions and overrides
│   ├── field-route.json       # tracked route and cohort per statue, fixed before scoring
│   ├── field-guide.md         # the shooting protocol for the field queries
│   ├── field-queries/         # ignored field photographs, one directory per statue
│   ├── discovery/             # ignored Wikidata/Commons caches, staging, and audits
│   ├── images/                # ignored cached research copies
│   ├── embeddings/            # ignored embedding cache, keyed by content and backbone
│   ├── splits/                # ignored generated evaluation split
│   └── manifest.json          # ignored generated manifest
├── docs/                      # the published GitHub Pages demo
│   ├── index.html             # the findings, and an identifier that runs in the browser
│   ├── app.js                 # ONNX inference, retrieval, and the co-location warning
│   ├── chart.js               # the figures the page draws from the result artifacts
│   ├── assets/                # generated reference vectors and thumbnails
│   ├── brand/                 # tracked lockups and the script that generates them
│   ├── demo/                  # the build that produces docs/assets
│   └── figures/               # figures selected for publication
├── src/krasnal_id/
│   ├── cli.py                 # unified Typer CLI
│   ├── config.py              # Hydra composition + Pydantic validation
│   ├── models.py              # manifest, split, review, and attribution schemas
│   ├── atomic.py              # shared atomic file replacement
│   ├── geometry.py            # distance on the ground, shared by pipeline and experiments
│   ├── statistics.py          # the rank statistic two experiments share
│   ├── logging.py             # structured run logging
│   ├── configs/               # packaged Hydra configuration groups
│   ├── data_pipeline/
│   │   ├── wikidata_query.py
│   │   ├── commons_discovery.py  # the classes Wikidata has no item for
│   │   ├── commons_fetch.py      # reviewed, cached Commons acquisition
│   │   ├── camera_metadata.py    # EXIF cameras, deliberately outside the staging chain
│   │   ├── license_templates.py  # the public-domain basis, also outside the chain
│   │   ├── field_queries.py      # field photographs staged as queries, never references
│   │   ├── build_manifest.py
│   │   └── build_split.py        # deterministic leave-one-out folds
│   ├── embeddings/
│   │   ├── backbone.py        # the adapter contract and its pinned identity
│   │   ├── dinov2.py
│   │   ├── clip.py
│   │   ├── extract.py         # resumable extraction for any local image record
│   │   ├── cache.py           # atomic, validated, content-addressed vector cache
│   │   └── store.py           # manifest-ordered access for evaluation code
│   ├── retrieval/
│   │   ├── knn.py
│   │   ├── rerank.py          # SIFT + RANSAC verification of the top candidates
│   │   └── query.py           # single-image retrieval against the reference set
│   ├── experiments/
│   │   ├── contracts.py       # serializable result schemas
│   │   ├── artifacts.py       # atomic result persistence
│   │   ├── baseline_accuracy.py
│   │   ├── pool_size_ablation.py
│   │   ├── geo_ablation.py    # does narrowing by location help?
│   │   ├── probe_baseline.py  # does a trained classifier beat retrieval?
│   │   ├── confusion_analysis.py
│   │   ├── open_set.py        # unknown-query rejection
│   │   ├── open_set_geometry.py # does geometry reject where similarity cannot?
│   │   ├── camera_gap.py      # the query-domain gap, lower-bounded from EXIF
│   │   ├── photographer_gap.py # statue or photographer? decomposed against a control
│   │   ├── rerank_ablation.py  # the geometric re-ranking sweep
│   │   ├── recall_curve.py     # the first stage's ceiling, and three rejected fixes
│   │   └── field_gap.py       # the query-domain gap, measured on field photographs
│   ├── export/
│   │   ├── schema.py          # arrow schemas, HF features, the shard plan
│   │   ├── rows.py            # the derived rights columns
│   │   ├── tables.py          # parquet writing
│   │   ├── card.py            # the dataset card, licences and credits
│   │   ├── huggingface.py     # building the export directory
│   │   └── push.py            # uploading it, behind an explicit flag
│   ├── demo/
│   │   └── app.py             # the local Gradio demo
│   └── viz/
│       ├── embedding_plot.py
│       ├── ablation_plot.py
│       └── open_set_plot.py
├── results/                   # ignored generated results
└── tests/                     # schemas, configs, CLI, experiments, and interface contracts
```

## 10. Engineering conventions

- Require Python 3.12 or newer. Use `uv` for environment management and commit `uv.lock`.
- Keep all importable code under the single `src/krasnal_id/` package. Use Hatchling as the build backend.
- Use Pydantic v2 for manifest and application contracts, packaged Hydra configuration groups for composition, and one Typer CLI (`krasnal-id`) for all pipeline stages.
- Pinned dependencies, structured (JSON) logging for experiment runs, cached embeddings — don't recompute what's already on disk.
- Keep PyTorch/Transformers, analysis libraries, and Gradio in separate optional dependency groups; core imports and CI must work without them.
- Pin backbone configurations to immutable upstream model revisions before inference.
- Every stored image record keeps license/attribution metadata; treat this as a schema requirement, not optional.
- Config-driven experiment parameters (pool sizes, seeds, thresholds) — no magic numbers buried in scripts.
- Enforce Ruff formatting/linting, strict mypy checks, pytest coverage of at least 85%, and the same checks in GitHub Actions on Python 3.12.
- Do not introduce DVC. Ignore downloaded data, manifests, embeddings, and generated results by default; selectively tracking final portfolio artifacts requires a later documented decision.
- Original source code uses the MIT License. It does not relicense downloaded Wikimedia assets.

## 11. Deliverables
- Clean GitHub repo with a README that tells the story: the problem, why naive classification is hard, the pool-size ablation as the key result, and what it implies for a real location-aware version.
- `results/` folder with the accuracy-vs-pool-size plot, confusion matrix, and embedding visualization as saved figures.
- Optional: a small live demo (Gradio) for anyone reviewing the portfolio to try it themselves.

## 12. Living documentation and handoffs
- Treat this file as the authoritative project brief and decision record. Update it in the same change whenever implementation work introduces or changes architecture, scope, schemas, conventions, build order, or other decisions that future contributors must follow.
- Keep `CHANGELOG.md` current throughout implementation. Every material addition, change, fix, removal, or documentation milestone must be recorded under an `Unreleased` section as part of the same change.
- Changelog entries must describe what is actually present and working in the repository, not planned work. This gives future AI agents an accurate handoff point.
- When cutting a version, move the relevant `Unreleased` entries into a dated version section and recreate an empty `Unreleased` section.
- Before completing an implementation task, verify whether both this file and `CHANGELOG.md` need corresponding updates. Documentation-only wording fixes do not require a new architectural decision, but should still be logged when material.

### 12.1 Current dataset-audit and implementation handoff (updated 2026-09-05)

Everything below is recomputed from the tracked artifacts on the date in the heading. Where a
number here disagrees with an artifact, the artifact wins.

**The manifest is 306 classes and 1,691 images**, rebuilt Commons-first on 2026-09-04 per §5.6.
482 category mappings carry a decision: 469 approved, 13 rejected. Staging holds 1,958 images
across 458 classes; six are excluded by image review, and the 306 classes that clear the
three-image threshold are exactly the manifest's. The 152 that miss it hold 261 images between
them — 43 classes with one photograph, 109 with two.

**294 of the 306 classes are placed**, 23 from Wikidata's `P625` and 271 derived from their own
photographs' camera positions per §5.7. Twelve remain unplaced. A derived position is never
recorded as an authoritative one; `DwarfRecord.coordinate_source` says which is which.

Three approvals were reversed after acquisition measured what they cost, and the reason
generalizes: a category byte-identical to another empties *both* sides through §5.5's cross-label
quarantine. `Papa Krasnal` duplicated `Q11823412` and wiped out a class that was in the published
results; the `Detektyw Magda i Rabusie` and `Doktor Basia i Krasnalątko` umbrellas emptied their
own members. Rejecting the three restored `Q11823412`. Where members still collide with each other
— three robbers photographed in one scene — the quarantine is correct and those classes stay
absent, as Troszka, Adoratorek, Tancerka Balerina and Śpiewak Operowy still are.

Live operational facts, each tied to a tracked artifact:

- `data/discovery/fetched-images.json` is authoritative for what is admitted. There are 2,118
  files under `data/images/`, so about 160 are quarantined or orphaned and must not enter a
  manifest through directory scanning. Do not delete them automatically.
- The deterministic image-level exclusion/override contract lives in tracked
  `data/image-review.json`, tied to the current `fetched-images.json` staging hash. Its seven
  decisions are still the live ones, and the reasoning behind each is not repeated anywhere else:
  - retain Papa Krasnal page `166491` as the canonical lower-page-ID duplicate winner;
  - exclude Papa Krasnal page `22381955` as its differently encoded duplicate;
  - exclude pages `22398133` (Papa Krasnal), `52890654` and `52890655` (Pralinka), and
    `89462414` (Binio) for low subject prominence;
  - exclude Capgeminiusz Programista page `134103757` as a heavily posterized,
    non-photographic reference.
- Tracked `data/category-review.json` stores durable display-name overrides for
  `Q136001294` -> `Abruzjusz`, `Q136001318` -> `Ossolinek`, and `Q136001344` -> `Demokracja`.
  Generated discovery files remain unchanged; downstream manifest construction must use the
  reviewed override when present.
- Tracked `data/field-route.json` fixes the fieldwork route and each statue's cohort before any
  photograph is scored, per §5.8 and §5.10. `data/field-queries/` holds the nineteen core
  directories and no photographs.
- Cross-label protection is working: perceptual hashing found no remaining cross-class
  near-duplicate candidate in staging.

Implementation state:

- `data build-manifest` consumes `fetched-images.json` plus both tracked review files, applies the
  three-image threshold, records discovery/staging/review provenance hashes, and writes the
  manifest atomically. It never scans the filesystem.
- `data build-split` consumes only the validated manifest, creates one deterministic leave-one-out
  fold per admitted image, records the canonical manifest hash, and writes the ignored split
  artifact atomically.
- `data camera-metadata` fetches EXIF cameras into `data/discovery/camera-metadata.json`,
  deliberately outside the staging chain per §5.9.
- **A backbone has two identities: `name` and `family`.** `name` owns the artifact — the result
  filename, the exported vector column, the cached-vector key — and `family` selects the adapter
  that runs the checkpoint. `create_backbone` dispatches on the family, so §7.9's three extra
  DINOv2 checkpoints are configuration rather than code. `BackboneName` is declared once in
  `config.py` and reused by every experiment field that names backbones in a list, because
  `fuse_backbones` having its own copy meant it could reject a checkpoint the pipeline could run.
- **Five backbones are packaged; two are first-class.** `dinov2` and `clip` are exported,
  visualized and shipped to the browser. `dinov2-large`, `dinov2-registers` and
  `dinov2-registers-large` exist for §7.9 and are named in no export or visualization list, so they
  cost the published artifacts nothing. Their vectors sit in the same content-addressed cache; the
  key includes the model id and revision, so no checkpoint can read another's vectors.
- `embeddings extract` is implemented for the pinned DINOv2 and CLIP configurations. It validates
  image checksums and dimensions, supports CPU/automatic CUDA selection and configured batching,
  reuses valid normalized `.npy` vectors, and keeps model loading lazy so CI remains offline. CI
  uses deterministic fake backbones; real weights are downloaded only on local ML runs.
  `--field-queries` points the same loop at the staged field photographs.
- Retrieval, baseline metrics, the candidate-pool and geographic ablations, confusion analysis,
  visualization, the trained-classifier comparison, open-set rejection, the camera gap, and both
  demos are implemented and have been run end to end on this dataset. No module raises
  `NotImplementedError`. See section 8 for what is open beyond this point, and `CHANGELOG.md` for
  the per-stage record.
- The field-query path of §5.10 is implemented and its stages have been exercised end to end
  against this dataset, but only on stand-in photographs derived from references — enough to prove
  the plumbing, worth nothing as a finding. Since there are no photographs,
  `results/field_gap-*.json` does not exist and no field-gap number is published anywhere. The
  first real run is the fieldwork.

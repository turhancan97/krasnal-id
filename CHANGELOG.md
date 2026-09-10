# Changelog

This file records material changes that are actually present in the repository so that human contributors and future AI agents can quickly establish the current implementation state.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and
[Semantic Versioning](https://semver.org/spec/v2.0.0.html). Version numbers track the build
order recorded in `AGENTS.md` section 8, so `0.3.0` is the release that completes v0.1-v0.3.
Releases after it mark a completed research direction from that section's open-questions list
rather than a build-order stage, so `0.4.0` is open-set rejection.

## [Unreleased]

## [0.16.0] - 2026-09-10

The dataset is published on Kaggle as well, and §5.11's last open question closes.

That section had left it open on the grounds that "the same export directory would serve".
Building it showed that it would not: file shape and config granularity are platform conventions,
not dataset properties. The Hub wants parquet whose image column is a `{bytes, path}` struct; a
Kaggle user opening an image dataset expects a folder of images beside a table describing them.
So Kaggle is a second writer, and only the rights artifacts are shared — both exports derive
their licence URL, SPDX identifier, per-file modification flag and credit line from the same
code, because §5.11's obligation is per file and the two platforms must not be able to disagree
about a photographer.

Three of the four defects this release fixes were invisible until something outside this
repository rejected them. Kaggle's CLI validates `resources` with `os.path.isfile` before it zips
anything, so declaring the `images/` directory — the most useful-looking entry — aborted the
upload. Its description field is markdown *with HTML parsing*, so `embeddings_<backbone>.npy`
was read as an unknown opening tag and silently swallowed the entire licensing section, leaving a
dataset tagged "Other (specified in description)" with nothing specifying it. And the Kaggle
account is not the GitHub one. Each cost a failed attempt to learn and each is now pinned by a
test.

Validated the way a dataset release should be: scoring leave-one-out from the published
`embeddings_dinov2.npy` and `images.csv` alone, with no project import, reproduces 93.1% top-1,
95.7% top-5 and 116 errors of 1,691 — the numbers in `RESULTS.md`, exactly.

### Added

- **The published Kaggle dataset is linked wherever the Hugging Face one is.** The README's
  status paragraph and findings list, the "Use the published dataset" and citation sections,
  `RESULTS.md`'s reproduction section, both pages' footers, and `.zenodo.json`'s related
  identifiers, so the DOI record points at both copies. Each mention says which shape is which:
  parquet with the images embedded on the Hub, images on disk beside CSV tables on Kaggle.
- **The Kaggle export now describes every file and every column, and draws a cover.**
  `resources[].schema.fields` carries a name, type and description per column across all four
  CSVs; `COLUMN_NOTES` holds one entry per column and the export refuses a column missing from
  it, so a new column cannot ship undescribed. `LICENSES.md`, `ATTRIBUTION.md` and
  `provenance.json` gained descriptions too. A 1200x600 cover image is written beside the upload
  directory — not inside it, since Kaggle sets the cover in the web UI and a file in the folder
  would publish as data. `docs/kaggle-starter.ipynb` reproduces the headline from the published
  files alone.
- **`data export-kaggle`: the dataset as a Kaggle dataset.** §5.11 left this open on the grounds
  that "the same export directory would serve"; building it showed that it would not. File shape
  and config granularity are platform conventions, not dataset properties — the Hub wants parquet
  whose image column is a `{bytes, path}` struct, and a Kaggle user opening an image dataset
  expects a folder of images beside a table describing them. So Kaggle gets
  `images/<dwarf_id>/`, `images.csv`, `classes.csv`, `folds.csv` and one
  `embeddings_<backbone>.npy` per backbone, and the Hub keeps its parquet.
- **The rights artifacts are shared, deliberately.** `LICENSES.md`, `ATTRIBUTION.md` and
  `credits.csv` are the same generated files both exports publish, and both derive their licence
  URL, SPDX identifier, per-file modification flag and credit line from `build_image_rows`.
  §5.11's obligation is per file, so the two platforms must not be able to disagree about a
  photographer.
- **The licence field says `other`, which is the honest answer.** Kaggle takes exactly one licence
  and this corpus has ten across four families: `CC-BY-SA-4.0` would assert 4.0 over the 3.0, 2.5
  and 2.0 files and assert a licence at all over the 52 public-domain and 7 CC0 ones. `other` is
  Kaggle's "Other (specified in description)", which is only honest if the description specifies
  them — so the generated description carries the family breakdown, the modified/unmodified
  split, the freedom-of-panorama disclosure and the `data/image-review.json` removal path, each
  asserted by a test.
- **Three things are refused rather than discovered late.** Kaggle rejects an over-long title,
  subtitle or slug instead of trimming, so the limits are checked before 676 MB is copied.
  `folds.csv` stores only the query because every fold's reference set is every other image — a
  rule the export verifies fold by fold and refuses if it ever stops holding, rather than
  publishing a table whose stated rule has quietly become false. And a `.npy` has no keys, so the
  vector order is compared against `images.csv` before writing: a mismatch there would not fail,
  it would make every downstream number wrong in silence.
- **Publishing stays a human step.** The command writes the directory and prints the
  `kaggle datasets create` and `kaggle datasets version` commands. §5.12 made `--push` opt-in for
  the Hub because a mistyped repo id becoming world-readable is unrecoverable; on Kaggle it is
  worse, because a dataset slug cannot be renamed after creation. No `kaggle` dependency is added
  and no credential path exists in this repository.
- The configured Kaggle id is `turhancankargin/wroclaw-dwarves`, which is a *different account*
  from the Hugging Face `turhancan97/wroclaw-dwarves`. Both are pinned by a test, because one
  configured id would send one of the two exports to an account that does not exist.
- **`resources` names only files, because that is all Kaggle's validator accepts.** It checks
  every entry with `os.path.isfile` against the source folder *before* it zips anything, so a
  `images/` directory entry fails the upload outright with "does not exist" — and `images.zip`
  would fail too, since `--dir-mode zip` creates that during the upload rather than in the
  folder. The photographs still upload, via the folder walk; `resources` is descriptive metadata
  for the flat files, and the directory is described in the prose instead. A test now asserts
  every declared resource is a real file, which is the CLI's precondition restated.
- **Kaggle renders the description as markdown *with HTML parsing*, which ate half of it.** The
  first published version showed only its opening paragraphs: `embeddings_<backbone>.npy` was
  read as an unknown opening tag and silently swallowed everything after it, backticks and all.
  What vanished was the whole licensing section — the CC family breakdown, the modification
  statement, the freedom-of-panorama disclosure and the removal path — which is exactly the text
  that makes an `other` licence tag mean anything. Fixed by naming both embedding files, and a
  test now asserts no angle bracket appears in any rendered field.
- **Validated by re-deriving the headline from the export alone.** Scoring leave-one-out from
  `embeddings_*.npy` and `images.csv` with no project import gives **93.1% top-1 / 95.7% top-5
  for DINOv2 and 82.9% for CLIP** — the published numbers exactly.

## [0.15.0] - 2026-09-10

The site becomes two pages, the identifier offers both backbones, and the README shows its
results instead of only describing them.

`index.html` had grown to 550 lines and the upload control a visitor comes for sat under a fold
of prose. The findings move to their own page and the identifier drops to about 100 lines.

The backbone switch is the substantive change, and the interesting part is what it must not say.
CLIP is 64 MB against DINOv2 q4's 56 MB and scores 82.4% against 93.2%: it is the larger download
*and* the weaker model, so offering it as the lighter option would put back on the front page the
exact premise `0.13.0` was written to disprove. It is there as a comparison instead. Most of what
this project has found is about the gap between these two representations, and switching re-ranks
the photograph already on screen, so a visitor sees that gap on their own photograph rather than
reading it off a chart. Both backbones reproduce their previous single-backbone builds exactly —
the DINOv2 vector file is byte-identical to `0.13.0`'s.

Three defects surfaced only because the pages were finally loaded in a real browser and
screenshotted rather than read. The co-located warning had been rendering as an empty red box
under almost every result, the headline chart's tooltip had never worked at all, and the demo
build had been running 17.6x slower than it needed to for as long as it has existed. None of
them fails loudly; each one produces a page that looks approximately right.

### Added

- **`visualize retrieval-examples`: the retrieval itself, not an average of it.** Every other
  figure here plots an aggregate. This one draws a query photograph beside the five dwarves each
  backbone ranks highest for it, green for the correct statue, so a reader can see what 93.1%
  top-1 is made of. Written to `results/retrieval-examples.jpg`.
- **The queries are chosen by a stated rule, because a hand-picked example of a model succeeding
  is worth nothing.** Every fold is scored under both backbones exactly as `experiment baseline`
  scores it, folds are grouped by which backbones ranked the right statue first — both right,
  DINOv2 only, DINOv2 missed, both wrong — and the first fold in image-ID order represents its
  group. Image IDs order by Commons page ID, which is unrelated to anything either model sees.
- The figure is a JPEG where every other one is a PNG: it is 44 photographs rather than a line
  plot, and PNG stores those exactly and at 3.7 MB against 756 KB.
- **The README now shows its results instead of only describing them.** It carried no figures at
  all, despite four already sitting in `docs/figures/`: the new contact sheet joins the pool-size
  ablation, the UMAP projection and the open-set rejection curve, each beside the section that
  produces it.

### Changed

- **The published page runs either backbone, and the switch is a comparison rather than a size
  tier.** DINOv2 stays the default: it is the pipeline's own model, 56 MB at `q4`, and 93.2%
  top-1 on the vectors it ships. CLIP sits beside it at 64 MB and 82.4%, which makes it the
  *larger* download and the weaker one — labelling it "lighter" would re-assert on the front page
  the premise `0.13.0` was written to disprove. It is offered because the gap between the two
  backbones is what sections 7, 9, 10 and 12 are largely about, and switching re-ranks the
  photograph already on screen, so a visitor sees that gap on their own photograph rather than
  reading it off a chart.
- **Each button's accuracy is what the build measured, never a figure copied across.** The build
  now embeds every reference photograph with both backbones and re-scores the leave-one-out
  protocol per backbone; the page reads those numbers out of `references.json`. Both reproduce
  their previous single-backbone builds exactly — DINOv2 93.2% / 95.9%, CLIP 82.4% / 90.4%.
- CLIP costs nothing unless it is chosen. Its weights and its `references-clip.bin` are fetched
  on selection, so the default page load is unchanged. Vectors are per backbone
  (`references-dinov2.bin`, 5.1 MB; `references-clip.bin`, 3.4 MB) and the metadata they share
  stays in one `references.json`; the single `references.bin` is gone.
- **`docs/backbones.mjs` describes the two models once, for both importers.** They differ in the
  model class, the pooling — DINOv2's CLS token against CLIP's `image_embeds` — and the shortest
  edge, 256 against 224. Those used to be constants typed into `build.mjs` and `app.js`
  separately, which held only while nobody edited one of them; disagreement between the two makes
  every cosine meaningless without making anything fail.
- **The build sets `intraOpNumThreads` explicitly, which is worth 17.6x.** Embedding one image
  with DINOv2 q4 took **7502 ms** at onnxruntime's default and **426 ms** at four threads — seven
  hours against twenty-four minutes for a two-backbone build. onnxruntime sizes its thread pool
  from the *host's* core count rather than the cpuset the process is confined to, so on a shared
  machine it opened roughly forty threads onto four usable CPUs. The `pthread_setaffinity_np`
  errors it has printed on every build since the demo was built are that mismatch.
- **The site is two pages: an identifier and a written result.** `index.html` had grown to 550
  lines, of which about 230 were the five findings sections, so the upload control a visitor comes
  for sat above a fold of prose most of them are not there for. The findings now live in
  `findings.html`, `index.html` is about 100 lines, and each page carries one card-sized link to
  the other. The stylesheet moved out of `index.html`'s `<style>` block into a shared `style.css`,
  because two pages that drift apart visually are worse than one long one.
- The root URL is still the identifier, which matters: `CITATION.cff`, the Zenodo record and the
  Hugging Face dataset card all point at it.

### Fixed

- **The co-located warning rendered as an empty red box under almost every result.** `.note` sets
  `display:flex` and the box is toggled with the `hidden` attribute, but `hidden` is only
  `display:none` in the *user-agent* stylesheet, so any author `display` rule beats it. Every
  identification of a statue that is not co-located — most of them — therefore showed an empty
  warning-styled box. `[hidden]{display:none!important}` now enforces it once for the whole
  stylesheet, which also covers the new `display:grid` model switch. Caught by screenshotting the
  page rather than by reading it.
- **The pool-size chart's tooltip and crosshair have never worked on the published site.**
  `chart.js` creates its pointer target as `<rect class="hit-area" id="hit">` and then looked it up
  with `getElementById("hit-area")` — the class, not the id. That returned `null`, so the first
  `addEventListener` call threw, none of the three pointer handlers was ever attached, and the
  page logged one uncaught `TypeError` on every load. Found by loading the page in a real browser
  while splitting it, which is the only way this class of bug shows up: the chart draws correctly
  and only the interaction is missing.

## [0.14.0] - 2026-09-10

Geometry does not reject what similarity cannot, and why it looked as though it would is worth
more than the answer.

Section 7.3 found no similarity threshold worth shipping at 306 classes. Section 7.6 then found
that geometry discriminates where similarity does not, which made rejection look like the obvious
next measurement. It is not. Blended against cosine with each query's own photographer withheld,
geometry buys **0.65 AUROC points for DINOv2 and 2.3 for CLIP**, and four-fifths of unknown
statues are still accepted either way. Inliers alone are *worse* than cosine there.

What makes that credible is the condition that produced it. A known query averages **144 inliers**
in the standard condition and **17** once its own photographer is withheld, while the unknown arm
barely moves. A pair yielding 144 inliers is the same frame from the same visit, not two
photographs of one statue — so geometry's apparent edge at rejection was largely re-identifying
the photographer's own shot, and **geometry is more photographer-dependent than appearance, not
less**. Reported without the disjoint condition this experiment would have concluded the opposite.

The gain is five times larger for CLIP than for DINOv2, which is this project's recurring pattern
rather than a point in geometry's favour: the linear probe was worth 3.1 points to CLIP and
nothing to DINOv2, and re-ranking gained CLIP 3.4 top-1 points against DINOv2's 0.9. Every add-on
measured here helps only where the representation is weak, substituting for a poor backbone rather
than extending a good one.

Both confidence signals this project computes have now been measured for rejection and both fail
at this scale. The published demo names a statue for every photograph, and that is a measured
position rather than an omission.

### Added

- **`experiment open-set-geometry`: geometry as a rejection signal, not just a re-ranker.**
  Section 7.3 found no similarity threshold worth shipping at 306 classes, because every statue is
  the same semantic category and a missing statue's nearest neighbour scores much like a present
  one's. Section 7.6 then showed that geometry discriminates where similarity does not, and nobody
  had asked whether it *rejects*. Four signals — `cosine` as the control, `inliers_top_1`,
  `inliers_best` and `blended` — are computed in one pass over one query population, so a
  difference between them is the signal rather than the harness, and each is reported as a
  threshold-free AUROC beside a leave-one-class-out calibrated operating point and an in-sample
  upper bound.
- **Every signal is measured with the query's own photographer withheld as well.** Section 7.6
  found the inlier separation inflated by same-visit near-duplicates, so a known query can match
  its own photographer's other frame from the same angle rather than the statue. The withholding
  applies to both arms, not only the known one: the arms would otherwise search galleries of
  different sizes, and gallery size is itself a difference in how hard the nearest wrong statue is
  to find.
- **The answer is no, and the disjoint condition is why it is credible.** Geometry buys rejection
  0.65 AUROC points and half a point of false acceptance over similarity with the photographer
  withheld — 74.2% to 73.7% of unknown statues still accepted — and inliers *alone* are worse than
  cosine there, 0.707 against 0.798. The mechanism is measured: a known query averages **144
  inliers** in the standard condition and **17** once its own photographer is withheld, while the
  unknown arm barely moves. A pair yielding 144 inliers is the same frame from the same visit, not
  two photographs of one statue, so **geometry is more photographer-dependent than appearance, not
  less**. Reported without the disjoint condition this experiment would have concluded the
  opposite. Recorded as section 7.8 and `RESULTS.md` section 12.
- **Measured on both backbones, and geometry helps CLIP five times as much — which is this
  project's recurring pattern rather than a point in geometry's favour.** Disjoint, the best
  geometric signal gains CLIP 2.3 AUROC points against DINOv2's 0.65. Section 4's linear probe was
  worth 3.1 points to CLIP and nothing to DINOv2, and section 10's re-ranking gained CLIP 3.4
  top-1 points against DINOv2's 0.9: every add-on measured here helps only where the representation
  is weak, substituting for a poor backbone rather than extending a good one.
- The two runs cross-check each other. SIFT reads pixels and knows nothing about the embedding, so
  only the candidate sets differ between them — and the known-arm inlier means agree within 3%
  (156.4 against 152.5 standard, 21.5 against 19.5 disjoint). The collapse is a property of this
  dataset's photographers rather than of either backbone.
- CLIP's disjoint `inliers_top_1` row records 100% false acceptance at 100% known acceptance, which
  is the calibration limit in full rather than a measurement: more than a tenth of known queries
  have zero inliers against their top-1, so the 90% quantile is zero and every query clears it.

## [0.13.0] - 2026-09-09

The published demo now runs the same model as the research pipeline, on a smaller download than
the one it replaced.

CLIP had been in the browser since the demo was built, on the premise that DINOv2 was too large to
ship. The premise was never measured and it was false: `Xenova/dinov2-base` at `q4` is 56 MB
against CLIP's 64 MB. Re-scored on exactly the vectors it ships, the page goes from 82.4% to
**93.2% top-1** — 0.1 points above the research pipeline and inside its confidence interval, so
4-bit quantisation and browser decode together cost nothing measurable at 306 classes.

The interesting part is what the false premise had been holding up. Three separate conclusions
elsewhere in the project rested on it: that giving the demo an "I don't know" answer required a
model port first, that CLIP's weakness was a deployment constraint to be tolerated, and that a
browser-sized alternative was an open research question. All three are now resolved, and none of
them needed the work they appeared to need.

Two guards came out of building it, both for failures that produce plausible numbers rather than
errors. A stale cached copy of one asset beside a fresh copy of the other would have sliced
768-wide vectors out of a 512-wide buffer and scored the result. And the self-test that was
supposed to catch a bad export demanded an agreement its own measured drift could not reach, so it
reported healthy browsers as broken while a genuinely broken export — `uint8`, at cosine 0.111 —
sat two regimes away from the bound.

### Changed

- **The published demo now runs DINOv2, the same backbone as the research pipeline, instead of
  CLIP.** The page had run CLIP since it was built, on the premise that DINOv2 was too large to
  ship to a browser. That premise was never measured and it was false: `Xenova/dinov2-base`
  quantised to `q4` is **56 MB against the 64 MB CLIP export it replaced**, so the demo now runs
  the pipeline's own model on a *smaller* download. Re-scored on exactly the vectors it ships,
  the page goes from **82.4% to 93.2% top-1** — 0.1 points *above* the research pipeline's 93.1%
  and inside its confidence interval, so 4-bit quantisation and browser decode together cost
  nothing measurable at 306 classes. `references.bin` grows from 3.5 MB to 5.2 MB with the wider
  768-dimensional vectors; total assets are 28 MB, still dominated by the 22 MB of thumbnails.
- **`uint8` must never be shipped for DINOv2, and only a measured check catches that.** Its cosine
  agreement with the Python pipeline is **0.111** — noise, not drift. The export loads, produces
  plausible vectors and retrieves nothing, which is section 6.3's `vision_model_quantized.onnx`
  trap under a different name. `q4` and `q4f16` agree at 0.935, `fp16` and `fp32` at 0.991. `q4`
  was chosen over `fp16` because `fp16` buys 1.0 point for 117 MB more download.
- Both `docs/demo/build.mjs` and `docs/app.js` now take the CLS token explicitly,
  `last_hidden_state[:, 0, :]`, because DINOv2 offers no pooled output to fall back on, and
  pre-scale to a 256-pixel shortest edge rather than CLIP's 224 — DINOv2's processor resizes to
  256 before centre-cropping 224, so scaling to 224 first would crop the border away and silently
  change the input.
- Three consequences of the old premise are corrected rather than left standing: section 7.2 no
  longer says the demo "cannot simply be given" a rejection threshold because it runs CLIP (the
  model is no longer the obstacle; the absence of any usable operating point at 306 classes is),
  section 7.7 no longer calls CLIP's presence a deployment constraint, and section 8's open
  question "a browser-sized model that is not CLIP" is closed. `RESULTS.md`'s demo caveats and
  `README.md`'s demo section follow. `docs/chart.js` is deliberately unchanged: it plots the
  research pool-size ablation, where CLIP is still a legitimate comparison arm.

### Fixed

- **`?selftest=1` no longer reports its own expected outcome as a failure.** It required cosine
  agreement above 0.99 on the *minimum* of 8 probes, a bound the decode drift documented in
  section 6.3 cannot clear — the probes re-embed the same thumbnail bytes the build embedded, so
  sharp and the browser canvas are the only difference, measured at 0.989 mean / 0.980 min for
  DINOv2 and 0.986 for CLIP before it, worth nothing in top-1 either time. The check is there to
  catch a broken export, which is not a subtle failure: `uint8` agrees at 0.111. The bound is now
  0.95, clear of both regimes, and the message says which of the two it is looking at.

- **The demo now refuses inconsistent reference data instead of scoring it.**
  `references.json` and `references.bin` are fetched separately and carry no version in their
  URLs, so a returning visitor can briefly hold a fresh copy of one and a cached copy of the
  other — a real window now that the vectors changed width. Slicing 768 floats per image out of a
  stale 512-wide CLIP buffer runs off the end and yields short vectors, which score as plausible
  nonsense rather than failing. The loader now checks the buffer length against
  `images x dimensions` and throws, naming the cache as the cause.

## [0.12.0] - 2026-09-08

Why re-ranking stops where it does, and a guard against the accident that measuring it
nearly caused. Three standard ways of raising the first stage's recall are measured and all
three fail; a result artifact now records the settings that produced it, and a run that would
discard a different run is refused before it is computed rather than after.

### Added

- **Every result artifact now records the experiment group that produced it.** An artifact
  previously could not say which pool sizes, weights or cut-offs it used, and two runs of one
  experiment under different settings were indistinguishable, because the filename carries only the
  experiment and the backbone — `visualize` globs it and expects one file per backbone. A
  `top_k=50` re-ranking sweep came one command away from silently overwriting the `top_k=10` result
  `RESULTS.md` section 10 cites.
- **A run whose artifact would discard a different run is refused before it is computed.** The
  pre-flight is the point: a re-ranking sweep takes forty minutes, and refusing at the end would
  waste exactly as much time as no check at all. Measured: the refusal now arrives in 2.8 seconds,
  names every differing setting (`max_keypoints: 60 -> 800, top_k: 3 -> 50`), and leaves the
  earlier artifact intact. `write_experiment_result` repeats the check as a backstop.
- Identical settings still overwrite freely, which is the ordinary case of re-running after
  re-extracting embeddings. An artifact written before this field cannot be compared against, so it
  is replaced rather than blocking its own regeneration — and the replacement arms the check. The
  19 existing artifacts are therefore unarmed until each is next regenerated; they are deliberately
  *not* backfilled from the packaged defaults, since the defaults are not necessarily what ran and
  a backfilled configuration would assert rather than record.

- Added `krasnal-id experiment recall`, which measures re-ranking's ceiling and tests three
  standard ways of raising it. **All three fail**, and the failures are the finding:
  - **Verifying 50 candidates instead of 10 buys 0.18 points** (CLIP disjoint, 57.04% to 57.22%),
    and is *worse* than k=10 at higher blend weights. The 10.6 points of headroom between k=10 and
    k=50 are not convertible because the failures are correlated — the ranking loses the statue on
    hard queries and geometry is weak on those same queries.
  - **Fusing the backbones does not beat the better one**: 79.6% at r@1 against DINOv2's 81.8%
    alone. Two models that fail on the same lookalike families do not decorrelate by averaging.
  - **Query expansion hurts by 7 points** for CLIP at r@10, and worsens with more neighbours. It
    assumes the top results are mostly right; at 54% precision they are near-identical statues, so
    the expanded query moves onto its own confuser. A technique that is standard elsewhere is
    harmful on a fine-grained set whose errors are lookalikes.
- The experiment reads cached vectors and no photographs, so it runs in seconds and can be
  consulted *before* a verification sweep that takes forty minutes — which is the order to run them
  in. Three arms (`full`, `answerable`, `disjoint`) line the columns up with the whole-dataset
  figures of section 2 and the answerable subset of sections 9 and 10, rather than leaving a reader
  to reconcile them.
- Added `RESULTS.md` section 11 and `AGENTS.md` 7.7. The conclusion is that the bottleneck is the
  representation rather than the amount of it searched, and that it belongs to one backbone:
  **DINOv2's first guess cross-photographer beats CLIP's tenth**. CLIP is in the project only
  because the browser demo needs a small model, so a distilled or quantised DINOv2 is now a
  recorded open question.

## [0.11.0] - 2026-09-07

The first accuracy improvement in this project that comes from method rather than data, and
the test of whether it was real. Geometric verification on the top candidates lifts DINOv2 to
94.0% and CLIP to 86.3%; withholding each query's own photographer shows the spectacular part
of the evidence was near-duplicate confirmation while 79% of the gain was not.

### Added

- Added `krasnal-id experiment rerank`, the first accuracy improvement in this project that comes
  from method rather than data. Counting RANSAC inliers between a query and each of its global
  top-10 candidates, and blending that into the cosine similarity, lifts DINOv2 from 93.1% to
  **94.0%** and CLIP from 82.9% to **86.3%**. Geometry is worth four times as much to CLIP, which
  is the same asymmetry the photographer-disjoint result found from the other side: CLIP leans on
  appearance, so supplying it with shape evidence is worth more.
- Weight zero in the sweep is a control rather than a setting, and it is checked rather than
  assumed — it reproduces the unranked baseline to the digit (93.14%). Ties in the blended score
  break by the global order, which is what makes that hold; without it the control would shuffle
  equal scores and no other column in the sweep would be readable as a difference.
- Geometry is blended into the similarity and never substituted for it, because a pilot over
  sampled pairs found the inlier distributions overlap and some correct pairs verify at zero.
  Sorting by inliers alone would demote correct answers that photograph badly. The inlier count is
  capped, so one spectacular match cannot dominate a score.
- Promotions and demotions are reported alongside the net. DINOv2's best weight fixes 19 queries
  and breaks 4; at twice that weight it fixes 20 and breaks 9 — the net barely moves while the
  churn doubles, and only the decomposition shows it.
- Added `experiment.photographer_disjoint`, which runs the sweep twice over the answerable queries
  — once on all references, once with the query's own photographer withheld — and settles whether
  the geometric gain was near-duplicate confirmation. **It was, for the evidence; it was not, for
  the gain.** The inlier separation collapses from 58–64 against 4 down to **6 against 4**, so the
  spectacular verification really was two frames from one visit. But **79% of the accuracy gain
  survives for both backbones** — DINOv2 keeps +0.43 of +0.61, CLIP +2.94 of +3.72 — because the
  blend is a tie-breaker on top of cosine rather than a replacement for it. Geometry does not need
  to be decisive to help; it needs to be uncorrelated with the mistake the embedding is making.
  CLIP cross-photographer goes from 54.1% to 57.0%.
- Each arm keeps its own weight-zero control, since the disjoint arm's baseline is not the ordinary
  one; comparing a disjoint gain against the standard baseline would be meaningless.
- Extracted `krasnal_id.photographers`, because two experiments now need the same reference-
  selection primitives and section 6 forbids one experiment importing another for shared logic —
  the same reason `geometry` and `statistics` exist.
- Added `RESULTS.md` section 10 and `AGENTS.md` 7.6, stating what now binds the result: in the
  disjoint arm **recall**, not verification, is the limit. The correct statue never enters the top
  10 for 107 DINOv2 and 288 CLIP queries, capping them at 90.8% and 75.1%, so raising `top_k` or
  improving the first stage is worth more than better geometry.
- Added a `rerank` extra (`opencv-python-headless`) and installed it in CI. SIFT rather than a
  learned matcher specifically so nothing downloads weights: the `ml` extra is kept out of CI for
  that reason, and this code would otherwise be the only pipeline stage never exercised there.

## [0.10.0] - 2026-09-07

Closes the photographer-disjoint question, and corrects the surfaces that were still
describing it as unmeasured — including the demo page, which runs the backbone the finding
hits hardest.

### Added

- Added `krasnal-id experiment photographer-gap`, which answers the question `RESULTS.md` had been
  calling unmeasured. Two photographers took 67.4% of the corpus and same-visit near-duplicates
  were never removed, so the headline could have been the model recognising a camera rather than a
  statue. Withholding each query's own photographer costs DINOv2 12.4 top-1 points — but a
  **size-matched random control pays 9.8 of them**, so only **2.6 points are attributable to the
  photographer**. CLIP's attributable gap is **13.2**, five times DINOv2's, which sharpens the
  existing ordering into a mechanism: CLIP's language alignment pulls toward appearance and style,
  and that is what covaries with who held the camera.
- The control arm is what makes the number mean anything, and it exists because this project's own
  headline finding is that accuracy rises as the pool shrinks: withholding a photographer removes
  distractors too, so an undecomposed drop would have overstated the leakage fivefold.
- Unanswerable queries are excluded and counted rather than scored as failures. **125 of the 306
  classes have a single photographer**, so 534 of the 1,691 queries have no correct reference at
  all under this protocol; counting them wrong would measure the dataset's coverage and report it
  as the model's weakness. The rates cover the 1,157 that can be asked.
- Added `RESULTS.md` section 9 and `AGENTS.md` 7.5, both stating the figures as a *lower* bound on
  the attributable gap: the disjoint arm ends up with a median 252 candidate classes against the
  control's 305, which by the pool-size result is an easier pool, and it still lost.

- Added a **Personal and sensitive information** section to the dataset card, which had none. The
  dataset documents sculpture rather than people, but these are street photographs, so passers-by
  may appear incidentally and may be identifiable; no face detection was applied and the frequency
  was not measured. The section says so, notes that EXIF is absent from the 1,538 re-encoded files
  and that the coordinates are camera positions already public on Commons, and extends the removal
  path to anyone who appears in a photograph.

### Changed

- The published demo page now carries the cross-photographer figure in its limitations. It runs
  CLIP and quotes 82.9%, where CLIP identifies a statue 54.1% of the time without the same
  photographer's other photographs — the page a visitor actually uses was the last surface
  still omitting a finding that `RESULTS.md` and the dataset card both carried.
- The dataset card no longer says the accuracy is "inflated by an unmeasured amount". It gives the
  measurement, and adds the number a reader actually needs: **cross-photographer accuracy is 81.8%
  for DINOv2 and 54.1% for CLIP**, against headlines of 93.1% and 82.9%. The published demo runs
  CLIP, so a visitor photographing a statue Commons documents through one contributor is closer to
  54% than to 83%.

## [0.9.1] - 2026-09-06

Cut so the Zenodo record has correct metadata. The `v0.9.0` tag predated
`.zenodo.json` and the ORCID, so the DOI minted from it credits an author with no
persistent identifier and links neither the dataset nor the demo. The concept DOI resolves to the
newest release, so archiving this one corrects what that DOI points at without changing it.

### Added

- Recorded the Zenodo concept DOI `10.5281/zenodo.22548023` in the README (badge plus a
  `## Citing this work` section the README previously had none of), `CITATION.cff`, and the
  dataset card's BibTeX. The *concept* DOI rather than the version DOI, so citations accumulate on
  one identifier instead of splitting across releases.
- Recorded ORCID `0000-0002-6751-4773` in `.zenodo.json`, `CITATION.cff` and the dataset card, so
  every surface that asserts authorship points at the same persistent identifier rather than at a
  name that can be spelled several ways. The checksum digit was verified before use.
- Added `.zenodo.json`, so the DOI minted when a GitHub release is archived describes the work
  rather than whatever Zenodo infers from the repository. It declares `mit`, and says in both the
  description and the notes that MIT covers the source code only: the release archive is 30 MB of
  which 25 MB is 1,691 thumbnails derived from Wikimedia Commons, which keep their own Public
  Domain, CC0, CC BY and CC BY-SA terms. A single `license` field would otherwise assert MIT over
  photographs nobody licensed that way — the same over-claim the dataset export refuses to make.
  It also links the published dataset (`isSourceOf`) and the demo (`isDocumentedBy`).

## [0.9.0] - 2026-09-06

### Added

- Added `krasnal-id data export-hf`, which builds the whole dataset into a Hugging Face repository
  — six configs (`default`, `metadata`, `classes`, `embeddings_dinov2`, `embeddings_clip`,
  `leave_one_out`), a generated dataset card, a licence inventory, credits grouped by photographer,
  a machine-readable `credits.csv`, and a `provenance.json` digesting every emitted file. Built on
  the real dataset: 1,691 photographs across 2 derived shards, 306 classes, 676 MB, in 12 seconds.
  **Both baselines re-derive from the exported artifact alone** — 93.1% and 82.9% top-1, with no
  code from this repository — which is what makes the export faithful rather than merely valid.
- Added `krasnal-id data export-hf --push`, opt-in and **private unless `--public` is passed**. The
  token is never passed, held or logged; `huggingface_hub` resolves it. Push failures exit 1,
  configuration failures exit 2, and a malformed repository id is refused before anything leaves
  the machine.
- Added `krasnal-id data license-templates`, which re-queries the Commons template behind each
  file labelled "Public domain". The Public Domain Mark is a label rather than a licence — it says
  a file is free of known copyright without saying why — and the fetcher discarded the template
  that does. 52 files is small enough that publishing an unverified rights claim would be a choice.
- Added `krasnal_id.atomic`, the atomic-write idiom that seven modules had been carrying verbatim.
  The existing seven are unchanged and can migrate later.
- Added a `CITATION.cff`, whose `license: MIT` covers the code and says so, because a bare MIT on a
  dataset citation would misstate the photographs' terms.
- Added `AGENTS.md` section 5.12 recording the export's own decisions, and section 5.11, the
  licence analysis behind the release. Measured from the manifest rather
  than assumed: 1,577 of the 1,691 images are CC BY-SA, 55 CC BY, 52 public domain and 7 CC0, and
  all 1,691 carry author, licence, licence URL and source URL with no gaps — so the attribution CC
  BY-SA requires is already satisfiable per file. Redistribution is permitted and the published
  demo already ships every photograph as a thumbnail, so the open question is which tier to release
  (metadata, metadata plus embeddings, or the photographs) and whether Polish freedom of panorama,
  which is what lets Commons host photographs of copyrighted sculptures, needs answering
  deliberately before pixels go to a US-hosted platform. Section 5.11 is now a decision rather
  than an analysis: tier 3, public and ungated, with the freedom-of-panorama question answered by
  disclosure plus a removal path named in the card and backed by `data/image-review.json`.

### Changed

- `RESULTS.md` gains a limitation that changes how to read the headline: **122 photographers
  contributed, but two of them took 67.4% of the corpus** — Pnapora 715 images, Fallaner 424.
  Near-duplicates from a single visit were never removed, so a method can score partly by
  recognising a photographer's camera and processing rather than the statue, and 93.1% is
  inflated by an unmeasured amount. A photographer-disjoint protocol would bound it; `AGENTS.md`
  section 8 now carries that as the open question replacing the release.
- `README.md`'s `## Data and licensing` said nothing about a published dataset and now carries the
  licence inventory, the modification split, the freedom-of-panorama position and the removal path.
- The published demo page now links the dataset beside the repository.

### Fixed

- Stopped recording `PD-Layout` as a public-domain basis. Commons describes it as a "table style
  formatting template", used *inside* the real licence tags, so it matched the `PD-` prefix while
  saying nothing about why a file is free — presentation markup in a rights field, where a reader
  would reasonably take it for an answer. The 52 public-domain files now state exactly one reason
  each: 35 `PD-author`, 16 `PD-user`, 1 `PD-self`. Filtered on read as well as on fetch, so an
  artifact retrieved before the exclusion does not need re-fetching to be clean.
- **The Hub rejected the dataset card**: `license_link` must be an absolute https URI, and it was
  the repository-relative `LICENSES.md`. It now resolves against the repository being published
  to, so it still points at the ten-licence inventory rather than at one licence among ten. The
  failure came after `repos/create` had already succeeded, which left an empty repository behind —
  so the card is now validated against the Hub's own validator **before** anything is created, and
  a card the Hub would reject is a message rather than an orphan.
- Added a test asserting `krasnal_id.__version__` equals the installed distribution version, so a
  release that bumps one and not the other fails instead of shipping.
- **`src/krasnal_id/__init__.py` still reported `0.7.0` after the 0.8.0 release.** Every previous
  release bumped it and cutting 0.8.0 missed it. The export's provenance receipt reads the
  installed distribution version rather than the module constant regardless, but the constant was
  wrong and is now correct.
- Corrected `AGENTS.md` section 5.11, which said the pipeline "stores downscaled 2,000-pixel
  copies, so they are adaptations". Measured against each file's recorded `commons_sha1`:
  **1,538 of the 1,691 are downscaled adaptations and 153 are byte-identical to the Commons
  original**, because the fetcher only resizes what exceeds the cap. Asserting modification over
  all of them would be a false statement in a rights field on 153 files, and would destroy the one
  signal telling a reuser which copies are exact. The export derives `modified` per file from the
  bytes.

## [0.8.0] - 2026-09-06

### Added

- Added the whole measuring path for the field-query experiment, built before the photographs
  exist so a day in Wrocław produces a result the same day rather than starting a build.
  `krasnal-id data field-queries` stages the photographs on disk into a generated query manifest,
  `krasnal-id embeddings extract --field-queries` embeds them with the pinned backbones, and
  `krasnal-id experiment field-gap` scores them. Verified end to end on the real 306-class dataset
  with DINOv2.
- Added tracked `data/field-route.json`: the 53 statues on the route, each filed as a member of a
  confused family or as a control, with the top-1 errors its Commons photographs already draw.
  Fixed before any photograph is scored, so the cohorts cannot be chosen after seeing the result.
- The field gap is reported against **the same statues' leave-one-out folds**, not against the
  headline 93.1%, so pool size and class difficulty are held fixed and only the query's origin
  varies. Overall, per-cohort, and per-statue rows are all written to the result artifact.
- Nothing in the new path writes to the manifest, the staging file or the split, so staging a
  photograph invalidates no published result. The embedding cache is shared because it is keyed by
  content hash, but `load_embedding_matrix` still reads the manifest alone, so a field vector
  cannot enter the reference set.

### Changed

- The extraction loop now works on any local image record rather than manifest images specifically,
  which is what lets field photographs reuse it without a second copy of the batching, validation
  and resume logic.

### Fixed

- Corrected the field guide and `AGENTS.md` section 5.8, which both described the core route's
  eight controls as statues "neither backbone has ever confused". Four of them — Syzyfki,
  Capgeminiusz Programista, Kowal and Śpioch — draw one to three top-1 errors each. A control is a
  statue in no confused family, and the contamination is now named where it matters: it makes the
  cohorts look more alike, so it understates a concentrated drop rather than manufacturing one.
- Corrected the coordinate-validation count in `RESULTS.md`'s limitations, which said the derived
  positions were checked against "the 23 that have both". 23 is how many classes Wikidata places;
  21 of those also carry enough photograph coordinates to derive a position, and 21 is what the
  section itself, `AGENTS.md` 5.7 and the 0.6.0 entry all say. Recomputed from the manifest: 21
  classes, median error 8.5 m, worst 147 m, 14 within 50 m.
- Noted in `AGENTS.md` 5.7 that its measured 295 placed classes became 294 when the drift check
  dropped a one-degree latitude typo in the same release, so the decision record no longer reads
  as if it disagreed with `RESULTS.md`.
- Refreshed the three documentation sections that had drifted behind the last two releases, so a
  reader is not told the repository is something it stopped being at `0.4.0`:
  - `AGENTS.md` 12.1, the handoff a future contributor reads first, was dated 2026-09-04 and still
    said 23 classes carry coordinates and the rest carry none. Every figure in it is now recomputed
    from the tracked artifacts: 469 approved and 13 rejected category mappings (it claimed 478 and
    4), 294 placed classes, 1,958 staged images across 458 classes, and 152 below the three-image
    threshold. The stale counts from the 23-class era are gone; the live image-review decisions,
    the cross-label duplicate lesson and the display-name overrides are kept, because the reasoning
    behind each is not written down anywhere else.
  - `AGENTS.md` 9's repository tree was missing eighteen modules, `docs/`, `RESULTS.md` and three
    data artifacts. It now lists what is actually there, and every path in it was checked to exist.
  - `README.md`'s project status stopped at `0.4.0` and read as a scaffold inventory. It now says
    what each release since closed, and states plainly that one question remains open and needs a
    day in Wrocław rather than more code.
- `RESULTS.md`'s dataset section said 478 of 482 category mappings were approved; the tracked
  review file says 469 approved and 13 rejected. Verified by rebuilding the manifest from the
  current review files, which reproduces the published 306 classes and 1,691 images exactly, so
  469 is the number behind the results.
- `AGENTS.md` 8 no longer describes the pool-size limitation in terms of a 23-class dataset that
  has not existed since `0.5.0`. The question is marked done, with the data question it leaves
  behind stated in current numbers.

## [0.7.0] - 2026-09-05

### Added

- Added `krasnal-id experiment camera-gap`, a lower bound on the query-domain gap that needs no
  fieldwork. Commons records each file's EXIF camera, so the 51 references shot on phones become
  queries and are scored against the same reference set as the 1,565 shot on cameras. **DINOv2
  loses 5.3 top-1 points and CLIP 15.6**; CLIP's intervals do not overlap, DINOv2's do, so its gap
  points the right way but 51 queries cannot prove one.
- Added `krasnal-id data camera-metadata`, writing `data/discovery/camera-metadata.json`. It sits
  outside the staging chain deliberately: anything added to `fetched-images.json` invalidates the
  manifest, the split, twelve result artifacts and the demo, and a camera model is read by one
  analysis rather than used to build the dataset. `AGENTS.md` 5.9 records the trade and when to
  reverse it.
- Added `RESULTS.md` section 8, reporting the confounds beside the result: phone queries belong to
  classes with *more* reference photographs (median 7 against 6), are not lower resolution, and
  are no likelier to be in an already-confused class — so the gap is not explained by which
  statues they happen to show.


## [0.6.0] - 2026-09-05

### Added

- Added derived coordinates, taking the geographic experiment from 23 of 306 classes to **294**.
  Wikidata's `P625` places only 23, so the rest are placed from the median of their own
  photographs' camera positions — 73.5% of Commons files here carry one. Validated against the 21
  classes that have both: median error **9 m**, against a smallest pool radius of 171 m.
- Added `DwarfRecord.coordinate_source`, so a derived position can never be mistaken for an
  authoritative one. The schema rejects a coordinate with no source, a source with no coordinate,
  and a Wikidata source on a dwarf that has no Wikidata item. Wikidata always wins where both
  exist, and `ImageRecord.coordinates` carries the per-photograph position it is derived from.
- Added pair separation to the confusion analysis: `ConfusionPair.separation_metres`, median
  separations for confused and merely-competing pairs, the share of each within 100 m and 300 m,
  and a rank statistic for whether proximity predicts confusion at all.
- Added `krasnal_id.geometry` and `krasnal_id.statistics`, so the data pipeline does not import an
  experiment to measure a distance and two experiments do not share a rank statistic by importing
  each other.
- The published demo detects 29 co-located groups where it previously found 2, because 294 statues
  are placed rather than 23, so its warning that a top match belongs to a co-located installation
  now fires where it should.

### Changed

- **The geographic finding now holds city-wide.** At 23 classes it rested on six statues in one
  themed installation; across 294 it holds at every measured pool size for both backbones, and the
  arm is exact rather than sampled, so it is not seed noise. DINOv2 peaks at −0.89 points at a pool
  of five, CLIP at −2.12 at a pool of ten.
- **The co-location mechanism is now measured rather than inferred, and it is weaker than the
  earlier wording implied.** Every competing dwarf pair carries a ground distance. Confused pairs
  are about twice as likely as merely-competing pairs to stand within 100 m (11.9% against 5.5%
  for DINOv2), but the enrichment decays to 1.2x by 300 m, the whole-population rank statistic is
  0.518, and 88% of confused pairs stand more than 100 m apart. Co-location acts at the scale of a
  shared plinth, not a shared neighbourhood, and explains a minority of confusion — enough to
  produce the observed penalty, not enough to be described as its main cause. `RESULTS.md` section
  3 and the published page are corrected accordingly.
- Extended the geographic ablation's pool sizes to 294, where it previously stopped at 23.
- Rewrote `RESULTS.md` section 3 and the published page's location section around the new result,
  including that most positions are inferred rather than stated.

### Fixed

- Dropped a derived coordinate that could not be a real position: one class's only geotagged
  photograph carried a one-degree latitude typo, placing it 111 km from Wrocław and setting the
  full-pool radius for every query. The bound lives in the thresholds config, the reference point
  is the median of the placed dwarves rather than a hardcoded city centre, and Wikidata
  coordinates are exempt from the check.


## [0.5.0] - 2026-09-04

### Added

- Rebuilt the dataset Commons-first: **306 classes and 1,691 images**, from 23 and 146. 482
  category mappings carry a decision, 1,958 images staged across 462 categories, and 306 clear the
  three-image threshold. 23 classes come from Wikidata and carry coordinates; 283 are
  Commons-only and carry none.
- Added `RESULTS.md` section 7, recording which conclusions the 13x larger dataset overturned
  rather than quietly replacing the numbers, and `AGENTS.md` 7.3 with the same for contributors.
- Extended the pool-size ablation to N=306, where it previously stopped at 23 and section 7.1 had
  to warn against extrapolating past it.

### Changed

- **Open-set rejection is retired as a working feature.** It reported 4.1% false acceptance at 23
  classes and reports 38.3% at 306, because an absent statue now has 305 chances to find a
  lookalike rather than 22. DINOv2's AUROC falls from 0.969 to 0.896. The 0.4.0 headline was an
  artefact of the small pool and the mechanism predicts no recovery at city scale.
- **The decay rates are revised in opposite directions.** DINOv2 loses 0.79 points per doubling
  against a predicted 0.96; CLIP loses 2.06 against a predicted 1.76 and is still worsening at the
  edge of the range. Section 7.1's warning that a small pool flatters the result was half right,
  and applying it as a blanket rule would have mispredicted DINOv2.
- **A linear probe now helps CLIP by 3.1 top-1 points and still does nothing for DINOv2** (−0.1),
  where at 23 classes both showed the same +0.7 non-effect. Class prototypes now cost about 8
  points to both.
- The dominant confusion cluster is the three *Słupniki* pillar dwarves rather than the
  water-themed trio, and the embedding projection selects the same families using centroid
  distance alone.

### Fixed

- The geographic ablation refused to run when any dwarf lacked coordinates, which made it
  unrunnable on a Commons-first manifest where 283 of 306 do. It is now scoped to the located
  subset and records `located_dwarfs`, `manifest_dwarfs` and `located_fraction`, so a result
  measured over 7.5% of the pool cannot be read as covering it.
- `measure_pool_size` built an index of images per dwarf that nothing used, and raised `KeyError`
  as soon as the geographic arm passed a dwarf universe narrower than the embedding matrix.
- The embedding projection named all 306 classes, overlapping into a wall of leader lines. Past a
  24-class budget it now names only the classes nearest another class and greys the rest.


### Added

- Added Commons-first discovery as `krasnal-id data query --include-commons`, which enumerates the
  481 per-dwarf categories under `Category:Dwarves in Wrocław by name` as a second source and
  merges them with Wikidata. Measured on 2026-09-04 it discovers 482 classes against the 41
  Wikidata alone can reach, because only 44 Wikidata items exist for those 481 categories. A
  Wikidata record wins any category it claims, since it carries a QID and the P625 coordinates the
  geographic ablation needs. Both sources share one discovery artifact and one recorded query
  hash, so every downstream consistency check keeps working, and without the flag the command
  reproduces the Wikidata-only artifact exactly.
- Added `data/discovery/commons-categories.json` caching with the same refresh, recovery and
  query-identity validation as the SPARQL cache, plus `unexpected_category_name`,
  `duplicate_category_slug` and `claimed_by_wikidata` audit reasons. A slug collision is excluded
  rather than merged, because two statues would otherwise share one image directory.

### Changed

- Widened the dataset contracts for statues Wikidata has no item for: `wikidata_url` is now
  optional, `DWARF_ID_PATTERN` is shared by the manifest and both review files, and a
  Commons-identified dwarf may not claim a Wikidata item or coordinates. Synthetic test fixtures
  became one-based, because `Q0` is not a QID and the pattern now says so.


## [0.4.0] - 2026-09-03

### Added

- Added open-set rejection as `krasnal-id experiment open-set`, answering the limitation
  `RESULTS.md` recorded: a query of a statue outside the reference set no longer has to be given a
  nearest neighbour. It scores two equal populations built from the manifest without sampling —
  the leave-one-out folds, and the same images queried against a gallery holding none of their own
  dwarf — and reports threshold-free AUROC alongside operating points whose thresholds are
  calibrated leave-one-class-out, so no query helps set the bar it must clear. A threshold fitted
  on all the data is reported too, labeled `in_sample`, as an optimistic reference. Per-dwarf rows
  record which statues slip through when removed and which statue covered for them.
- Added `OpenSetRejectionResult` and `DwarfRejection` to the experiment contracts, the
  `experiment=open_set` Hydra group with configurable acceptance targets, and the
  `thresholds.open_set_top_rejections` cap on reported per-dwarf rows.
- Added the rejection tradeoff figure as `krasnal-id visualize open-set`, drawing every saved
  backbone's curve on one axis with its calibrated operating points marked on it. The open-set
  artifact now carries that curve as `RejectionOperatingPoint` rows, swept over every observed
  score, so any threshold's cost is readable from the artifact rather than only the three
  configured targets. The curve is descriptive and the metrics stay the calibrated source of
  truth. Published to `docs/figures/open-set-rejection.png` under the section 6.4 policy.
- Added `RESULTS.md` section 6 with the measured outcome: DINOv2 separates present from absent
  statues at 0.969 AUROC and rejects 95.9% of unknown queries while still identifying 89.0% of
  known ones, costing about 3.4 points against its closed-set top-1 of 95.9%. CLIP reaches 0.898
  AUROC and has no useful operating point. Being reliably identifiable turns out not to imply
  being reliably rejectable: *Kowal* is never misidentified while present but is the worst dwarf
  to reject once removed.

## [0.3.0] - 2026-09-03

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

- Added dark-ground raster plates to `docs/brand/`: a 1280x640 `social-preview.png` carrying the
  tagline, a plain variant without it, and a 2x stacked lockup for slides.
- Added the project identity under `docs/brand/`: a mark and horizontal and stacked lockups, each
  in light and dark variants, with the wordmark converted to outlines so the files render without
  a font installed, plus the generator that produces them and the usage rules.
- Added a published findings summary linked from the README: an illustrated short read of the
  headline results, with `RESULTS.md` remaining the complete written record.
- Added `RESULTS.md`, the written research record: dataset construction, the four experiments,
  what the numbers mean, limitations, and a reproduction script.
- Added the accuracy-versus-pool-size figure as `krasnal-id visualize ablation`, drawing every
  saved ablation curve on one log-scaled axis with seed-spread error bars and the fitted slope.
- Added tracked `docs/figures/` holding the figures published in `RESULTS.md`, under the result
  publication policy now recorded in `AGENTS.md` section 6.4.
- Added the Gradio demonstration as `krasnal-id demo [--top-k N] [--port P] [--share]`: upload a
  photograph to see ranked candidate dwarves with similarity scores and their closest reference
  photographs, with the reference set loaded once per session.
- Published the demo at <https://turhancan97.github.io/krasnal-id/> from `docs/` on `main`, and
  linked it from the README in place of the retired findings artifact.
- Added a static browser demo under `docs/`, deployable to GitHub Pages: upload or photograph a
  statue and identify it entirely on-device, with attribution beside every match and a note when
  the top match belongs to a co-located installation.
- Added the demo's build at `docs/demo/build.mjs`, producing the reference vectors, thumbnails and
  attribution metadata. It runs the same library, model and dtype the browser uses, re-scores the
  leave-one-out protocol on the vectors it ships, and records self-test probes so the page can
  verify its own pipeline in a real browser.
- Added `docs/resize.mjs`, one Lanczos-3 resampler imported by both the browser and the build, so
  a query and a reference are preprocessed by identical arithmetic rather than by two platforms'
  differing built-in resamplers.
- Added the geographic ablation as `krasnal-id experiment geo-ablation`: candidate pools built
  from real Wikidata coordinates by nearest-neighbour proximity, scored against randomly sampled
  pools of matched size, with the median and maximum radius each pool spans.
- Added a `geo_ablation` Hydra experiment group and its validated `GeoAblationConfig`.
- Added the trained-classifier comparison as `krasnal-id experiment probe`: per-class prototype
  and per-fold linear-probe classifiers scored against a retrieval arm on the same folds, with
  Wilson intervals, mean reciprocal rank, and an explicit top-1 gain over retrieval.
- Added a `probe` Hydra experiment group and its validated `ProbeExperimentConfig`.
- Added single-image retrieval as `krasnal-id retrieve <image> [--top-k N]`, reporting ranked
  candidate dwarves with similarity scores and the reference image each matched, reusing a cached
  vector for identical file content and withholding every byte-identical copy of the query.
- Added confusion analysis reporting directed most-confused dwarf pairs with query counts,
  misidentification counts, and mean cosine margins, plus a `ConfusionPair` contract, summary
  margin metrics, and a wired `krasnal-id experiment confusion` command.
- Added UMAP and t-SNE embedding visualization with lazy analysis-dependency loading, a
  headless backend, deterministic seeding, per-class colors and markers beyond a twenty-color
  palette, decluttered centroid labels with leader lines, and a wired
  `krasnal-id visualize embeddings` command.
- Added the candidate-pool-size ablation: per-pool top-1 and MRR across seeds with the observed
  seed spread as error bars, a fitted accuracy-points-per-doubling slope, dataset-aware pool-size
  resolution, atomic result artifacts, and a wired `krasnal-id experiment pool-ablation` command.
- Added the full-pool retrieval baseline: dwarf-level and image-level top-k accuracy with 95%
  Wilson intervals, mean reciprocal rank, split/manifest hash verification, atomic result
  artifacts under `results/`, and a wired `krasnal-id experiment baseline` command.
- Added a shared embedding store that loads manifest-ordered cached vectors and holds the single
  definition of the embedding cache key used by both extraction and evaluation.
- Added cosine-similarity k-NN retrieval that rescales inputs to unit length, breaks equal
  similarities by ascending image ID for reproducible rankings, clamps similarities into the
  validated range, truncates to `top_k`, and returns a whole small pool rather than failing.
- Added deterministic leave-one-out split generation, validated split contracts, atomic split output,
  lazy DINOv2 and CLIP adapters, normalized resumable embedding caches, strict manifest-image
  validation, and offline fake-backbone extraction coverage.

### Changed

- Removed the CLI placeholder helper, which no command referenced once every stage was
  implemented.
- Set the default linear-probe regularization to `C=100`. At the conventional `C=1.0` the probe
  underfits L2-normalized embeddings badly, scoring 66% top-1 against 96%.
- Declared `threadpoolctl` in the `analysis` extra and hold BLAS to one thread while fitting
  per-fold classifiers, which took the probe sweep from over six minutes to twelve seconds.
- Extended strict `mypy` coverage to `tests`, and fixed the 35 errors that surfaced: fixtures now
  build `HttpUrl`, `datetime` and `Path` values explicitly instead of relying on Pydantic
  coercion, the CLIP output-shape test uses typed fakes rather than stacked ignores, retry tests
  patch the `time` module directly, and the CLI command tree is walked through a typed helper.
- Made the baseline's Wilson interval helpers public so confusion analysis reuses them.
- Installed the analysis extra in CI so the visualization code is exercised there.
- Recorded the dataset-scale decision in `AGENTS.md` §7.1: keep the >=3-image threshold and
  report degradation per doubling, with the measured evidence for rejecting a lower threshold
  and for treating the excluded duplicates as unrecoverable.
- Widened the default ablation pool sizes to `[2, 3, 5, 8, 10, 15, 20, 50, 100]`; sizes above the
  available class count are skipped with a warning.
- Indexed `EmbeddingMatrix` image IDs, which the ablation resolves hundreds of thousands of times.
- Moved the shared synthetic dataset builders used by evaluation tests into `tests/helpers.py`.
- Pinned CLIP to the immutable `c237dc49a33fc61debc9276459120b7eac67e7ef` safetensors-conversion
  revision so the loaded weights match the revision recorded in the embedding cache key.
- Added `torchvision` to the `ml` extra, which `transformers` requires for image processing.
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

- Fixed CLIP feature extraction to read `pooler_output` from the vision-output object that
  `transformers` 5 returns from `get_image_features`, with regression coverage for both shapes.
- Made CLI help tests inspect generated command metadata instead of environment-dependent Rich-rendered text.
- Fixed Commons response parsing to accept scalar-valued extension metadata returned by the
  live API while continuing to require string-valued attribution and license fields.
- Fixed Commons downloads whose responsive derivative exceeds the requested bound by locally
  downscaling static images to the configured maximum while preserving their aspect ratio.

## Current state

This section is a rolling summary of what the repository does today, kept outside the version
sections above so that a new contributor or agent can read one place to orient themselves.

- Repository scaffolding, Wikidata discovery, and reviewed Commons image acquisition are
  complete, with automated checks passing.
- Audited manifest construction is implemented and produces 23 classes and 146 images from
  the current local artifacts.
- Deterministic evaluation split generation and resumable DINOv2/CLIP embedding extraction are
  implemented and have been run end to end on the local dataset, caching 146 normalized
  768-dimensional DINOv2 vectors and 146 normalized 512-dimensional CLIP vectors.
- Cosine k-NN retrieval and the full-pool baseline are implemented. On the current 23-class,
  146-image dataset the baseline reports dwarf-level top-1 of 95.9% for DINOv2 and 92.5% for
  CLIP, with top-5 at 99.3% and 98.6% and MRR at 0.9714 and 0.9506.
- The candidate-pool-size ablation is implemented and run. Top-1 falls from 98.9% at a pool of
  two to 95.9% at the full pool of 23 for DINOv2, and from 98.9% to 92.5% for CLIP, giving fitted
  slopes of -0.96 and -1.76 accuracy points per doubling.
- Confusion analysis and embedding visualization are implemented and run. DINOv2 misidentifies
  6 of 146 queries against CLIP's 11, and both backbones agree that the Puszczajacy Stateczki,
  Zbierajacy Wode, and Karmiacy Ptaki water-themed dwarves are the systematically confused
  cluster, which the projections show as a single tight neighborhood.
- The geographic ablation is implemented and run, answering the question `AGENTS.md` section 5.2
  left open. Real proximity pools are consistently harder than random pools of the same size
  (DINOv2 -1.6 points at a pool of five, CLIP -2.5), because six of the 23 dwarves stand within
  one metre of each other as one themed installation and are the same statues the confusion
  analysis flags.
- Single-image retrieval is implemented and reproduces the confusion finding interactively: a
  Puszczajacy Stateczki query ranks Zbierajacy Wode first under both backbones, with the water-
  themed cluster filling the top three candidates.
- The trained-classifier comparison is implemented and run. Neither trained method meaningfully
  beats retrieval: the linear probe gains 0.7 top-1 points over retrieval for both backbones
  (DINOv2 96.6% against 95.9%, CLIP 93.2% against 92.5%) while class prototypes lose 2.7 and 2.1
  points, and every difference sits inside the confidence intervals.
- The Gradio demonstration is implemented, completing the v0.1-v0.3 build order. Every
  scaffolded stage now has real behavior and no module raises `NotImplementedError`.
- A static browser demo is published at <https://turhancan97.github.io/krasnal-id/> from
  `docs/` on `main`, embedding a query with a quantised ONNX CLIP on the visitor's own device
  and scoring 91.8% top-1 and 98.6% top-5 on the vectors it ships.
- Everything above is released as `0.3.0`. The planned build order is finished, so further
  work is a new research direction rather than a remaining stage; the limitations listed in
  `RESULTS.md` name the open questions.
- Open-set rejection was implemented and released as `0.4.0`, then **retired as a working feature
  by the larger dataset**: false acceptance rises from 4.1% at 23 classes to 38.3% at 306. The demo
  still shows its ranking unconditionally, which is now the honest behavior rather than a gap.
- The dataset is rebuilt Commons-first at **306 classes and 1,691 images**, released as `0.5.0`,
  and every experiment, figure and the published demo are regenerated at that scale. `RESULTS.md`
  section 7 records which conclusions the rebuild overturned.
- Coordinates are derived from photographs as well as Wikidata, so the geographic experiment covers
  294 of 306 classes rather than 23, and the co-location mechanism behind it is measured rather
  than inferred. Released as `0.6.0`.
- The query-domain gap now has a **lower bound** rather than being wholly unmeasured: the 51
  references shot on phones lose 5.3 top-1 points for DINOv2 and 15.6 for CLIP when used as
  queries. Released as `0.7.0`. It does not close the question — these are still Commons uploads,
  and a casual snapshot is harder than any photograph in this dataset.
- **Fieldwork remains the only clean measurement**, and `data/field-guide.md` holds a nineteen-statue
  route chosen from the data: every member of the three clusters both backbones confuse, plus
  controls in the same streets. `data/field-queries/` has the directories waiting.

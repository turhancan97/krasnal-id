# Doppeldwarf-306 explainer — review package

This is the first approval stage of a two-minute explainer. Open [review.html](review.html) to
review the script, eight storyboard frames, two 15-second voice samples and a 10-second motion
study. The complete video has **not** been rendered and the project page has **not** been changed.

The story is 226 words, with a 120-second production budget including eight seconds of credits.
Timings in `story.json` are budgets, not measured timings for the full narration. The short preview
uses `af_heart` provisionally; the final voice has not been chosen. Captions are already burned into
the preview, with matching downloadable SRT and WebVTT. The samples contain the same words but use
each voice's natural delivery; end padding makes both files exactly 15 seconds without truncation.

## Review locally

From the repository root:

```bash
python3 -m http.server 8765 --directory .
```

Open `http://localhost:8765/video/explainer/review.html`. A repository-root server is necessary
because the review's fonts and source links refer to existing repository files. This review page
is outside `docs/` and is not part of the published website. Python's basic server is sufficient
for viewing and playback. Seek testing needs an HTTP server with byte-range support; the browser
checks use a local range-capable server, matching the static hosting transport.

Please review the narration and factual emphasis, the motion and light-background visual style,
and the two voices. Choose `af_heart` or `am_michael`, or request a revision. Listen specifically to
Wrocław, Doppeldwarf-306 and cosine similarity. Technical checks cannot approve pronunciation.

## Sources and rights

- `story.json` is the editable narration, scene budget, visual treatment and exact eight-colour
  palette. `script.md` and `review/transcript.txt` are generated from it.
- `claims.json` holds existing research table values and source-file digests. The build refuses
  changed sources until the claims have been reviewed deliberately. The pool-size plot uses the
  rounded published table, without inventing intermediate measurements or uncertainty bands.
- `credits.json` names the four existing teaser photographs with author, licence, source,
  modifications and checksums. See [CREDITS.md](CREDITS.md) for linked attribution and depicted-work
  disclosure. Original code is MIT; photograph terms stay separate.
- `model-lock.json` pins the Kokoro repository revision and SHA-256 of its configuration, weights
  and both stock voices. The model and voice files remain outside this repository.
- `review/audio-receipt.json` records voice, speed, speech length and audio hashes.
  `review/checks.json` records actual frame, codec, duration and source-integrity checks.

The vector directions, similarity bars, geographic layout and feature connections are explicitly
illustrative. Research location results are simulations over 294 placed statues. The browser
identifier has its own measurements and its own close-call calibration. The script does not
promise that field photographs score like the Commons dataset, or that all error comes from
lookalike families or photographer style.

## Reproduce in an isolated environment

Do not install these packages into the research environment. On Linux, reproduce the native
prefix using `native-linux-64.lock`; it records the exact conda-forge builds used here.
For another platform, start from `native-environment.yml`, then resolve a native lock there.

```bash
conda create --prefix /path/to/explainer-env --file video/explainer/native-linux-64.lock
export EXPLAINER_ENV=/path/to/explainer-env
export PKG_CONFIG_PATH="$EXPLAINER_ENV/lib/pkgconfig:$EXPLAINER_ENV/share/pkgconfig"
uv pip sync --python "$EXPLAINER_ENV/bin/python" --torch-backend cpu video/explainer/requirements.lock
```

`requirements.lock` locks Python packages with hashes; its CPU-only torch index is explicit.
The original inputs are `requirements.in`. Only intentional dependency updates should resolve it
again with `uv pip compile --torch-backend cpu --generate-hashes`.

On gmum, use your own running four-CPU allocation:

```bash
export EXPLAINER_ENV=/shared/results/common/kargin/projects/krasnal-id/cache/explainer-env
export EXPLAINER_CACHE=/shared/results/common/kargin/projects/krasnal-id/cache/explainer-review
video/explainer/run_review.sh YOUR_CPU_JOB_ID all
```

The wrapper limits threads, runs inside the allocation, and stores models and intermediate renders
in the external cache. The existing research package, data, results, embedding binaries and model
descriptors are never written. `sources` and `docs` need only the Python standard library:

```bash
python3 video/explainer/build_review.py sources
python3 video/explainer/build_review.py docs
```

The individual allocated stages are `voices`, `frames`, `preview` and `check`. They must run in
that order after a source change. The eight frames use Manim Community with bundled TeX Gyre Heros
fonts; no LaTeX installation or external animation plugin is needed. Only selected review outputs
are tracked; all intermediate frames, downloaded weights and render caches stay outside source.

## After approval

Implement the complete film from the approved script and selected voice, first measuring audio
and subtitle timing. Keep the full film at or below 120 seconds; trim prose rather than rush the
voice. Export 1920×1080 at 30 fps, H.264/AAC, fast-start MP4, with a target below 25 MB, plus captions,
transcript and linked credits. The final native player belongs between the existing teaser and
abstract, with poster, controls, inline playback, no autoplay and `preload="none"`.

That second stage includes complete soundtrack and transition review, mobile caption checks,
playback/seeking/fullscreen/keyboard tests and project-page screenshots in both themes. There is
deliberately no full-film render or publication command in this review builder.

# Doppeldwarf-306 explainer

The owner approved the 226-word script, eight storyboard frames, motion style and **af_heart**
voice. The complete film is `docs/media/explainer.mp4`, embedded between the existing teaser and
abstract on the project page. It is 120 seconds at 1920×1080, 30 fps, H.264/AAC, with fast-start
playback, natural-speed narration and burned-in English subtitles. Credits occupy the final eight
seconds. SRT, WebVTT, a transcript and linked photograph/narration credits accompany the movie.

The original [review package](review.html), two 15-second voice auditions and ten-second motion
study remain as the approved reference. `story.json` preserves that review snapshot;
`approval.json` is the current approval record, bound to the snapshot's SHA-256. The narration
text has not been edited after approval.

## View locally

From the repository root, run `python3 -m http.server 8765 --directory docs` and open
`http://localhost:8765/#explainer`. Python's basic server is sufficient for playback. Seek tests
need an HTTP server with byte-range support, matching the static hosting transport.

For the historical review page, serve the repository root instead and open
`http://localhost:8765/video/explainer/review.html`. That page refers to fonts and source documents
outside `docs/` and is not part of the published site.

## Source and provenance

- `story.json` defines the unchanged narration, original timing budget and exact eight-colour
  palette; `script.md` is its readable review copy.
- `approval.json` records the owner's choice and refuses a changed story. `pronunciations.json`
  supplies explicit phonemes for names/acronyms while keeping the original subtitle words.
- `film_audio.py` synthesizes seven chapters at speed 1.0, caches each by spoken text/model/voice,
  and allocates whole video frames around measured speech. Spare frames fill the 112-second
  content budget, followed by eight stationary seconds of credits. Speech is neither cut nor
  accelerated; narration exceeding the budget is refused.
- `production/timing.json` records chapter boundaries, synthesis word timestamps and caption
  cues. `film_scenes.py` uses those word positions to time the visual highlights and transitions.
- `film.py` exports the movie, captions, poster and accessible companion pages, then checks actual
  codec, duration, dimensions, frame count, byte size, subtitle words and fast-start layout.
- `claims.json` holds the existing measured pool-size values, source hashes and protected demo/
  research inputs. Only the project-page edit is permitted by the full-film approval; the other
  protected hashes must still match. `docs/media/explainer-build.json` records the shipped assets.
- `credits.json` selects the four existing teaser photographs with full attribution, source,
  licence, modification and checksum fields. See [CREDITS.md](CREDITS.md). Photograph licences do
  not grant rights in the depicted sculptures; the existing disclosure and removal path apply.
- `model-lock.json` pins the exact Kokoro configuration, weights and stock voices. Models and
  intermediate WAVs/renders remain in the external task cache, outside tracked source.

The vector directions, similarity bars, geographic positions and feature connections are labelled
illustrative. Pool curves contain the published measured points. Location findings are labelled
as simulations over 294 placed statues. Research uncertainty evaluation and the browser demo's
close-call hint are distinguished. The film makes no street-photo accuracy claim.

## Reproduce without changing the research environment

Create a standalone prefix using the exact native Linux lock. Other platforms can start from
`native-environment.yml` and resolve their own native lock.

```bash
conda create --prefix /path/to/explainer-env --file video/explainer/native-linux-64.lock
export EXPLAINER_ENV=/path/to/explainer-env
export PKG_CONFIG_PATH="$EXPLAINER_ENV/lib/pkgconfig:$EXPLAINER_ENV/share/pkgconfig"
uv pip sync --python "$EXPLAINER_ENV/bin/python" --torch-backend cpu video/explainer/requirements.lock
```

Python dependencies are hash-locked independently of the root project, with CPU-only torch.
Manim uses bundled TeX Gyre Heros fonts without a LaTeX installation. On gmum, run inside your own
four-CPU allocation:

```bash
export EXPLAINER_ENV=/shared/results/common/kargin/projects/krasnal-id/cache/explainer-env
export EXPLAINER_CACHE=/shared/results/common/kargin/projects/krasnal-id/cache/explainer-review
video/explainer/run_film.sh YOUR_CPU_JOB_ID all
```

Individual build stages are `audio`, `assets`, `render`, `check`, `inspect` and `delivery`, in that order.
`inspect` samples every caption and chapter transition; `delivery` compares decoded AAC with the
approved master audio. Both run after the final encode and write verification receipts.
The optional `transcribe` stage reruns the pinned local Whisper verification over the cached WAVs. The wrapper limits
threads, uses the existing allocation and stores model/render caches outside the repository.
The build never writes dataset, results, embedding binaries or model descriptors.

Offline protocol tests need only the Python standard library:

```bash
python3 -m unittest discover -s video/explainer -p test_film.py -v
```

To rebuild the historical review documentation after the approved page edit, use
`python3 video/explainer/build_review.py docs --allow-page-change`. That flag verifies the approval
and exempts only the project page from the original review snapshot guard. `run_review.sh` remains
the original review-stage recipe; it deliberately checks the original page unless the flag is
passed to the Python stages directly.

## Validation records

The media receipt records the actual delivered movie and its hashes. Final export takes audio
directly from the lossless master, avoiding the encoder delay of Manim's intermediate soundtrack.
To rerun Chromium checks, install Playwright in an external scratch directory and serve `docs/`
with byte-range HTTP support, then run:

```bash
PLAYWRIGHT_MODULE=/path/to/scratch/node_modules/playwright \
EXPLAINER_QA=/path/to/external/screenshots \
EXPLAINER_SITE_URL=http://127.0.0.1:8765/ \
node video/explainer/verify_browser.cjs
```

The browser receipt includes the movie hash and screenshot paths. Browser checks exercise
playback, byte-range seeking, keyboard play/pause, fullscreen, downloads and the four desktop/
mobile theme layouts. Frame samples cover each chapter and transition. Audio signal/timing and
an independent local automatic transcription are separate checks; automatic transcription is
not a human listening pass. Nothing in these scripts pushes or publishes the website.

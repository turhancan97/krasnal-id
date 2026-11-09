# Demonstrations

[Back to the README](../README.md) · [All guides](README.md)

## Interactive demonstration

The published demo at <https://turhancan97.github.io/krasnal-id/demo.html> is a static page: it embeds an
uploaded photograph with a quantised ONNX backbone in the visitor's own browser and ranks it
against reference vectors built by the same code, so nothing is uploaded and there is no backend.
The [project page](../docs/index.html) introduces the research and dataset; the identifier lives
at [`demo.html`](../docs/demo.html). The complete written record stays in [RESULTS.md](../RESULTS.md).

![The published identifier after running one of its example photographs: a DINOv2/CLIP model switch showing 56 MB at 93.2% and 64 MB at 82.4%, the query photograph, the five closest dwarves with similarity scores and photographer credits, and a warning that this statue stands with six others installed as one themed group.](../docs/figures/demo-ui.png)

*The identifier, mid-result. Everything shown was computed in the browser.*

**The page runs either backbone, and the switch is a comparison rather than a size tier.** DINOv2
is the default — it is the pipeline's own model, 56 MB at `q4`, and 93.2% top-1 on the vectors it
ships. CLIP is offered beside it at 64 MB and 82.4%, which makes it the *larger* download and the
less accurate one; it is there because the gap between the two is what most of this project's
findings are about, and switching re-ranks the photograph already on screen so a visitor can see
that gap on their own photograph instead of reading it off a chart. CLIP's weights and its
vectors are fetched only if it is selected, so the default page load is unchanged.

Rebuild the data after the manifest changes:

    cd docs/demo && npm install && node build.mjs

The build embeds every reference photograph with both backbones, writes one
`assets/references-<backbone>.bin` per backbone beside the shared `assets/references.json`, and
re-scores the leave-one-out protocol on exactly the vectors it just wrote — the accuracy on each
button is that measurement, never a figure copied from the research pipeline. It also records
probes so the page can verify itself: `?selftest=1` reports cosine agreement with the build for
the selected backbone, and `?selftest=full` re-embeds all 1,691 reference photographs in the
browser and scores the protocol there.

Both sides read `docs/backbones.mjs`, which is the only place the two models are described. They
differ in three ways that must agree between the build and the page or every cosine is
meaningless: the model class, the pooling (DINOv2's CLS token against CLIP's `image_embeds`), and
the shortest edge (256 against 224).

The build sets `intraOpNumThreads` explicitly. onnxruntime sizes its thread pool from the host's
core count rather than the cpuset it is confined to, and on a shared machine that difference cost
**7502 ms per image against 426 ms** — seven hours against twenty-four minutes for a full build.
Set it from `nproc` and re-measure on the machine that runs the build.

There is also a local Gradio version, which uses the research pipeline rather than the browser one:

    uv sync --extra demo
    uv run krasnal-id demo
    uv run krasnal-id demo --top-k 3 --port 7860 --override backbone=clip

Upload a photograph to see the ranked candidate dwarves, their similarity scores, and the
reference photographs each one matched. The manifest and cached vectors load once per session
rather than per query. Uploading a photograph that is already in the dataset withholds its own
reference copies and says so, so it cannot simply match itself.

The demo serves locally and reports nothing to any external service. It writes its scratch files
to a per-user directory rather than the shared `/tmp/gradio`, which fails on a multi-user machine
where another account created that path first; set `GRADIO_TEMP_DIR` to override.

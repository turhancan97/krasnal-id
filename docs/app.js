/**
 * In-browser identification, on either of the two backbones.
 *
 * The reference vectors in assets/references-<backbone>.bin were produced by
 * this same library, model and dtype, and with the same antialiased
 * pre-downscale applied below. That matters: measured while this page ran CLIP,
 * Python-preprocessed references cost 3.5 points of top-1 against
 * browser-preprocessed queries, and skipping the pre-downscale costs another
 * two, because transformers.js resizes a large photograph in one aliasing step.
 *
 * DINOv2 is the default: it is the model the research pipeline runs, and at q4
 * it is both smaller and 10.8 points better than the CLIP it replaced. CLIP is
 * here as a comparison arm rather than a lighter option — switching re-ranks
 * the same photograph, and the gap between the two rankings is what most of
 * this project's findings are about. Its weights and vectors are fetched only
 * if a visitor asks for them.
 *
 * Nothing is uploaded. The photograph is decoded, scaled, embedded and compared
 * entirely on this device.
 */
import {
  AutoModel,
  AutoProcessor,
  CLIPVisionModelWithProjection,
  RawImage,
  env,
} from "https://cdn.jsdelivr.net/npm/@huggingface/transformers@3.8.1";
import { BACKBONES, DEFAULT_BACKBONE, backbone, unitVector, vectorFile } from "./backbones.mjs";
import { resizeToShortestEdge } from "./resize.mjs";

const TOP_K = 5;

// Resolved here rather than in backbones.mjs because the build loads
// transformers.js from npm and this file loads it from a CDN.
const CLASSES = { AutoModel, CLIPVisionModelWithProjection };

env.allowLocalModels = false;

const el = (id) => document.getElementById(id);
const statusText = el("status-text");
const statusRow = el("status");
const bar = el("bar");
const barFill = el("bar-fill");

let meta = null;
let current = DEFAULT_BACKBONE;
/** Per-backbone caches, so switching back is instant and never re-downloads. */
const vectorCache = new Map();
const modelCache = new Map();
const pending = new Map();
/** The last thing identified, so switching backbones can re-rank it. */
let lastQuery = null;
let objectUrl = null;

function say(message, isError = false) {
  statusText.textContent = message;
  statusRow.classList.toggle("err", isError);
}

function progress(fraction) {
  if (fraction === null) {
    bar.hidden = true;
    return;
  }
  bar.hidden = false;
  barFill.style.width = `${Math.round(fraction * 100)}%`;
}

/** The backbones this build actually shipped vectors for. */
function available() {
  return BACKBONES.filter((spec) => meta.backbones?.[spec.id]);
}

/** Load the shared metadata. Small, so always eager. */
async function loadMetadata() {
  const response = await fetch("assets/references.json");
  if (!response.ok) throw new Error("reference data is unavailable");
  meta = await response.json();
  if (!available().length) throw new Error("this build shipped no reference vectors");
  if (!meta.backbones[current]) current = available()[0].id;
}

/** Load one backbone's reference vectors, once. */
async function loadVectors(id) {
  if (vectorCache.has(id)) return vectorCache.get(id);
  const info = meta.backbones[id];
  const response = await fetch(`assets/${info.vectors ?? vectorFile(id)}`);
  if (!response.ok) throw new Error(`reference vectors for ${id} are unavailable`);
  const raw = new Float32Array(await response.arrayBuffer());
  const dim = info.dimensions;
  // references.json and the .bin files are fetched separately and carry no version
  // in their URLs, so a returning visitor can briefly hold a fresh one and a cached
  // other. Silently slicing a mismatch is the dangerous outcome: reads of the wrong
  // width run off the end and return short vectors, which score as plausible
  // nonsense rather than failing. Check the length instead.
  if (raw.length !== meta.images.length * dim) {
    throw new Error(
      `reference data is inconsistent: ${raw.length} floats for ` +
        `${meta.images.length} x ${dim}. A stale copy is cached — reload without cache.`,
    );
  }
  const vectors = [];
  for (let i = 0; i < meta.images.length; i += 1) {
    vectors.push(raw.subarray(i * dim, (i + 1) * dim));
  }
  vectorCache.set(id, vectors);
  return vectors;
}

/** Download one backbone's weights on first use, reporting progress. */
async function loadModel(id) {
  if (modelCache.has(id)) return modelCache.get(id);
  if (pending.has(id)) return pending.get(id);
  const spec = backbone(id);
  const task = (async () => {
    const seen = new Map();
    const onProgress = (item) => {
      if (item.status === "progress" && item.total) {
        seen.set(item.file, item.loaded / item.total);
        const mean = [...seen.values()].reduce((a, b) => a + b, 0) / seen.size;
        progress(mean);
        say(
          `Downloading ${spec.label} — ${Math.round(mean * 100)}% of ${spec.downloadMb} MB. ` +
            "This happens once.",
        );
      }
    };
    const ModelClass = CLASSES[spec.className];
    if (!ModelClass) throw new Error(`no model class named ${spec.className}`);
    const [processor, model] = await Promise.all([
      AutoProcessor.from_pretrained(spec.modelId, { progress_callback: onProgress }),
      ModelClass.from_pretrained(spec.modelId, {
        dtype: spec.dtype,
        progress_callback: onProgress,
      }),
    ]);
    progress(null);
    const ready = { processor, model };
    modelCache.set(id, ready);
    return ready;
  })();
  pending.set(id, task);
  try {
    return await task;
  } finally {
    pending.delete(id);
  }
}

/**
 * Decode, then downscale with the same resampler the build used.
 *
 * The canvas is only a decoder here. Its own scaling is browser-dependent and
 * measurably disagreed with the build (0.98 cosine, where 1.00 is wanted), so
 * the pixels go through resize.mjs instead and both sides match by construction.
 *
 * The edge is the backbone's own: DINOv2 wants 256 before its centre-crop,
 * CLIP wants 224, and using one for the other silently changes the input.
 */
async function readScaled(source, shortestEdge) {
  const blob = typeof source === "string" ? await (await fetch(source)).blob() : source;
  // Decode verbatim. Left to its defaults a browser may apply the display colour
  // profile and premultiply alpha, both of which shift pixel values away from
  // what the build saw and cost cosine agreement against the shipped vectors.
  const bitmap = await createImageBitmap(blob, {
    colorSpaceConversion: "none",
    premultiplyAlpha: "none",
  });
  const canvas = document.createElement("canvas");
  canvas.width = bitmap.width;
  canvas.height = bitmap.height;
  const context = canvas.getContext("2d", {
    willReadFrequently: true,
    colorSpace: "srgb",
    alpha: true,
  });
  context.imageSmoothingEnabled = false;
  context.drawImage(bitmap, 0, 0);
  bitmap.close?.();

  const pixels = context.getImageData(0, 0, canvas.width, canvas.height, {
    colorSpace: "srgb",
  });
  canvas.width = 0;
  canvas.height = 0;
  const scaled = resizeToShortestEdge(
    new Uint8ClampedArray(pixels.data),
    pixels.width,
    pixels.height,
    shortestEdge,
  );
  return new RawImage(scaled.data, scaled.width, scaled.height, 4);
}

async function embedSource(source, id = current) {
  const spec = backbone(id);
  const { processor, model } = await loadModel(id);
  const output = await model(await processor(await readScaled(source, spec.shortestEdge)));
  // Pooled the way this backbone's references were pooled — the CLS token for
  // DINOv2, the projection for CLIP. Anything else would not compare.
  return unitVector(spec.pool(output));
}

/** Rank distinct dwarves by their best-matching reference photograph. */
function rank(query, vectors) {
  const best = new Map();
  for (let i = 0; i < vectors.length; i += 1) {
    const reference = vectors[i];
    let dot = 0;
    for (let d = 0; d < query.length; d += 1) dot += query[d] * reference[d];
    const image = meta.images[i];
    const currentBest = best.get(image.dwarf);
    if (!currentBest || dot > currentBest.score) best.set(image.dwarf, { score: dot, image });
  }
  return [...best.values()].sort((a, b) => b.score - a.score).slice(0, TOP_K);
}

function renderHits(hits) {
  const list = el("hits");
  list.textContent = "";
  hits.forEach((hit, index) => {
    const item = document.createElement("li");
    item.className = index === 0 ? "hit top" : "hit";
    const credit = `${hit.image.author} · ${hit.image.license}`;
    item.innerHTML = `
      <span class="rank">${index + 1}</span>
      <img src="assets/thumbs/${hit.image.thumb}" alt="" loading="lazy">
      <span class="who">
        <span class="name"></span>
        <span class="credit">
          <a href="" target="_blank" rel="noopener noreferrer nofollow"></a>
        </span>
      </span>
      <span class="score">${hit.score.toFixed(3)}</span>`;
    item.querySelector(".name").textContent = hit.image.name;
    const link = item.querySelector(".credit a");
    link.textContent = credit;
    link.href = hit.image.source_url;
    list.appendChild(item);
  });
}

/** Flag the finding: a top match inside a co-located group is the hard case. */
function renderCoLocated(hits) {
  const box = el("colocated");
  const top = hits[0]?.image.dwarf;
  const group = (meta.co_located_groups || []).find((members) => members.includes(top));
  if (!group || group.length < 2) {
    box.hidden = true;
    return;
  }
  const names = new Map(meta.dwarfs.map((d) => [d.id, d.name]));
  const others = group.filter((id) => id !== top).map((id) => names.get(id));
  el("colocated-text").innerHTML =
    `<b>This one is genuinely hard.</b> It stands at the same spot as ` +
    `${others.length} other ${others.length === 1 ? "dwarf" : "dwarves"} — ` +
    `${others.join(", ")} — and they were installed as one themed group. ` +
    `They are the statues this project's error analysis finds most confusable, ` +
    `and knowing where you are does not separate them.`;
  box.hidden = false;
}

/**
 * Verify that this browser reproduces the shipped vectors.
 *
 * The build recorded vectors for a handful of the thumbnails this page serves,
 * so the page can re-embed those exact bytes and report the agreement. Reachable
 * at ?selftest=1 — the resampler here is the canvas, not the one that built the
 * references, and this is the only way to measure what that costs.
 */
async function runSelfTest() {
  const spec = backbone(current);
  const probes = meta.backbones[current].self_test ?? [];
  if (!probes.length) {
    say(`This build shipped no self-test probes for ${spec.label}.`, true);
    return;
  }
  const agreements = [];
  for (const [index, probe] of probes.entries()) {
    say(`Self-test ${index + 1}/${probes.length} on ${spec.label}…`);
    const vector = await embedSource(`assets/thumbs/${probe.thumb}`);
    let dot = 0;
    for (let d = 0; d < vector.length; d += 1) dot += vector[d] * probe.vector[d];
    agreements.push(dot);
  }
  agreements.sort((a, b) => a - b);
  const mean = agreements.reduce((a, b) => a + b, 0) / agreements.length;
  // These probes re-embed the same thumbnail bytes the build embedded, so the only
  // difference is the decoder: sharp in Node against the canvas here. That drift is
  // real, documented and harmless — measured at 0.989 mean / 0.980 min for DINOv2 and
  // 0.986 for CLIP, worth nothing in top-1 either time. So a threshold of 0.99 on the
  // *minimum*, as this check used to carry, reported the expected outcome as a failure.
  //
  // What the check is actually for is a broken export, and that failure is not subtle:
  // `uint8` DINOv2 agrees with the pipeline at cosine 0.111. Anything above 0.9 is
  // decode drift; anything near zero is a model that loaded and produces nonsense.
  // 0.95 sits an order of magnitude clear of both regimes.
  const BROKEN_BELOW = 0.95;
  const broken = agreements[0] < BROKEN_BELOW;
  say(
    `Self-test on ${spec.label}: ${probes.length} probes, cosine agreement mean ` +
      `${mean.toFixed(4)}, min ${agreements[0].toFixed(4)} — ` +
      (broken
        ? "far below the 0.95 this should never cross. The model export is wrong, " +
          "not merely decoding differently."
        : "this browser reproduces the build, within the decode drift " +
          "sharp and the canvas are known to differ by."),
    broken,
  );
}

/**
 * Embed every reference thumbnail here and score the leave-one-out protocol.
 *
 * Slow, and deliberately available: cosine agreement says how far this browser
 * drifts from the build, but only this says what that drift costs. Reachable at
 * ?selftest=full.
 */
async function runFullSelfTest() {
  const spec = backbone(current);
  const vectors = await loadVectors(current);
  const started = performance.now();
  const local = [];
  for (const [index, image] of meta.images.entries()) {
    if (index % 10 === 0) {
      say(`Full self-test on ${spec.label}: embedding ${index + 1}/${meta.images.length}…`);
      progress(index / meta.images.length);
    }
    local.push(await embedSource(`assets/thumbs/${image.thumb}`));
  }
  progress(null);

  let top1 = 0;
  let top5 = 0;
  for (let q = 0; q < local.length; q += 1) {
    const best = new Map();
    for (let r = 0; r < vectors.length; r += 1) {
      if (r === q) continue;
      let dot = 0;
      for (let d = 0; d < vectors[r].length; d += 1) dot += local[q][d] * vectors[r][d];
      const dwarf = meta.images[r].dwarf;
      if (!best.has(dwarf) || dot > best.get(dwarf)) best.set(dwarf, dot);
    }
    const ranked = [...best.entries()].sort((a, b) => b[1] - a[1]).map(([dwarf]) => dwarf);
    const at = ranked.indexOf(meta.images[q].dwarf);
    if (at === 0) top1 += 1;
    if (at < 5) top5 += 1;
  }
  const n = local.length;
  const measured = meta.backbones[current].measured;
  const seconds = ((performance.now() - started) / 1000).toFixed(0);
  say(
    `Full self-test on ${spec.label} in this browser: top-1 ${((100 * top1) / n).toFixed(1)}%, ` +
      `top-5 ${((100 * top5) / n).toFixed(1)}% over ${n} thumbnails in ${seconds}s. ` +
      `The build measured ${(measured.top_1 * 100).toFixed(1)}% and ` +
      `${(measured.top_5 * 100).toFixed(1)}%.`,
  );
}

async function identify(source, label) {
  lastQuery = { source, label };
  try {
    say("Preparing…");
    const [vectors] = await Promise.all([loadVectors(current), loadModel(current)]);
    say("Looking…");
    const query = await embedSource(source);
    const hits = rank(query, vectors);

    if (typeof source === "string") {
      el("query-img").src = source;
    } else {
      if (objectUrl) URL.revokeObjectURL(objectUrl);
      objectUrl = URL.createObjectURL(source);
      el("query-img").src = objectUrl;
    }
    el("query-sub").textContent = label;
    renderHits(hits);
    renderCoLocated(hits);
    el("result").hidden = false;
    say(
      `${backbone(current).label} compared it against ${meta.images.length} reference ` +
        `photographs of ${meta.dwarfs.length} dwarves.`,
    );
  } catch (error) {
    console.error(error);
    progress(null);
    // The previous ranking is deliberately left on screen while a new one is
    // computed, so switching backbones does not flash an empty card. That makes
    // hiding it on failure necessary: a stale ranking under an error message
    // reads as the answer to the thing that just failed.
    el("result").hidden = true;
    say(
      `Could not identify that photograph: ${error.message}. ` +
        "If this device is low on memory, try a desktop browser.",
      true,
    );
  }
}

/** Say what the page is waiting for, given what is already downloaded. */
function readyMessage() {
  const spec = backbone(current);
  return modelCache.has(current)
    ? `${spec.label} is loaded and ready.`
    : `${spec.label} downloads once, on your first identification — ${spec.downloadMb} MB.`;
}

/**
 * Switch backbones, and re-rank whatever is on screen.
 *
 * Re-ranking is the point of offering the choice at all: the same photograph
 * under both models is this project's central comparison, and a visitor should
 * not have to pick their photograph again to see it.
 */
async function selectBackbone(id) {
  if (id === current) return;
  current = id;
  renderModelChoice();
  renderFacts();
  if (lastQuery) await identify(lastQuery.source, lastQuery.label);
  else say(readyMessage());
}

/** The model switch, labelled from what the build actually measured. */
function renderModelChoice() {
  const row = el("models");
  const options = available();
  // One backbone is not a choice; a build that ships one should not imply two.
  row.hidden = options.length < 2;
  el("model-note").hidden = options.length < 2;
  if (row.hidden) return;
  row.textContent = "";
  for (const spec of options) {
    const info = meta.backbones[spec.id];
    const button = document.createElement("button");
    button.type = "button";
    button.className = spec.id === current ? "model on" : "model";
    button.setAttribute("aria-pressed", String(spec.id === current));
    button.innerHTML = '<span class="ml"></span><span class="mr"></span><span class="ms"></span>';
    button.querySelector(".ml").textContent = spec.label;
    button.querySelector(".mr").textContent = spec.role;
    button.querySelector(".ms").textContent =
      `${spec.downloadMb} MB · ${(info.measured.top_1 * 100).toFixed(1)}% top-1`;
    button.addEventListener("click", () => selectBackbone(spec.id));
    row.appendChild(button);
  }
}

/** Keep the header's fact strip on the backbone actually selected. */
function renderFacts() {
  const info = meta.backbones[current];
  el("fact-model").textContent = backbone(current).label;
  el("fact-acc").textContent = `${(info.measured.top_1 * 100).toFixed(1)}% top-1`;
}

function wireInputs() {
  const drop = el("drop");
  for (const id of ["file", "camera"]) {
    el(id).addEventListener("change", (event) => {
      const file = event.target.files?.[0];
      if (file) identify(file, file.name);
    });
  }
  el("pick").addEventListener("click", () => el("file").click());
  el("shoot").addEventListener("click", () => el("camera").click());

  // Offer the camera only where it means something.
  if (/Android|iPhone|iPad|iPod/i.test(navigator.userAgent)) {
    el("camera-label").hidden = false;
  }

  drop.addEventListener("dragover", (event) => {
    event.preventDefault();
    drop.classList.add("over");
  });
  drop.addEventListener("dragleave", () => drop.classList.remove("over"));
  drop.addEventListener("drop", (event) => {
    event.preventDefault();
    drop.classList.remove("over");
    const file = event.dataTransfer.files?.[0];
    if (file) identify(file, file.name);
  });
}

/** Offer a few reference photographs so the demo can be tried without one. */
function renderExamples() {
  const row = el("examples");
  const picks = ["Q136290068", "Q11823412", "Q136341163", "Q65742089"]
    .map((id) => meta.images.find((image) => image.dwarf === id))
    .filter(Boolean);
  for (const image of picks) {
    const thumb = document.createElement("img");
    thumb.src = `assets/thumbs/${image.thumb}`;
    thumb.alt = `Example: ${image.name}`;
    thumb.title = image.name;
    thumb.addEventListener("click", () =>
      identify(`assets/thumbs/${image.thumb}`, `example — ${image.name}`),
    );
    row.appendChild(thumb);
  }
}

loadMetadata()
  .then(() => {
    renderModelChoice();
    renderFacts();
    wireInputs();
    renderExamples();
    say(readyMessage());
    const mode = new URLSearchParams(location.search).get("selftest");
    if (mode === "full") runFullSelfTest();
    else if (mode !== null) runSelfTest();
  })
  .catch((error) => {
    console.error(error);
    say(`The reference data could not be loaded: ${error.message}`, true);
  });

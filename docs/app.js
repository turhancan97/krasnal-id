/**
 * In-browser identification.
 *
 * The reference vectors in assets/references.bin were produced by this same
 * library, model and dtype, and with the same antialiased pre-downscale applied
 * below. That matters: Python-preprocessed references cost 3.5 points of top-1
 * against browser-preprocessed queries, and skipping the pre-downscale costs
 * another two, because transformers.js resizes a large photograph in one
 * aliasing step.
 *
 * Nothing is uploaded. The photograph is decoded, scaled, embedded and compared
 * entirely on this device.
 */
import {
  AutoProcessor,
  CLIPVisionModelWithProjection,
  RawImage,
  env,
} from "https://cdn.jsdelivr.net/npm/@huggingface/transformers@3.8.1";
import { resizeToShortestEdge } from "./resize.mjs";

const MODEL_ID = "Xenova/clip-vit-base-patch32";
const DTYPE = "q4"; // 64 MB, and indistinguishable from full precision here.
const TOP_K = 5;
const SHORTEST_EDGE = 224; // Must match the build's EMBED_SHORTEST_EDGE.

env.allowLocalModels = false;

const el = (id) => document.getElementById(id);
const statusText = el("status-text");
const statusRow = el("status");
const bar = el("bar");
const barFill = el("bar-fill");

let meta = null;
let vectors = null;
let extractor = null;
let loading = null;

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

/** Load the reference vectors and their metadata. Small, so always eager. */
async function loadReferences() {
  const [metaResponse, binResponse] = await Promise.all([
    fetch("assets/references.json"),
    fetch("assets/references.bin"),
  ]);
  if (!metaResponse.ok || !binResponse.ok) throw new Error("reference data is unavailable");
  meta = await metaResponse.json();
  const raw = new Float32Array(await binResponse.arrayBuffer());
  const dim = meta.model.dimensions;
  vectors = [];
  for (let i = 0; i < meta.images.length; i += 1) {
    vectors.push(raw.subarray(i * dim, (i + 1) * dim));
  }
  el("fact-acc").textContent = `${(meta.measured.top_1 * 100).toFixed(1)}% top-1`;
}

/** Download the model on first use, reporting progress. */
async function loadModel() {
  if (extractor) return extractor;
  if (loading) return loading;
  loading = (async () => {
    const seen = new Map();
    const onProgress = (item) => {
      if (item.status === "progress" && item.total) {
        seen.set(item.file, item.loaded / item.total);
        const mean = [...seen.values()].reduce((a, b) => a + b, 0) / seen.size;
        progress(mean);
        say(`Downloading the model — ${Math.round(mean * 100)}%. This happens once.`);
      }
    };
    const [processor, model] = await Promise.all([
      AutoProcessor.from_pretrained(MODEL_ID, { progress_callback: onProgress }),
      CLIPVisionModelWithProjection.from_pretrained(MODEL_ID, {
        dtype: DTYPE,
        progress_callback: onProgress,
      }),
    ]);
    progress(null);
    extractor = { processor, model };
    return extractor;
  })();
  return loading;
}

/**
 * Decode, then downscale with the same resampler the build used.
 *
 * The canvas is only a decoder here. Its own scaling is browser-dependent and
 * measurably disagreed with the build (0.98 cosine, where 1.00 is wanted), so
 * the pixels go through resize.mjs instead and both sides match by construction.
 */
async function readScaled(source) {
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
    SHORTEST_EDGE,
  );
  return new RawImage(scaled.data, scaled.width, scaled.height, 4);
}

function normalise(values) {
  let sum = 0;
  for (const value of values) sum += value * value;
  const norm = Math.sqrt(sum) || 1;
  return Float32Array.from(values, (value) => value / norm);
}

/** Rank distinct dwarves by their best-matching reference photograph. */
function rank(query) {
  const best = new Map();
  for (let i = 0; i < vectors.length; i += 1) {
    const reference = vectors[i];
    let dot = 0;
    for (let d = 0; d < query.length; d += 1) dot += query[d] * reference[d];
    const image = meta.images[i];
    const current = best.get(image.dwarf);
    if (!current || dot > current.score) best.set(image.dwarf, { score: dot, image });
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

async function embedSource(source) {
  const { processor, model } = await loadModel();
  const output = await model(await processor(await readScaled(source)));
  return normalise(output.image_embeds.data);
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
  const probes = meta.self_test ?? [];
  if (!probes.length) {
    say("This build shipped no self-test probes.", true);
    return;
  }
  const agreements = [];
  for (const [index, probe] of probes.entries()) {
    say(`Self-test ${index + 1}/${probes.length}…`);
    const vector = await embedSource(`assets/thumbs/${probe.thumb}`);
    let dot = 0;
    for (let d = 0; d < vector.length; d += 1) dot += vector[d] * probe.vector[d];
    agreements.push(dot);
  }
  agreements.sort((a, b) => a - b);
  const mean = agreements.reduce((a, b) => a + b, 0) / agreements.length;
  const matched = agreements[0] > 0.99;
  say(
    `Self-test: ${probes.length} probes, cosine agreement mean ` +
      `${mean.toFixed(4)}, min ${agreements[0].toFixed(4)} — this browser ` +
      `${matched ? "matches" : "differs from"} the build.`,
    !matched,
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
  const started = performance.now();
  const local = [];
  for (const [index, image] of meta.images.entries()) {
    if (index % 10 === 0) {
      say(`Full self-test: embedding ${index + 1}/${meta.images.length}…`);
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
  const seconds = ((performance.now() - started) / 1000).toFixed(0);
  say(
    `Full self-test in this browser: top-1 ${((100 * top1) / n).toFixed(1)}%, ` +
      `top-5 ${((100 * top5) / n).toFixed(1)}% over ${n} thumbnails in ${seconds}s. ` +
      `The build measured ${(meta.measured.top_1 * 100).toFixed(1)}% and ` +
      `${(meta.measured.top_5 * 100).toFixed(1)}%.`,
  );
}

async function identify(source, label) {
  try {
    el("result").hidden = true;
    say("Preparing…");
    await loadModel();
    say("Looking…");
    const query = await embedSource(source);
    const hits = rank(query);

    el("query-img").src = typeof source === "string" ? source : URL.createObjectURL(source);
    el("query-sub").textContent = label;
    renderHits(hits);
    renderCoLocated(hits);
    el("result").hidden = false;
    say(`Compared against ${meta.images.length} reference photographs of ${meta.dwarfs.length} dwarves.`);
  } catch (error) {
    console.error(error);
    progress(null);
    say(
      `Could not identify that photograph: ${error.message}. ` +
        "If this device is low on memory, try a desktop browser.",
      true,
    );
  }
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

loadReferences()
  .then(() => {
    wireInputs();
    renderExamples();
    const mode = new URLSearchParams(location.search).get("selftest");
    if (mode === "full") runFullSelfTest();
    else if (mode !== null) runSelfTest();
  })
  .catch((error) => {
    console.error(error);
    say(`The reference data could not be loaded: ${error.message}`, true);
  });

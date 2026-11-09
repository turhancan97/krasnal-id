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
import { BACKBONES, DEFAULT_BACKBONE, backbone, unitVector, vectorFile } from "./backbones.mjs";
import { resizeToShortestEdge } from "./resize.mjs";
import { rankStatues, weightByLocation, distanceMetres } from "./retrieval.mjs";

const TOP_K = 5;

// Examples use stored vectors, so the model library is unnecessary until an
// uploaded photo or an embedding self-test needs it.
let transformerTask;
async function transformerLibrary() {
  transformerTask ??= import("https://cdn.jsdelivr.net/npm/@huggingface/transformers@3.8.1").then((library) => {
    library.env.allowLocalModels = false;
    return library;
  }).catch((error) => { transformerTask = null; throw error; });
  return transformerTask;
}

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
let lastRanking = null;
let requestVersion = 0;
let locationRequest = 0;
let userPosition = null;
let mapModule = null;
let dwarfById = new Map();

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
  if (!meta.location_prior || !meta.close_call || !meta.examples?.image_ids) {
    throw new Error("reference metadata is outdated — reload without cache");
  }
  dwarfById = new Map(meta.dwarfs.map((d) => [d.id, d]));
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
    const library = await transformerLibrary();
    const { AutoProcessor } = library;
    const seen = new Map();
    const onProgress = (item) => {
      if (id !== current) return;
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
    const ModelClass = library[spec.className];
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
  const { RawImage } = await transformerLibrary();
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

function renderHits(hits) {
  const list = el("hits");
  list.textContent = "";
  hits.forEach((hit, index) => {
    const item = document.createElement("li");
    item.className = index === 0 ? "hit top" : "hit";
    item.dataset.imageId = hit.image.id;
    item.dataset.dwarf = hit.image.dwarf;
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
    item.querySelector(".score").title = `Photograph cosine ${hit.cosine.toFixed(3)}; location bonus ${(hit.score-hit.cosine).toFixed(3)}`;
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
    `<b>Hard case.</b> It stands with ${others.join(", ")} as one themed group, ` +
    `and location cannot tell them apart.`;
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
  const vectors = await loadVectors(spec.id);
  const position = meta.dwarfs.find((d) => Number.isFinite(d.lat));
  for (const imageId of meta.examples.image_ids) {
    const index = meta.images.findIndex((image) => image.id === imageId);
    const plain = rankStatues(vectors[index], vectors, meta.images, index);
    const control = weightByLocation(plain, dwarfById, position, { ...meta.location_prior, weight:0 });
    if (control.length !== plain.length || control.some((hit,i) => hit.image.id !== plain[i].image.id || hit.score !== plain[i].score)) {
      throw new Error("Weight-0 location control changed the plain cosine ranking");
    }
  }
  const agreements = [];
  for (const [index, probe] of probes.entries()) {
    say(`Self-test ${index + 1}/${probes.length} on ${spec.label}…`);
    const vector = await embedSource(`assets/thumbs/${probe.thumb}`, spec.id);
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
      `${mean.toFixed(4)}, min ${agreements[0].toFixed(4)}; weight-0 location control passed for ${meta.examples.image_ids.length} examples — ` +
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

/** Cache this query by backbone: location toggles never load a model or re-embed it. */
async function identify(source, label) {
  lastQuery = { source, label, vectors: new Map() };
  await identifyQuery(lastQuery);
}

async function identifyExample(imageId) {
  const index = meta.images.findIndex((image) => image.id === imageId);
  if (index < 0) throw new Error("Example is absent from the reference metadata");
  const image = meta.images[index];
  lastQuery = { source: `assets/thumbs/${image.thumb}`, label: `Example — ${image.name}`, index, image, vectors: new Map() };
  for (const button of el("examples").querySelectorAll("button")) {
    button.setAttribute("aria-pressed", String(button.dataset.imageId === imageId));
  }
  await identifyQuery(lastQuery);
}

async function identifyQuery(query) {
  const id = current;
  const request = ++requestVersion;
  try {
    say(query.image ? "Scoring the held-out example…" : "Preparing…");
    el("result").setAttribute("aria-busy", "true");
    const vectors = await loadVectors(id);
    let vector = query.vectors.get(id);
    if (!vector) {
      vector = query.image ? vectors[query.index] : await embedSource(query.source, id);
      query.vectors.set(id, vector);
    }
    if (request !== requestVersion) return;
    const plain = rankStatues(vector, vectors, meta.images, query.index ?? -1);
    lastRanking = { plain, query, backbone: id };
    if (typeof query.source === "string") {
      el("query-img").src = query.source;
    } else {
      if (objectUrl) URL.revokeObjectURL(objectUrl);
      objectUrl = URL.createObjectURL(query.source);
      el("query-img").src = objectUrl;
    }
    el("query-sub").textContent = query.label;
    const credit = el("query-credit");
    credit.replaceChildren(); credit.hidden = !query.image;
    if (query.image) appendCredit(credit, query.image);
    else for (const button of el("examples").querySelectorAll("button")) button.setAttribute("aria-pressed", "false");
    el("result").hidden = false;
    renderCurrentRanking();
    say(`Compared with ${(meta.images.length - (query.image ? 1 : 0)).toLocaleString("en")} photos of ${meta.dwarfs.length} statues using ${backbone(id).label}.`);
  } catch (error) {
    if (request !== requestVersion) return;
    console.error(error); progress(null); lastRanking = null;
    el("result").hidden = true;
    say(`Could not identify that photograph: ${error.message}. If this device is low on memory, try a desktop browser.`, true);
  } finally {
    if (request === requestVersion) el("result").setAttribute("aria-busy", "false");
  }
}

function renderCurrentRanking() {
  if (!lastRanking || lastRanking.backbone !== current) return;
  const { plain } = lastRanking;
  const position = el("use-location").checked ? userPosition : null;
  const hits = weightByLocation(plain, dwarfById, position, meta.location_prior).slice(0, TOP_K);
  renderHits(hits); renderCoLocated(hits);
  el("ranking-note").textContent = position ? "Ranked by photo and distance." : "";
  const alone = el("photo-alone");
  alone.hidden = hits[0].image.dwarf === plain[0].image.dwarf;
  alone.textContent = `The photograph alone ranked ${plain[0].image.name} first.`;
  renderCloseCall(plain, Boolean(position), hits[0].image.dwarf !== plain[0].image.dwarf);
  mapModule ??= import("./matches-map.mjs");
  mapModule.then((module) => module.renderMatchesMap(hits, meta, position)).catch(() => {
    mapModule = null; el("map-status").textContent = "Map unavailable. The photograph matches above still work.";
  });
}

function renderCloseCall(plain, weighted, winnerChanged) {
  const calibration = meta.close_call[current];
  const margin = plain[0].cosine - plain[1].cosine;
  if (margin >= calibration.threshold) return;
  const note = document.createElement("div"); note.className = "close-call";
  note.dataset.margin = String(margin);
  const heading = document.createElement("p");
  // A geographic winner can be the photo runner-up. Name both photo candidates
  // explicitly in that case, rather than suggesting the displayed winner again.
  if (winnerChanged) {
    heading.append("Close call on the photo alone — ");
    const first = document.createElement("strong"); first.textContent = plain[0].image.name;
    heading.append(first, " or ");
  } else {
    heading.append(weighted ? "Close call on the photo alone — could also be " : "Close call — could also be ");
  }
  const name = document.createElement("strong"); name.textContent = plain[1].image.name;
  heading.append(name, ".");
  const detail = document.createElement("p");
  detail.textContent = `On held-out dataset photos this close, the first guess was right ${(100 * calibration.top_1_below).toFixed(1)}% of the time.`;
  note.append(heading, detail); el("hits").firstElementChild.append(note);
}

function wireLocation() {
  const control = el("use-location");
  const status = el("location-status");
  control.addEventListener("change", () => {
    const ticket = ++locationRequest;
    userPosition = null; renderCurrentRanking();
    if (!control.checked) {
      status.textContent = ""; return;
    }
    if (!navigator.geolocation) {
      status.textContent = "Location is unavailable in this browser; photograph alone is used."; return;
    }
    status.textContent = "Getting your location…";
    navigator.geolocation.getCurrentPosition((fix) => {
      if (ticket !== locationRequest || !control.checked) return;
      const { latitude:lat, longitude:lon, accuracy } = fix.coords;
      if (![lat,lon,accuracy].every(Number.isFinite) || accuracy < 0 || accuracy > 100) {
        status.textContent = "Location accuracy is worse than 100 m; photograph alone is used.";
      } else {
        const position = {lat,lon};
        const nearest = Math.min(...meta.dwarfs.filter((d)=>Number.isFinite(d.lat)&&Number.isFinite(d.lon)).map((d)=>distanceMetres(position,d)));
        if (nearest > 2000) status.textContent = "You seem to be away from Wrocław’s statues; location not used.";
        else { userPosition = position; status.textContent = "Nearby statues get a small boost; every statue stays in the running."; }
      }
      renderCurrentRanking();
    }, (error) => {
      if (ticket !== locationRequest || !control.checked) return;
      status.textContent = error.code === 1 ? "Location permission was denied; photograph alone is used." : "Could not get your location; photograph alone is used.";
      renderCurrentRanking();
    }, { enableHighAccuracy:true, maximumAge:0, timeout:10000 });
  });
}

/** Say what the page is waiting for, given what is already downloaded. */
function readyMessage() {
  const spec = backbone(current);
  return modelCache.has(current)
    ? `${spec.label} is loaded and ready.`
    : `${spec.label} downloads once, when you identify an uploaded photo — ${spec.downloadMb} MB.`;
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
  if (lastQuery) await identifyQuery(lastQuery);
  else say(readyMessage());
}

/** The model switch, labelled from what the build actually measured. */
function renderModelChoice() {
  const row = el("models");
  const options = available();
  // One backbone is not a choice; a build that ships one should not imply two.
  row.hidden = options.length < 2;
  if (row.hidden) return;
  row.textContent = "";
  for (const spec of options) {
    const info = meta.backbones[spec.id];
    const button = document.createElement("button");
    button.type = "button";
    button.className = spec.id === current ? "model on" : "model";
    button.setAttribute("aria-pressed", String(spec.id === current));
    button.innerHTML = '<span class="ml"></span><span class="ms"></span>';
    button.querySelector(".ml").textContent = spec.label;
    button.querySelector(".ms").textContent =
      `${(info.measured.top_1 * 100).toFixed(1)}% · ${spec.downloadMb} MB`;
    button.addEventListener("click", () => selectBackbone(spec.id));
    row.appendChild(button);
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

/** Each thumbnail and selected query retains its own photographer and licence. */
function appendCredit(parent, image, stacked = false) {
  const author = document.createElement("a");
  author.href = image.source_url; author.textContent = image.author; author.title = image.author;
  const licence = document.createElement("a");
  licence.href = image.license_url; licence.textContent = image.license;
  for (const link of [author,licence]) { link.target = "_blank"; link.rel = "noopener noreferrer"; }
  if (stacked) parent.append(author, licence);
  else parent.append(author, document.createTextNode(" · "), licence);
}

function renderExamples() {
  const row = el("examples"); row.replaceChildren();
  const credits = el("example-credits"); credits.replaceChildren();
  for (const id of meta.examples.image_ids) {
    const image = meta.images.find((entry) => entry.id === id);
    if (!image) throw new Error(`Missing example ${id}`);
    const tile = document.createElement("div"); tile.className = "example-tile";
    const button = document.createElement("button"); button.className = "example-button"; button.type = "button";
    button.dataset.imageId = id; button.setAttribute("aria-pressed", "false");
    button.setAttribute("aria-label", `Try ${image.name}`); button.title = `${image.name} · ${image.author} · ${image.license}`;
    const thumb = document.createElement("img"); thumb.src = `assets/thumbs/${image.thumb}`; thumb.alt = image.name;
    button.append(thumb); button.addEventListener("click", () => identifyExample(id));
    tile.append(button); row.append(tile);
    const credit = document.createElement("p"); credit.append(`${image.name}: `); appendCredit(credit, image); credits.append(credit);
  }
}

loadMetadata()
  .then(() => {
    renderModelChoice();
      wireInputs();
    wireLocation();
    renderExamples();
    say(readyMessage());
    const mode = new URLSearchParams(location.search).get("selftest");
    if (mode === "full") return runFullSelfTest();
    else if (mode !== null) return runSelfTest();
  })
  .catch((error) => {
    console.error(error);
    say(`The reference data or self-test could not be completed: ${error.message}`, true);
  });

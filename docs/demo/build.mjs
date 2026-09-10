/**
 * Build the static demo's data files, for every backbone the page can run.
 *
 * The reference vectors are produced by *the same library, model and dtype the
 * browser runs*, not by the Python pipeline. That is not a preference: measured
 * on this dataset while the page ran CLIP, transformers.js preprocesses
 * differently enough that Python-built references cost 3.5 points of top-1 when
 * compared against browser-built queries. Anything that embeds a query must
 * embed the references, and that holds whichever model is in use.
 *
 * Both sides also pre-downscale to the model's own shortest edge with a proper
 * antialiased resampler before its processor sees the image. Without that,
 * transformers.js downscales a 2000px photograph in one aliasing step and loses
 * roughly two points of accuracy outright.
 *
 * Each backbone gets its own `references-<id>.bin`; `references.json` carries
 * the metadata they share and one entry per backbone describing its vectors.
 * Every photograph is decoded once and scaled once per backbone, because the
 * two want different shortest edges but the same pixels behind them.
 *
 *   cd docs/demo && npm install && node build.mjs
 *
 * Model weights are fetched from the Hugging Face CDN on first run and cached.
 */
import {
  AutoModel,
  AutoProcessor,
  CLIPVisionModelWithProjection,
  RawImage,
  env,
} from "@huggingface/transformers";
import { mkdirSync, readFileSync, readdirSync, statSync, writeFileSync } from "node:fs";
import { join, resolve } from "node:path";
import sharp from "sharp";
import { BACKBONES, unitVector, vectorFile } from "../backbones.mjs";
import { resizeToShortestEdge } from "../resize.mjs";

env.allowLocalModels = false;

const REPO = resolve(import.meta.dirname, "../..");
const OUT = join(REPO, "docs/assets");
const THUMBS = join(OUT, "thumbs");

const THUMB_LONG_SIDE = 320;
const CO_LOCATED_METRES = 25;
const SELF_TEST_COUNT = 8;

// Resolved here rather than in backbones.mjs because the browser loads
// transformers.js from a CDN and this file loads it from npm.
const CLASSES = { AutoModel, CLIPVisionModelWithProjection };

// Measured on this dataset, embedding DINOv2 q4 where the build actually had
// four usable CPUs:
//
//   default   7502 ms/image   (~7 hours for both backbones)
//   1 thread   613 ms/image
//   4 threads  426 ms/image   (~24 minutes)
//   8 threads 1636 ms/image
//  16 threads 3497 ms/image
//
// onnxruntime-node sizes its thread pool from the host's core count rather than
// from the cpuset the process is confined to, so on a shared or containerised
// machine it opens a thread per host core, cannot pin any of them — the
// pthread_setaffinity_np errors it prints are exactly that — and spends its
// time in contention instead of in matrix multiplies. Past the number of CPUs
// really available, every further thread is pure overhead.
//
// Set this to the CPUs the build can actually use (`nproc`), not to the ones
// `/proc/cpuinfo` lists, and re-measure rather than assuming this number
// transfers to another machine.
const ONNX_THREADS = 4;

const manifest = JSON.parse(readFileSync(join(REPO, "data/manifest.json"), "utf8"));
const images = [...manifest.images].sort((a, b) => a.image_id.localeCompare(b.image_id));
const dwarfs = [...manifest.dwarfs].sort((a, b) => a.dwarf_id.localeCompare(b.dwarf_id));
const names = new Map(dwarfs.map((d) => [d.dwarf_id, d.display_name]));

console.log(`${images.length} reference photographs, ${dwarfs.length} dwarves`);
console.log(`backbones: ${BACKBONES.map((b) => `${b.label} (${b.dtype})`).join(", ")}`);

// --- models -----------------------------------------------------------------
const loaded = new Map();
for (const spec of BACKBONES) {
  const ModelClass = CLASSES[spec.className];
  if (!ModelClass) throw new Error(`no model class named ${spec.className}`);
  loaded.set(spec.id, {
    processor: await AutoProcessor.from_pretrained(spec.modelId),
    model: await ModelClass.from_pretrained(spec.modelId, {
      dtype: spec.dtype,
      session_options: { intraOpNumThreads: ONNX_THREADS },
    }),
  });
  console.log(`model ready: ${spec.modelId} (${spec.dtype})`);
}

/**
 * Decode once, keeping the raw pixels so each backbone can scale its own way.
 *
 * sharp only decodes here. Its own resize is deliberately not used: it does not
 * match a browser's, and the query and the reference have to agree.
 */
async function decode(file) {
  const { data, info } = await sharp(file).ensureAlpha().raw().toBuffer({ resolveWithObject: true });
  return { pixels: new Uint8ClampedArray(data), width: info.width, height: info.height };
}

/** Embed already-decoded pixels with one backbone, at that backbone's edge. */
async function embedDecoded(spec, decoded) {
  const scaled = resizeToShortestEdge(
    decoded.pixels,
    decoded.width,
    decoded.height,
    spec.shortestEdge,
  );
  const image = new RawImage(scaled.data, scaled.width, scaled.height, 4);
  const { processor, model } = loaded.get(spec.id);
  return unitVector(spec.pool(await model(await processor(image))));
}

const embedFile = async (spec, file) => embedDecoded(spec, await decode(file));

// --- thumbnails -------------------------------------------------------------
// Model-independent, so built once regardless of how many backbones ship.
mkdirSync(THUMBS, { recursive: true });
const thumbs = new Map();
for (const image of images) {
  const name = `${image.image_id}.webp`;
  await sharp(join(REPO, image.local_path))
    .resize({ width: THUMB_LONG_SIDE, height: THUMB_LONG_SIDE, fit: "inside", kernel: "lanczos3" })
    .webp({ quality: 80, effort: 6 })
    .toFile(join(THUMBS, name));
  thumbs.set(image.image_id, name);
}
console.log(`wrote ${thumbs.size} thumbnails`);

// --- reference vectors ------------------------------------------------------
const vectors = new Map(BACKBONES.map((spec) => [spec.id, []]));
for (const [index, image] of images.entries()) {
  const decoded = await decode(join(REPO, image.local_path));
  for (const spec of BACKBONES) {
    vectors.get(spec.id).push(await embedDecoded(spec, decoded));
  }
  if ((index + 1) % 40 === 0) console.log(`  embedded ${index + 1}/${images.length}`);
}

// --- leave-one-out score, on exactly the vectors being shipped --------------
function score(reference, queries) {
  const dim = reference[0].length;
  let top1 = 0;
  let top5 = 0;
  let reciprocal = 0;
  for (let q = 0; q < queries.length; q += 1) {
    const best = new Map();
    for (let r = 0; r < reference.length; r += 1) {
      if (r === q) continue;
      let dot = 0;
      for (let d = 0; d < dim; d += 1) dot += queries[q][d] * reference[r][d];
      const dwarf = images[r].dwarf_id;
      if (!best.has(dwarf) || dot > best.get(dwarf)) best.set(dwarf, dot);
    }
    const ranked = [...best.entries()].sort((a, b) => b[1] - a[1]).map(([dwarf]) => dwarf);
    const at = ranked.indexOf(images[q].dwarf_id);
    if (at === 0) top1 += 1;
    if (at < 5) top5 += 1;
    reciprocal += 1 / (at + 1);
  }
  const n = queries.length;
  return { top_1: top1 / n, top_5: top5 / n, mrr: reciprocal / n, folds: n };
}

// --- self test: vectors for thumbnails the page can re-fetch ---------------
// Lets the page verify its own pipeline in a real browser, where the resampler
// is the canvas rather than sharp. Model-specific, so one set per backbone.
const step = Math.max(1, Math.floor(images.length / SELF_TEST_COUNT));
const probes = images.filter((_, index) => index % step === 0).slice(0, SELF_TEST_COUNT);

const backbones = {};
for (const spec of BACKBONES) {
  const built = vectors.get(spec.id);
  const dim = built[0].length;
  const measured = score(built, built);
  console.log(
    `${spec.label}: top-1 ${(measured.top_1 * 100).toFixed(1)}%, ` +
      `top-5 ${(measured.top_5 * 100).toFixed(1)}%, MRR ${measured.mrr.toFixed(4)}`,
  );

  const selfTest = [];
  for (const image of probes) {
    const vector = await embedFile(spec, join(THUMBS, thumbs.get(image.image_id)));
    selfTest.push({
      thumb: thumbs.get(image.image_id),
      vector: [...vector].map((v) => Number(v.toFixed(6))),
    });
  }

  const buffer = Buffer.alloc(built.length * dim * 4);
  built.forEach((vector, i) =>
    vector.forEach((value, d) => buffer.writeFloatLE(value, (i * dim + d) * 4)),
  );
  writeFileSync(join(OUT, vectorFile(spec.id)), buffer);

  backbones[spec.id] = {
    repo: spec.modelId,
    dtype: spec.dtype,
    dimensions: dim,
    shortest_edge: spec.shortestEdge,
    vectors: vectorFile(spec.id),
    bytes: buffer.length,
    measured,
    self_test: selfTest,
  };
}
console.log(`wrote ${probes.length} self-test probes per backbone`);

// --- co-located installations, derived not listed --------------------------
const R = 6371008.8;
const rad = (deg) => (deg * Math.PI) / 180;
function metres(a, b) {
  const dLat = rad(b[0] - a[0]);
  const dLon = rad(b[1] - a[1]);
  const inner =
    Math.sin(dLat / 2) ** 2 +
    Math.cos(rad(a[0])) * Math.cos(rad(b[0])) * Math.sin(dLon / 2) ** 2;
  return 2 * R * Math.asin(Math.sqrt(Math.min(1, inner)));
}
const located = dwarfs.filter((d) => d.coordinates);
const parent = new Map(located.map((d) => [d.dwarf_id, d.dwarf_id]));
const find = (id) => (parent.get(id) === id ? id : (parent.set(id, find(parent.get(id))), parent.get(id)));
for (let i = 0; i < located.length; i += 1) {
  for (let j = i + 1; j < located.length; j += 1) {
    const a = located[i];
    const b = located[j];
    const distance = metres(
      [a.coordinates.latitude, a.coordinates.longitude],
      [b.coordinates.latitude, b.coordinates.longitude],
    );
    if (distance <= CO_LOCATED_METRES) parent.set(find(b.dwarf_id), find(a.dwarf_id));
  }
}
const clusters = new Map();
for (const dwarf of located) {
  const root = find(dwarf.dwarf_id);
  clusters.set(root, [...(clusters.get(root) ?? []), dwarf.dwarf_id]);
}
const coLocated = [...clusters.values()].filter((g) => g.length > 1).map((g) => g.sort());
console.log(`co-located groups: ${coLocated.map((g) => g.length).join(", ")}`);

// --- write ------------------------------------------------------------------
writeFileSync(
  join(OUT, "references.json"),
  `${JSON.stringify(
    {
      generated_at: new Date().toISOString(),
      backbones,
      // The staging hash, not a hash of the manifest: it is what ties these vectors
      // to the exact acquisition run that produced their images.
      staging_sha256: manifest.staging_sha256,
      co_located_groups: coLocated,
      dwarfs: dwarfs.map((d) => ({ id: d.dwarf_id, name: d.display_name })),
      images: images.map((image) => ({
        id: image.image_id,
        dwarf: image.dwarf_id,
        name: names.get(image.dwarf_id),
        thumb: thumbs.get(image.image_id),
        author: image.author,
        license: image.license,
        license_url: image.license_url,
        source_url: image.source_url,
      })),
    },
    null,
    1,
  )}\n`,
);

const thumbBytes = readdirSync(THUMBS).reduce((sum, f) => sum + statSync(join(THUMBS, f)).size, 0);
for (const spec of BACKBONES) {
  console.log(`  ${vectorFile(spec.id).padEnd(22)} ${(backbones[spec.id].bytes / 1024).toFixed(0)} KB`);
}
console.log(`  references.json        ${(statSync(join(OUT, "references.json")).size / 1024).toFixed(0)} KB`);
console.log(`  thumbs/                ${(thumbBytes / 1e6).toFixed(1)} MB`);

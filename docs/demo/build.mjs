/**
 * Build the static demo's data files.
 *
 * The reference vectors are produced by *the same library, model and dtype the
 * browser runs*, not by the Python pipeline. That is not a preference: measured
 * on this dataset, transformers.js preprocesses differently enough that
 * Python-built references cost 3.5 points of top-1 when compared against
 * browser-built queries. Anything that embeds a query must embed the references.
 *
 * Both sides also pre-downscale to a 224 shortest edge with a proper antialiased
 * resampler before the model's own processor sees the image. Without that,
 * transformers.js downscales a 2000px photograph in one aliasing step and loses
 * roughly two points of accuracy outright.
 *
 *   cd docs/demo && npm install && node build.mjs
 *
 * Model weights are fetched from the Hugging Face CDN on first run and cached.
 */
import { AutoProcessor, CLIPVisionModelWithProjection, RawImage, env } from "@huggingface/transformers";
import { mkdirSync, readFileSync, readdirSync, statSync, writeFileSync } from "node:fs";
import { join, resolve } from "node:path";
import sharp from "sharp";
import { resizeToShortestEdge } from "../resize.mjs";

env.allowLocalModels = false;

const REPO = resolve(import.meta.dirname, "../..");
const OUT = join(REPO, "docs/assets");
const THUMBS = join(OUT, "thumbs");

const MODEL_ID = "Xenova/clip-vit-base-patch32";
const DTYPE = "q4";               // 64 MB, and lossless in practice on this dataset.
const EMBED_SHORTEST_EDGE = 224;  // What the model wants; we resize to it ourselves.
const THUMB_LONG_SIDE = 320;
const CO_LOCATED_METRES = 25;
const SELF_TEST_COUNT = 8;

const manifest = JSON.parse(readFileSync(join(REPO, "data/manifest.json"), "utf8"));
const images = [...manifest.images].sort((a, b) => a.image_id.localeCompare(b.image_id));
const dwarfs = [...manifest.dwarfs].sort((a, b) => a.dwarf_id.localeCompare(b.dwarf_id));
const names = new Map(dwarfs.map((d) => [d.dwarf_id, d.display_name]));

console.log(`${images.length} reference photographs, ${dwarfs.length} dwarves`);

/**
 * Decode, then downscale with the resampler the browser also uses.
 *
 * sharp only decodes here. Its own resize is deliberately not used: it does not
 * match a browser's, and the query and the reference have to agree.
 */
async function readScaled(file) {
  const { data, info } = await sharp(file)
    .ensureAlpha()
    .raw()
    .toBuffer({ resolveWithObject: true });
  const scaled = resizeToShortestEdge(
    new Uint8ClampedArray(data),
    info.width,
    info.height,
    EMBED_SHORTEST_EDGE,
  );
  return new RawImage(scaled.data, scaled.width, scaled.height, 4);
}

const unit = (values) => {
  let sum = 0;
  for (const value of values) sum += value * value;
  const norm = Math.sqrt(sum) || 1;
  return Float32Array.from(values, (value) => value / norm);
};

const processor = await AutoProcessor.from_pretrained(MODEL_ID);
const model = await CLIPVisionModelWithProjection.from_pretrained(MODEL_ID, { dtype: DTYPE });
console.log(`model ready: ${MODEL_ID} (${DTYPE})`);

async function embed(file) {
  const output = await model(await processor(await readScaled(file)));
  return unit(output.image_embeds.data);
}

// --- thumbnails -------------------------------------------------------------
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
const vectors = [];
for (const [index, image] of images.entries()) {
  vectors.push(await embed(join(REPO, image.local_path)));
  if ((index + 1) % 40 === 0) console.log(`  embedded ${index + 1}/${images.length}`);
}
const dim = vectors[0].length;

// --- leave-one-out score, on exactly the vectors being shipped --------------
function score(queries) {
  let top1 = 0;
  let top5 = 0;
  let reciprocal = 0;
  for (let q = 0; q < queries.length; q += 1) {
    const best = new Map();
    for (let r = 0; r < vectors.length; r += 1) {
      if (r === q) continue;
      let dot = 0;
      for (let d = 0; d < dim; d += 1) dot += queries[q][d] * vectors[r][d];
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
const measured = score(vectors);
console.log(
  `shipped vectors score top-1 ${(measured.top_1 * 100).toFixed(1)}%, ` +
    `top-5 ${(measured.top_5 * 100).toFixed(1)}%, MRR ${measured.mrr.toFixed(4)}`,
);

// --- self test: vectors for thumbnails the page can re-fetch ---------------
// Lets the page verify its own pipeline in a real browser, where the resampler
// is the canvas rather than sharp.
const step = Math.max(1, Math.floor(images.length / SELF_TEST_COUNT));
const probes = images.filter((_, index) => index % step === 0).slice(0, SELF_TEST_COUNT);
const selfTest = [];
for (const image of probes) {
  const vector = await embed(join(THUMBS, thumbs.get(image.image_id)));
  selfTest.push({ thumb: thumbs.get(image.image_id), vector: [...vector].map((v) => Number(v.toFixed(6))) });
}
console.log(`wrote ${selfTest.length} self-test probes`);

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
const buffer = Buffer.alloc(vectors.length * dim * 4);
vectors.forEach((vector, i) =>
  vector.forEach((value, d) => buffer.writeFloatLE(value, (i * dim + d) * 4)),
);
writeFileSync(join(OUT, "references.bin"), buffer);

writeFileSync(
  join(OUT, "references.json"),
  `${JSON.stringify(
    {
      generated_at: new Date().toISOString(),
      model: { repo: MODEL_ID, dtype: DTYPE, dimensions: dim, shortest_edge: EMBED_SHORTEST_EDGE },
      manifest_sha256: manifest.staging_sha256,
      measured,
      co_located_groups: coLocated,
      self_test: selfTest,
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
console.log(`  references.bin   ${(buffer.length / 1024).toFixed(0)} KB`);
console.log(`  references.json  ${(statSync(join(OUT, "references.json")).size / 1024).toFixed(0)} KB`);
console.log(`  thumbs/          ${(thumbBytes / 1e6).toFixed(1)} MB`);

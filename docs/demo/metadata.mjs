/** Metadata calibration uses shipped vectors only. No image decoder or model imports. */
import { createHash } from 'node:crypto';
import { readFileSync, writeFileSync, renameSync, unlinkSync } from 'node:fs';
import { join } from 'node:path';
import { BACKBONES, vectorFile } from '../backbones.mjs';
import { rankStatues } from '../retrieval.mjs';

const PRIOR_ARTIFACT = 'results/cvww_sens_prior-exponential-200/workshop_validation-dinov2.json';
const hash = (bytes) => createHash('sha256').update(bytes).digest('hex');
const compareIds = (a, b) => a < b ? -1 : a > b ? 1 : 0;

export function calibrate(vectors, images) {
  let top1 = 0, top5 = 0, reciprocal = 0;
  const queries = images.map((image, q) => {
    const ranked = rankStatues(vectors[q], vectors, images, q);
    const at = ranked.findIndex((hit) => hit.image.dwarf === image.dwarf);
    if (at < 0) throw new Error(`No remaining reference for ${image.id}`);
    top1 += Number(at === 0); top5 += Number(at < 5); reciprocal += 1 / (at + 1);
    return { id: image.id, dwarf: image.dwarf, margin: ranked[0].cosine - ranked[1].cosine, correct: at === 0 };
  });
  const n = queries.length;
  return { measured: { top_1: top1 / n, top_5: top5 / n, mrr: reciprocal / n, folds: n }, queries };
}

export function closeCallStatistics(queries) {
  // Candidate t values are observed distinct margins. Strict < t includes whole
  // tie groups, never individual members of a tie. Both sides must be nonempty.
  // Scan every boundary: prefix accuracy need not be monotonic. Keep the largest
  // t whose prefix accuracy is strictly below 80%. No interpolation or rounding.
  const sorted = [...queries].sort((a, b) => a.margin - b.margin || compareIds(a.id, b.id));
  const totalCorrect = sorted.filter((q) => q.correct).length;
  let correct = 0, selected = null;
  for (let i = 0; i < sorted.length - 1; i += 1) {
    correct += Number(sorted[i].correct);
    const count = i + 1;
    if (sorted[i].margin === sorted[i + 1].margin) continue;
    if (correct * 5 < count * 4) selected = {
      threshold: sorted[i + 1].margin, share_of_queries: count / sorted.length,
      top_1_below: correct / count, top_1_above: (totalCorrect - correct) / (sorted.length - count),
    };
  }
  if (!selected) throw new Error('No nonempty close-call group below 80% accuracy');
  return selected;
}

export function selectExamples(queries, dwarfs) {
  const teaser = queries.find((q) => q.id === 'commons-4074360');
  if (!teaser) throw new Error('Missing fixed teaser query');
  const slupniki = new Set(dwarfs.filter((d) => d.name.startsWith('Słupniki')).map((d) => d.id));
  const sorted = queries.filter((q) => !slupniki.has(q.dwarf))
    .sort((a, b) => a.margin - b.margin || compareIds(a.id, b.id));
  const used = new Set([teaser.dwarf]);
  const ids = [teaser.id];
  for (const percentile of [0.1, 0.3, 0.5, 0.7, 0.9]) {
    const target = Math.ceil(percentile * sorted.length) - 1;
    const candidates = sorted.map((q, index) => ({ ...q, deviation: Math.abs(index - target) }))
      .filter((q) => !used.has(q.dwarf))
      .sort((a, b) => a.deviation - b.deviation || compareIds(a.id, b.id));
    const chosen = candidates[0];
    if (!chosen) throw new Error('Too few distinct statues for examples');
    ids.push(chosen.id); used.add(chosen.dwarf);
  }
  return { image_ids: ids, rule: 'Fixed query commons-4074360; then nearest-rank 10th, 30th, 50th, 70th and 90th percentiles of DINOv2 leave-one-out cosine margins among non-Słupniki images. Sort by margin then ASCII image ID; choose the nearest unused-statue rank at each percentile, ties by lower ASCII image ID.' };
}

export function enrichMetadata(metadata, manifest, repo, out) {
  const images = [...manifest.images].sort((a, b) => a.image_id.localeCompare(b.image_id));
  if (metadata.staging_sha256 !== manifest.staging_sha256 || images.length !== metadata.images.length ||
      images.some((image, i) => image.image_id !== metadata.images[i].id || image.dwarf_id !== metadata.images[i].dwarf)) {
    throw new Error('Manifest and shipped image order/labels/staging differ; metadata-only update refused');
  }
  const manifestDwarfs = new Map(manifest.dwarfs.map((d) => [d.dwarf_id, d]));
  if (manifestDwarfs.size !== metadata.dwarfs.length) throw new Error('Statue sets differ');
  const dwarfs = metadata.dwarfs.map((d) => {
    const record = manifestDwarfs.get(d.id);
    if (!record) throw new Error(`Unknown statue ${d.id}`);
    const { lat, lon, coordinate_source, ...base } = d;
    if (!record.coordinates) return base;
    if (!['wikidata', 'commons_camera'].includes(record.coordinate_source)) throw new Error(`Unknown coordinate source for ${d.id}`);
    return { ...base, lat: Number(record.coordinates.latitude.toFixed(5)), lon: Number(record.coordinates.longitude.toFixed(5)), coordinate_source: record.coordinate_source };
  });
  const placed = dwarfs.filter((d) => Number.isFinite(d.lat)).length;
  if (dwarfs.length !== 306 || placed !== 294 || dwarfs.length - placed !== 12) throw new Error('Expected 294 placed and 12 unplaced statues');
  console.log(`Coordinates: ${placed} placed, ${dwarfs.length - placed} unplaced`);
  const artifact = JSON.parse(readFileSync(join(repo, PRIOR_ARTIFACT), 'utf8'));
  const metrics = new Map(artifact.metrics.map((m) => [m.name, m.value]));
  const weights = Array.from({ length: 5 }, (_, i) => metrics.get(`dinov2_soft_prior_gps100_fold_${i}_weight`));
  if (weights.some((w) => !Number.isFinite(w)) || weights.filter((w) => w === 0.2).length !== 4) throw new Error('Location-prior fold provenance does not hold');
  const location_prior = { kernel: 'exponential', scale_m: 200, weight: 0.2, source: `${PRIOR_ARTIFACT}, metrics dinov2_soft_prior_gps100_fold_*_weight (4 of 5 held-out folds chose 0.2)` };
  console.log(`Verified location_prior: ${JSON.stringify(location_prior)}; fold weights ${weights.join(', ')}`);
  const imageOrderHash = hash(JSON.stringify(metadata.images.map((i) => i.id)));
  const backbones = structuredClone(metadata.backbones), close_call = {};
  let dinoQueries;
  for (const spec of BACKBONES) {
    const info = backbones[spec.id];
    if (!info || info.repo !== spec.modelId || info.dtype !== spec.dtype || info.shortest_edge !== spec.shortestEdge || info.vectors !== vectorFile(spec.id)) throw new Error(`Backbone identity mismatch: ${spec.id}`);
    const buffer = readFileSync(join(out, info.vectors));
    const checksum = hash(buffer);
    if (buffer.length !== metadata.images.length * info.dimensions * 4 || buffer.length !== info.bytes) throw new Error(`Vector length mismatch: ${spec.id}`);
    if (info.image_order_sha256 && info.image_order_sha256 !== imageOrderHash) throw new Error(`Vector image order mismatch: ${spec.id}`);
    if (info.vectors_sha256 && info.vectors_sha256 !== checksum) throw new Error(`Vector checksum mismatch: ${spec.id}`);
    const flat = new Float32Array(metadata.images.length * info.dimensions);
    for (let i = 0; i < flat.length; i += 1) {
      flat[i] = buffer.readFloatLE(i * 4);
      if (!Number.isFinite(flat[i])) throw new Error(`Nonfinite vector: ${spec.id}`);
    }
    const vectors = metadata.images.map((_, i) => flat.subarray(i * info.dimensions, (i + 1) * info.dimensions));
    const result = calibrate(vectors, metadata.images);
    for (const [key, value] of Object.entries(result.measured)) {
      if (!Number.isFinite(info.measured[key]) || Math.abs(value - info.measured[key]) > 1e-12) throw new Error(`Shipped measurement differs: ${spec.id}.${key}`);
    }
    console.log(`${spec.label}: measured reproduced to 1e-12: ${JSON.stringify(result.measured)}`);
    info.image_order_sha256 = imageOrderHash; info.vectors_sha256 = checksum;
    close_call[spec.id] = closeCallStatistics(result.queries);
    console.log(`${spec.label} close_call: ${JSON.stringify(close_call[spec.id])}`);
    if (spec.id === 'dinov2') dinoQueries = result.queries;
  }
  const examples = selectExamples(dinoQueries, dwarfs);
  console.log(`Examples: ${JSON.stringify(examples)}`);
  return { ...metadata, metadata_updated_at: new Date().toISOString(), backbones, dwarfs, location_prior, close_call, examples };
}

export function writeMetadataAtomic(path, metadata) {
  const temporary = `${path}.${process.pid}.tmp`;
  try {
    writeFileSync(temporary, `${JSON.stringify(metadata, null, 1)}\n`, { flag: 'wx' });
    renameSync(temporary, path);
  } finally { try { unlinkSync(temporary); } catch (error) { if (error.code !== 'ENOENT') throw error; } }
}

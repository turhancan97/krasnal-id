/**
 * One resize implementation, shared by the browser and the build.
 *
 * The reference vectors and a visitor's query must be preprocessed identically or
 * retrieval degrades, and neither platform's built-in resampler can provide that:
 * a canvas downscale is browser-dependent and differs from sharp's lanczos3 by
 * enough to cost cosine agreement (0.98 measured, where 1.00 is needed). So the
 * pixels are resized here, in arithmetic that is identical everywhere.
 *
 * Two stages, both deterministic:
 *   1. An integer box reduction, which is exact averaging and cheap. This does
 *      most of the work for a phone photograph.
 *   2. A Lanczos-3 pass to the exact target size.
 *
 * Operates on tightly packed RGBA bytes and returns the same.
 */

const LANCZOS_A = 3;

function lanczos(x) {
  if (x === 0) return 1;
  const abs = Math.abs(x);
  if (abs >= LANCZOS_A) return 0;
  const px = Math.PI * abs;
  return (LANCZOS_A * Math.sin(px) * Math.sin(px / LANCZOS_A)) / (px * px);
}

/** Average non-overlapping factor x factor blocks. Exact, and fast. */
function boxReduce(data, width, height, factor) {
  const outWidth = Math.floor(width / factor);
  const outHeight = Math.floor(height / factor);
  const out = new Uint8ClampedArray(outWidth * outHeight * 4);
  const area = factor * factor;
  for (let y = 0; y < outHeight; y += 1) {
    for (let x = 0; x < outWidth; x += 1) {
      let r = 0;
      let g = 0;
      let b = 0;
      let a = 0;
      for (let dy = 0; dy < factor; dy += 1) {
        let row = ((y * factor + dy) * width + x * factor) * 4;
        for (let dx = 0; dx < factor; dx += 1) {
          r += data[row];
          g += data[row + 1];
          b += data[row + 2];
          a += data[row + 3];
          row += 4;
        }
      }
      const at = (y * outWidth + x) * 4;
      out[at] = Math.round(r / area);
      out[at + 1] = Math.round(g / area);
      out[at + 2] = Math.round(b / area);
      out[at + 3] = Math.round(a / area);
    }
  }
  return { data: out, width: outWidth, height: outHeight };
}

/** Precompute the source indices and weights for one separable axis. */
function axisWeights(sourceLength, targetLength) {
  const scale = sourceLength / targetLength;
  const support = scale > 1 ? LANCZOS_A * scale : LANCZOS_A;
  const rows = [];
  for (let i = 0; i < targetLength; i += 1) {
    const centre = (i + 0.5) * scale - 0.5;
    const first = Math.max(0, Math.ceil(centre - support));
    const last = Math.min(sourceLength - 1, Math.floor(centre + support));
    const indices = [];
    const weights = [];
    let total = 0;
    for (let j = first; j <= last; j += 1) {
      const weight = lanczos(scale > 1 ? (j - centre) / scale : j - centre);
      if (weight === 0) continue;
      indices.push(j);
      weights.push(weight);
      total += weight;
    }
    if (total === 0) {
      indices.push(Math.min(sourceLength - 1, Math.max(0, Math.round(centre))));
      weights.push(1);
      total = 1;
    }
    for (let k = 0; k < weights.length; k += 1) weights[k] /= total;
    rows.push({ indices, weights });
  }
  return rows;
}

function resamplePass(data, width, height, targetLength, horizontal) {
  const rows = axisWeights(horizontal ? width : height, targetLength);
  const outWidth = horizontal ? targetLength : width;
  const outHeight = horizontal ? height : targetLength;
  const out = new Uint8ClampedArray(outWidth * outHeight * 4);
  for (let y = 0; y < outHeight; y += 1) {
    for (let x = 0; x < outWidth; x += 1) {
      const { indices, weights } = rows[horizontal ? x : y];
      let r = 0;
      let g = 0;
      let b = 0;
      let a = 0;
      for (let k = 0; k < indices.length; k += 1) {
        const at = horizontal
          ? (y * width + indices[k]) * 4
          : (indices[k] * width + x) * 4;
        const weight = weights[k];
        r += data[at] * weight;
        g += data[at + 1] * weight;
        b += data[at + 2] * weight;
        a += data[at + 3] * weight;
      }
      const to = (y * outWidth + x) * 4;
      out[to] = r;
      out[to + 1] = g;
      out[to + 2] = b;
      out[to + 3] = a;
    }
  }
  return { data: out, width: outWidth, height: outHeight };
}

/**
 * Scale RGBA pixels so the shorter side is exactly `target`, preserving aspect.
 * Never enlarges beyond what the aspect requires.
 */
export function resizeToShortestEdge(data, width, height, target) {
  let current = { data, width, height };

  const factor = Math.floor(Math.min(current.width, current.height) / target);
  if (factor >= 2) {
    current = boxReduce(current.data, current.width, current.height, factor);
  }

  const scale = target / Math.min(current.width, current.height);
  const outWidth = Math.max(target, Math.round(current.width * scale));
  const outHeight = Math.max(target, Math.round(current.height * scale));
  if (outWidth === current.width && outHeight === current.height) return current;

  current = resamplePass(current.data, current.width, current.height, outWidth, true);
  current = resamplePass(current.data, current.width, current.height, outHeight, false);
  return current;
}

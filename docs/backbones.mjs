/**
 * The two backbones this site can run, described once for both sides.
 *
 * `demo/build.mjs` embeds the references with these settings and `app.js`
 * embeds the query with them, and the two must agree exactly or the cosine
 * scores are meaningless. They used to agree by having the same constants typed
 * into both files, which held only as long as nobody edited one of them; now
 * there is one description and two importers.
 *
 * Neither the model class nor the pooling rule is shared between the two, which
 * is the whole reason this file exists:
 *
 *   - DINOv2 has no pooled output, so its vector is the CLS token, position 0
 *     of `last_hidden_state`. CLIP has a projection head and its vector is
 *     `image_embeds`. Mean-pooling DINOv2's patches instead would be a
 *     different embedding that does not compare with the shipped references.
 *   - DINOv2's processor resizes the shortest edge to 256 and then centre-crops
 *     224; CLIP's goes straight to 224. Pre-scaling DINOv2 to 224 through
 *     resize.mjs would crop the border away and silently change the input.
 *
 * `class` is a name rather than the class itself because the browser loads
 * transformers.js from a CDN and the build loads it from npm, so each importer
 * resolves it against its own copy of the library.
 */

/**
 * DINOv2 is the default and the page says so: it is the model the research
 * pipeline uses, and it is both smaller and better here. CLIP is offered
 * because the project's most repeated finding is about the gap between them,
 * and a visitor watching it on their own photograph learns more than a chart
 * tells them.
 *
 * Download sizes are measured, not estimated — see AGENTS.md section 6.5. CLIP
 * is the *larger* of the two, which is the opposite of the premise the demo was
 * originally built on.
 */
export const BACKBONES = [
  {
    id: "dinov2",
    label: "DINOv2",
    role: "the pipeline's model",
    modelId: "Xenova/dinov2-base",
    dtype: "q4",
    shortestEdge: 256,
    className: "AutoModel",
    downloadMb: 56,
    /** The CLS token — exactly what `embeddings/extract.py` takes. */
    pool: (output) => {
      const hidden = output.last_hidden_state;
      const width = hidden.dims[hidden.dims.length - 1];
      return hidden.data.subarray(0, width);
    },
  },
  {
    id: "clip",
    label: "CLIP",
    role: "the comparison arm",
    modelId: "Xenova/clip-vit-base-patch32",
    dtype: "q4",
    shortestEdge: 224,
    className: "CLIPVisionModelWithProjection",
    downloadMb: 64,
    /** The projected image embedding, which is CLIP's own output vector. */
    pool: (output) => output.image_embeds.data,
  },
];

/** The backbone the page starts on, and the one the research pipeline runs. */
export const DEFAULT_BACKBONE = "dinov2";

/** Look one up by id, failing loudly rather than returning undefined. */
export function backbone(id) {
  const found = BACKBONES.find((entry) => entry.id === id);
  if (!found) throw new Error(`unknown backbone: ${id}`);
  return found;
}

/** The file holding one backbone's reference vectors. */
export function vectorFile(id) {
  return `references-${id}.bin`;
}

/** Scale a vector to unit length so a dot product is a cosine. */
export function unitVector(values) {
  let sum = 0;
  for (const value of values) sum += value * value;
  const norm = Math.sqrt(sum) || 1;
  return Float32Array.from(values, (value) => value / norm);
}

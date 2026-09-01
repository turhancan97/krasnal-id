"""Build the static demo's data files from the manifest.

The browser embeds an uploaded photograph with a quantised ONNX export of CLIP,
then compares it against the reference vectors this script writes. Those
references are therefore computed with **the same ONNX model**, not with the
Python pipeline: a query and a reference that came from different pipelines
would disagree in ways nothing surfaces.

The script re-scores the leave-one-out protocol on the vectors it just wrote and
records the result in the metadata, so the accuracy the site reports is the
accuracy the site actually delivers.

    KRASNAL_ONNX_MODEL=/path/to/vision_model_q4.onnx \
      uv run --extra ml --with onnxruntime python docs/demo/generate.py

The model is not vendored. Download it from
https://huggingface.co/Xenova/clip-vit-base-patch32/resolve/main/onnx/vision_model_q4.onnx
Do not substitute `vision_model_quantized.onnx`: it is the one export that
measurably degrades retrieval (87.7% top-1 against 93.2% for this one).
"""

import json
import os
from collections import defaultdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from krasnal_id.embeddings.store import EmbeddingMatrix
from krasnal_id.experiments.baseline_accuracy import evaluate_baseline
from krasnal_id.experiments.geo_ablation import dwarf_locations, haversine_metres
from krasnal_id.models import DatasetManifest, EvaluationSplit

MODEL_ID = "Xenova/clip-vit-base-patch32"
MODEL_FILE = os.environ.get("KRASNAL_ONNX_MODEL", "vision_model_q4.onnx")
CLIP_REVISION = "c237dc49a33fc61debc9276459120b7eac67e7ef"

OUT = Path("docs/assets")
THUMBS = OUT / "thumbs"
THUMB_LONG_SIDE = 320
# Statues closer together than this are treated as one installation.
CO_LOCATED_METRES = 25.0


def load_inputs() -> tuple[DatasetManifest, EvaluationSplit]:
    """Read the generated manifest and its split."""
    manifest = DatasetManifest.model_validate(
        json.loads(Path("data/manifest.json").read_text(encoding="utf-8"))
    )
    split = EvaluationSplit.model_validate(
        json.loads(Path("data/splits/leave-one-out.json").read_text(encoding="utf-8"))
    )
    return manifest, split


def preprocess(paths: list[Path]) -> np.ndarray:
    """Apply CLIP's own image processor, so the vectors match the pinned model."""
    from transformers import AutoProcessor

    processor = AutoProcessor.from_pretrained(
        "openai/clip-vit-base-patch32", revision=CLIP_REVISION
    )
    batches = []
    for path in paths:
        with Image.open(path) as image:
            batches.append(
                processor(images=[image.convert("RGB")], return_tensors="np")["pixel_values"]
            )
    return np.concatenate(batches).astype(np.float32)


def embed(pixel_values: np.ndarray, model_path: Path) -> np.ndarray:
    """Run the browser's own ONNX export over every reference image."""
    import onnxruntime as ort

    options = ort.SessionOptions()
    options.intra_op_num_threads = 4
    options.inter_op_num_threads = 1
    options.log_severity_level = 3
    session = ort.InferenceSession(str(model_path), options, providers=["CPUExecutionProvider"])

    chunks = []
    for start in range(0, len(pixel_values), 16):
        chunk = pixel_values[start : start + 16]
        chunks.append(session.run(["image_embeds"], {"pixel_values": chunk})[0])
    vectors = np.concatenate(chunks).astype(np.float32)
    return np.asarray(vectors / np.linalg.norm(vectors, axis=1, keepdims=True), dtype=np.float32)


def co_located_groups(manifest: DatasetManifest) -> list[list[str]]:
    """Group dwarves that stand close enough to be one installation."""
    locations = dwarf_locations(manifest)
    ids = sorted(locations)
    parent = {dwarf: dwarf for dwarf in ids}

    def find(dwarf: str) -> str:
        while parent[dwarf] != dwarf:
            parent[dwarf] = parent[parent[dwarf]]
            dwarf = parent[dwarf]
        return dwarf

    for index, first in enumerate(ids):
        for second in ids[index + 1 :]:
            if haversine_metres(locations[first], locations[second]) <= CO_LOCATED_METRES:
                parent[find(second)] = find(first)

    grouped: dict[str, list[str]] = defaultdict(list)
    for dwarf in ids:
        grouped[find(dwarf)].append(dwarf)
    return [sorted(members) for members in grouped.values() if len(members) > 1]


def write_thumbnails(manifest: DatasetManifest) -> dict[str, str]:
    """Write one web-sized thumbnail per reference image."""
    THUMBS.mkdir(parents=True, exist_ok=True)
    names: dict[str, str] = {}
    for record in manifest.images:
        name = f"{record.image_id}.webp"
        with Image.open(record.local_path) as image:
            copy = image.convert("RGB")
            copy.thumbnail((THUMB_LONG_SIDE, THUMB_LONG_SIDE), Image.LANCZOS)
            copy.save(THUMBS / name, "WEBP", quality=80, method=6)
        names[record.image_id] = name
    return names


def main() -> None:
    """Build every file the static demo loads."""
    model_path = Path(MODEL_FILE)
    if not model_path.is_file():
        raise SystemExit(
            f"ONNX model not found: {model_path}\n"
            "Download vision_model_q4.onnx from "
            f"https://huggingface.co/{MODEL_ID}/resolve/main/onnx/vision_model_q4.onnx "
            "and set KRASNAL_ONNX_MODEL to its path. See this module's docstring."
        )

    manifest, split = load_inputs()
    records = sorted(manifest.images, key=lambda image: image.image_id)
    OUT.mkdir(parents=True, exist_ok=True)

    vectors = embed(preprocess([record.local_path for record in records]), model_path)

    # Score the protocol on exactly the vectors being shipped.
    matrix = EmbeddingMatrix(
        image_ids=tuple(record.image_id for record in records),
        dwarf_ids=tuple(record.dwarf_id for record in records),
        vectors=vectors,
    )
    measured = {metric.name: metric.value for metric in evaluate_baseline(split, matrix, (1, 5))}
    print(
        f"shipped vectors score top-1 {measured['top_1']:.1%}, "
        f"top-5 {measured['top_5']:.1%}, MRR {measured['mrr']:.4f}"
    )

    (OUT / "references.bin").write_bytes(vectors.astype("<f4").tobytes())
    thumbs = write_thumbnails(manifest)

    names = {dwarf.dwarf_id: dwarf.display_name for dwarf in manifest.dwarfs}
    metadata: dict[str, Any] = {
        "generated_at": datetime.now(UTC).isoformat(),
        "model": {"repo": MODEL_ID, "file": model_path.name, "dimensions": int(vectors.shape[1])},
        "manifest_sha256": split.manifest_sha256,
        "measured": {
            "top_1": measured["top_1"],
            "top_5": measured["top_5"],
            "mrr": measured["mrr"],
            "folds": int(measured["evaluated_folds"]),
        },
        "co_located_groups": co_located_groups(manifest),
        "dwarfs": [
            {"id": dwarf.dwarf_id, "name": dwarf.display_name}
            for dwarf in sorted(manifest.dwarfs, key=lambda d: d.dwarf_id)
        ],
        "images": [
            {
                "id": record.image_id,
                "dwarf": record.dwarf_id,
                "name": names[record.dwarf_id],
                "thumb": thumbs[record.image_id],
                "author": record.author,
                "license": record.license,
                "license_url": str(record.license_url),
                "source_url": str(record.source_url),
            }
            for record in records
        ],
    }
    (OUT / "references.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=1, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    thumb_bytes = sum(path.stat().st_size for path in THUMBS.iterdir())
    print(f"  references.bin   {(OUT / 'references.bin').stat().st_size / 1024:7.0f} KB")
    print(f"  references.json  {(OUT / 'references.json').stat().st_size / 1024:7.0f} KB")
    print(f"  thumbs/          {thumb_bytes / 1e6:7.1f} MB  ({len(thumbs)} files)")


if __name__ == "__main__":
    main()

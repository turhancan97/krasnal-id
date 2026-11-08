"""Region vectors for a multi-vector baseline: the CLS token plus grid-pooled patches.

The second internal review (AGENTS.md section 7.18) asked whether a compact
multi-vector representation, in the spirit of patch-wise retrieval and region
tokens, closes part of the gap that pairwise verification closes. This is the
simplest training-free version of that idea and not a reproduction of either
method: each image keeps its CLS token and the mean of its final-layer patch
tokens over each cell of a 2x2 and a 3x3 grid, every vector L2-normalized.

The vectors live in one `.npz` beside the embedding cache rather than in it: the
cache is keyed to one vector per image, and nothing but this baseline reads them.
"""

from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt

from krasnal_id.atomic import atomic_bytes
from krasnal_id.config import BackboneConfig
from krasnal_id.embeddings.backbone import import_optional_ml, resolve_device
from krasnal_id.embeddings.extract import _load_and_validate_image
from krasnal_id.models import DatasetManifest

GRIDS: tuple[int, ...] = (2, 3)


class RegionExtractionError(ValueError):
    """Raised when region vectors cannot be extracted or read safely."""


def region_count(grids: Sequence[int] = GRIDS) -> int:
    """The number of vectors per image: the CLS token plus every grid cell."""
    return 1 + sum(g * g for g in grids)


def pool_regions(
    cls: npt.NDArray[np.floating[Any]],
    patches: npt.NDArray[np.floating[Any]],
    grids: Sequence[int] = GRIDS,
) -> npt.NDArray[np.float32]:
    """CLS plus grid-cell means of a patch map, each row L2-normalized.

    `cls` is (n, d) and `patches` (n, h, w, d). A cell spans the patches whose
    row and column fall in its share of the map, by `np.array_split`, so an
    uneven map still covers every patch exactly once per grid.
    """
    if patches.ndim != 4 or cls.ndim != 2 or cls.shape[0] != patches.shape[0]:
        raise RegionExtractionError(
            f"expected cls (n, d) and patches (n, h, w, d), got {cls.shape} and {patches.shape}"
        )
    _, h, w, _ = patches.shape
    rows: list[npt.NDArray[np.float64]] = [cls.astype(np.float64)]
    for g in grids:
        if g > h or g > w:
            raise RegionExtractionError(f"a {g}x{g} grid needs at least {g} patches a side")
        for band in np.array_split(np.arange(h), g):
            for column in np.array_split(np.arange(w), g):
                cell = patches[:, band[0] : band[-1] + 1, column[0] : column[-1] + 1, :]
                rows.append(cell.astype(np.float64).mean(axis=(1, 2)))
    stacked = np.stack(rows, axis=1)
    norms = np.linalg.norm(stacked, axis=2, keepdims=True)
    if np.any(norms == 0):
        raise RegionExtractionError("a region vector has zero norm")
    return (stacked / norms).astype(np.float32)


def late_interaction(
    query: npt.NDArray[np.float32], references: npt.NDArray[np.float32]
) -> npt.NDArray[np.float64]:
    """Score references by the mean, over query regions, of the best reference region.

    `query` is (r, d) and `references` (n, r, d); the result is (n,). This is the
    ColBERT-style MaxSim, symmetric in nothing: every query region must find a
    counterpart, which is what lets a small distinctive region count.
    """
    similarity = np.einsum("qd,nrd->nqr", query.astype(np.float64), references.astype(np.float64))
    scores: npt.NDArray[np.float64] = similarity.max(axis=2).mean(axis=1)
    return scores


class RegionExtractor:
    """Run a CLS-token checkpoint once and keep its patch tokens as well as its CLS."""

    def __init__(self, config: BackboneConfig) -> None:
        if config.family not in {"dinov2", "dinov3"}:
            raise RegionExtractionError("region vectors need a dinov2- or dinov3-family backbone")
        self._config = config
        transformers = import_optional_ml("transformers")
        self._torch: Any = import_optional_ml("torch")
        self._device = resolve_device(config)
        self._processor = transformers.AutoImageProcessor.from_pretrained(
            config.model_id, revision=config.revision
        )
        self._model = transformers.AutoModel.from_pretrained(
            config.model_id, revision=config.revision
        )
        self._model.to(self._device)
        self._model.eval()
        self._patch = int(self._model.config.patch_size)

    def regions(self, images: Sequence[Any]) -> tuple[npt.NDArray[np.float32], ...]:
        """Return (cls, regions) for a batch: the plain CLS vector and the region stack."""
        inputs = self._processor(images=[i.convert("RGB") for i in images], return_tensors="pt")
        pixels = inputs["pixel_values"].to(self._device)
        with self._torch.inference_mode():
            hidden = self._model(pixel_values=pixels).last_hidden_state
        h, w = pixels.shape[2] // self._patch, pixels.shape[3] // self._patch
        tokens = hidden.detach().float().cpu().numpy()
        # Everything before the patch map is CLS and, for DINOv3, its registers.
        prefix = tokens.shape[1] - h * w
        if prefix < 1:
            raise RegionExtractionError(f"no CLS token before a {h}x{w} patch map")
        cls = tokens[:, 0, :]
        patches = tokens[:, prefix:, :].reshape(tokens.shape[0], h, w, tokens.shape[2])
        return cls.astype(np.float32), pool_regions(cls, patches)


def extract_regions(
    manifest: DatasetManifest, config: BackboneConfig, output: Path, batch_size: int = 16
) -> Path:
    """Write every manifest image's region stack to one `.npz`, in image-ID order."""
    records = tuple(sorted(manifest.images, key=lambda image: image.image_id))
    extractor = RegionExtractor(config)
    cls_rows: list[npt.NDArray[np.float32]] = []
    region_rows: list[npt.NDArray[np.float32]] = []
    for start in range(0, len(records), batch_size):
        batch = records[start : start + batch_size]
        cls, regions = extractor.regions([_load_and_validate_image(r) for r in batch])
        cls_rows.append(cls)
        region_rows.append(regions)
    payload = {
        "image_ids": np.asarray([r.image_id for r in records]),
        "cls": np.concatenate(cls_rows),
        "regions": np.concatenate(region_rows),
        "model_id": np.asarray(config.model_id),
        "revision": np.asarray(config.revision),
        "grids": np.asarray(GRIDS),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    with atomic_bytes(output, RegionExtractionError) as handle:
        np.savez(
            handle,
            image_ids=payload["image_ids"],
            cls=payload["cls"],
            regions=payload["regions"],
            model_id=payload["model_id"],
            revision=payload["revision"],
            grids=payload["grids"],
        )
    return output


def load_regions(
    path: Path, image_ids: Sequence[str], config: BackboneConfig
) -> npt.NDArray[np.float32]:
    """Read the region stacks in the given image order, refusing a stale or foreign file."""
    try:
        with np.load(path) as data:
            stored = [str(i) for i in data["image_ids"]]
            regions = np.asarray(data["regions"], dtype=np.float32)
            model_id, revision = str(data["model_id"]), str(data["revision"])
    except (OSError, KeyError, ValueError) as error:
        raise RegionExtractionError(f"cannot read region vectors {path}: {error}") from error
    if (model_id, revision) != (config.model_id, config.revision):
        raise RegionExtractionError(
            f"{path} was written by {model_id}@{revision}, not {config.model_id}@{config.revision}"
        )
    position = {image_id: index for index, image_id in enumerate(stored)}
    missing = [i for i in image_ids if i not in position]
    if missing:
        raise RegionExtractionError(f"{path} lacks {len(missing)} manifest images")
    return regions[[position[i] for i in image_ids]]

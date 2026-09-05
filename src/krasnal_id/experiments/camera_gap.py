"""Are photographs taken on phones harder queries than photographs taken on cameras?

`RESULTS.md` names the gap between a Commons upload and a casual phone photograph
as the largest untested thing in the project. Fieldwork is the clean way to
measure it. This is the proxy available without leaving the desk: Commons records
each file's EXIF camera, so the existing references split into phone-originated
and camera-originated queries and the two can be scored against the same
references.

The result is a **lower bound**. These are still Commons uploads — chosen,
often composed, taken by someone who meant to document the statue — so a real
snapshot is a harder query than anything measured here. See `AGENTS.md` 5.9.
"""

import re
from dataclasses import dataclass
from datetime import UTC, datetime
from statistics import median

from krasnal_id.config import AppConfig, CameraGapExperimentConfig
from krasnal_id.data_pipeline.camera_metadata import CameraMetadataFile
from krasnal_id.embeddings.store import EmbeddingMatrix, load_embedding_matrix
from krasnal_id.experiments.baseline_accuracy import (
    BaselineExperimentError,
    accuracy_metric,
    evaluate_fold,
    load_evaluation_inputs,
)
from krasnal_id.experiments.contracts import ExperimentResult, MetricSummary
from krasnal_id.models import DatasetManifest, EvaluationSplit

# Manufacturer and model strings that identify a phone. A heuristic, not a fact:
# it will mis-file an unusual device either way, which is why the experiment
# reports its group sizes and an `unknown` bucket rather than asking to be trusted.
PHONE_PATTERN = re.compile(
    r"iphone|ipad|\bsm-[ag]|galaxy|\bpixel\b|xiaomi|redmi|huawei|honor|oneplus|oppo|"
    r"\bvivo\b|motorola|moto |xperia|realme|poco|gt-i9|nexus",
    re.IGNORECASE,
)


class CameraGapError(ValueError):
    """Raised when camera-gap inputs are missing, invalid, or inconsistent."""


@dataclass(frozen=True, slots=True)
class GroupOutcome:
    """How one camera group's queries ranked, and what else was true of them."""

    group: str
    ranks: tuple[int, ...]
    references_per_class: tuple[int, ...]

    def top_k_hits(self, k: int) -> int:
        """Count queries whose correct dwarf ranked within the first k candidates."""
        return sum(1 for rank in self.ranks if rank <= k)


def classify_camera(camera: str | None) -> str:
    """Sort one camera description into phone, camera, or unknown."""
    if not camera or not camera.strip():
        return "unknown"
    return "phone" if PHONE_PATTERN.search(camera) else "camera"


def group_queries(
    split: EvaluationSplit,
    manifest: DatasetManifest,
    matrix: EmbeddingMatrix,
    metadata: CameraMetadataFile,
) -> tuple[GroupOutcome, ...]:
    """Score every fold and file it under the camera its query was taken on."""
    if not split.folds:
        raise CameraGapError("split contains no folds")

    page_by_image = {image.image_id: image.commons_page_id for image in manifest.images}
    references_per_class: dict[str, int] = {}
    for image in manifest.images:
        references_per_class[image.dwarf_id] = references_per_class.get(image.dwarf_id, 0) + 1

    ranks: dict[str, list[int]] = {"phone": [], "camera": [], "unknown": []}
    sizes: dict[str, list[int]] = {"phone": [], "camera": [], "unknown": []}
    for fold in split.folds:
        page_id = page_by_image.get(fold.query_image_id)
        group = classify_camera(metadata.cameras.get(str(page_id)) if page_id is not None else None)
        outcome = evaluate_fold(
            fold.query_image_id, fold.query_dwarf_id, fold.reference_image_ids, matrix
        )
        ranks[group].append(outcome.dwarf_rank)
        sizes[group].append(references_per_class[fold.query_dwarf_id])

    return tuple(
        GroupOutcome(
            group=group, ranks=tuple(ranks[group]), references_per_class=tuple(sizes[group])
        )
        for group in ("phone", "camera", "unknown")
        if ranks[group]
    )


def summarize_camera_gap(
    outcomes: tuple[GroupOutcome, ...],
    top_k: tuple[int, ...],
) -> tuple[MetricSummary, ...]:
    """Report each group, the gap between phone and camera, and the confounds.

    The confound a reader will reach for first is that phone photographs might
    belong to harder classes rather than being harder photographs, so the median
    references per class is reported per group beside the accuracy.
    """
    if not outcomes:
        raise CameraGapError("no queries were grouped")

    metrics: list[MetricSummary] = []
    for outcome in outcomes:
        total = len(outcome.ranks)
        for k in sorted(set(top_k)):
            metrics.append(
                accuracy_metric(f"{outcome.group}_top_{k}", outcome.top_k_hits(k), total)
            )
        metrics.append(MetricSummary(name=f"{outcome.group}_queries", value=float(total)))
        metrics.append(
            MetricSummary(
                name=f"{outcome.group}_median_references",
                value=float(median(outcome.references_per_class)),
            )
        )

    by_group = {outcome.group: outcome for outcome in outcomes}
    phone, camera = by_group.get("phone"), by_group.get("camera")
    if phone is not None and camera is not None:
        for k in sorted(set(top_k)):
            phone_rate = phone.top_k_hits(k) / len(phone.ranks)
            camera_rate = camera.top_k_hits(k) / len(camera.ranks)
            # Positive means phone queries did worse, which is the expected direction.
            metrics.append(MetricSummary(name=f"top_{k}_gap", value=camera_rate - phone_rate))
    return tuple(metrics)


def run_camera_gap(config: AppConfig, metadata: CameraMetadataFile) -> ExperimentResult:
    """Compare phone-originated and camera-originated queries on one split."""
    if not isinstance(config.experiment, CameraGapExperimentConfig):
        raise CameraGapError(
            f"the camera gap requires experiment=camera_gap, got {config.experiment.kind}"
        )

    try:
        manifest, split = load_evaluation_inputs(
            config.paths.manifest_path, config.paths.evaluation_split_path
        )
    except BaselineExperimentError as error:
        raise CameraGapError(str(error)) from error

    matrix = load_embedding_matrix(manifest, config.backbone, config.paths.embeddings_dir)
    outcomes = group_queries(split, manifest, matrix, metadata)

    return ExperimentResult(
        experiment="camera_gap",
        backbone=config.backbone.name,
        created_at=datetime.now(UTC),
        seed=config.experiment.seed,
        metrics=summarize_camera_gap(outcomes, config.experiment.top_k),
    )

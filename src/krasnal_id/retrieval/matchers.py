"""Local matchers behind one interface, so section 14 can be asked again.

Section 14 concluded that local features cannot serve as a first stage, on the
evidence of SIFT alone. That qualification is load-bearing: SIFT is a
hand-designed detector from 1999, bronze is close to its worst case -- specular,
low-texture, few stable corners -- and the failure it measured is concentrated
in exactly the wide-baseline, cross-illumination regime that learned matchers
were built for. Answering the question properly needs a second matcher behind
the same protocol, which is what this module provides.

The interface is deliberately the shape `FeatureCache` already had: describe an
image once, then score a pair. Everything expensive is per image, and a
candidate statue is proposed for many queries, so per-image work must be cached
and per-pair work must not be.
"""

from pathlib import Path
from typing import Any, Protocol, runtime_checkable

import numpy as np
import numpy.typing as npt

from krasnal_id.retrieval.rerank import (
    DETECT_LONG_SIDE,
    MINIMUM_CORRESPONDENCES,
    RANSAC_TOLERANCE,
    RerankError,
)

# Keypoints kept per image. Matched to SIFT's 800 would understate a learned
# detector, which is designed to propose more and let the matcher reject; 2048
# is the value LightGlue's own evaluations use.
LEARNED_KEYPOINTS = 2048


@runtime_checkable
class LocalMatcher(Protocol):
    """Describe images once, score pairs many times."""

    @property
    def name(self) -> str:
        """Return the matcher's artifact identity."""
        ...

    def get(self, image_id: str, path: Path) -> Any:
        """Return one image's cached description."""
        ...

    def inliers(self, query: Any, candidate: Any) -> int:
        """Return how many correspondences survive a RANSAC homography."""
        ...


def _ransac_inliers(
    source: npt.NDArray[np.float32],
    target: npt.NDArray[np.float32],
) -> int:
    """Count correspondences consistent with one homography.

    Shared with SIFT rather than reimplemented, so a difference between matchers
    is a difference in correspondences and not in how they were verified.
    """
    import cv2

    if source.shape[0] < MINIMUM_CORRESPONDENCES:
        return 0
    _, mask = cv2.findHomography(
        source.reshape(-1, 1, 2), target.reshape(-1, 1, 2), cv2.RANSAC, RANSAC_TOLERANCE
    )
    return 0 if mask is None else int(mask.sum())


class DiskLightGlueMatcher:
    """DISK keypoints matched by LightGlue, via kornia.

    DISK rather than SuperPoint on purpose. SuperPoint's published weights are
    research-only, and section 8 requires that whatever reaches `docs/` be
    licensed for it -- so the matcher that gets measured here is one that could
    also ship. Kornia's DISK weights are Apache-2.0 like kornia itself.
    """

    def __init__(self, device: str = "auto", max_keypoints: int = LEARNED_KEYPOINTS) -> None:
        self._device_request = device
        self.max_keypoints = max_keypoints
        self._disk: Any | None = None
        self._matcher: Any | None = None
        self._torch: Any | None = None
        self._device: Any | None = None
        self._described: dict[str, tuple[Any, Any]] = {}

    @property
    def name(self) -> str:
        """Return the matcher's artifact identity."""
        return "disk-lightglue"

    def _ensure_loaded(self) -> None:
        """Import kornia and load weights once, only when a pair is scored."""
        if self._disk is not None:
            return
        try:
            import torch
            from kornia.feature import DISK, LightGlueMatcher
        except ImportError as error:  # pragma: no cover - exercised by the extra's absence
            raise RerankError(
                "the disk-lightglue matcher needs the match extra; run uv sync --extra match"
            ) from error
        self._torch = torch
        resolved = self._device_request
        if resolved == "auto":
            resolved = "cuda" if torch.cuda.is_available() else "cpu"
        if resolved == "cuda" and not torch.cuda.is_available():
            raise RerankError("CUDA was requested for the matcher but is not available")
        self._device = torch.device(resolved)
        self._disk = DISK.from_pretrained("depth").to(self._device).eval()
        self._matcher = LightGlueMatcher("disk").to(self._device).eval()

    def _read(self, path: Path) -> Any:
        """Load one image as a normalized RGB tensor at the detection scale.

        OpenCV rather than `kornia.io`, whose Rust backend is out of step with the
        pinned kornia and raises on `read_image_jpegturbo`. The rest of the
        pipeline reads images with OpenCV anyway, so this also keeps one decoder.
        """
        import cv2

        assert self._torch is not None
        image = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if image is None:
            raise RerankError(f"could not read {path} as an image")
        height, width = image.shape[:2]
        longest = max(height, width)
        if longest > DETECT_LONG_SIDE:
            scale = DETECT_LONG_SIDE / longest
            image = cv2.resize(
                image, (int(width * scale), int(height * scale)), interpolation=cv2.INTER_AREA
            )
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        return self._torch.from_numpy(rgb).permute(2, 0, 1)[None].to(self._device)

    def get(self, image_id: str, path: Path) -> tuple[Any, Any]:
        """Return one image's keypoints and descriptors, describing it on first use."""
        cached = self._described.get(image_id)
        if cached is not None:
            return cached
        self._ensure_loaded()
        assert self._disk is not None
        assert self._torch is not None
        with self._torch.inference_mode():
            features = self._disk(
                self._read(path),
                n=self.max_keypoints,
                window_size=5,
                score_threshold=0.0,
                pad_if_not_divisible=True,
            )[0]
        described = (features.keypoints, features.descriptors)
        self._described[image_id] = described
        return described

    def inliers(self, query: tuple[Any, Any], candidate: tuple[Any, Any]) -> int:
        """Return how many LightGlue correspondences survive a RANSAC homography."""
        self._ensure_loaded()
        assert self._matcher is not None
        assert self._torch is not None
        from kornia.feature import laf_from_center_scale_ori

        (points_a, descriptors_a), (points_b, descriptors_b) = query, candidate
        if len(points_a) < MINIMUM_CORRESPONDENCES or len(points_b) < MINIMUM_CORRESPONDENCES:
            return 0
        lafs_a = laf_from_center_scale_ori(
            points_a[None], self._torch.ones(1, len(points_a), 1, 1, device=self._device)
        )
        lafs_b = laf_from_center_scale_ori(
            points_b[None], self._torch.ones(1, len(points_b), 1, 1, device=self._device)
        )
        with self._torch.inference_mode():
            _, indices = self._matcher(descriptors_a, descriptors_b, lafs_a, lafs_b)
        if indices.shape[0] < MINIMUM_CORRESPONDENCES:
            return 0
        source = points_a[indices[:, 0]].cpu().numpy().astype(np.float32)
        target = points_b[indices[:, 1]].cpu().numpy().astype(np.float32)
        return _ransac_inliers(source, target)

    def __len__(self) -> int:
        """Return how many images have been described so far."""
        return len(self._described)


def create_matcher(name: str, max_keypoints: int, device: str = "auto") -> LocalMatcher:
    """Build one matcher by name, without importing what it does not need."""
    if name == "sift":
        from krasnal_id.retrieval.rerank import FeatureCache

        return FeatureCache(max_keypoints)
    if name == "disk-lightglue":
        return DiskLightGlueMatcher(device=device)
    raise RerankError(f"unsupported matcher: {name}")

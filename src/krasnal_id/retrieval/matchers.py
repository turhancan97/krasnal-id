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
    Correspondences,
    RerankError,
    empty_correspondences,
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

    def correspondences(self, query: Any, candidate: Any) -> Correspondences:
        """Return where the two images matched, and which matches a homography kept."""
        ...


def _verify(
    source: npt.NDArray[np.float32],
    target: npt.NDArray[np.float32],
) -> Correspondences:
    """Mark the correspondences consistent with one homography.

    The same verification SIFT uses, at the same tolerance, so a difference
    between matchers is a difference in correspondences and not in how they were
    checked.
    """
    import cv2

    if source.shape[0] < MINIMUM_CORRESPONDENCES:
        return empty_correspondences()
    _, mask = cv2.findHomography(
        source.reshape(-1, 1, 2), target.reshape(-1, 1, 2), cv2.RANSAC, RANSAC_TOLERANCE
    )
    if mask is None:
        return empty_correspondences()  # type: ignore[unreachable]
    return Correspondences(source=source, target=target, inlier_mask=mask.ravel().astype(bool))


def usable_cuda(torch: Any) -> bool:
    """Is there a GPU this build can actually run a kernel on?

    `torch.cuda.is_available()` answers a different and weaker question: whether a
    driver and a device exist. A build compiled for sm_75 and up reports True on
    an sm_70 V100 and then raises `no kernel image is available for execution on
    the device` at the first real operation -- which here meant losing the SIFT
    arm's four minutes before the learned one had started. One tiny operation
    settles it in milliseconds, so it is asked before any weights are loaded.
    """
    if not torch.cuda.is_available():
        return False
    try:
        probe = torch.zeros(8, 8, device="cuda")
        torch.nn.functional.pad(probe, (0, 1, 0, 1)).sum().item()
    except Exception:
        # Deliberately broad: the failure arrives as a driver-level error whose
        # class varies by torch version, and any of them means the same thing.
        return False
    return True


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
        self._device = resolve_torch_device(self._device_request, torch)
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
        return self.correspondences(query, candidate).inliers

    def correspondences(
        self, query: tuple[Any, Any], candidate: tuple[Any, Any]
    ) -> Correspondences:
        """Return where LightGlue matched, and which matches the homography kept."""
        self._ensure_loaded()
        assert self._matcher is not None
        assert self._torch is not None
        from kornia.feature import laf_from_center_scale_ori

        (points_a, descriptors_a), (points_b, descriptors_b) = query, candidate
        if len(points_a) < MINIMUM_CORRESPONDENCES or len(points_b) < MINIMUM_CORRESPONDENCES:
            return empty_correspondences()
        lafs_a = laf_from_center_scale_ori(
            points_a[None], self._torch.ones(1, len(points_a), 1, 1, device=self._device)
        )
        lafs_b = laf_from_center_scale_ori(
            points_b[None], self._torch.ones(1, len(points_b), 1, 1, device=self._device)
        )
        with self._torch.inference_mode():
            _, indices = self._matcher(descriptors_a, descriptors_b, lafs_a, lafs_b)
        if indices.shape[0] < MINIMUM_CORRESPONDENCES:
            return empty_correspondences()
        source = points_a[indices[:, 0]].cpu().numpy().astype(np.float32)
        target = points_b[indices[:, 1]].cpu().numpy().astype(np.float32)
        return _verify(source, target)

    def __len__(self) -> int:
        """Return how many images have been described so far."""
        return len(self._described)


# LoFTR's evidence settings. They are part of what an inlier count *means*, so they
# are fixed constants rather than options: changing either must change the matcher's
# name, and with it the journal identity, or a resume would mix two definitions.
LOFTR_LONG_SIDE = 840  # LoFTR's own outdoor evaluation scale; memory grows with tokens
# kornia's own coarse-match threshold, so no extra filter: the matcher runs as
# released, as DISK runs at LightGlue's defaults. A stricter 0.5 separated the one
# Słupniki pair used as a smoke test better (5 inliers to 0, against 15 to 11), and
# choosing it on that evidence would have been tuning on a test query.
LOFTR_MIN_CONFIDENCE = 0.2


def resolve_torch_device(request: str, torch: Any) -> Any:
    """Turn a device request into a device this torch build can really run on."""
    resolved = request
    if resolved == "auto":
        resolved = "cuda" if usable_cuda(torch) else "cpu"
    if resolved == "cuda" and not torch.cuda.is_available():
        raise RerankError("CUDA was requested for the matcher but is not available")
    if resolved == "cuda" and not usable_cuda(torch):
        raise RerankError(
            "CUDA is present but this torch build has no kernels for the device: "
            f"{torch.cuda.get_device_capability(0)} against {torch.cuda.get_arch_list()}"
        )
    return torch.device(resolved)


class LoftrMatcher:
    """Detector-free dense matching by LoFTR, via kornia (AGENTS.md section 7.15).

    The third matcher, added because a reviewer asked whether a dense,
    detector-free matcher does better than keypoints on specular, low-texture
    bronze. It does not fit "describe once, score many": LoFTR's network runs on
    the *pair*, so `get` caches only the prepared grayscale image and all the
    work happens in `correspondences`. Matches below `LOFTR_MIN_CONFIDENCE` (LoFTR's
    own threshold) are dropped, at most `max_keypoints` of the rest are kept by confidence, and the
    survivors go through the same homography check as SIFT and DISK.

    Kornia's LoFTR "outdoor" weights are released under Apache-2.0 with kornia;
    they reach no `docs/` deliverable either way.
    """

    def __init__(self, device: str = "auto", max_keypoints: int = LEARNED_KEYPOINTS) -> None:
        self._device_request = device
        self.max_keypoints = max_keypoints
        self._model: Any | None = None
        self._torch: Any | None = None
        self._device: Any | None = None
        self._prepared: dict[str, Any] = {}

    @property
    def name(self) -> str:
        """Return the matcher's artifact identity."""
        return "loftr"

    def _ensure_loaded(self) -> None:
        """Import kornia and load weights once, only when a pair is scored."""
        if self._model is not None:
            return
        try:
            import torch
            from kornia.feature import LoFTR
        except ImportError as error:  # pragma: no cover - exercised by the extra's absence
            raise RerankError(
                "the loftr matcher needs the match extra; run uv sync --extra match"
            ) from error
        self._torch = torch
        self._device = resolve_torch_device(self._device_request, torch)
        self._model = LoFTR(pretrained="outdoor").to(self._device).eval()

    def get(self, image_id: str, path: Path) -> Any:
        """Return one image as a grayscale tensor, kept on the CPU until a pair needs it."""
        cached = self._prepared.get(image_id)
        if cached is not None:
            return cached
        import cv2

        self._ensure_loaded()
        assert self._torch is not None
        image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if image is None:
            raise RerankError(f"could not read {path} as an image")
        height, width = image.shape[:2]
        scale = min(1.0, LOFTR_LONG_SIDE / max(height, width))
        # LoFTR's coarse level is one eighth of the input, so both sides must divide by 8.
        size = (max(8, int(width * scale) // 8 * 8), max(8, int(height * scale) // 8 * 8))
        image = cv2.resize(image, size, interpolation=cv2.INTER_AREA)
        prepared = self._torch.from_numpy(image.astype(np.float32) / 255.0)[None, None]
        self._prepared[image_id] = prepared
        return prepared

    def inliers(self, query: Any, candidate: Any) -> int:
        """Return how many LoFTR correspondences survive a RANSAC homography."""
        return self.correspondences(query, candidate).inliers

    def correspondences(self, query: Any, candidate: Any) -> Correspondences:
        """Return where LoFTR matched, and which matches the homography kept."""
        self._ensure_loaded()
        assert self._model is not None
        assert self._torch is not None
        with self._torch.inference_mode():
            found = self._model(
                {"image0": query.to(self._device), "image1": candidate.to(self._device)}
            )
        confidence = found["confidence"].float().cpu().numpy()
        keep = np.flatnonzero(confidence >= LOFTR_MIN_CONFIDENCE)
        keep = keep[np.argsort(-confidence[keep], kind="stable")][: self.max_keypoints]
        if keep.size < MINIMUM_CORRESPONDENCES:
            return empty_correspondences()
        source = found["keypoints0"].cpu().numpy()[keep].astype(np.float32)
        target = found["keypoints1"].cpu().numpy()[keep].astype(np.float32)
        return _verify(source, target)

    def __len__(self) -> int:
        """Return how many images have been prepared so far."""
        return len(self._prepared)


def create_matcher(name: str, max_keypoints: int, device: str = "auto") -> LocalMatcher:
    """Build one matcher by name, without importing what it does not need."""
    if name == "sift":
        from krasnal_id.retrieval.rerank import FeatureCache

        return FeatureCache(max_keypoints)
    if name == "disk-lightglue":
        return DiskLightGlueMatcher(device=device, max_keypoints=max_keypoints)
    if name == "loftr":
        return LoftrMatcher(device=device, max_keypoints=max_keypoints)
    raise RerankError(f"unsupported matcher: {name}")

"""Geometric verification of a global ranking's top candidates.

A cosine ranking over whole-image embeddings compares appearance. Two
photographs of the *same* statue additionally admit a consistent geometry
between their matched keypoints, and two photographs of different-but-similar
statues do not — which is exactly the discrimination the errors here need, since
they concentrate on families of near-identical sculptures.

The method is the classical one: SIFT keypoints, a ratio-tested nearest-neighbour
match, and a RANSAC homography whose inlier count scores the pair. SIFT rather
than a learned detector on purpose — it downloads no weights, so the code is
exercised in CI like the rest of the pipeline instead of being skipped.

**The evidence is weak enough that it must not simply replace the ranking.**
Measured over sampled pairs, the same statue yields a median of 10 inliers and a
different statue 4, but the distributions overlap and some correct pairs yield
none. Replacing cosine order with inlier order would therefore demote correct
answers that happen to verify poorly. The inlier count is blended into the
similarity instead, with the weight swept — and a weight of zero has to reproduce
the unranked baseline exactly, which is what makes the rest of the sweep
believable.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt

# Above this, an inlier count carries no more information than "verified".
# Capping keeps one spectacular match from dominating a blended score.
INLIER_CAP = 30
# The long side every image is scaled to before detection. Full 2,000-pixel
# frames cost several times the time for no measured gain at this cap.
DETECT_LONG_SIDE = 1024
# Lowe's ratio, as published.
RATIO = 0.75
# RANSAC reprojection tolerance in pixels, at the detection scale.
RANSAC_TOLERANCE = 5.0
MINIMUM_CORRESPONDENCES = 4


class RerankError(ValueError):
    """Raised when local features cannot be computed or compared."""


@dataclass(frozen=True, slots=True)
class LocalFeatures:
    """One image's keypoint positions and descriptors.

    Descriptors are stored as `uint8`, which is lossless for SIFT's 0-255 range
    and keeps the whole corpus resident: 1,691 images at 800 keypoints is 173 MB
    as bytes against 693 MB as float32.
    """

    points: npt.NDArray[np.float32]
    descriptors: npt.NDArray[np.uint8]

    def __len__(self) -> int:
        """Return the number of keypoints."""
        return int(self.points.shape[0])


def _detector(max_keypoints: int) -> Any:
    """Build a SIFT detector, importing OpenCV only when features are wanted."""
    try:
        import cv2
    except ImportError as error:  # pragma: no cover - exercised by the extra's absence
        raise RerankError(
            "geometric re-ranking needs the rerank extra; run uv sync --extra rerank"
        ) from error
    # `cv2.SIFT.create` rather than the older `cv2.SIFT_create`: only the former
    # appears in OpenCV's own type stubs.
    return cv2.SIFT.create(nfeatures=max_keypoints)


def extract_features(path: Path, max_keypoints: int) -> LocalFeatures:
    """Detect and describe one image's keypoints."""
    import cv2

    image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise RerankError(f"could not read {path} as an image")

    height, width = image.shape
    longest = max(height, width)
    if longest > DETECT_LONG_SIDE:
        scale = DETECT_LONG_SIDE / longest
        image = cv2.resize(
            image, (int(width * scale), int(height * scale)), interpolation=cv2.INTER_AREA
        )

    keypoints, descriptors = _detector(max_keypoints).detectAndCompute(image, None)
    if descriptors is None or not keypoints:
        return LocalFeatures(
            points=np.zeros((0, 2), dtype=np.float32),
            descriptors=np.zeros((0, 128), dtype=np.uint8),
        )
    return LocalFeatures(
        points=np.asarray([point.pt for point in keypoints], dtype=np.float32),
        descriptors=np.asarray(descriptors, dtype=np.uint8),
    )


def count_inliers(query: LocalFeatures, candidate: LocalFeatures) -> int:
    """Return how many correspondences survive a RANSAC homography.

    Zero means "no geometry consistent with the same object was found", which
    covers both a genuine mismatch and an image too smooth or too small to
    describe. The experiment therefore treats zero as absence of evidence rather
    than evidence of absence.
    """
    import cv2

    if len(query) < MINIMUM_CORRESPONDENCES or len(candidate) < MINIMUM_CORRESPONDENCES:
        return 0

    pairs = cv2.BFMatcher().knnMatch(
        query.descriptors.astype(np.float32), candidate.descriptors.astype(np.float32), k=2
    )
    good = [
        first
        for first, second in (pair for pair in pairs if len(pair) == 2)
        if first.distance < RATIO * second.distance
    ]
    if len(good) < MINIMUM_CORRESPONDENCES:
        return 0

    source = query.points[[match.queryIdx for match in good]].reshape(-1, 1, 2)
    target = candidate.points[[match.trainIdx for match in good]].reshape(-1, 1, 2)
    _, mask = cv2.findHomography(source, target, cv2.RANSAC, RANSAC_TOLERANCE)
    return 0 if mask is None else int(mask.sum())


def blended_score(cosine: float, inliers: int, weight: float) -> float:
    """Combine appearance and geometry into one score.

    At `weight` zero this is the cosine similarity unchanged, which is the
    property that lets the sweep be read as a difference from the baseline rather
    than as a separate pipeline.
    """
    if weight < 0.0:
        raise RerankError(f"the blend weight cannot be negative, got {weight}")
    return cosine + weight * min(inliers, INLIER_CAP) / INLIER_CAP


class FeatureCache:
    """Lazily computed local features, kept for the run.

    A candidate statue is proposed for many different queries, so recomputing its
    features per query would dominate the cost. The whole corpus fits, so nothing
    is evicted.
    """

    def __init__(self, max_keypoints: int) -> None:
        self.max_keypoints = max_keypoints
        self._features: dict[str, LocalFeatures] = {}

    def get(self, image_id: str, path: Path) -> LocalFeatures:
        """Return one image's features, computing them on first use."""
        cached = self._features.get(image_id)
        if cached is None:
            cached = extract_features(path, self.max_keypoints)
            self._features[image_id] = cached
        return cached

    def __len__(self) -> int:
        """Return how many images have been described so far."""
        return len(self._features)

"""Several local matchers through one re-ranking protocol."""

from pathlib import Path
from unittest import mock

import pytest

from krasnal_id.config import MatcherRerankConfig, RerankAblationConfig, load_config
from krasnal_id.experiments.matcher_rerank import (
    MatcherRerankError,
    MatcherRun,
    best_weight,
    run_matcher_rerank,
    summarize,
)
from krasnal_id.experiments.rerank_ablation import Candidate, QueryEvidence
from krasnal_id.retrieval.matchers import (
    DiskLightGlueMatcher,
    create_matcher,
    usable_cuda,
)
from krasnal_id.retrieval.rerank import INLIER_CAP, FeatureCache, RerankError


def _evidence(query: str, cosine: tuple[float, ...], inliers: tuple[int, ...]) -> QueryEvidence:
    """One query whose first candidate is the correct statue."""
    return QueryEvidence(
        query_image_id=query,
        baseline_rank=1,
        candidates=tuple(
            Candidate(dwarf_id=f"Q{i}", cosine=c, inliers=n, correct=i == 0)
            for i, (c, n) in enumerate(zip(cosine, inliers, strict=True))
        ),
    )


def test_sift_and_a_learned_matcher_share_one_interface() -> None:
    """The seam that lets section 14's question be asked of something else.

    `collect_evidence` takes whatever has `name`, `get` and `inliers`, so the
    protocol is identical across matchers and a difference between columns is a
    difference in correspondences rather than in procedure.
    """
    sift = create_matcher("sift", max_keypoints=200)
    learned = create_matcher("disk-lightglue", max_keypoints=200)

    assert isinstance(sift, FeatureCache)
    assert isinstance(learned, DiskLightGlueMatcher)
    for matcher in (sift, learned):
        assert matcher.name in {"sift", "disk-lightglue"}
        assert callable(matcher.get)
        assert callable(matcher.inliers)


def test_an_unknown_matcher_is_refused() -> None:
    with pytest.raises(RerankError, match="unsupported matcher"):
        create_matcher("superglue", max_keypoints=200)


def test_the_learned_matcher_loads_nothing_until_a_pair_is_scored() -> None:
    """CI has no weights and no network, so construction must stay inert."""
    matcher = DiskLightGlueMatcher()

    assert len(matcher) == 0
    assert matcher.name == "disk-lightglue"


def test_the_best_weight_breaks_ties_toward_the_control() -> None:
    """A matcher that gains nothing must be reported at zero, not at a lucky tie."""
    # Geometry agrees with cosine, so every weight scores the same.
    evidence = (_evidence("a", (0.9, 0.5), (30, 1)),)
    truths = {"a": "Q0"}

    weight, accuracy = best_weight(evidence, truths, (0.0, 0.05, 0.2), cut_off=1)

    assert weight == 0.0
    assert accuracy == pytest.approx(1.0)


def test_a_weight_that_actually_helps_is_chosen() -> None:
    """Cosine puts the wrong statue first; geometry at weight 0.2 reverses it."""
    evidence = (_evidence("a", (0.50, 0.55), (30, 0)),)
    truths = {"a": "Q0"}

    weight, accuracy = best_weight(evidence, truths, (0.0, 0.2), cut_off=1)

    assert weight == 0.2
    assert accuracy == pytest.approx(1.0)


def test_matchers_are_paired_against_the_first_one() -> None:
    """Two matchers over the same queries are differenced, not compared by interval."""
    truths = {"a": "Q0", "b": "Q0"}
    # SIFT finds neither; the learned matcher finds both.
    # The cosine gap must be inside the blend's reach or no weight can flip it:
    # `blended_score` adds at most `weight` for a fully-verified candidate.
    sift = MatcherRun(
        "sift", (_evidence("a", (0.5, 0.55), (0, 0)), _evidence("b", (0.5, 0.55), (0, 0)))
    )
    learned = MatcherRun(
        "disk-lightglue",
        (_evidence("a", (0.5, 0.55), (30, 0)), _evidence("b", (0.5, 0.55), (30, 0))),
    )

    metrics = {m.name: m.value for m in summarize((sift, learned), truths, (0.0, 0.2), (1,))}

    assert metrics["sift_best_top_1"] == pytest.approx(0.0)
    assert metrics["disk-lightglue_best_top_1"] == pytest.approx(1.0)
    assert metrics["disk-lightglue_vs_sift_top_1_wins"] == 2
    assert metrics["disk-lightglue_vs_sift_top_1_losses"] == 0
    assert metrics["queries"] == 2


def test_matchers_scored_on_different_queries_are_refused() -> None:
    """An unpaired comparison is worse than none, so it is an error."""
    truths = {"a": "Q0", "b": "Q0"}
    sift = MatcherRun("sift", (_evidence("a", (0.9,), (1,)), _evidence("b", (0.9,), (1,))))
    short = MatcherRun("disk-lightglue", (_evidence("a", (0.9,), (1,)),))

    with pytest.raises(MatcherRerankError, match="would not be paired"):
        summarize((sift, short), truths, (0.0,), (1,))

    with pytest.raises(MatcherRerankError, match="at least one matcher"):
        summarize((), truths, (0.0,), (1,))


def test_run_requires_the_matcher_rerank_experiment_group() -> None:
    with pytest.raises(MatcherRerankError, match="requires experiment=matcher_rerank"):
        run_matcher_rerank(load_config(["experiment=baseline"]))


def test_packaged_defaults_compare_sift_against_a_learned_matcher() -> None:
    """The published run: section 7.6's settings, in the regime that matters."""
    experiment = load_config(["experiment=matcher_rerank"]).experiment
    assert isinstance(experiment, MatcherRerankConfig)

    assert experiment.matchers[0] == "sift", "SIFT is the baseline others are paired against"
    assert len(experiment.matchers) >= 2
    assert experiment.photographer_disjoint, "section 7.8 says this is the regime that matters"
    # Matched to the re-ranking sweep, or the columns do not land against its 94.0%.
    rerank = load_config(["experiment=rerank_ablation"]).experiment
    assert isinstance(rerank, RerankAblationConfig)
    assert experiment.top_k == rerank.top_k
    assert experiment.weights == rerank.weights

    for broken, message in (
        ({"weights": (0.1,)}, "zero control weight"),
        ({"weights": (-1.0, 0.0)}, "cannot be negative"),
        ({"matchers": ("sift", "sift")}, "cannot contain duplicates"),
        ({"top_k_metrics": (0,)}, "must be positive"),
    ):
        with pytest.raises(ValueError, match=message):
            MatcherRerankConfig.model_validate(
                {
                    "kind": "matcher_rerank",
                    "seed": 1,
                    "matchers": ("sift", "disk-lightglue"),
                    "top_k": 10,
                    "max_keypoints": 800,
                    "weights": (0.0, 0.1),
                    "top_k_metrics": (1,),
                    **broken,
                }
            )


def test_the_learned_matcher_refuses_cuda_it_cannot_have() -> None:
    """An explicit device request must fail loudly rather than fall back silently."""
    import torch

    if torch.cuda.is_available():  # pragma: no cover - depends on the host
        pytest.skip("CUDA is available, so the refusal cannot be exercised")
    with pytest.raises(RerankError, match="CUDA was requested"):
        DiskLightGlueMatcher(device="cuda").get("x", Path("nonexistent.jpg"))


def test_an_unusable_gpu_is_not_a_usable_one() -> None:
    """`cuda.is_available()` answers a weaker question than callers assume.

    It reports a driver and a device, not whether this build has kernels for that
    device. A torch compiled for sm_75 and up says True on an sm_70 card and then
    raises at the first real operation -- which cost a run its completed SIFT arm
    before the learned matcher had started. The probe runs one tiny operation.
    """
    absent = mock.Mock()
    absent.cuda.is_available.return_value = False
    assert usable_cuda(absent) is False

    present_but_wrong = mock.Mock()
    present_but_wrong.cuda.is_available.return_value = True
    present_but_wrong.zeros.side_effect = RuntimeError(
        "CUDA error: no kernel image is available for execution on the device"
    )
    assert usable_cuda(present_but_wrong) is False


def test_a_working_gpu_passes_the_probe() -> None:
    working = mock.Mock()
    working.cuda.is_available.return_value = True

    assert usable_cuda(working) is True


def test_the_learned_matcher_gets_its_own_keypoint_budget() -> None:
    """Holding a learned detector to SIFT's 800 would handicap what is under test.

    1024 rather than LightGlue's usual 2048 costs nothing here: `blended_score`
    caps inliers at `INLIER_CAP`, so every pair scoring above 30 blends
    identically, and 1024 already clears it on the pairs that separate.
    """
    experiment = load_config(["experiment=matcher_rerank"]).experiment
    assert isinstance(experiment, MatcherRerankConfig)

    assert experiment.learned_keypoints != experiment.max_keypoints
    assert experiment.learned_keypoints > INLIER_CAP

    learned = create_matcher("disk-lightglue", experiment.learned_keypoints)
    assert isinstance(learned, DiskLightGlueMatcher)
    assert learned.max_keypoints == experiment.learned_keypoints

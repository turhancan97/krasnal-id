"""Full-candidate-pool top-1, top-5, and MRR baseline."""

from krasnal_id.config import AppConfig
from krasnal_id.experiments.contracts import ExperimentResult


def run_baseline(config: AppConfig) -> ExperimentResult:
    """Run the configured full-pool retrieval baseline."""
    raise NotImplementedError("Baseline evaluation is scheduled for v0.1")

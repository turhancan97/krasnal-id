"""Accuracy-versus-candidate-pool-size headline experiment."""

from krasnal_id.config import AppConfig
from krasnal_id.experiments.contracts import ExperimentResult


def run_pool_size_ablation(config: AppConfig) -> tuple[ExperimentResult, ...]:
    """Evaluate repeated synthetic candidate pools for every configured size."""
    raise NotImplementedError("Pool-size ablation is scheduled for v0.2")

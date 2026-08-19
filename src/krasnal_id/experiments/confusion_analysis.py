"""Most-confused dwarf-pair analysis."""

from krasnal_id.config import AppConfig
from krasnal_id.experiments.contracts import ExperimentResult


def run_confusion_analysis(config: AppConfig) -> ExperimentResult:
    """Identify and summarize systematic cross-class retrieval errors."""
    raise NotImplementedError("Confusion analysis is scheduled for v0.2")

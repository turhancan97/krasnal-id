"""Atomic persistence for structured experiment results."""

import json
import os
from pathlib import Path
from tempfile import NamedTemporaryFile

from krasnal_id.experiments.contracts import ExperimentResult


class ExperimentArtifactError(ValueError):
    """Raised when a result artifact cannot be written."""


def experiment_result_path(results_dir: Path, result: ExperimentResult) -> Path:
    """Return the deterministic artifact path for one experiment run."""
    return results_dir / f"{result.experiment}-{result.backbone}.json"


def write_experiment_result(path: Path, result: ExperimentResult) -> None:
    """Write a validated experiment result atomically as formatted JSON."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    try:
        with NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temporary_path = Path(handle.name)
            json.dump(
                result.model_dump(mode="json"),
                handle,
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
            )
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_path, path)
    except OSError as error:
        raise ExperimentArtifactError(f"could not write result {path}: {error}") from error
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)

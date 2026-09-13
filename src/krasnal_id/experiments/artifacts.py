"""Atomic persistence for structured experiment results."""

import json
import os
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import TYPE_CHECKING

from krasnal_id.experiments.contracts import ExperimentResult

if TYPE_CHECKING:  # pragma: no cover - imported for typing only
    # Deferred: config imports nothing from here, and this keeps it that way.
    from krasnal_id.config import AppConfig


class ExperimentArtifactError(ValueError):
    """Raised when a result artifact cannot be written."""


def result_variant(experiment: object) -> str | None:
    """Return the artifact variant an experiment configuration declares, if any.

    Read from the configuration rather than passed around, so the pre-flight guard
    and the write path cannot disagree about which file a run belongs in.
    """
    variant = getattr(experiment, "artifact_variant", None)
    return variant if isinstance(variant, str) and variant else None


def experiment_result_path(results_dir: Path, result: ExperimentResult) -> Path:
    """Return the deterministic artifact path for one experiment run.

    The name carries the experiment, an optional variant and the backbone. The
    variant exists because two genuinely different measurements — a re-ranking
    sweep with the photographer withheld and one without — are not two settings of
    one run, and making them share a filename left the overwrite guard as the only
    thing keeping them apart. That guard is blind to an artifact written before it
    existed, which is exactly how the run section 10 cites was lost.

    **Only experiments the visualizations do not glob may declare a variant.**
    `pool_size_ablation-*.json` and `open_set-*.json` are globbed and expect one
    file per backbone, so those configurations declare none and a test pins it.
    """
    stem = "-".join(part for part in (result.experiment, result.variant, result.backbone) if part)
    return results_dir / f"{stem}.json"


def _recorded_configuration(path: Path) -> tuple[bool, dict[str, object] | None]:
    """Return whether `path` holds a readable result, and the config it records."""
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        # Unreadable or absent: there is nothing to protect.
        return False, None
    if not isinstance(payload, dict):
        return False, None
    existing = payload.get("configuration")
    return True, existing if isinstance(existing, dict) else None


def _describe_difference(existing: dict[str, object], replacement: dict[str, object] | None) -> str:
    """Name the settings that differ, so the message says what would be lost."""
    new = replacement or {}
    keys = sorted(set(existing) | set(new))
    changed = [key for key in keys if existing.get(key) != new.get(key)]
    return ", ".join(f"{key}: {existing.get(key)!r} -> {new.get(key)!r}" for key in changed)


def replaces_a_different_run(path: Path, result: ExperimentResult) -> str | None:
    """Return a description of the clash if writing would discard another run.

    Two runs of one experiment under different settings produce the same filename,
    so without this a `top_k=50` sweep silently overwrites the `top_k=10` result a
    published section cites. Identical settings overwrite freely, which is the
    ordinary case of re-running after re-extracting embeddings.

    An artifact written before configurations were recorded cannot be compared, so
    it is replaced rather than blocking every re-run — and the replacement records
    a configuration, which arms the check from then on.
    """
    present, existing = _recorded_configuration(path)
    if not present or existing is None:
        return None
    if existing == result.configuration:
        return None
    return _describe_difference(existing, result.configuration)


def guard_result_path(config: "AppConfig") -> None:
    """Refuse a run whose result could not be saved, before it is computed.

    The write-time check is the backstop; this is the one that matters. A
    re-ranking sweep takes forty minutes, and discovering at the end that the
    artifact cannot be written without discarding another run is the same wasted
    time as no check at all.

    The artifact's `experiment` field equals its configuration group's `kind` for
    every experiment, so the destination is derivable from the configuration alone
    and this needs no knowledge of which command is running.
    """
    kind = getattr(config.experiment, "kind", None)
    if not isinstance(kind, str):  # pragma: no cover - every group declares one
        return
    stem = "-".join(
        part for part in (kind, result_variant(config.experiment), config.backbone.name) if part
    )
    path = config.paths.results_dir / f"{stem}.json"
    present, existing = _recorded_configuration(path)
    if not present or existing is None:
        return
    replacement = config.experiment.model_dump(mode="json")
    if existing == replacement:
        return
    raise ExperimentArtifactError(
        f"{path} holds a {kind} run with different settings "
        f"({_describe_difference(existing, replacement)}); this run would discard it. "
        "Point paths.results_dir elsewhere to keep both, or delete that file to replace it."
    )


def write_experiment_result(
    path: Path, result: ExperimentResult, *, allow_replace: bool = False
) -> None:
    """Write a validated experiment result atomically as formatted JSON.

    Refuses to discard an artifact produced by different settings unless
    `allow_replace` says to. See `replaces_a_different_run`.
    """
    if not allow_replace and (clash := replaces_a_different_run(path, result)) is not None:
        raise ExperimentArtifactError(
            f"{path} holds a {result.experiment} run with different settings ({clash}); "
            "writing would discard it. Point paths.results_dir elsewhere to keep both, "
            "or delete that file to replace it."
        )
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

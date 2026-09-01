"""Build and persist the deterministic evaluation split."""

import json
import os
from datetime import UTC, datetime
from pathlib import Path
from tempfile import NamedTemporaryFile

from pydantic import ValidationError

from krasnal_id.data_pipeline.build_manifest import canonical_json_sha256
from krasnal_id.models import DatasetManifest, EvaluationSplit, EvaluationSplitFold


class SplitConfigurationError(ValueError):
    """Raised when a manifest or split configuration is invalid."""


def build_evaluation_split(
    manifest: DatasetManifest,
    generated_at: datetime,
) -> EvaluationSplit:
    """Build one leave-one-out fold for every admitted image."""
    images = tuple(sorted(manifest.images, key=lambda image: image.image_id))
    folds = tuple(
        EvaluationSplitFold(
            query_image_id=query.image_id,
            query_dwarf_id=query.dwarf_id,
            reference_image_ids=tuple(
                image.image_id for image in images if image.image_id != query.image_id
            ),
        )
        for query in images
    )
    return EvaluationSplit(
        schema_version="1.0",
        strategy="leave_one_out",
        manifest_sha256=canonical_json_sha256(manifest.model_dump(mode="json")),
        generated_at=generated_at,
        folds=folds,
    )


def _read_manifest(path: Path) -> DatasetManifest:
    """Read and strictly validate the generated manifest."""
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        return DatasetManifest.model_validate(payload)
    except (OSError, json.JSONDecodeError, ValidationError, TypeError, ValueError) as error:
        raise SplitConfigurationError(f"invalid manifest {path}: {error}") from error


def build_split_from_artifact(
    manifest_path: Path,
    generated_at: datetime | None = None,
) -> EvaluationSplit:
    """Load a manifest and build its deterministic leave-one-out split."""
    return build_evaluation_split(
        _read_manifest(manifest_path),
        generated_at or datetime.now(UTC),
    )


def write_evaluation_split(path: Path, split: EvaluationSplit) -> None:
    """Write a validated split atomically."""
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
        ) as temporary:
            temporary_path = Path(temporary.name)
            json.dump(
                split.model_dump(mode="json"),
                temporary,
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
            )
            temporary.write("\n")
            temporary.flush()
            os.fsync(temporary.fileno())
        os.replace(temporary_path, path)
    except OSError as error:
        raise SplitConfigurationError(f"could not write split {path}: {error}") from error
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)

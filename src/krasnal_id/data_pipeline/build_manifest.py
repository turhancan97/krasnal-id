"""Assemble and validate a versioned local dataset manifest."""

from datetime import datetime

from krasnal_id.models import DatasetManifest, DwarfRecord, ImageRecord


def build_dataset_manifest(
    dwarfs: tuple[DwarfRecord, ...],
    images: tuple[ImageRecord, ...],
    generated_at: datetime,
    minimum_images_per_dwarf: int,
) -> DatasetManifest:
    """Filter insufficient classes and return a validated dataset manifest."""
    raise NotImplementedError("Manifest construction is scheduled for v0.1")

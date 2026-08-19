"""Fetch licensed image metadata and files from Wikimedia Commons."""

from pathlib import Path

from krasnal_id.config import WikimediaDataConfig
from krasnal_id.models import DwarfRecord, ImageRecord


def fetch_images(
    dwarfs: tuple[DwarfRecord, ...],
    destination: Path,
    config: WikimediaDataConfig,
) -> tuple[ImageRecord, ...]:
    """Download usable Commons images and preserve their attribution metadata."""
    raise NotImplementedError("Commons image fetching is scheduled for v0.1")

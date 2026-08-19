"""Discover Wroclaw dwarf entities and optional coordinates through Wikidata."""

from krasnal_id.config import WikimediaDataConfig
from krasnal_id.models import DwarfRecord


def query_dwarfs(config: WikimediaDataConfig) -> tuple[DwarfRecord, ...]:
    """Return normalized dwarf records discovered from Wikidata."""
    raise NotImplementedError("Wikidata querying is scheduled for v0.1")

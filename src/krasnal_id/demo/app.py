"""Gradio upload-and-retrieve demonstration."""

import getpass
import importlib
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from krasnal_id.config import AppConfig
from krasnal_id.embeddings.store import EmbeddingStoreError, load_embedding_matrix
from krasnal_id.retrieval.query import (
    QueryError,
    load_query_vector,
    rank_candidates,
    read_manifest,
)

_DESCRIPTION = """
Upload a photograph of a Wrocław dwarf statue to see which one it most resembles.

Candidates are ranked by cosine similarity between embeddings of the uploaded photo and
every reference photo in the dataset. This is a research prototype: the reference set
covers only the dwarves with enough usable Creative Commons photographs, so a statue
outside that set will still return its nearest neighbours rather than saying it is unknown.
"""


class DemoError(ValueError):
    """Raised when the demonstration cannot be configured or launched."""


def ensure_private_temp_dir() -> Path:
    """Point Gradio at a per-user scratch directory unless one is configured.

    Gradio defaults to a shared `/tmp/gradio`, which fails outright on a multi-user
    machine where another account created it first. An explicit GRADIO_TEMP_DIR is
    always respected.
    """
    configured = os.environ.get("GRADIO_TEMP_DIR")
    if configured:
        return Path(configured)
    try:
        user = getpass.getuser()
    except OSError:  # pragma: no cover - only when the account has no name
        user = str(os.getpid())
    private = Path(tempfile.gettempdir()) / f"gradio-krasnal-id-{user}"
    private.mkdir(parents=True, exist_ok=True)
    os.environ["GRADIO_TEMP_DIR"] = str(private)
    return private


def import_optional_demo(module_name: str) -> Any:
    """Import an optional demo dependency only when the demo is launched."""
    try:
        return importlib.import_module(module_name)
    except ImportError as error:
        raise DemoError(
            "demo dependencies are required to launch the interface; run uv sync --extra demo"
        ) from error


@dataclass(frozen=True, slots=True)
class DemoBundle:
    """Everything one demo session needs, loaded once rather than per query."""

    config: AppConfig
    manifest: Any
    matrix: Any
    top_k: int

    def image_path_for(self, image_id: str) -> Path | None:
        """Return the local reference photo for one image ID, when it still exists."""
        for record in self.manifest.images:
            if record.image_id == image_id:
                return record.local_path if record.local_path.is_file() else None
        return None


def load_bundle(config: AppConfig, top_k: int) -> DemoBundle:
    """Load the manifest and cached vectors once for the whole session."""
    if top_k <= 0:
        raise DemoError(f"top_k must be positive, got {top_k}")
    try:
        manifest = read_manifest(config.paths.manifest_path)
        matrix = load_embedding_matrix(manifest, config.backbone, config.paths.embeddings_dir)
    except (QueryError, EmbeddingStoreError) as error:
        raise DemoError(str(error)) from error
    return DemoBundle(config=config, manifest=manifest, matrix=matrix, top_k=top_k)


def identify(
    bundle: DemoBundle,
    image_path: str | None,
) -> tuple[list[list[str]], list[tuple[str, str]], str]:
    """Rank candidates for one uploaded photo.

    Returns a rows/gallery/status triple rather than raising, because a Gradio
    callback that raises shows the user a stack trace instead of an explanation.
    """
    if not image_path:
        return [], [], "Upload a photograph to identify it."

    try:
        vector, digest, reused = load_query_vector(
            Path(image_path),
            bundle.config.backbone,
            bundle.config.paths.embeddings_dir,
        )
        outcome = rank_candidates(vector, digest, bundle.manifest, bundle.matrix, bundle.top_k)
    except (QueryError, EmbeddingStoreError) as error:
        return [], [], f"Could not identify this photograph: {error}"

    rows = [
        [
            str(candidate.rank),
            candidate.display_name,
            candidate.dwarf_id,
            f"{candidate.cosine_similarity:.4f}",
        ]
        for candidate in outcome.candidates
    ]
    gallery: list[tuple[str, str]] = []
    for candidate in outcome.candidates:
        path = bundle.image_path_for(candidate.matched_image_id)
        if path is not None:
            gallery.append((str(path), f"{candidate.rank}. {candidate.display_name}"))

    source = "reused a cached embedding" if reused else "embedded with the backbone"
    status = (
        f"Ranked {len(outcome.candidates)} candidates against "
        f"{len(bundle.matrix.image_ids)} reference photos "
        f"using {bundle.config.backbone.name} ({source})."
    )
    if outcome.excluded_reference_image_ids:
        status += (
            " This photograph is already in the dataset, so its own reference copies "
            f"({', '.join(outcome.excluded_reference_image_ids)}) were withheld."
        )
    return rows, gallery, status


def build_interface(bundle: DemoBundle) -> Any:
    """Build the Gradio interface for one loaded bundle."""
    ensure_private_temp_dir()
    gradio = import_optional_demo("gradio")

    classes = len({record.dwarf_id for record in bundle.manifest.images})
    # Analytics are off by default: launching a local research demo should not
    # report to an external service. Pass analytics_enabled to override.
    with gradio.Blocks(title="Krasnal-ID", analytics_enabled=False) as interface:
        gradio.Markdown(f"# Krasnal-ID\n{_DESCRIPTION}")
        gradio.Markdown(
            f"Reference set: **{classes} dwarves**, "
            f"**{len(bundle.matrix.image_ids)} photographs**, "
            f"backbone **{bundle.config.backbone.name}**."
        )
        with gradio.Row():
            with gradio.Column():
                upload = gradio.Image(type="filepath", label="Photograph", sources=["upload"])
                submit = gradio.Button("Identify", variant="primary")
            with gradio.Column():
                table = gradio.Dataframe(
                    headers=["Rank", "Dwarf", "Wikidata ID", "Similarity"],
                    label=f"Top {bundle.top_k} candidates",
                    interactive=False,
                    wrap=True,
                )
                status = gradio.Markdown()
        gallery = gradio.Gallery(label="Closest reference photographs", columns=5, height=220)

        outputs = [table, gallery, status]
        submit.click(lambda path: identify(bundle, path), inputs=upload, outputs=outputs)
        upload.change(lambda path: identify(bundle, path), inputs=upload, outputs=outputs)
    return interface


def launch(
    config: AppConfig | None = None,
    top_k: int = 5,
    share: bool = False,
    server_port: int | None = None,
) -> Any:
    """Launch the optional browser demonstration."""
    if config is None:
        from krasnal_id.config import load_config

        config = load_config()
    interface = build_interface(load_bundle(config, top_k))
    interface.launch(share=share, server_port=server_port)
    return interface

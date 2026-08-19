"""Unified command-line interface for all Krasnal-ID pipeline stages."""

import logging
from typing import Annotated

import typer

from krasnal_id.config import load_config
from krasnal_id.data_pipeline.wikidata_query import (
    WikidataConfigurationError,
    WikidataQueryError,
    discovery_paths,
    query_dwarfs,
)
from krasnal_id.logging import configure_logging
from krasnal_id.models import AuditDisposition

OverrideOption = Annotated[
    list[str] | None,
    typer.Option("--override", "-o", help="Hydra override; repeat for multiple values."),
]
LimitOption = Annotated[
    int | None,
    typer.Option("--limit", min=1, help="Emit only the first N eligible records by QID."),
]
RefreshOption = Annotated[
    bool,
    typer.Option("--refresh", help="Ignore a valid cache and query Wikidata again."),
]

app = typer.Typer(help="Fine-grained retrieval research for Wroclaw dwarf statues.")
data_app = typer.Typer(help="Discover and cache licensed Wikimedia data.")
embeddings_app = typer.Typer(help="Extract and cache image embeddings.")
experiment_app = typer.Typer(help="Run reproducible retrieval experiments.")
visualize_app = typer.Typer(help="Create saved research visualizations.")

app.add_typer(data_app, name="data")
app.add_typer(embeddings_app, name="embeddings")
app.add_typer(experiment_app, name="experiment")
app.add_typer(visualize_app, name="visualize")


def _run_placeholder(
    stage: str,
    overrides: list[str] | None,
    required_overrides: list[str] | None = None,
) -> None:
    """Validate configuration, initialize logging, and report an unimplemented stage."""
    merged_overrides = [*(required_overrides or []), *(overrides or [])]
    config = load_config(merged_overrides)
    configure_logging(config.logging)
    logging.getLogger(__name__).info("validated placeholder stage: %s", stage)
    typer.echo(f"{stage} is not implemented yet.", err=True)
    raise typer.Exit(code=2)


@data_app.command("query")
def query_wikidata(
    override: OverrideOption = None,
    limit: LimitOption = None,
    refresh: RefreshOption = False,
) -> None:
    """Discover Wroclaw dwarf records through Wikidata."""
    config = load_config(override or [])
    configure_logging(config.logging)
    try:
        result = query_dwarfs(
            config.data,
            config.paths.discovery_dir,
            limit=limit,
            refresh=refresh,
        )
    except WikidataConfigurationError as error:
        typer.echo(f"Configuration error: {error}", err=True)
        raise typer.Exit(code=2) from error
    except WikidataQueryError as error:
        typer.echo(f"Query failed: {error}", err=True)
        raise typer.Exit(code=1) from error

    excluded = sum(record.disposition is AuditDisposition.EXCLUDED for record in result.audit)
    warnings = sum(record.disposition is AuditDisposition.WARNING for record in result.audit)
    paths = discovery_paths(config.paths.discovery_dir)
    typer.echo(
        "Wikidata discovery complete: "
        f"cache={result.cache_status} eligible={result.eligible_total} "
        f"emitted={len(result.records)} excluded={excluded} warnings={warnings}"
    )
    typer.echo(f"Records: {paths.dwarfs}")
    typer.echo(f"Audit: {paths.audit}")


@data_app.command("fetch")
def fetch_commons(override: OverrideOption = None) -> None:
    """Fetch licensed image metadata and files from Wikimedia Commons."""
    _run_placeholder("data fetch", override)


@data_app.command("build-manifest")
def build_manifest(override: OverrideOption = None) -> None:
    """Build and validate the versioned dataset manifest."""
    _run_placeholder("data build-manifest", override)


@embeddings_app.command("extract")
def extract_embeddings(override: OverrideOption = None) -> None:
    """Extract and cache embeddings for every admitted image."""
    _run_placeholder("embeddings extract", override)


@app.command("retrieve")
def retrieve(override: OverrideOption = None) -> None:
    """Retrieve the nearest dwarf candidates for a query image."""
    _run_placeholder("retrieve", override)


@experiment_app.command("baseline")
def baseline_experiment(override: OverrideOption = None) -> None:
    """Measure full-pool top-k accuracy and mean reciprocal rank."""
    _run_placeholder("experiment baseline", override, ["experiment=baseline"])


@experiment_app.command("pool-ablation")
def pool_ablation_experiment(override: OverrideOption = None) -> None:
    """Measure accuracy across synthetic candidate-pool sizes."""
    _run_placeholder("experiment pool-ablation", override, ["experiment=pool_size_ablation"])


@experiment_app.command("confusion")
def confusion_experiment(override: OverrideOption = None) -> None:
    """Find and summarize the most-confused dwarf pairs."""
    _run_placeholder("experiment confusion", override, ["experiment=confusion"])


@visualize_app.command("embeddings")
def visualize_embeddings(override: OverrideOption = None) -> None:
    """Project cached embeddings into a saved two-dimensional figure."""
    _run_placeholder("visualize embeddings", override, ["experiment=visualization"])


@app.command("demo")
def demo(override: OverrideOption = None) -> None:
    """Launch the optional Gradio retrieval demonstration."""
    _run_placeholder("demo", override)


if __name__ == "__main__":
    app()

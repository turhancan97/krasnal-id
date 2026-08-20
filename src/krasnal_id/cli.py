"""Unified command-line interface for all Krasnal-ID pipeline stages."""

import logging
from typing import Annotated

import typer

from krasnal_id.config import load_config
from krasnal_id.data_pipeline.build_manifest import (
    ManifestConfigurationError,
    build_manifest_from_artifacts,
    write_dataset_manifest,
)
from krasnal_id.data_pipeline.commons_fetch import (
    CommonsConfigurationError,
    fetch_images,
    fetch_paths,
    prepare_category_review,
)
from krasnal_id.data_pipeline.wikidata_query import (
    WikidataConfigurationError,
    WikidataQueryError,
    discovery_paths,
    query_dwarfs,
)
from krasnal_id.logging import configure_logging
from krasnal_id.models import (
    AuditDisposition,
    CategoryReviewStatus,
    FetchAuditDisposition,
)

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
PrepareReviewOption = Annotated[
    bool,
    typer.Option("--prepare-review", help="Create or update the offline category review file."),
]
MaxImagesOption = Annotated[
    int | None,
    typer.Option(
        "--max-images-per-dwarf",
        min=1,
        help="Download at most N eligible images for each approved dwarf.",
    ),
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
def fetch_commons(
    override: OverrideOption = None,
    prepare_review: PrepareReviewOption = False,
    max_images_per_dwarf: MaxImagesOption = None,
    refresh: bool = typer.Option(
        False,
        "--refresh",
        help="Refresh Commons metadata and changed image revisions.",
    ),
) -> None:
    """Fetch licensed image metadata and files from Wikimedia Commons."""
    config = load_config(override or [])
    configure_logging(config.logging)
    if prepare_review and (max_images_per_dwarf is not None or refresh):
        typer.echo(
            "Configuration error: --prepare-review cannot be combined with download options.",
            err=True,
        )
        raise typer.Exit(code=2)

    try:
        if prepare_review:
            review = prepare_category_review(
                config.paths.discovery_dir,
                config.paths.category_review_path,
            )
        else:
            result = fetch_images(
                config.paths.discovery_dir,
                config.paths.category_review_path,
                config.paths.images_dir,
                config.data,
                max_images_per_dwarf=max_images_per_dwarf,
                refresh=refresh,
            )
    except CommonsConfigurationError as error:
        typer.echo(f"Configuration error: {error}", err=True)
        raise typer.Exit(code=2) from error

    if prepare_review:
        pending = sum(record.status is CategoryReviewStatus.PENDING for record in review.records)
        typer.echo(f"Category review prepared: records={len(review.records)} pending={pending}")
        typer.echo(f"Review: {config.paths.category_review_path}")
        return

    excluded = sum(record.disposition is FetchAuditDisposition.EXCLUDED for record in result.audit)
    paths = fetch_paths(config.paths.discovery_dir)
    typer.echo(
        "Commons fetch complete: "
        f"approved={result.approved_categories} rejected={result.rejected_categories} "
        f"pending={result.pending_categories} discovered={result.discovered_images} "
        f"eligible={result.eligible_images} downloaded={result.downloaded_images} "
        f"reused={result.reused_images} excluded={excluded} "
        f"errors={result.operational_failures}"
    )
    typer.echo(f"Images: {paths.fetched_images}")
    typer.echo(f"Audit: {paths.audit}")
    if result.operational_failures:
        raise typer.Exit(code=1)


@data_app.command("build-manifest")
def build_manifest(override: OverrideOption = None) -> None:
    """Build and validate the versioned dataset manifest."""
    config = load_config(override or [])
    configure_logging(config.logging)
    try:
        manifest = build_manifest_from_artifacts(
            config.paths.discovery_dir / "dwarfs.json",
            config.paths.discovery_dir / "fetched-images.json",
            config.paths.category_review_path,
            config.paths.image_review_path,
            config.thresholds.minimum_images_per_dwarf,
        )
        write_dataset_manifest(config.paths.manifest_path, manifest)
    except ManifestConfigurationError as error:
        typer.echo(f"Manifest configuration error: {error}", err=True)
        raise typer.Exit(code=2) from error

    typer.echo(
        "Manifest build complete: "
        f"dwarfs={len(manifest.dwarfs)} images={len(manifest.images)} "
        f"threshold={manifest.minimum_images_per_dwarf} "
        f"output={config.paths.manifest_path}"
    )


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

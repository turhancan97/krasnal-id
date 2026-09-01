"""Unified command-line interface for all Krasnal-ID pipeline stages."""

from pathlib import Path
from typing import Annotated

import typer

from krasnal_id.config import load_config
from krasnal_id.data_pipeline.build_manifest import (
    ManifestConfigurationError,
    build_manifest_from_artifacts,
    write_dataset_manifest,
)
from krasnal_id.data_pipeline.build_split import (
    SplitConfigurationError,
    build_split_from_artifact,
    write_evaluation_split,
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
from krasnal_id.demo.app import DemoError
from krasnal_id.demo.app import launch as launch_demo
from krasnal_id.embeddings.backbone import EmbeddingConfigurationError
from krasnal_id.embeddings.extract import EmbeddingExtractionError, extract_from_artifact
from krasnal_id.embeddings.store import EmbeddingStoreError
from krasnal_id.experiments.artifacts import (
    ExperimentArtifactError,
    experiment_result_path,
    write_experiment_result,
)
from krasnal_id.experiments.baseline_accuracy import BaselineExperimentError, run_baseline
from krasnal_id.experiments.confusion_analysis import (
    ConfusionAnalysisError,
    run_confusion_analysis,
)
from krasnal_id.experiments.pool_size_ablation import PoolAblationError, run_pool_size_ablation
from krasnal_id.experiments.probe_baseline import ProbeExperimentError, run_probe_comparison
from krasnal_id.logging import configure_logging
from krasnal_id.models import (
    AuditDisposition,
    CategoryReviewStatus,
    FetchAuditDisposition,
)
from krasnal_id.retrieval.query import QueryError, retrieve_image
from krasnal_id.viz.ablation_plot import create_ablation_plot
from krasnal_id.viz.embedding_plot import VisualizationError, create_embedding_plot

OverrideOption = Annotated[
    list[str] | None,
    typer.Option("--override", "-o", help="Hydra override; repeat for multiple values."),
]
TopKOption = Annotated[
    int,
    typer.Option("--top-k", min=1, help="Number of candidate dwarves to report."),
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


@data_app.command("build-split")
def build_split(override: OverrideOption = None) -> None:
    """Build the deterministic leave-one-out evaluation split."""
    config = load_config(override or [])
    configure_logging(config.logging)
    try:
        split = build_split_from_artifact(config.paths.manifest_path)
        write_evaluation_split(config.paths.evaluation_split_path, split)
    except SplitConfigurationError as error:
        typer.echo(f"Split configuration error: {error}", err=True)
        raise typer.Exit(code=2) from error

    typer.echo(
        "Evaluation split complete: "
        f"strategy={split.strategy} folds={len(split.folds)} "
        f"output={config.paths.evaluation_split_path}"
    )


@embeddings_app.command("extract")
def extract_embeddings(override: OverrideOption = None) -> None:
    """Extract and cache embeddings for every admitted image."""
    config = load_config(override or [])
    configure_logging(config.logging)
    try:
        summary = extract_from_artifact(
            config.paths.manifest_path,
            config.backbone,
            config.paths.embeddings_dir,
        )
    except (EmbeddingConfigurationError, EmbeddingExtractionError) as error:
        typer.echo(f"Embedding extraction error: {error}", err=True)
        raise typer.Exit(code=2) from error

    typer.echo(
        "Embedding extraction complete: "
        f"backbone={config.backbone.name} total={summary.total} "
        f"reused={summary.reused} computed={summary.computed} "
        f"cache={config.paths.embeddings_dir}"
    )


@app.command("retrieve")
def retrieve(
    image: Annotated[
        Path,
        typer.Argument(help="Path to the query photograph.", show_default=False),
    ],
    top_k: TopKOption = 5,
    override: OverrideOption = None,
) -> None:
    """Retrieve the nearest dwarf candidates for a query image."""
    config = load_config(override or [])
    configure_logging(config.logging)
    try:
        outcome = retrieve_image(image, config, top_k)
    except (QueryError, EmbeddingStoreError) as error:
        typer.echo(f"Retrieval error: {error}", err=True)
        raise typer.Exit(code=2) from error

    source = "cached vector" if outcome.reused_cached_vector else "freshly embedded"
    typer.echo(
        f"Top {len(outcome.candidates)} candidates for {image} "
        f"(backbone={config.backbone.name}, {source})"
    )
    if outcome.excluded_reference_image_ids:
        typer.echo(
            "  withheld the query's own reference copies: "
            f"{', '.join(outcome.excluded_reference_image_ids)}"
        )
    for candidate in outcome.candidates:
        typer.echo(
            f"  {candidate.rank}. {candidate.display_name} ({candidate.dwarf_id}) "
            f"similarity {candidate.cosine_similarity:.4f} "
            f"via {candidate.matched_image_id}"
        )


@experiment_app.command("baseline")
def baseline_experiment(override: OverrideOption = None) -> None:
    """Measure full-pool top-k accuracy and mean reciprocal rank."""
    config = load_config(["experiment=baseline", *(override or [])])
    configure_logging(config.logging)
    try:
        result = run_baseline(config)
        path = experiment_result_path(config.paths.results_dir, result)
        write_experiment_result(path, result)
    except (
        BaselineExperimentError,
        EmbeddingStoreError,
        ExperimentArtifactError,
    ) as error:
        typer.echo(f"Baseline experiment error: {error}", err=True)
        raise typer.Exit(code=2) from error

    typer.echo(f"Baseline experiment complete: backbone={result.backbone} result={path}")
    for metric in result.metrics:
        if metric.lower_bound is None or metric.upper_bound is None:
            typer.echo(f"  {metric.name}: {metric.value:.4f}")
        else:
            typer.echo(
                f"  {metric.name}: {metric.value:.4f} "
                f"[95% CI {metric.lower_bound:.4f}-{metric.upper_bound:.4f}]"
            )


@experiment_app.command("pool-ablation")
def pool_ablation_experiment(override: OverrideOption = None) -> None:
    """Measure accuracy across synthetic candidate-pool sizes."""
    config = load_config(["experiment=pool_size_ablation", *(override or [])])
    configure_logging(config.logging)
    try:
        result = run_pool_size_ablation(config)
        path = experiment_result_path(config.paths.results_dir, result)
        write_experiment_result(path, result)
    except (
        PoolAblationError,
        EmbeddingStoreError,
        ExperimentArtifactError,
    ) as error:
        typer.echo(f"Pool-size ablation error: {error}", err=True)
        raise typer.Exit(code=2) from error

    typer.echo(f"Pool-size ablation complete: backbone={result.backbone} result={path}")
    for metric in result.metrics:
        if metric.lower_bound is None or metric.upper_bound is None:
            typer.echo(f"  {metric.name}: {metric.value:.4f}")
        else:
            typer.echo(
                f"  {metric.name}: {metric.value:.4f} "
                f"[seeds {metric.lower_bound:.4f}-{metric.upper_bound:.4f}]"
            )


@experiment_app.command("probe")
def probe_experiment(override: OverrideOption = None) -> None:
    """Compare trained prototype and linear-probe classifiers against retrieval."""
    config = load_config(["experiment=probe", *(override or [])])
    configure_logging(config.logging)
    try:
        result = run_probe_comparison(config)
        path = experiment_result_path(config.paths.results_dir, result)
        write_experiment_result(path, result)
    except (
        ProbeExperimentError,
        EmbeddingStoreError,
        ExperimentArtifactError,
    ) as error:
        typer.echo(f"Probe comparison error: {error}", err=True)
        raise typer.Exit(code=2) from error

    typer.echo(f"Probe comparison complete: backbone={result.backbone} result={path}")
    for metric in result.metrics:
        if metric.lower_bound is None or metric.upper_bound is None:
            typer.echo(f"  {metric.name}: {metric.value:+.4f}")
        else:
            typer.echo(
                f"  {metric.name}: {metric.value:.4f} "
                f"[95% CI {metric.lower_bound:.4f}-{metric.upper_bound:.4f}]"
            )


@experiment_app.command("confusion")
def confusion_experiment(override: OverrideOption = None) -> None:
    """Find and summarize the most-confused dwarf pairs."""
    config = load_config(["experiment=confusion", *(override or [])])
    configure_logging(config.logging)
    try:
        result = run_confusion_analysis(config)
        path = experiment_result_path(config.paths.results_dir, result)
        write_experiment_result(path, result)
    except (
        ConfusionAnalysisError,
        EmbeddingStoreError,
        ExperimentArtifactError,
    ) as error:
        typer.echo(f"Confusion analysis error: {error}", err=True)
        raise typer.Exit(code=2) from error

    typer.echo(f"Confusion analysis complete: backbone={result.backbone} result={path}")
    for metric in result.metrics:
        typer.echo(f"  {metric.name}: {metric.value:.4f}")
    typer.echo(f"  most-confused pairs (of {len(result.pairs)} reported):")
    for pair in result.pairs[:10]:
        typer.echo(
            f"    {pair.true_display_name} -> {pair.confused_display_name}: "
            f"{pair.misidentifications} of {pair.queries} queries misidentified, "
            f"mean margin {pair.mean_margin:+.4f}"
        )


@visualize_app.command("embeddings")
def visualize_embeddings(override: OverrideOption = None) -> None:
    """Project cached embeddings into a saved two-dimensional figure."""
    config = load_config(["experiment=visualization", *(override or [])])
    configure_logging(config.logging)
    try:
        path = create_embedding_plot(config)
    except (VisualizationError, EmbeddingStoreError) as error:
        typer.echo(f"Embedding visualization error: {error}", err=True)
        raise typer.Exit(code=2) from error

    typer.echo(f"Embedding visualization complete: figure={path}")


@visualize_app.command("ablation")
def visualize_ablation(override: OverrideOption = None) -> None:
    """Draw the accuracy-versus-pool-size curve from saved ablation results."""
    config = load_config(["experiment=visualization", *(override or [])])
    configure_logging(config.logging)
    try:
        path = create_ablation_plot(config)
    except VisualizationError as error:
        typer.echo(f"Ablation visualization error: {error}", err=True)
        raise typer.Exit(code=2) from error

    typer.echo(f"Ablation visualization complete: figure={path}")


@app.command("demo")
def demo(
    top_k: TopKOption = 5,
    share: Annotated[
        bool,
        typer.Option("--share", help="Expose a temporary public Gradio link."),
    ] = False,
    port: Annotated[
        int | None,
        typer.Option("--port", min=1, max=65535, help="Port to serve the interface on."),
    ] = None,
    override: OverrideOption = None,
) -> None:
    """Launch the optional Gradio retrieval demonstration."""
    config = load_config(override or [])
    configure_logging(config.logging)
    try:
        launch_demo(config, top_k=top_k, share=share, server_port=port)
    except (DemoError, EmbeddingStoreError) as error:
        typer.echo(f"Demo error: {error}", err=True)
        raise typer.Exit(code=2) from error


if __name__ == "__main__":
    app()

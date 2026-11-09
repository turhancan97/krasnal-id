"""Unified command-line interface for all Krasnal-ID pipeline stages."""

import json
import shlex
from pathlib import Path
from typing import Annotated

import typer

from krasnal_id.config import WorkshopValidationConfig, backbone_config, load_config
from krasnal_id.data_pipeline.build_manifest import (
    ManifestConfigurationError,
    build_manifest_from_artifacts,
    write_dataset_manifest,
)
from krasnal_id.data_pipeline.build_split import (
    SplitConfigurationError,
    build_split_from_artifact,
    read_evaluation_split,
    write_evaluation_split,
)
from krasnal_id.data_pipeline.camera_metadata import (
    CameraMetadataError,
    camera_metadata_path,
    fetch_camera_metadata,
    load_camera_metadata,
)
from krasnal_id.data_pipeline.commons_discovery import CommonsDiscoveryError
from krasnal_id.data_pipeline.commons_fetch import (
    CommonsConfigurationError,
    fetch_images,
    fetch_paths,
    prepare_category_review,
)
from krasnal_id.data_pipeline.field_queries import (
    FieldQueryError,
    field_query_manifest_path,
    load_field_query_manifest,
    load_field_route,
    queries_by_cohort,
    stage_field_queries,
    write_field_query_manifest,
)
from krasnal_id.data_pipeline.field_sort import (
    FieldSortError,
    SortStatus,
    field_sort_report_path,
    sort_field_inbox,
    write_sort_report,
)
from krasnal_id.data_pipeline.license_templates import (
    LicenseTemplateError,
    fetch_license_templates,
    license_template_path,
    load_license_templates,
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
from krasnal_id.embeddings.cache import EmbeddingCache
from krasnal_id.embeddings.extract import (
    EmbeddingExtractionError,
    create_backbone,
    extract_from_artifact,
)
from krasnal_id.embeddings.extract import extract_embeddings as extract_record_embeddings
from krasnal_id.embeddings.store import EmbeddingStoreError
from krasnal_id.experiments.artifacts import (
    ExperimentArtifactError,
    experiment_result_path,
    guard_result_path,
    write_experiment_result,
)
from krasnal_id.experiments.baseline_accuracy import BaselineExperimentError, run_baseline
from krasnal_id.experiments.camera_gap import CameraGapError, run_camera_gap
from krasnal_id.experiments.confusion_analysis import (
    ConfusionAnalysisError,
    run_confusion_analysis,
)
from krasnal_id.experiments.field_gap import FieldGapError, run_field_gap
from krasnal_id.experiments.geo_ablation import GeoAblationError, run_geo_ablation
from krasnal_id.experiments.geometry_first import GeometryFirstError, run_geometry_first
from krasnal_id.experiments.matcher_rerank import MatcherRerankError, run_matcher_rerank
from krasnal_id.experiments.open_set import OpenSetExperimentError, run_open_set_rejection
from krasnal_id.experiments.open_set_geometry import (
    OpenSetGeometryError,
    run_open_set_geometry,
)
from krasnal_id.experiments.photographer_gap import (
    PhotographerGapError,
    run_photographer_gap,
)
from krasnal_id.experiments.pool_size_ablation import PoolAblationError, run_pool_size_ablation
from krasnal_id.experiments.probe_baseline import ProbeExperimentError, run_probe_comparison
from krasnal_id.experiments.recall_curve import RecallCurveError, run_recall_curve
from krasnal_id.experiments.rerank_ablation import RerankAblationError, run_rerank_ablation
from krasnal_id.experiments.workshop_validation import (
    WorkshopValidationError,
    run_workshop_validation,
)
from krasnal_id.export.huggingface import HuggingFaceExportError, build_export
from krasnal_id.export.kaggle import KaggleExportError, build_kaggle_export
from krasnal_id.export.push import PushConfigurationError, PushError, push_export
from krasnal_id.logging import configure_logging
from krasnal_id.models import (
    AuditDisposition,
    CategoryReviewStatus,
    DatasetManifest,
    FetchAuditDisposition,
)
from krasnal_id.retrieval.query import QueryError, retrieve_image
from krasnal_id.retrieval.rerank import DETECT_LONG_SIDE, RerankError
from krasnal_id.viz.ablation_plot import create_ablation_plot
from krasnal_id.viz.embedding_plot import VisualizationError, create_embedding_plot
from krasnal_id.viz.figure_style import FigureStyleError, layout_named
from krasnal_id.viz.match_plot import MatchPlotError, draw_match_figure, match_pair
from krasnal_id.viz.open_set_plot import create_open_set_plot
from krasnal_id.viz.paper_figures import PUBLICATION_DPI, PaperFigureError, Sources, draw_all
from krasnal_id.viz.paper_tables import write_all as write_paper_tables
from krasnal_id.viz.retrieval_examples import create_retrieval_examples_plot
from krasnal_id.viz.workshop_tables import (
    WorkshopTableError,
    write_extra_numbers,
    write_sensitivity_outputs,
    write_workshop_outputs,
)


def _read_manifest(path: Path) -> DatasetManifest:
    """Read the generated manifest for a command that needs the dataset itself."""
    try:
        return DatasetManifest.model_validate(json.loads(path.read_text(encoding="utf-8")))
    except (OSError, ValueError) as error:
        raise ManifestConfigurationError(f"invalid manifest {path}: {error}") from error


PushOption = Annotated[
    bool,
    typer.Option("--push", help="Upload the built export to the Hugging Face Hub."),
]
PublicOption = Annotated[
    bool,
    typer.Option("--public", help="Create the pushed repository public rather than private."),
]
RepoIdOption = Annotated[
    str | None,
    typer.Option("--repo-id", help="Hugging Face repository id; defaults to the configured one."),
]
EmbeddingsOption = Annotated[
    bool,
    typer.Option("--embeddings/--no-embeddings", help="Include the cached backbone vectors."),
]
DatasetIdOption = Annotated[
    str | None,
    typer.Option("--dataset-id", help="Kaggle dataset id; defaults to the configured one."),
]
ImagesOption = Annotated[
    bool,
    typer.Option("--images/--no-images", help="Include the photographs themselves."),
]

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
    typer.Option("--refresh", help="Ignore a valid cache and query the sources again."),
]
IncludeCommonsOption = Annotated[
    bool,
    typer.Option(
        "--include-commons",
        help="Also discover Commons per-dwarf categories Wikidata has no item for.",
    ),
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
    include_commons: IncludeCommonsOption = False,
) -> None:
    """Discover Wroclaw dwarf records through Wikidata, optionally plus Commons."""
    config = load_config(override or [])
    configure_logging(config.logging)
    try:
        result = query_dwarfs(
            config.data,
            config.paths.discovery_dir,
            limit=limit,
            refresh=refresh,
            include_commons=include_commons,
        )
    except WikidataConfigurationError as error:
        typer.echo(f"Configuration error: {error}", err=True)
        raise typer.Exit(code=2) from error
    except (WikidataQueryError, CommonsDiscoveryError) as error:
        typer.echo(f"Query failed: {error}", err=True)
        raise typer.Exit(code=1) from error

    excluded = sum(record.disposition is AuditDisposition.EXCLUDED for record in result.audit)
    warnings = sum(record.disposition is AuditDisposition.WARNING for record in result.audit)
    paths = discovery_paths(config.paths.discovery_dir)
    sources = "wikidata+commons" if include_commons else "wikidata"
    typer.echo(
        f"Discovery complete ({sources}): "
        f"cache={result.cache_status} eligible={result.eligible_total} "
        f"emitted={len(result.records)} excluded={excluded} warnings={warnings}"
    )
    if include_commons:
        commons_only = sum(1 for record in result.records if record.dwarf_id.startswith("C-"))
        placed = sum(1 for record in result.records if record.coordinates is not None)
        typer.echo(
            f"Sources: {len(result.records) - commons_only} from Wikidata, "
            f"{commons_only} Commons-only; {placed} carry coordinates"
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


@data_app.command("camera-metadata")
def build_camera_metadata(override: OverrideOption = None) -> None:
    """Fetch the EXIF camera model behind every manifest image."""
    import httpx

    from krasnal_id.data_pipeline.wikidata_query import _contact_user_agent, _request_commons_page

    config = load_config(override or [])
    configure_logging(config.logging)
    try:
        manifest = DatasetManifest.model_validate(
            json.loads(config.paths.manifest_path.read_text(encoding="utf-8"))
        )
    except (OSError, ValueError) as error:
        typer.echo(f"Camera metadata error: invalid manifest: {error}", err=True)
        raise typer.Exit(code=2) from error

    user_agent = _contact_user_agent()
    try:
        with httpx.Client() as client:

            def request(parameters: dict[str, str]) -> object:
                return _request_commons_page(config.data, user_agent, client, parameters)

            metadata = fetch_camera_metadata(manifest, config.data, request)
    except (CameraMetadataError, CommonsConfigurationError) as error:
        typer.echo(f"Camera metadata error: {error}", err=True)
        raise typer.Exit(code=1) from error

    path = camera_metadata_path(config.paths.discovery_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(metadata.model_dump(mode="json"), ensure_ascii=False, indent=2, sort_keys=True)
        + "\n",
        encoding="utf-8",
    )
    known = sum(1 for value in metadata.cameras.values() if value)
    typer.echo(
        f"Camera metadata complete: pages={len(metadata.cameras)} with_camera={known} output={path}"
    )


@data_app.command("license-templates")
def build_license_templates(override: OverrideOption = None) -> None:
    """Fetch the public-domain basis behind every label-licensed image."""
    import httpx

    from krasnal_id.data_pipeline.wikidata_query import _contact_user_agent, _request_commons_page

    config = load_config(override or [])
    configure_logging(config.logging)
    try:
        manifest = _read_manifest(config.paths.manifest_path)
    except ManifestConfigurationError as error:
        typer.echo(f"Licence template error: {error}", err=True)
        raise typer.Exit(code=2) from error

    user_agent = _contact_user_agent()
    try:
        with httpx.Client() as client:

            def request(parameters: dict[str, str]) -> object:
                return _request_commons_page(config.data, user_agent, client, parameters)

            templates = fetch_license_templates(manifest, config.data, request)
    except (LicenseTemplateError, CommonsConfigurationError) as error:
        typer.echo(f"Licence template error: {error}", err=True)
        raise typer.Exit(code=1) from error

    path = license_template_path(config.paths.discovery_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(templates.model_dump(mode="json"), ensure_ascii=False, indent=2, sort_keys=True)
        + "\n",
        encoding="utf-8",
    )
    known = sum(1 for value in templates.templates.values() if value)
    typer.echo(
        f"Licence templates complete: pages={len(templates.templates)} "
        f"with_basis={known} output={path}"
    )


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
            max_coordinate_drift_km=config.thresholds.max_coordinate_drift_km,
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


@data_app.command("field-sort")
def propose_field_sort(override: OverrideOption = None) -> None:
    """Propose a route statue for each unsorted field photograph from its GPS.

    Nothing is moved. The commands printed for unambiguous photographs are
    suggestions to run after looking at each one; the directory a photograph ends
    up in is what ties it to a statue, never this proposal.
    """
    config = load_config(override or [])
    configure_logging(config.logging)
    try:
        route = load_field_route(config.paths.field_route_path)
        manifest = _read_manifest(config.paths.manifest_path)
        report = sort_field_inbox(
            config.paths.field_inbox_dir,
            route,
            manifest,
            config.thresholds.field_sort_max_distance_m,
            config.thresholds.field_sort_ambiguity_m,
        )
        path = field_sort_report_path(config.paths.data_dir)
        write_sort_report(path, report)
    except (FieldQueryError, FieldSortError, ManifestConfigurationError) as error:
        typer.echo(f"Field sort error: {error}", err=True)
        raise typer.Exit(code=2) from error

    counts = " ".join(f"{status}={report.count(status)}" for status in SortStatus)
    typer.echo(f"Field sort proposal: photographs={len(report.proposals)} {counts} output={path}")
    typer.echo("  Nothing was moved. Look at each photograph before filing it.")
    for proposal in report.proposals:
        nearest = proposal.candidates[0] if proposal.candidates else None
        if proposal.status == SortStatus.PROPOSED and nearest is not None:
            target = config.paths.field_queries_dir / nearest.dwarf_id
            typer.echo(
                f"  # {proposal.path.name}: {nearest.display_name} at "
                f"{nearest.distance_m:.0f} m\n"
                f"  mkdir -p {shlex.quote(str(target))} && "
                f"mv -n {shlex.quote(str(proposal.path))} {shlex.quote(str(target))}/"
            )
        elif proposal.status == SortStatus.AMBIGUOUS:
            options = ", ".join(
                f"{candidate.display_name} [{candidate.dwarf_id}] {candidate.distance_m:.0f} m"
                for candidate in proposal.candidates
            )
            typer.echo(f"  # {proposal.path.name}: ambiguous, decide by eye between {options}")
        elif proposal.status == SortStatus.OFF_ROUTE and nearest is not None:
            typer.echo(
                f"  # {proposal.path.name}: off the route, nearest is "
                f"{nearest.display_name} at {nearest.distance_m:.0f} m"
            )
        else:
            typer.echo(f"  # {proposal.path.name}: {proposal.status}, {proposal.detail}")


@data_app.command("field-queries")
def stage_field_query_manifest(override: OverrideOption = None) -> None:
    """Stage the field photographs on disk into a query manifest."""
    config = load_config(override or [])
    configure_logging(config.logging)
    try:
        route = load_field_route(config.paths.field_route_path)
        manifest = _read_manifest(config.paths.manifest_path)
        staged = stage_field_queries(config.paths.field_queries_dir, manifest, route)
        path = field_query_manifest_path(config.paths.data_dir)
        write_field_query_manifest(path, staged)
    except (FieldQueryError, ManifestConfigurationError) as error:
        typer.echo(f"Field query error: {error}", err=True)
        raise typer.Exit(code=2) from error

    cohorts = queries_by_cohort(staged.queries)
    statues = len({query.dwarf_id for query in staged.queries})
    typer.echo(
        f"Field query staging complete: photographs={len(staged.queries)} "
        f"statues={statues} of {len(route.entries)} on the route output={path}"
    )
    for cohort, records in sorted(cohorts.items()):
        typer.echo(
            f"  {cohort}: {len(records)} photographs of "
            f"{len({record.dwarf_id for record in records})} statues"
        )


@data_app.command("export-hf")
def export_huggingface(
    override: OverrideOption = None,
    push: PushOption = False,
    repo_id: RepoIdOption = None,
    public: PublicOption = False,
    embeddings: EmbeddingsOption = True,
) -> None:
    """Build the Hugging Face dataset directory, and optionally publish it."""
    config = load_config(override or [])
    configure_logging(config.logging)
    target = repo_id or config.export.repo_id

    try:
        manifest = _read_manifest(config.paths.manifest_path)
        split = read_evaluation_split(config.paths.evaluation_split_path)
        templates = None
        template_path = license_template_path(config.paths.discovery_dir)
        if template_path.is_file():
            templates = load_license_templates(template_path).templates
        result = build_export(
            config,
            manifest,
            split,
            backbones=tuple(
                backbone_config(name, override or []) for name in config.export.backbones
            ),
            license_templates=templates,
            with_embeddings=embeddings,
        )
    except (
        HuggingFaceExportError,
        LicenseTemplateError,
        ManifestConfigurationError,
        SplitConfigurationError,
        EmbeddingStoreError,
    ) as error:
        typer.echo(f"Hugging Face export error: {error}", err=True)
        raise typer.Exit(code=2) from error

    if templates is None:
        typer.echo(
            "  note: no licence templates found, so public-domain rows carry no recorded "
            "basis; run krasnal-id data license-templates first",
            err=True,
        )
    typer.echo(
        f"Hugging Face export complete: images={result.images} classes={result.classes} "
        f"folds={result.folds} shards={result.shards} modified={result.modified} "
        f"unmodified={result.unmodified} backbones={','.join(result.backbones) or 'none'} "
        f"manifest={result.manifest_sha256[:12]} output={result.paths.root}"
    )

    if not push:
        typer.echo(
            f"  not published. To publish: krasnal-id data export-hf --push --repo-id {target}"
        )
        return

    try:
        outcome = push_export(
            result.paths.root,
            target,
            private=not public,
            commit_message=f"Export manifest {result.manifest_sha256[:12]}",
        )
    except PushConfigurationError as error:
        typer.echo(f"Hugging Face push error: {error}", err=True)
        raise typer.Exit(code=2) from error
    except PushError as error:
        typer.echo(f"Hugging Face push error: {error}", err=True)
        raise typer.Exit(code=1) from error

    visibility = "public" if public else "private"
    typer.echo(f"Pushed: repo={outcome.repo_id} visibility={visibility} url={outcome.url}")


@data_app.command("export-kaggle")
def export_kaggle(
    override: OverrideOption = None,
    dataset_id: DatasetIdOption = None,
    embeddings: EmbeddingsOption = True,
    images: ImagesOption = True,
) -> None:
    """Build the Kaggle dataset directory, ready for `kaggle datasets create`."""
    config = load_config(override or [])
    configure_logging(config.logging)

    try:
        manifest = _read_manifest(config.paths.manifest_path)
        split = read_evaluation_split(config.paths.evaluation_split_path)
        templates = None
        template_path = license_template_path(config.paths.discovery_dir)
        if template_path.is_file():
            templates = load_license_templates(template_path).templates
        result = build_kaggle_export(
            config,
            manifest,
            split,
            backbones=tuple(
                backbone_config(name, override or []) for name in config.export.backbones
            ),
            dataset_id=dataset_id,
            license_templates=templates,
            with_embeddings=embeddings,
            with_images=images,
        )
    except (
        KaggleExportError,
        LicenseTemplateError,
        ManifestConfigurationError,
        SplitConfigurationError,
        EmbeddingStoreError,
    ) as error:
        typer.echo(f"Kaggle export error: {error}", err=True)
        raise typer.Exit(code=2) from error

    if templates is None:
        typer.echo(
            "  note: no licence templates found, so public-domain rows carry no recorded "
            "basis; run krasnal-id data license-templates first",
            err=True,
        )

    typer.echo(
        f"Kaggle export complete: images={result.images} classes={result.classes} "
        f"folds={result.folds} modified={result.modified} unmodified={result.unmodified} "
        f"backbones={','.join(result.backbones) or 'none'} "
        f"manifest={result.manifest_sha256[:12]} output={result.paths.root}"
    )
    if result.cover:
        typer.echo(f"  cover image for the web UI: {result.cover}")
    if result.image_bytes:
        typer.echo(
            f"  photographs: {result.image_bytes / 1e6:.0f} MB under {result.paths.images_dir}"
        )

    # Publishing is deliberately not automated. A Kaggle dataset slug cannot be
    # renamed after creation, so the one irreversible step stays a human one.
    typer.echo(
        f"  not published. To create it:\n"
        f"    kaggle datasets create -p {result.paths.root} --dir-mode zip\n"
        f"  To publish a new version of {result.dataset_id}:\n"
        f"    kaggle datasets version -p {result.paths.root} --dir-mode zip "
        f'-m "manifest {result.manifest_sha256[:12]}"'
    )


@embeddings_app.command("extract-regions")
def extract_region_vectors(override: OverrideOption = None) -> None:
    """Write CLS plus 2x2 and 3x3 grid-pooled patch vectors for every admitted image.

    For the multi-vector baseline of AGENTS.md section 7.18 only; the file sits
    beside the embedding cache as `regions-<backbone>.npz` and nothing else reads it.
    """
    from krasnal_id.embeddings.regions import RegionExtractionError, extract_regions

    config = load_config(override or [])
    configure_logging(config.logging)
    output = Path(config.paths.embeddings_dir) / f"regions-{config.backbone.name}.npz"
    try:
        written = extract_regions(
            _read_manifest(config.paths.manifest_path),
            config.backbone,
            output,
            config.backbone.batch_size,
        )
    except (RegionExtractionError, EmbeddingExtractionError) as error:
        typer.echo(f"Region extraction error: {error}", err=True)
        raise typer.Exit(code=2) from error
    typer.echo(f"Region vectors complete: {written}")


@embeddings_app.command("extract")
def extract_embeddings(
    override: OverrideOption = None,
    field_queries: Annotated[
        bool,
        typer.Option(
            "--field-queries",
            help="Embed the staged field photographs instead of the reference set.",
        ),
    ] = False,
) -> None:
    """Extract and cache embeddings for every admitted image.

    The cache is content-addressed, so field photographs share it with the
    references without joining them: `--field-queries` embeds the staged queries
    and touches neither the manifest nor the split.
    """
    config = load_config(override or [])
    configure_logging(config.logging)
    try:
        if field_queries:
            staged = load_field_query_manifest(
                field_query_manifest_path(config.paths.data_dir),
                _read_manifest(config.paths.manifest_path),
            )
            summary = extract_record_embeddings(
                staged.queries,
                create_backbone(config.backbone),
                EmbeddingCache(config.paths.embeddings_dir),
                config.backbone.batch_size,
            )
        else:
            summary = extract_from_artifact(
                config.paths.manifest_path,
                config.backbone,
                config.paths.embeddings_dir,
            )
    except (
        EmbeddingConfigurationError,
        EmbeddingExtractionError,
        FieldQueryError,
        ManifestConfigurationError,
    ) as error:
        typer.echo(f"Embedding extraction error: {error}", err=True)
        raise typer.Exit(code=2) from error

    typer.echo(
        "Embedding extraction complete: "
        f"source={'field-queries' if field_queries else 'manifest'} "
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
    guard_result_path(config)
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
    guard_result_path(config)
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


@experiment_app.command("geo-ablation")
def geo_ablation_experiment(override: OverrideOption = None) -> None:
    """Compare real proximity-based candidate pools against randomly sampled ones."""
    config = load_config(["experiment=geo_ablation", *(override or [])])
    configure_logging(config.logging)
    guard_result_path(config)
    try:
        result = run_geo_ablation(config)
        path = experiment_result_path(config.paths.results_dir, result)
        write_experiment_result(path, result)
    except (
        GeoAblationError,
        EmbeddingStoreError,
        ExperimentArtifactError,
    ) as error:
        typer.echo(f"Geographic ablation error: {error}", err=True)
        raise typer.Exit(code=2) from error

    typer.echo(f"Geographic ablation complete: backbone={result.backbone} result={path}")
    for metric in result.metrics:
        if metric.name.startswith("geo_radius_metres"):
            typer.echo(
                f"  {metric.name}: {metric.value:.0f} m median"
                + (f", {metric.upper_bound:.0f} m max" if metric.upper_bound else "")
            )
        elif metric.lower_bound is None or metric.upper_bound is None:
            typer.echo(f"  {metric.name}: {metric.value:+.4f}")
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
    guard_result_path(config)
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


@experiment_app.command("open-set")
def open_set_experiment(override: OverrideOption = None) -> None:
    """Measure whether a similarity threshold can reject dwarves outside the dataset."""
    config = load_config(["experiment=open_set", *(override or [])])
    configure_logging(config.logging)
    guard_result_path(config)
    try:
        result = run_open_set_rejection(config)
        path = experiment_result_path(config.paths.results_dir, result)
        write_experiment_result(path, result)
    except (
        OpenSetExperimentError,
        EmbeddingStoreError,
        ExperimentArtifactError,
    ) as error:
        typer.echo(f"Open-set rejection error: {error}", err=True)
        raise typer.Exit(code=2) from error

    typer.echo(f"Open-set rejection complete: backbone={result.backbone} result={path}")
    for metric in result.metrics:
        if metric.name.endswith("_threshold") or metric.name == "mean_similarity_gap":
            typer.echo(
                f"  {metric.name}: {metric.value:+.4f}"
                + (
                    f" [{metric.lower_bound:+.4f}-{metric.upper_bound:+.4f}]"
                    if metric.lower_bound is not None and metric.upper_bound is not None
                    else ""
                )
            )
        elif metric.lower_bound is None or metric.upper_bound is None:
            typer.echo(f"  {metric.name}: {metric.value:.4f}")
        else:
            typer.echo(
                f"  {metric.name}: {metric.value:.4f} "
                f"[95% CI {metric.lower_bound:.4f}-{metric.upper_bound:.4f}]"
            )
    typer.echo(f"  hardest dwarves to reject (of {len(result.rejections)} reported):")
    for rejection in result.rejections[:10]:
        typer.echo(
            f"    {rejection.display_name}: {rejection.false_accepts} of "
            f"{rejection.unknown_queries} unknown queries accepted, mean top similarity "
            f"{rejection.mean_top_similarity:+.4f}, usually matched "
            f"{rejection.nearest_display_name}"
        )


@experiment_app.command("open-set-geometry")
def open_set_geometry_experiment(override: OverrideOption = None) -> None:
    """Compare appearance and geometry as signals for rejecting unknown dwarves."""
    config = load_config(["experiment=open_set_geometry", *(override or [])])
    configure_logging(config.logging)
    guard_result_path(config)
    try:
        result = run_open_set_geometry(config)
        path = experiment_result_path(config.paths.results_dir, result)
        write_experiment_result(path, result)
    except (
        OpenSetGeometryError,
        EmbeddingStoreError,
        ExperimentArtifactError,
    ) as error:
        typer.echo(f"Geometric open-set rejection error: {error}", err=True)
        raise typer.Exit(code=2) from error

    typer.echo(f"Geometric open-set rejection complete: backbone={result.backbone} result={path}")
    for condition in dict.fromkeys(row.condition for row in result.signals):
        rows = [row for row in result.signals if row.condition == condition]
        typer.echo(
            f"  {condition} ({rows[0].known_queries} known, {rows[0].unknown_queries} unknown):"
        )
        for row in rows:
            typer.echo(
                f"    {row.signal:<14} AUROC {row.auroc:.4f}  "
                f"false accepts {row.false_acceptance:6.1%} at "
                f"{row.achieved_known_acceptance:.1%} known  "
                f"best balanced {row.in_sample_balanced_accuracy:.4f}"
            )


@experiment_app.command("camera-gap")
def camera_gap_experiment(override: OverrideOption = None) -> None:
    """Compare phone-originated queries against camera-originated ones."""
    config = load_config(["experiment=camera_gap", *(override or [])])
    configure_logging(config.logging)
    guard_result_path(config)
    try:
        metadata = load_camera_metadata(camera_metadata_path(config.paths.discovery_dir))
        result = run_camera_gap(config, metadata)
        path = experiment_result_path(config.paths.results_dir, result)
        write_experiment_result(path, result)
    except (
        CameraGapError,
        CameraMetadataError,
        EmbeddingStoreError,
        ExperimentArtifactError,
    ) as error:
        typer.echo(f"Camera gap error: {error}", err=True)
        raise typer.Exit(code=2) from error

    typer.echo(f"Camera gap complete: backbone={result.backbone} result={path}")
    for metric in result.metrics:
        if metric.lower_bound is None or metric.upper_bound is None:
            typer.echo(f"  {metric.name}: {metric.value:+.4f}")
        else:
            typer.echo(
                f"  {metric.name}: {metric.value:.4f} "
                f"[95% CI {metric.lower_bound:.4f}-{metric.upper_bound:.4f}]"
            )


@experiment_app.command("field-gap")
def field_gap_experiment(override: OverrideOption = None) -> None:
    """Compare field photographs against the same statues' Commons queries."""
    config = load_config(["experiment=field_gap", *(override or [])])
    configure_logging(config.logging)
    guard_result_path(config)
    try:
        staged = load_field_query_manifest(
            field_query_manifest_path(config.paths.data_dir),
            _read_manifest(config.paths.manifest_path),
        )
        result = run_field_gap(config, staged)
        path = experiment_result_path(config.paths.results_dir, result)
        write_experiment_result(path, result)
    except (
        FieldGapError,
        FieldQueryError,
        EmbeddingStoreError,
        ExperimentArtifactError,
        ManifestConfigurationError,
    ) as error:
        typer.echo(f"Field gap error: {error}", err=True)
        raise typer.Exit(code=2) from error

    typer.echo(f"Field gap complete: backbone={result.backbone} result={path}")
    for metric in result.metrics:
        if metric.lower_bound is None or metric.upper_bound is None:
            typer.echo(f"  {metric.name}: {metric.value:+.4f}")
        else:
            typer.echo(
                f"  {metric.name}: {metric.value:.4f} "
                f"[95% CI {metric.lower_bound:.4f}-{metric.upper_bound:.4f}]"
            )
    typer.echo(f"  hardest statues to identify in the street (of {len(result.classes)}):")
    for row in result.classes[:10]:
        typer.echo(
            f"    {row.display_name} ({row.cohort}): {row.field_top_1_hits} of "
            f"{row.field_queries} field photographs identified, against "
            f"{row.commons_top_1_hits} of {row.commons_queries} Commons queries "
            f"({row.commons_disjoint_top_1_hits} of {row.commons_disjoint_queries} "
            f"photographer-disjoint), mean rank {row.field_mean_rank:.1f}"
        )


@experiment_app.command("workshop-validation")
def workshop_validation_experiment(override: OverrideOption = None) -> None:
    """Re-check published results from evidence on disk, with no model or matcher.

    Results go to `paths.results_dir/<output_subdir>`, so no released artifact can
    be overwritten; the matcher journals are read from `paths.results_dir/journals`.
    """
    config = load_config(["experiment=workshop_validation", *(override or [])])
    configure_logging(config.logging)
    assert isinstance(config.experiment, WorkshopValidationConfig)
    journals_dir = Path(config.paths.results_dir) / "journals"
    output = config.model_copy(
        update={
            "paths": config.paths.model_copy(
                update={
                    "results_dir": Path(config.paths.results_dir) / config.experiment.output_subdir
                }
            )
        }
    )
    guard_result_path(output)
    try:
        result = run_workshop_validation(output, journals_dir)
        path = experiment_result_path(output.paths.results_dir, result)
        write_experiment_result(path, result)
    except (
        WorkshopValidationError,
        ExperimentArtifactError,
        ManifestConfigurationError,
    ) as error:
        typer.echo(f"Workshop validation error: {error}", err=True)
        raise typer.Exit(code=2) from error

    typer.echo(f"Workshop validation complete: backbone={result.backbone} result={path}")
    for metric in result.metrics:
        if metric.lower_bound is None or metric.upper_bound is None:
            typer.echo(f"  {metric.name}: {metric.value:g}")
        else:
            typer.echo(
                f"  {metric.name}: {metric.value:.4f} "
                f"[95% CI {metric.lower_bound:.4f}-{metric.upper_bound:.4f}]"
            )
    typer.echo("  weights chosen on the other folds:")
    for row in result.fold_weights:
        typer.echo(
            f"    {row.matcher} fold {row.fold}: weight {row.weight:g} "
            f"(selection top-1 {row.selection_top_1:.4f}; held out "
            f"{row.held_out_queries} queries of {row.held_out_statues} statues)"
        )


@experiment_app.command("photographer-gap")
def photographer_gap_experiment(override: OverrideOption = None) -> None:
    """Measure how much accuracy survives changing the photographer."""
    config = load_config(["experiment=photographer_gap", *(override or [])])
    configure_logging(config.logging)
    guard_result_path(config)
    try:
        result = run_photographer_gap(config)
        path = experiment_result_path(config.paths.results_dir, result)
        write_experiment_result(path, result)
    except (
        PhotographerGapError,
        EmbeddingStoreError,
        ExperimentArtifactError,
    ) as error:
        typer.echo(f"Photographer gap error: {error}", err=True)
        raise typer.Exit(code=2) from error

    typer.echo(f"Photographer gap complete: backbone={result.backbone} result={path}")
    for metric in result.metrics:
        if metric.lower_bound is None or metric.upper_bound is None:
            typer.echo(f"  {metric.name}: {metric.value:+.4f}")
        else:
            typer.echo(
                f"  {metric.name}: {metric.value:.4f} "
                f"[95% CI {metric.lower_bound:.4f}-{metric.upper_bound:.4f}]"
            )


@experiment_app.command("recall")
def recall_experiment(override: OverrideOption = None) -> None:
    """Measure the first stage's recall, and whether two standard fixes help."""
    config = load_config(["experiment=recall_curve", *(override or [])])
    configure_logging(config.logging)
    guard_result_path(config)
    try:
        result = run_recall_curve(config)
        path = experiment_result_path(config.paths.results_dir, result)
        write_experiment_result(path, result)
    except (
        RecallCurveError,
        EmbeddingStoreError,
        ExperimentArtifactError,
    ) as error:
        typer.echo(f"Recall curve error: {error}", err=True)
        raise typer.Exit(code=2) from error

    typer.echo(f"Recall curve complete: backbone={result.backbone} result={path}")
    for metric in result.metrics:
        if metric.lower_bound is None or metric.upper_bound is None:
            typer.echo(f"  {metric.name}: {metric.value:+.4f}")
        else:
            typer.echo(
                f"  {metric.name}: {metric.value:.4f} "
                f"[95% CI {metric.lower_bound:.4f}-{metric.upper_bound:.4f}]"
            )


@experiment_app.command("geometry-first")
def geometry_first_experiment(override: OverrideOption = None) -> None:
    """Rank every reference by geometry alone, with no appearance involved."""
    config = load_config(["experiment=geometry_first", *(override or [])])
    configure_logging(config.logging)
    guard_result_path(config)
    try:
        result = run_geometry_first(config)
        path = experiment_result_path(config.paths.results_dir, result)
        write_experiment_result(path, result)
    except (
        GeometryFirstError,
        RerankError,
        EmbeddingStoreError,
        ExperimentArtifactError,
    ) as error:
        typer.echo(f"Geometry-first error: {error}", err=True)
        raise typer.Exit(code=2) from error

    typer.echo(f"Geometry-first complete: backbone={result.backbone} result={path}")
    for metric in result.metrics:
        if metric.lower_bound is None or metric.upper_bound is None:
            typer.echo(f"  {metric.name}: {metric.value:+.4f}")
        else:
            typer.echo(
                f"  {metric.name}: {metric.value:.4f} "
                f"[95% CI {metric.lower_bound:.4f}-{metric.upper_bound:.4f}]"
            )


@experiment_app.command("matcher-rerank")
def matcher_rerank_experiment(override: OverrideOption = None) -> None:
    """Compare local matchers through the geometric re-ranking protocol."""
    config = load_config(["experiment=matcher_rerank", *(override or [])])
    configure_logging(config.logging)
    guard_result_path(config)
    try:
        result = run_matcher_rerank(config)
        path = experiment_result_path(config.paths.results_dir, result)
        write_experiment_result(path, result)
    except (
        MatcherRerankError,
        RerankAblationError,
        RerankError,
        EmbeddingStoreError,
        ExperimentArtifactError,
    ) as error:
        typer.echo(f"Matcher re-rank error: {error}", err=True)
        raise typer.Exit(code=2) from error

    typer.echo(f"Matcher re-rank complete: backbone={result.backbone} result={path}")
    for metric in result.metrics:
        if metric.lower_bound is None or metric.upper_bound is None:
            typer.echo(f"  {metric.name}: {metric.value:+.4f}")
        else:
            typer.echo(
                f"  {metric.name}: {metric.value:.4f} "
                f"[95% CI {metric.lower_bound:.4f}-{metric.upper_bound:.4f}]"
            )


@experiment_app.command("rerank")
def rerank_experiment(override: OverrideOption = None) -> None:
    """Sweep how much geometric verification adds to the global ranking."""
    config = load_config(["experiment=rerank_ablation", *(override or [])])
    configure_logging(config.logging)
    guard_result_path(config)
    try:
        result = run_rerank_ablation(config)
        path = experiment_result_path(config.paths.results_dir, result)
        write_experiment_result(path, result)
    except (
        RerankAblationError,
        RerankError,
        EmbeddingStoreError,
        ExperimentArtifactError,
    ) as error:
        typer.echo(f"Re-ranking error: {error}", err=True)
        raise typer.Exit(code=2) from error

    typer.echo(f"Re-ranking complete: backbone={result.backbone} result={path}")
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
    guard_result_path(config)
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
def visualize_embeddings(
    output: Annotated[Path | None, typer.Option(help="Where to write the figure.")] = None,
    dpi: Annotated[int, typer.Option(min=72, help="Output density.")] = 200,
    override: OverrideOption = None,
) -> None:
    """Project cached embeddings into a saved two-dimensional figure."""
    config = load_config(["experiment=visualization", *(override or [])])
    configure_logging(config.logging)
    try:
        path = create_embedding_plot(config, output, dpi)
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


@visualize_app.command("open-set")
def visualize_open_set(override: OverrideOption = None) -> None:
    """Draw the rejection tradeoff curve from saved open-set results."""
    config = load_config(["experiment=visualization", *(override or [])])
    configure_logging(config.logging)
    try:
        path = create_open_set_plot(config)
    except VisualizationError as error:
        typer.echo(f"Open-set visualization error: {error}", err=True)
        raise typer.Exit(code=2) from error

    typer.echo(f"Open-set visualization complete: figure={path}")


@visualize_app.command("paper-figures")
def visualize_paper_figures(
    output_dir: Annotated[Path, typer.Option(help="Where to write the figures.")] = Path(
        "paper/figures"
    ),
    layout: Annotated[
        str, typer.Option(help="Figure sizes and backbones: journal or cvww.")
    ] = "journal",
    validation: Annotated[
        Path | None,
        typer.Option(help="The validation artifact the short-paper figures draw from."),
    ] = None,
    override: OverrideOption = None,
) -> None:
    """Draw the manuscript's data-derived figures at publication density.

    The manuscript is untracked; this command is not. A figure nobody can
    regenerate from a committed command is a figure nobody can check.
    """
    config = load_config(list(override or []))
    configure_logging(config.logging)
    try:
        sources = Sources(
            results_dir=Path(config.paths.results_dir),
            manifest=Path(config.paths.manifest_path),
            validation=validation,
        )
        written = draw_all(sources, output_dir, layout_named(layout))
    except (PaperFigureError, FigureStyleError) as error:
        typer.echo(f"Paper figure error: {error}", err=True)
        raise typer.Exit(code=2) from error
    for path in written:
        typer.echo(f"  {path}")
    typer.echo(f"Paper figures complete: {len(written)} written at {PUBLICATION_DPI} dpi")


@visualize_app.command("paper-tables")
def visualize_paper_tables(
    output_dir: Annotated[Path, typer.Option(help="Where to write the fragments.")] = Path(
        "paper/tables"
    ),
    override: OverrideOption = None,
) -> None:
    """Emit the Supplementary Information's tables as LaTeX, from the artifacts.

    Fourteen tables of numbers is exactly the material that acquires a wrong
    digit in transcription, so none of them is transcribed.
    """
    config = load_config(list(override or []))
    configure_logging(config.logging)
    try:
        sources = Sources(
            results_dir=Path(config.paths.results_dir),
            manifest=Path(config.paths.manifest_path),
        )
        written = write_paper_tables(sources, output_dir)
    except PaperFigureError as error:
        typer.echo(f"Paper table error: {error}", err=True)
        raise typer.Exit(code=2) from error
    for path in written:
        typer.echo(f"  {path}")
    typer.echo(f"Paper tables complete: {len(written)} written")


@visualize_app.command("workshop-tables")
def visualize_workshop_tables(
    output_dir: Annotated[Path, typer.Option(help="Where to write the fragments.")] = Path(
        "paper-cvww/generated"
    ),
    extra: Annotated[
        list[str] | None,
        typer.Option(
            "--extra",
            help="PREFIX=PATH: another validation artifact whose metrics are written to "
            "extra-numbers.tex as \\wv{PREFIX_<metric>}. Repeat for several.",
        ),
    ] = None,
    override: OverrideOption = None,
) -> None:
    """Emit the paper's numbers and tables from the validation artifact.

    The prose cites intervals by metric name through the macro file this writes,
    so none of them is transcribed. `--extra` adds the numbers of other runs, such
    as the DINOv3 re-ranking check, under a prefix so they cannot collide.
    """
    config = load_config(["experiment=workshop_validation", *(override or [])])
    configure_logging(config.logging)
    assert isinstance(config.experiment, WorkshopValidationConfig)
    validation = (
        Path(config.paths.results_dir)
        / config.experiment.output_subdir
        / f"workshop_validation-{config.backbone.name}.json"
    )
    try:
        written = write_workshop_outputs(validation, output_dir)
        if extra:
            written += (write_extra_numbers(extra, output_dir),)
        # The second reviewer's sensitivity runs, when any exist beside the artifact.
        written += write_sensitivity_outputs(Path(config.paths.results_dir), output_dir)
    except WorkshopTableError as error:
        typer.echo(f"Workshop table error: {error}", err=True)
        raise typer.Exit(code=2) from error
    for path in written:
        typer.echo(f"  {path}")
    typer.echo(f"Workshop tables complete: {len(written)} written")


@visualize_app.command("family")
def visualize_family(
    root: Annotated[
        Path, typer.Option(help="The family study's results directory (section 7.17).")
    ] = Path("results/family"),
    figure: Annotated[Path, typer.Option(help="Where to write the forest plot.")] = Path(
        "paper-cvww/figures/fig-family.png"
    ),
    output_dir: Annotated[Path, typer.Option(help="Where to write the TeX fragments.")] = Path(
        "paper-cvww/generated"
    ),
    width_cm: Annotated[float, typer.Option("--width-cm", help="Printed width.")] = 17.0,
    curves: Annotated[
        Path, typer.Option(help="Where to write the supplement's full curves, by paradigm.")
    ] = Path("paper-cvww/figures/figS-family-curves.png"),
    dpi: Annotated[int, typer.Option(min=72, help="Output density.")] = 300,
) -> None:
    """Draw the pretraining-family figure and write its numbers and table.

    Reads the family validation (`<root>/validation/`) and each backbone's released
    artifacts under `<root>/<backbone>/`; nothing is computed here.
    """
    from krasnal_id.viz.family_plot import render_family_curves, render_family_figure
    from krasnal_id.viz.workshop_tables import (
        WorkshopTableError,
        load_metrics,
        write_family_outputs,
    )

    validation = root / "validation" / "workshop_validation-dinov2.json"
    try:
        written = write_family_outputs(validation, root, output_dir)
        metrics = {
            name: {"value": m["value"], "lower_bound": m["lower"], "upper_bound": m["upper"]}
            for name, m in load_metrics(validation).items()
        }
        path = render_family_figure(metrics, figure, width_cm / 2.54, dpi)
        curves_path = render_family_curves(root, metrics, curves, width_cm / 2.54, dpi)
    except (WorkshopTableError, VisualizationError) as error:
        typer.echo(f"Family figure error: {error}", err=True)
        raise typer.Exit(code=2) from error
    for item in (*written, path, curves_path):
        typer.echo(f"  {item}")
    typer.echo("Family figure complete")


@visualize_app.command("matches")
def visualize_matches(
    query: Annotated[Path, typer.Option(help="The photograph to match from.")],
    candidate: Annotated[Path, typer.Option(help="The photograph to match against.")],
    output: Annotated[Path, typer.Option(help="Where to write the figure.")] = Path(
        "results/matches.png"
    ),
    matcher: Annotated[
        list[str] | None,
        typer.Option("--matcher", "-m", help="Repeat to compare matchers side by side."),
    ] = None,
    keypoints: Annotated[int, typer.Option(help="Keypoint budget per image.")] = 1024,
    device: Annotated[str, typer.Option(help="auto, cpu or cuda.")] = "auto",
    dpi: Annotated[int, typer.Option(min=72, help="Output density.")] = 150,
) -> None:
    """Draw the correspondences two photographs share, per matcher.

    The smallest way into this code: section 15's comparison needs a manifest,
    cached embeddings and about two hours, and this needs two image files.
    """
    names = tuple(matcher or ("sift", "disk-lightglue"))
    try:
        panels = tuple(match_pair(query, candidate, name, keypoints, device) for name in names)
        path = draw_match_figure(
            query,
            candidate,
            panels,
            output,
            long_side=DETECT_LONG_SIDE,
            caption=f"{query.name}  ·  {candidate.name}",
            dpi=dpi,
        )
    except (MatchPlotError, RerankError) as error:
        typer.echo(f"Match figure error: {error}", err=True)
        raise typer.Exit(code=2) from error

    for panel in panels:
        typer.echo(f"  {panel.matcher}: {panel.inliers} verified matches")
    typer.echo(f"Match figure written: {path}")


@visualize_app.command("teaser")
def visualize_teaser(
    family: Annotated[
        list[str],
        typer.Option("--family", help="Repeat once per statue of one lookalike family."),
    ],
    output: Annotated[Path, typer.Option(help="Where to write the figure.")] = Path(
        "paper-cvww/figures/fig0-teaser.jpg"
    ),
    dpi: Annotated[int, typer.Option(min=72, help="Output density.")] = 300,
    width_cm: Annotated[
        float | None,
        typer.Option("--width-cm", help="Printed width; defaults to a short-paper column."),
    ] = None,
    override: OverrideOption = None,
) -> None:
    """Draw the page-1 teaser: the family query answered correctly by the smallest margin."""
    from krasnal_id.viz.teaser import create_teaser

    config = load_config(["experiment=visualization", *(override or [])])
    configure_logging(config.logging)
    try:
        path, teaser = create_teaser(config, family, output, dpi, width_cm)
    except (VisualizationError, EmbeddingStoreError) as error:
        typer.echo(f"Teaser error: {error}", err=True)
        raise typer.Exit(code=2) from error
    typer.echo(f"Teaser query {teaser.query_image_id}, margin {teaser.margin:.3f}: {path}")
    for candidate in teaser.candidates:
        mark = "correct" if candidate.correct else "wrong"
        typer.echo(f"  {candidate.display_name}: {candidate.similarity:.3f} ({mark})")


@visualize_app.command("teaser-matches")
def visualize_teaser_matches(
    family: Annotated[
        list[str],
        typer.Option("--family", help="The teaser's lookalike family, as for visualize teaser."),
    ],
    output: Annotated[Path, typer.Option(help="Where to write the figure.")] = Path(
        "paper-cvww/figures/figS-matches.jpg"
    ),
    matcher: Annotated[
        list[str] | None,
        typer.Option("--matcher", "-m", help="Repeat once per matcher column."),
    ] = None,
    width_cm: Annotated[float, typer.Option("--width-cm", help="Printed width.")] = 17.0,
    device: Annotated[str, typer.Option(help="auto, cpu or cuda.")] = "auto",
    dpi: Annotated[int, typer.Option(min=72, help="Output density.")] = 300,
    override: OverrideOption = None,
) -> None:
    """The teaser's query matched against its correct statue and its lookalike.

    The pair is the teaser's, chosen by its stated rule, so nothing here is picked
    by eye: row one is the correct statue's best photograph, row two the closest
    other statue's.
    """
    from krasnal_id.config import WorkshopValidationConfig
    from krasnal_id.embeddings.store import load_embedding_matrix
    from krasnal_id.experiments.baseline_accuracy import load_evaluation_inputs
    from krasnal_id.experiments.workshop_validation import (
        eligible_queries,
        journal_path,
        load_journal_strictly,
        packaged,
    )
    from krasnal_id.viz.match_plot import draw_match_grid
    from krasnal_id.viz.matcher_examples import matched_photograph, select_rescue
    from krasnal_id.viz.teaser import create_teaser_selection

    config = load_config(["experiment=visualization", *(override or [])])
    configure_logging(config.logging)
    try:
        manifest, split = load_evaluation_inputs(
            config.paths.manifest_path, config.paths.evaluation_split_path
        )
        matrix = load_embedding_matrix(manifest, config.backbone, Path(config.paths.embeddings_dir))
        names = {dwarf.dwarf_id: dwarf.display_name for dwarf in manifest.dwarfs}
        paths = {image.image_id: Path(image.local_path) for image in manifest.images}
        settings = packaged(WorkshopValidationConfig, "workshop_validation")
        evidence = load_journal_strictly(
            journal_path(
                Path(config.paths.results_dir) / "journals",
                split,
                config.backbone.name,
                "disk-lightglue",
                settings,
            ),
            eligible_queries(split, manifest),
        )
        rescue = select_rescue(evidence)
        query_path = paths[rescue.query_image_id]
        teaser = create_teaser_selection(config, family)
        right, wrong = teaser.candidates[0], teaser.candidates[1]
        if not right.correct:
            raise VisualizationError("the teaser's best candidate is not the correct statue")
        rows = (
            (
                f"Rescue, correct: {names[rescue.correct_dwarf_id]}",
                query_path,
                matched_photograph(
                    rescue.query_image_id, rescue.correct_dwarf_id, split, manifest, matrix
                ),
            ),
            (
                f"Rescue, cosine's choice: {names[rescue.cosine_choice_dwarf_id]}",
                query_path,
                matched_photograph(
                    rescue.query_image_id, rescue.cosine_choice_dwarf_id, split, manifest, matrix
                ),
            ),
            (f"Failure, correct: {right.display_name}", teaser.query_local_path, right.local_path),
            (
                f"Failure, lookalike: {wrong.display_name}",
                teaser.query_local_path,
                wrong.local_path,
            ),
        )
        typer.echo(
            f"Rescue: {rescue.query_image_id}, {rescue.correct_inliers} verified matches for the "
            f"correct statue against at most {rescue.best_wrong_inliers} for any other"
        )
        path, counts = draw_match_grid(
            rows,
            tuple(matcher or ("sift", "disk-lightglue")),
            output,
            DETECT_LONG_SIDE,
            width_cm / 2.54,
            device=device,
            dpi=dpi,
        )
    except (VisualizationError, EmbeddingStoreError, MatchPlotError, RerankError) as error:
        typer.echo(f"Teaser match error: {error}", err=True)
        raise typer.Exit(code=2) from error
    for (row, name), inliers in sorted(counts.items()):
        typer.echo(f"  {rows[row][0]} | {name}: {inliers} verified matches")
    typer.echo(f"Teaser match figure written: {path}")


@visualize_app.command("retrieval-examples")
def visualize_retrieval_examples(
    group: Annotated[
        list[str] | None,
        typer.Option("--group", help="Repeat to keep only these agreement groups."),
    ] = None,
    output: Annotated[Path | None, typer.Option(help="Where to write the sheet.")] = None,
    dpi: Annotated[int, typer.Option(min=72, help="Output density.")] = 150,
    top_k: Annotated[
        int, typer.Option("--top-k", help="5 for the full sheet, 1 for the compact one.")
    ] = 5,
    width_cm: Annotated[
        float | None,
        typer.Option(
            "--width-cm",
            help="Printed width; defaults to a short-paper column (top 1) or page (top 5).",
        ),
    ] = None,
    override: OverrideOption = None,
) -> None:
    """Draw query photographs beside the five dwarves each backbone ranks highest.

    The manuscript's figure 4 is this sheet at three rows and publication
    density, which is why the groups and the density are options rather than
    constants.
    """
    config = load_config(["experiment=visualization", *(override or [])])
    configure_logging(config.logging)
    try:
        path = create_retrieval_examples_plot(config, group, output, dpi, top_k, width_cm)
    except (VisualizationError, EmbeddingStoreError) as error:
        typer.echo(f"Retrieval example visualization error: {error}", err=True)
        raise typer.Exit(code=2) from error

    typer.echo(f"Retrieval example visualization complete: figure={path}")


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

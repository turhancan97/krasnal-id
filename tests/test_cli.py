"""Command discovery and placeholder behavior tests."""

import pytest
from typer.testing import CliRunner

from krasnal_id.cli import app

runner = CliRunner()


def test_root_help_lists_pipeline_groups() -> None:
    result = runner.invoke(app, ["--help"])

    assert result.exit_code == 0
    for command in ["data", "embeddings", "retrieve", "experiment", "visualize", "demo"]:
        assert command in result.stdout


@pytest.mark.parametrize(
    "arguments",
    [
        ["data", "build-manifest"],
        ["embeddings", "extract"],
        ["retrieve"],
        ["experiment", "baseline"],
        ["experiment", "pool-ablation"],
        ["experiment", "confusion"],
        ["visualize", "embeddings"],
        ["demo"],
    ],
)
def test_placeholder_commands_validate_then_fail_clearly(arguments: list[str]) -> None:
    result = runner.invoke(app, [*arguments, "--override", "logging.json_output=false"])

    assert result.exit_code == 2
    assert "is not implemented yet" in result.output
    assert result.exception is not None


@pytest.mark.parametrize("group", ["data", "embeddings", "experiment", "visualize"])
def test_nested_help(group: str) -> None:
    result = runner.invoke(app, [group, "--help"])

    assert result.exit_code == 0

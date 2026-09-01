"""Command discovery and placeholder behavior tests."""

import pytest
from typer.testing import CliRunner

from krasnal_id.cli import app

runner = CliRunner()


def test_root_help_lists_pipeline_groups() -> None:
    result = runner.invoke(app, ["--help"], env={"COLUMNS": "120"})

    assert result.exit_code == 0
    for command in ["data", "embeddings", "retrieve", "experiment", "visualize", "demo"]:
        assert command in result.stdout


@pytest.mark.parametrize("group", ["data", "embeddings", "experiment", "visualize"])
def test_nested_help(group: str) -> None:
    result = runner.invoke(app, [group, "--help"])

    assert result.exit_code == 0

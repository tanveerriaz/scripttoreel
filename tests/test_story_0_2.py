"""
MVP 0, Story 0.2 — Project Status Command tests.
"""
import json
import sys
from pathlib import Path

import pytest
from click.testing import CliRunner

sys.path.insert(0, str(Path(__file__).parent.parent))


def test_status_shows_all_modules(tmp_path):
    """--status prints all 6 module names."""
    from src.project_manager import create_project
    from main import cli

    projects_root = tmp_path / "projects"
    create_project("Status Test Topic", duration_min=2, projects_root=projects_root)

    runner = CliRunner()
    result = runner.invoke(cli, [
        "--status", "--project", "status_test_topic",
        "--projects-root", str(projects_root),
    ])
    assert result.exit_code == 0, result.output
    # Output renders "Module 1 —", "Module 2 —", etc.
    for n in ["1", "2", "3", "4", "5", "6"]:
        assert f"module {n}" in result.output.lower(), f"Module {n} missing from status output"


def test_status_shows_topic(tmp_path):
    from src.project_manager import create_project
    from main import cli

    projects_root = tmp_path / "projects"
    create_project("My Cool Video", duration_min=3, projects_root=projects_root)

    runner = CliRunner()
    result = runner.invoke(cli, [
        "--status", "--project", "my_cool_video",
        "--projects-root", str(projects_root),
    ])
    assert "My Cool Video" in result.output


def test_status_nonexistent_project_exits_nonzero(tmp_path):
    from main import cli

    runner = CliRunner()
    result = runner.invoke(cli, [
        "--status", "--project", "does_not_exist",
        "--projects-root", str(tmp_path / "projects"),
    ])
    assert result.exit_code != 0

"""
MVP 0, Story 0.1 — Project Initialization tests.

Tests (all must pass before implementation is considered Done):
- test_init_creates_directory
- test_init_creates_project_json
- test_init_slug_format
- test_init_duplicate_topic_suffix
"""
import json
import re
import sys
from pathlib import Path

import pytest

# Allow imports from project root
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.project_manager import create_project, slug_from_topic
from src.utils.json_schemas import ProjectMetadata


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

@pytest.fixture
def tmp_projects(tmp_path):
    """A temporary projects/ directory."""
    p = tmp_path / "projects"
    p.mkdir()
    return p


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_slug_format_basic():
    """Slug should be snake_case, lowercase, no special chars."""
    slug = slug_from_topic("Haunted Places in Pakistan")
    assert slug == "haunted_places_in_pakistan"
    assert re.match(r"^[a-z0-9_]+$", slug), f"Slug has invalid chars: {slug!r}"


def test_slug_format_special_chars():
    slug = slug_from_topic("Top 10: Best (Free!) APIs")
    assert re.match(r"^[a-z0-9_]+$", slug)


def test_init_creates_directory(tmp_projects):
    meta = create_project("Test Topic", duration_min=3, projects_root=tmp_projects)
    project_dir = tmp_projects / meta.project_id
    assert project_dir.exists(), f"Project dir not created: {project_dir}"


def test_init_creates_project_json(tmp_projects):
    meta = create_project("My Video Topic", duration_min=5, projects_root=tmp_projects)
    project_json = tmp_projects / meta.project_id / "project.json"
    assert project_json.exists(), "project.json not created"

    raw = json.loads(project_json.read_text())
    loaded = ProjectMetadata(**raw)  # validates schema
    assert loaded.topic == "My Video Topic"
    assert loaded.duration_min == 5
    assert loaded.duration_sec == 300


def test_init_slug_in_metadata(tmp_projects):
    meta = create_project("Haunted Places in Pakistan", duration_min=5, projects_root=tmp_projects)
    assert meta.project_id == "haunted_places_in_pakistan"


def test_init_duplicate_topic_suffix(tmp_projects):
    meta1 = create_project("Same Topic", duration_min=2, projects_root=tmp_projects)
    meta2 = create_project("Same Topic", duration_min=2, projects_root=tmp_projects)
    assert meta1.project_id != meta2.project_id
    assert meta2.project_id == "same_topic_2"


def test_init_creates_expected_subdirs(tmp_projects):
    meta = create_project("Dir Test", duration_min=1, projects_root=tmp_projects)
    project_dir = tmp_projects / meta.project_id
    for sub in ["assets/raw/video", "assets/raw/image", "assets/raw/audio", "output"]:
        assert (project_dir / sub).exists(), f"Missing subdir: {sub}"

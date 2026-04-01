"""
Project lifecycle management — create, load, update project.json.
"""
from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from src.utils.json_schemas import ModuleStatus, ProjectMetadata, ProjectPipeline

_PROJECTS_ROOT = Path(__file__).parent.parent / "projects"


# ---------------------------------------------------------------------------
# Public helpers
# ---------------------------------------------------------------------------

def slug_from_topic(topic: str) -> str:
    """Convert a topic string to a safe snake_case project ID."""
    slug = topic.lower()
    slug = re.sub(r"[^a-z0-9\s]", "", slug)  # strip non-alphanumeric
    slug = re.sub(r"\s+", "_", slug.strip())  # spaces → underscore
    slug = re.sub(r"_+", "_", slug)            # collapse double underscores
    return slug


def create_project(
    topic: str,
    duration_min: float,
    projects_root: Optional[Path] = None,
) -> ProjectMetadata:
    """
    Initialise a new VideoForge project directory.

    Returns the populated ProjectMetadata object.
    Handles duplicate slugs by appending _2, _3 …
    """
    root = Path(projects_root) if projects_root else _PROJECTS_ROOT
    root.mkdir(parents=True, exist_ok=True)

    base_slug = slug_from_topic(topic)
    slug = base_slug
    counter = 1
    while (root / slug).exists():
        counter += 1
        slug = f"{base_slug}_{counter}"

    project_dir = root / slug
    now = datetime.now(timezone.utc).isoformat()

    meta = ProjectMetadata(
        project_id=slug,
        topic=topic,
        duration_min=duration_min,
        duration_sec=duration_min * 60,
        created_at=now,
        updated_at=now,
        project_dir=str(project_dir),
        pipeline=ProjectPipeline(),
    )

    # Create directory structure
    for sub in [
        "assets/raw/video",
        "assets/raw/image",
        "assets/raw/audio",
        "assets/processed",
        "output",
    ]:
        (project_dir / sub).mkdir(parents=True, exist_ok=True)

    _save_project_json(meta, project_dir)
    return meta


def load_project(project_id: str, projects_root: Optional[Path] = None) -> ProjectMetadata:
    """Load project.json → ProjectMetadata. Raises FileNotFoundError if missing."""
    root = Path(projects_root) if projects_root else _PROJECTS_ROOT
    path = root / project_id / "project.json"
    if not path.exists():
        raise FileNotFoundError(f"Project not found: {project_id!r} (looked at {path})")
    raw = json.loads(path.read_text())
    return ProjectMetadata(**raw)


def update_pipeline_status(
    project_id: str,
    module: str,
    status: ModuleStatus,
    projects_root: Optional[Path] = None,
    **extra_fields,
) -> ProjectMetadata:
    """Update a single module's pipeline status in project.json."""
    root = Path(projects_root) if projects_root else _PROJECTS_ROOT
    meta = load_project(project_id, root)

    setattr(meta.pipeline, module, status)
    meta.updated_at = datetime.now(timezone.utc).isoformat()
    for k, v in extra_fields.items():
        setattr(meta, k, v)

    project_dir = root / project_id
    _save_project_json(meta, project_dir)
    return meta


# ---------------------------------------------------------------------------
# Internal
# ---------------------------------------------------------------------------

def _save_project_json(meta: ProjectMetadata, project_dir: Path) -> None:
    path = project_dir / "project.json"
    path.write_text(json.dumps(meta.model_dump(), indent=2, default=str))

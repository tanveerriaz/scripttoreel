"""
ScriptToReel CLI — local script-to-video pipeline.

Usage:
    python main.py --init --topic "Haunted Places in Pakistan" --duration 5
    python main.py --run --project haunted_places_in_pakistan
    python main.py --run --no-plan --project haunted_places_in_pakistan
    python main.py --module 1 --project haunted_places_in_pakistan
    python main.py --status --project haunted_places_in_pakistan
    python main.py --validate --project haunted_places_in_pakistan

Pipeline order (new):
    --init  →  create project dirs  →  AI Director (production_plan.json)
    --run   →  Module 1 (Research)  →  2 (Metadata)  →  3 (Script+TTS)
            →  4 (Orchestration)   →  5 (Render)     →  6 (Validation)
"""
from __future__ import annotations

import os
import shutil
import sys
from pathlib import Path

import click
from rich.console import Console
from rich.table import Table
from rich import box

sys.path.insert(0, str(Path(__file__).parent))

if sys.version_info < (3, 11):
    print(
        f"Error: Python {sys.version_info.major}.{sys.version_info.minor} detected. "
        "ScriptToReel requires Python 3.11+ (torch/SDXL/Kokoro need it).\n"
        "Run with:  /opt/homebrew/bin/python3 main.py"
    )
    sys.exit(1)


# Ensure Homebrew binaries are on PATH so ffmpeg/ffprobe are always found on macOS
_HOMEBREW_BIN = "/opt/homebrew/bin"
if Path(_HOMEBREW_BIN).is_dir() and _HOMEBREW_BIN not in os.environ.get("PATH", ""):
    os.environ["PATH"] = _HOMEBREW_BIN + os.pathsep + os.environ.get("PATH", "")

from src.project_manager import create_project, load_project
from src.utils.json_schemas import ModuleStatus

console = Console()

_STATUS_ICONS = {
    ModuleStatus.PENDING: "[dim]⬜ pending[/dim]",
    ModuleStatus.RUNNING: "[yellow]🔄 running[/yellow]",
    ModuleStatus.COMPLETE: "[green]✅ complete[/green]",
    ModuleStatus.FAILED: "[red]❌ failed[/red]",
    ModuleStatus.SKIPPED: "[blue]⏭  skipped[/blue]",
}

_MODULE_LABELS = {
    "module_1_research": "Module 1 — Research & Asset Discovery",
    "module_2_metadata": "Module 2 — Metadata Extraction",
    "module_3_script": "Module 3 — Script & Voiceover",
    "module_4_orchestration": "Module 4 — Scene Orchestration",
    "module_5_render": "Module 5 — FFmpeg Rendering",
    "module_6_validation": "Module 6 — Quality Validation",
}

# New module execution order when using production plan:
# Script first → then asset search (using b_roll keywords) → metadata → orchestration → render → validate
_PLAN_ORDER = [3, 1, 2, 4, 5, 6]
# Legacy order (--no-plan): original 1→2→3→4→5→6
_LEGACY_ORDER = [1, 2, 3, 4, 5, 6]


@click.command()
@click.option("--init", "action", flag_value="init", help="Create a new project")
@click.option("--run", "action", flag_value="run", help="Run all modules end-to-end")
@click.option("--status", "action", flag_value="status", help="Show pipeline status")
@click.option("--validate", "action", flag_value="validate", help="Run validation only")
@click.option("--module", type=int, default=None, help="Run a specific module (1-6)")
@click.option("--topic", default=None, help="Video topic (for --init)")
@click.option("--duration", default=5, type=float, show_default=True, help="Duration in minutes (for --init)")
@click.option("--project", default=None, help="Project ID")
@click.option("--no-plan", "no_plan", is_flag=True, default=False,
              help="Skip production plan generation and use legacy module order (1→2→3→4→5→6)")
@click.option("--projects-root", default=None, hidden=True, help="Override projects directory (for tests)")
@click.option("--skip-director", is_flag=True, default=False, help="Skip AI Director review passes (faster runs)")
def cli(action, module, topic, duration, project, no_plan, projects_root, skip_director):
    """ScriptToReel — topic to 1080p MP4 pipeline."""
    projects_path = Path(projects_root) if projects_root else (Path(__file__).parent / "projects")

    # --init: create project dirs + run AI Director
    if action == "init":
        if not topic:
            console.print("[red]Error:[/red] --topic is required with --init")
            sys.exit(1)
        meta = create_project(topic, duration_min=duration, projects_root=projects_path)
        console.print(f"\n[green]✅ Project created:[/green] projects/{meta.project_id}/")
        console.print(f"   Project ID : [bold]{meta.project_id}[/bold]")
        console.print(f"   Topic      : {meta.topic}")
        console.print(f"   Duration   : {meta.duration_min} min ({meta.duration_sec:.0f}s)")

        if not no_plan:
            # Generate production plan via AI Director
            project_dir = projects_path / meta.project_id
            _generate_production_plan(meta.topic, meta.duration_min, project_dir)
            console.print(f"\n   [bold yellow]Next step:[/bold yellow] Edit [bold]projects/{meta.project_id}/production_plan.json[/bold] then run:")
            console.print(f"   python main.py --run --project {meta.project_id}\n")
        else:
            console.print(f"\n   Next step  : python main.py --run --no-plan --project {meta.project_id}\n")
        return

    # --module N (standalone)
    if module is not None and action is None:
        action = "module"

    # All remaining actions need --project
    if not project:
        console.print("[red]Error:[/red] --project is required")
        sys.exit(1)

    # --status
    if action == "status":
        _print_status(project, projects_path)
        return

    # --module N
    if action == "module" or module is not None:
        _run_module(project, module, projects_path, skip_director=skip_director)
        return

    # --run (all modules)
    if action == "run":
        order = _LEGACY_ORDER if no_plan else _PLAN_ORDER
        if not no_plan:
            # Ensure production plan exists before running
            project_dir = projects_path / project
            try:
                meta = load_project(project, projects_path)
            except FileNotFoundError as e:
                console.print(f"[red]Error:[/red] {e}")
                sys.exit(1)
            _ensure_production_plan(meta.topic, meta.duration_min, project_dir)
        for m in order:
            _run_module(project, m, projects_path, skip_director=skip_director)
        return

    # --validate
    if action == "validate":
        _run_module(project, 6, projects_path)
        return

    console.print("[red]No action specified.[/red] Use --help for usage.")
    sys.exit(1)


# ---------------------------------------------------------------------------
# Production plan helpers
# ---------------------------------------------------------------------------

def _generate_production_plan(topic: str, duration_min: float, project_dir: Path) -> None:
    """Generate production_plan.json via LLM. Prints status to console."""
    console.print("\n   [cyan]Generating production plan...[/cyan]")
    try:
        from src.production_plan import ProductionPlanModule
        pm = ProductionPlanModule(project_dir)
        pm.generate(topic, duration_min)
        console.print("   [green]✅ production_plan.json created[/green]")
    except Exception as e:
        console.print(f"   [yellow]⚠  Production plan generation failed: {e}[/yellow]")
        console.print("   [dim]Continuing with topic-based defaults.[/dim]")


def _ensure_production_plan(topic: str, duration_min: float, project_dir: Path) -> None:
    """Generate production_plan.json if it doesn't exist yet."""
    plan_path = project_dir / "production_plan.json"
    if plan_path.exists():
        console.print("[dim]   production_plan.json found — using existing plan[/dim]")
        return
    console.print("[cyan]   No production_plan.json found — generating now...[/cyan]")
    _generate_production_plan(topic, duration_min, project_dir)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _print_status(project_id: str, projects_path: Path) -> None:
    try:
        meta = load_project(project_id, projects_path)
    except FileNotFoundError as e:
        console.print(f"[red]Error:[/red] {e}")
        sys.exit(1)

    console.print(f"\n[bold]ScriptToReel Project Status[/bold]")
    console.print(f"  Project ID : {meta.project_id}")
    console.print(f"  Topic      : {meta.topic}")
    console.print(f"  Duration   : {meta.duration_min} min")
    console.print(f"  Created    : {meta.created_at[:19].replace('T', ' ')} UTC")

    # Show whether production plan exists
    projects_path_obj = projects_path
    plan_path = projects_path_obj / project_id / "production_plan.json"
    plan_status = "[green]✅ exists[/green]" if plan_path.exists() else "[dim]not generated[/dim]"
    console.print(f"  Plan       : {plan_status}")

    table = Table(box=box.ROUNDED, show_header=True, header_style="bold cyan")
    table.add_column("Module", style="bold")
    table.add_column("Status", justify="left")

    pipeline = meta.pipeline
    module_map = {
        "module_1_research": pipeline.module_1_research,
        "module_2_metadata": pipeline.module_2_metadata,
        "module_3_script": pipeline.module_3_script,
        "module_4_orchestration": pipeline.module_4_orchestration,
        "module_5_render": pipeline.module_5_render,
        "module_6_validation": pipeline.module_6_validation,
    }

    for key, status in module_map.items():
        label = _MODULE_LABELS.get(key, key)
        table.add_row(label, _STATUS_ICONS.get(status, str(status)))

    console.print(table)

    if meta.output_file:
        console.print(f"\n  Output: [green]{meta.output_file}[/green]")
    console.print()


def _check_dependencies(module_num: int) -> None:
    """Exit early with a clear error if required binaries are missing."""
    missing = []
    for binary in ("ffmpeg", "ffprobe"):
        if not shutil.which(binary):
            missing.append(binary)
    if missing:
        console.print(f"[red]Error:[/red] Missing required binaries: {', '.join(missing)}")
        console.print("Install ffmpeg: [bold]brew install ffmpeg[/bold]")
        sys.exit(1)

    # Ollama is only needed for Module 3, and only when OpenRouter is NOT configured
    if module_num in (3,):
        from src.utils.config_loader import load_api_keys as _load_keys
        _keys = _load_keys()
        _use_openrouter = _keys.get("USE_OPENROUTER", "").lower() == "true"
        _has_or_key = bool(_keys.get("OPENROUTER_API_KEY", ""))
        if not (_use_openrouter and _has_or_key):
            import requests as _req
            try:
                _req.get("http://localhost:11434", timeout=3)
            except Exception:
                console.print("[red]Error:[/red] Ollama is not reachable at http://localhost:11434")
                console.print("Start it with: [bold]ollama serve[/bold]")
                console.print("Or configure OpenRouter in config/api_keys.env")
                sys.exit(1)


def _run_module(
    project_id: str,
    module_num: int,
    projects_path: Path,
    skip_director: bool = False,
) -> None:
    """Dispatch to the appropriate module runner."""
    _check_dependencies(module_num)
    try:
        meta = load_project(project_id, projects_path)
    except FileNotFoundError as e:
        console.print(f"[red]Error:[/red] {e}")
        sys.exit(1)

    console.print(f"\n[bold cyan]Running Module {module_num}[/bold cyan] for project [bold]{project_id}[/bold]")

    project_dir = projects_path / project_id

    if module_num == 1:
        from src.module_1_research import ResearchModule
        m = ResearchModule(project_dir)
        m.run()
    elif module_num == 2:
        from src.module_2_metadata import MetadataModule
        m = MetadataModule(project_dir)
        m.run()
    elif module_num == 3:
        from src.module_3_script_voiceover import ScriptModule
        m = ScriptModule(project_dir, skip_director=skip_director)
        m.run()
    elif module_num == 4:
        from src.module_4_orchestration import OrchestrationModule
        m = OrchestrationModule(project_dir, skip_director=skip_director)
        m.run()
    elif module_num == 5:
        from src.module_5_ffmpeg_render import RenderModule
        m = RenderModule(project_dir)
        m.run()
    elif module_num == 6:
        from src.module_6_validation import ValidationModule
        m = ValidationModule(project_dir)
        report = m.run()
        sys.exit(0 if report.passed else 1)
    else:
        console.print(f"[red]Unknown module: {module_num}. Must be 1-6.[/red]")
        sys.exit(1)


if __name__ == "__main__":
    cli()

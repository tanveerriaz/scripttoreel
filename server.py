"""
ScriptToReel Web Server — Flask backend for the dashboard UI.

Endpoints:
  POST /api/generate       Start a new video generation job
  GET  /api/status/<id>    Poll pipeline status (SSE stream)
  GET  /api/projects       List all projects
  GET  /api/video/<id>     Download the final MP4
  GET  /                   Serve dashboard.html
"""
from __future__ import annotations

import json
import logging
import queue
import subprocess
import sys
import threading
import time
from pathlib import Path

from flask import Flask, Response, jsonify, request, send_file, send_from_directory

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

app = Flask(
    __name__,
    static_folder=str(ROOT),
    static_url_path="/static",
)
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# In-memory job registry: project_id → {"status": ..., "log": [...], "error": None}
_jobs: dict[str, dict] = {}
_jobs_lock = threading.Lock()

MODULE_NAMES = {
    1: "Research & Asset Discovery",
    2: "Metadata Extraction",
    3: "Script & Voiceover",
    4: "Scene Orchestration",
    5: "FFmpeg Rendering",
    6: "Quality Validation",
}


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.route("/")
def index():
    return send_from_directory(str(ROOT), "dashboard.html")


@app.route("/assets/<path:subpath>")
def serve_assets(subpath: str):
    """Serve files under assets/ so dashboard can use relative URLs (same as file:// opens)."""
    return send_from_directory(str(ROOT / "assets"), subpath)


@app.route("/api/projects")
def list_projects():
    projects_dir = ROOT / "projects"
    projects = []
    if projects_dir.exists():
        for p in sorted(projects_dir.iterdir()):
            pj = p / "project.json"
            if pj.exists():
                try:
                    meta = json.loads(pj.read_text())
                    video = p / "output" / "final_video.mp4"
                    projects.append({
                        "id": meta["project_id"],
                        "topic": meta.get("topic", ""),
                        "duration_min": meta.get("duration_min", 0),
                        "created_at": meta.get("created_at", ""),
                        "has_video": video.exists(),
                        "pipeline": meta.get("pipeline", {}),
                    })
                except Exception:
                    pass
    return jsonify(projects)


@app.route("/api/generate", methods=["POST"])
def generate():
    body = request.get_json(force=True)
    topic = (body.get("topic") or "").strip()
    duration = float(body.get("duration", 3))

    if not topic:
        return jsonify({"error": "topic is required"}), 400
    if duration < 0.5 or duration > 30:
        return jsonify({"error": "duration must be between 0.5 and 30 minutes"}), 400

    # Init the project synchronously so we get the project_id immediately
    result = subprocess.run(
        [sys.executable, "main.py", "--init", "--topic", topic, "--duration", str(duration)],
        capture_output=True, text=True, cwd=str(ROOT)
    )
    if result.returncode != 0:
        return jsonify({"error": result.stderr or "init failed"}), 500

    # Parse project_id from output
    project_id = None
    for line in result.stdout.splitlines():
        if "Project ID" in line:
            project_id = line.split(":")[-1].strip()
            break
    if not project_id:
        return jsonify({"error": "could not determine project_id"}), 500

    with _jobs_lock:
        _jobs[project_id] = {
            "project_id": project_id,
            "topic": topic,
            "current_module": 0,
            "current_module_name": "Initializing",
            "module_statuses": {},
            "log": [],
            "done": False,
            "success": False,
            "error": None,
        }

    # Run pipeline in background thread
    t = threading.Thread(target=_run_pipeline, args=(project_id,), daemon=True)
    t.start()

    return jsonify({"project_id": project_id})


@app.route("/api/status/<project_id>")
def status_stream(project_id):
    """Server-Sent Events stream for live pipeline progress."""
    def generate_events():
        last_log_idx = 0
        while True:
            with _jobs_lock:
                job = _jobs.get(project_id)
            if not job:
                yield f"data: {json.dumps({'error': 'unknown project'})}\n\n"
                return

            payload = {
                "project_id": project_id,
                "current_module": job["current_module"],
                "current_module_name": job["current_module_name"],
                "module_statuses": job["module_statuses"],
                "log": job["log"][last_log_idx:],
                "done": job["done"],
                "success": job["success"],
                "error": job["error"],
            }
            last_log_idx = len(job["log"])
            yield f"data: {json.dumps(payload)}\n\n"

            if job["done"]:
                return
            time.sleep(0.8)

    return Response(generate_events(), mimetype="text/event-stream",
                    headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


@app.route("/api/video/<project_id>")
def download_video(project_id):
    video = ROOT / "projects" / project_id / "output" / "final_video.mp4"
    if not video.exists():
        return jsonify({"error": "video not found"}), 404
    return send_file(
        str(video),
        as_attachment=True,
        download_name=f"{project_id}.mp4",
        mimetype="video/mp4",
    )


# ---------------------------------------------------------------------------
# Background pipeline runner
# ---------------------------------------------------------------------------

def _update_job(project_id: str, **kwargs):
    with _jobs_lock:
        if project_id in _jobs:
            _jobs[project_id].update(kwargs)


def _append_log(project_id: str, msg: str):
    with _jobs_lock:
        if project_id in _jobs:
            _jobs[project_id]["log"].append(msg)


def _run_pipeline(project_id: str):
    for module_num in range(1, 7):
        _update_job(
            project_id,
            current_module=module_num,
            current_module_name=MODULE_NAMES[module_num],
        )
        _append_log(project_id, f"▶ Starting Module {module_num}: {MODULE_NAMES[module_num]}…")

        proc = subprocess.Popen(
            [sys.executable, "main.py", "--module", str(module_num), "--project", project_id],
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, cwd=str(ROOT),
        )

        lines = []
        for line in proc.stdout:
            line = line.rstrip()
            if line:
                lines.append(line)
                _append_log(project_id, line)
        proc.wait()

        status = "complete" if proc.returncode == 0 else "failed"
        with _jobs_lock:
            if project_id in _jobs:
                _jobs[project_id]["module_statuses"][module_num] = status

        if proc.returncode != 0:
            _append_log(project_id, f"❌ Module {module_num} failed")
            _update_job(project_id, done=True, success=False,
                        error=f"Module {module_num} failed. Check logs.")
            return

        _append_log(project_id, f"✅ Module {module_num} complete")

    _update_job(project_id, done=True, success=True, current_module=7,
                current_module_name="Done")
    _append_log(project_id, "🎉 Video generation complete! Download your video below.")


if __name__ == "__main__":
    print("ScriptToReel Dashboard → http://localhost:8080")
    app.run(host="0.0.0.0", port=8080, debug=False, threaded=True)

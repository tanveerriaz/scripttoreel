#!/opt/homebrew/bin/python3
"""
ScriptToReel Web Server — Flask backend for the dashboard UI.

Endpoints:
  POST   /api/generate          Start a new video generation job
  GET    /api/status/<id>       Poll pipeline status (SSE stream)
  GET    /api/projects          List all projects
  DELETE /api/projects/<id>     Delete a project and all its files
  GET    /api/video/<id>        Download the final MP4
  GET    /                      Serve dashboard.html
"""
from __future__ import annotations

import json
import logging
import queue
import shutil
import subprocess
import sys
import threading
import time
from pathlib import Path

from flask import Flask, Response, jsonify, request, send_file, send_from_directory

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

from src.utils.llm_client import call_llm
from src.utils.config_loader import load_api_keys
from src.hook_engine import HookEngine

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
                        "script_title": meta.get("script_title", ""),
                        "duration_min": meta.get("duration_min", 0),
                        "created_at": meta.get("created_at", ""),
                        "has_video": video.exists(),
                        "pipeline": meta.get("pipeline", {}),
                        "pipeline_elapsed_sec": meta.get("pipeline_elapsed_sec"),
                        "module_timings": meta.get("module_timings"),
                    })
                except Exception:
                    pass
    return jsonify(projects)


@app.route("/api/search-topics", methods=["POST"])
def search_topics():
    """Web search + LLM to suggest 3-5 engaging video topics from a query."""
    body = request.get_json(force=True)
    query = (body.get("query") or "").strip()
    if not query:
        return jsonify({"error": "query is required"}), 400

    # Step 1: Web search via DuckDuckGo (news first for real stories, then text)
    search_results = []
    search_context = ""
    search_failed = False
    try:
        from duckduckgo_search import DDGS
        ddgs = DDGS()
        # Try news search first — returns real, recent, sourced articles
        try:
            news = ddgs.news(query, max_results=8)
            for r in news:
                search_results.append({
                    "title": r.get("title", ""),
                    "body": r.get("body", ""),
                    "url": r.get("url", ""),
                    "source_name": r.get("source", ""),
                    "date": r.get("date", ""),
                })
        except Exception:
            pass
        # Supplement with text search if news didn't return enough
        if len(search_results) < 5:
            text_results = ddgs.text(query, max_results=8)
            for r in text_results:
                search_results.append({
                    "title": r.get("title", ""),
                    "body": r.get("body", ""),
                    "url": r.get("href", ""),
                    "source_name": "",
                    "date": "",
                })
        snippets = []
        for i, r in enumerate(search_results):
            source_info = f" [{r['source_name']}]" if r.get("source_name") else ""
            date_info = f" ({r['date'][:10]})" if r.get("date") else ""
            snippets.append(f"[{i+1}] \"{r['title']}\"{source_info}{date_info} — {r['body']} (source: {r['url']})")
        search_context = "\n".join(snippets)[:4000]
        logger.info("Found %d search results (%d news + text)", len(search_results), len(search_results))
    except Exception as e:
        logger.warning("DuckDuckGo search failed: %s — falling back to LLM only", e)
        search_failed = True

    # Step 2: LLM selects and reframes REAL stories from search results
    logger.info("Search results fed to LLM:\n%s", search_context[:500] if search_context else "(none)")
    system_prompt = (
        "You are a video topic curator. You MUST ONLY use information from the search results below. "
        "Do NOT generate topics from your own knowledge.\n\n"
        "RULES:\n"
        "1. Each topic MUST correspond to a specific numbered search result [1], [2], etc.\n"
        "2. Copy the exact source URL from that search result into the 'source' field.\n"
        "3. The 'angle' field must summarize what the search result actually says — no embellishment.\n"
        "4. You may write a catchy 'title', but it must accurately reflect the source content.\n"
        "5. Do NOT invent any facts, names, statistics, or events not in the search results.\n"
        "6. Pick 3-5 of the most interesting results.\n\n"
        "Return ONLY a JSON array (no other text):\n"
        "[{\"title\": \"catchy title based on the real story\", "
        "\"angle\": \"summary of what the article actually reports\", "
        "\"why_viral\": \"why this would make a good video\", "
        "\"source\": \"exact URL from the search result\"}]"
    )
    if search_context:
        user_prompt = (
            f"User searched for: \"{query}\"\n\n"
            f"SEARCH RESULTS:\n{search_context}\n\n"
            "Pick 3-5 of the most compelling results from above. "
            "For each, copy the source URL exactly as shown. "
            "Do NOT add information beyond what the search results contain."
        )
    else:
        # No search results — cannot guarantee factual topics
        user_prompt = (
            f"User wants to create a video about: \"{query}\"\n\n"
            "I could not retrieve web search results. Suggest 3-5 GENERAL topic angles "
            "the user could research further. Mark each with \"needs_verification\": true "
            "since these are not backed by specific sources. "
            "Do NOT present any specific facts, statistics, or claims as real."
        )

    try:
        raw = call_llm(system_prompt, user_prompt, temperature=0.3, max_tokens=1024)
        # Parse JSON — strip markdown fences, extract the JSON array
        import re
        raw = re.sub(r"```(?:json)?\s*", "", raw).strip().rstrip("`")
        # Always use regex to find the array — handles trailing text after ]
        match = re.search(r"\[.*\]", raw, re.DOTALL)
        if not match:
            raise ValueError("No JSON array found in LLM response")
        suggestions = json.loads(match.group())
        # Filter out any empty or invalid suggestions
        suggestions = [s for s in suggestions if s.get("title")]
    except Exception as e:
        logger.warning("LLM topic suggestion failed: %s — returning raw query", e)
        suggestions = []

    # If LLM returned nothing useful, build suggestions directly from search results
    if not suggestions and search_results:
        for r in search_results[:5]:
            if r.get("title"):
                suggestions.append({
                    "title": r["title"],
                    "angle": r.get("body", ""),
                    "why_viral": "Real story from the web",
                    "source": r.get("url", ""),
                })

    # Last resort fallback
    if not suggestions:
        suggestions = [{"title": query, "angle": "Direct topic", "why_viral": "Your original idea", "needs_verification": True}]

    return jsonify({"suggestions": suggestions, "raw_query": query, "search_failed": search_failed})


@app.route("/api/generate-hook", methods=["POST"])
def generate_hook():
    """Generate a viral hook title for a given topic using HookEngine."""
    body = request.get_json(force=True)
    topic = (body.get("topic") or "").strip()
    if not topic:
        return jsonify({"error": "topic is required"}), 400

    try:
        engine = HookEngine(api_keys=load_api_keys())
        hook = engine.select_best_hook(topic)
        return jsonify({
            "hook_title": hook.get("text", topic),
            "pattern": hook.get("pattern", ""),
            "score": hook.get("score", 0),
        })
    except Exception as e:
        logger.warning("Hook generation failed: %s — using topic as title", e)
        return jsonify({"hook_title": topic, "pattern": "fallback", "score": 0})


@app.route("/api/generate", methods=["POST"])
def generate():
    body = request.get_json(force=True)
    topic = (body.get("topic") or "").strip()
    duration = float(body.get("duration", 3))
    hook_title = (body.get("hook_title") or "").strip()

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

    # Write hook title to project.json as script_title
    if hook_title:
        try:
            pj = ROOT / "projects" / project_id / "project.json"
            if pj.exists():
                meta = json.loads(pj.read_text())
                meta["script_title"] = hook_title
                pj.write_text(json.dumps(meta, indent=2, default=str))
        except Exception as e:
            logger.warning("Failed to save hook title to project.json: %s", e)

    with _jobs_lock:
        _jobs[project_id] = {
            "project_id": project_id,
            "topic": topic,
            "current_module": 0,
            "current_module_name": "Initializing",
            "module_statuses": {},
            "module_timings": {},           # module_num → seconds
            "log": [],
            "done": False,
            "success": False,
            "error": None,
            "started_at": time.time(),
            "elapsed_sec": 0,
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

            # Compute live elapsed time if still running
            elapsed = job.get("elapsed_sec", 0)
            if not job["done"] and job.get("started_at"):
                elapsed = round(time.time() - job["started_at"], 1)

            payload = {
                "project_id": project_id,
                "current_module": job["current_module"],
                "current_module_name": job["current_module_name"],
                "module_statuses": job["module_statuses"],
                "module_timings": job.get("module_timings", {}),
                "elapsed_sec": elapsed,
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


@app.route("/api/projects/<project_id>", methods=["DELETE"])
def delete_project(project_id: str):
    """Delete a project directory and all its assets/output files."""
    # Sanitise: reject any path traversal attempts
    if ".." in project_id or "/" in project_id:
        return jsonify({"error": "invalid project_id"}), 400

    project_dir = ROOT / "projects" / project_id
    if not project_dir.exists():
        return jsonify({"error": "project not found"}), 404

    try:
        shutil.rmtree(project_dir)
    except Exception as exc:
        logger.error("Failed to delete project %s: %s", project_id, exc)
        return jsonify({"error": str(exc)}), 500

    # Remove from in-memory job registry if present
    with _jobs_lock:
        _jobs.pop(project_id, None)

    logger.info("Deleted project %s", project_id)
    return jsonify({"deleted": project_id})


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


@app.route("/api/video/<project_id>/stream")
def stream_video(project_id):
    """Serve video inline for HTML5 <video> playback with Range request support."""
    video = ROOT / "projects" / project_id / "output" / "final_video.mp4"
    if not video.exists():
        return jsonify({"error": "video not found"}), 404

    file_size = video.stat().st_size
    range_header = request.headers.get("Range")

    if range_header:
        # Parse Range: bytes=start-end
        byte_range = range_header.replace("bytes=", "").split("-")
        start = int(byte_range[0])
        end = int(byte_range[1]) if byte_range[1] else file_size - 1
        length = end - start + 1

        with open(video, "rb") as f:
            f.seek(start)
            data = f.read(length)

        resp = Response(
            data,
            status=206,
            mimetype="video/mp4",
            direct_passthrough=True,
        )
        resp.headers["Content-Range"] = f"bytes {start}-{end}/{file_size}"
        resp.headers["Accept-Ranges"] = "bytes"
        resp.headers["Content-Length"] = str(length)
        return resp

    return send_file(str(video), as_attachment=False, mimetype="video/mp4")


@app.route("/api/thumbnail/<project_id>")
def get_thumbnail(project_id):
    """Serve project thumbnail (auto-generated by Module 6)."""
    thumb = ROOT / "projects" / project_id / "output" / "thumbnail.jpg"
    if not thumb.exists():
        return jsonify({"error": "thumbnail not found"}), 404
    return send_file(str(thumb), mimetype="image/jpeg")


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


_PLAN_ORDER = [3, 1, 2, 4, 5, 6]  # Script first → then research (uses script's SDXL prompts)

def _run_pipeline(project_id: str):
    pipeline_start = time.time()

    for module_num in _PLAN_ORDER:
        _update_job(
            project_id,
            current_module=module_num,
            current_module_name=MODULE_NAMES[module_num],
        )
        _append_log(project_id, f"▶ Starting Module {module_num}: {MODULE_NAMES[module_num]}…")

        module_start = time.time()
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
        module_elapsed = round(time.time() - module_start, 1)

        status = "complete" if proc.returncode == 0 else "failed"
        with _jobs_lock:
            if project_id in _jobs:
                _jobs[project_id]["module_statuses"][module_num] = status
                _jobs[project_id]["module_timings"][module_num] = module_elapsed
                _jobs[project_id]["elapsed_sec"] = round(time.time() - pipeline_start, 1)

        if proc.returncode != 0:
            _append_log(project_id, f"❌ Module {module_num} failed ({module_elapsed:.1f}s)")
            _update_job(project_id, done=True, success=False,
                        error=f"Module {module_num} failed. Check logs.")
            return

        _append_log(project_id, f"✅ Module {module_num} complete ({module_elapsed:.1f}s)")

    total_elapsed = round(time.time() - pipeline_start, 1)
    with _jobs_lock:
        if project_id in _jobs:
            _jobs[project_id]["elapsed_sec"] = total_elapsed

    # Save timing to project.json
    try:
        pj = ROOT / "projects" / project_id / "project.json"
        if pj.exists():
            meta = json.loads(pj.read_text())
            meta["pipeline_elapsed_sec"] = total_elapsed
            meta["module_timings"] = {str(k): v for k, v in _jobs.get(project_id, {}).get("module_timings", {}).items()}
            pj.write_text(json.dumps(meta, indent=2, default=str))
    except Exception as e:
        logger.warning("Failed to save timing to project.json: %s", e)

    mins, secs = divmod(int(total_elapsed), 60)
    _update_job(project_id, done=True, success=True, current_module=7,
                current_module_name="Done")
    _append_log(project_id, f"🎉 Video generation complete in {mins}m {secs}s! Download your video below.")


if __name__ == "__main__":
    print("ScriptToReel Dashboard → http://localhost:8080")
    app.run(host="0.0.0.0", port=8080, debug=False, threaded=True)

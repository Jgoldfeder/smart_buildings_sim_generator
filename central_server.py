#!/usr/bin/env python3
"""Central task queue server for distributed smart buildings baselines.

Usage:
    # Start server with default jobs (all buildings × all weathers × all policies)
    API_KEY=your-secret-key python central_server.py

    # Or set buildings/weathers/policies via environment
    BUILDINGS_DIR=buildings API_KEY=secret python central_server.py
"""

import os
import glob
import base64
import time
import threading
from datetime import datetime, timedelta
from flask import Flask, request, jsonify, abort
from queue import Queue
from threading import Lock
import pickle
from pathlib import Path

app = Flask(__name__)

# Configuration from environment
VALID_API_KEY = os.environ.get("API_KEY", "smart-buildings-key-123")
BUILDINGS_DIR = os.environ.get("BUILDINGS_DIR", "buildings")
RESULTS_DIR = os.environ.get("RESULTS_DIR", "distributed_results")

# Thread-safe in-memory task queue
task_queue = Queue()
lock = Lock()

# Track completed jobs
completed_jobs = set()
failed_jobs = []

# Weather presets (matching scenario_generator.py)
WEATHER_PRESETS = [
    "hot_summer", "mild_summer", "cold_winter",
    "mild_winter", "tropical", "temperate"
]

POLICIES = ["all_off", "all_on", "bang_bang"]


def get_job_key(task):
    """Create unique key for a job."""
    return f"{task['building']}:{task['weather']}:{task['policy']}"


def result_exists(task):
    """Check if result already exists on disk."""
    building_name = Path(task['building']).stem
    path = Path(RESULTS_DIR) / building_name / task['weather'] / task['policy'] / "summary.json"
    return path.exists()


def populate_queue():
    """Populate queue with all building × weather × policy combinations."""
    building_files = sorted(glob.glob(os.path.join(BUILDINGS_DIR, "*.yaml")))

    if not building_files:
        print(f"Warning: No building configs found in {BUILDINGS_DIR}")
        return

    commands = []
    skipped = 0

    for building_path in building_files:
        for weather in WEATHER_PRESETS:
            for policy in POLICIES:
                task = {
                    "building": building_path,
                    "weather": weather,
                    "policy": policy,
                }

                # Skip if already completed
                if result_exists(task):
                    skipped += 1
                    completed_jobs.add(get_job_key(task))
                else:
                    commands.append(task)

    for command in commands:
        task_queue.put(command)

    print(f"Buildings: {len(building_files)}")
    print(f"Weathers: {len(WEATHER_PRESETS)}")
    print(f"Policies: {len(POLICIES)}")
    print(f"Total possible: {len(building_files) * len(WEATHER_PRESETS) * len(POLICIES)}")
    print(f"Already completed: {skipped}")
    print(f"Queued: {task_queue.qsize()}")


# Populate on startup
populate_queue()


@app.before_request
def require_api_key():
    """Enforce API key on every request."""
    key = request.headers.get("X-API-KEY") or request.args.get("key")
    if key != VALID_API_KEY:
        abort(401, description="Unauthorized: invalid or missing API key")


@app.route('/get_task', methods=['GET'])
def get_task():
    """Worker pulls one task; returns 204 if queue is empty."""
    with lock:
        if task_queue.empty():
            return jsonify({"task": None}), 204
        task = task_queue.get()
    return jsonify({"task": task}), 200


@app.route('/add_task', methods=['POST'])
def add_task():
    """Allow adding new tasks. Expects JSON body: {"task": {...}}"""
    data = request.get_json(silent=True) or {}
    task = data.get("task")
    if not task:
        return jsonify(error="no task provided"), 400
    with lock:
        task_queue.put(task)
    return jsonify(ok=True), 201


@app.route('/report', methods=['POST'])
def report():
    """Worker sends a report dict as JSON; returns 200 on success."""
    data = request.get_json(silent=True)

    if not data or not isinstance(data, dict):
        return jsonify(error="expected JSON object"), 400

    # Extract job info
    building = data.get('building')
    weather = data.get('weather')
    policy = data.get('policy')
    status = data.get('status', 'unknown')

    if not all([building, weather, policy]):
        return jsonify(error="missing building, weather, or policy"), 400

    job_key = f"{building}:{weather}:{policy}"

    if status == 'success':
        with lock:
            completed_jobs.add(job_key)
        app.logger.info(f"Completed: {job_key} - reward={data.get('total_reward', 'N/A')}")
    else:
        with lock:
            failed_jobs.append({
                'job_key': job_key,
                'error': data.get('error', 'unknown error')
            })
        app.logger.warning(f"Failed: {job_key} - {data.get('error')}")

    return jsonify(ok=True), 200


# Track active workers and their progress
active_workers = {}
STALE_TIMEOUT = 300  # 5 minutes - requeue job if no heartbeat

@app.route('/status', methods=['GET'])
def status():
    """Get current queue status."""
    with lock:
        return jsonify({
            "queued": task_queue.qsize(),
            "completed": len(completed_jobs),
            "failed": len(failed_jobs),
            "failed_jobs": failed_jobs[-10:],  # Last 10 failures
            "active_workers": active_workers,
        }), 200


@app.route('/heartbeat', methods=['POST'])
def heartbeat():
    """Worker sends progress update."""
    data = request.get_json(silent=True)
    if not data:
        return jsonify(error="expected JSON"), 400

    worker_id = data.get('worker_id')
    step = data.get('step')
    total_steps = data.get('total_steps')
    job = data.get('job')
    task = data.get('task')  # Full task dict for recovery

    with lock:
        active_workers[worker_id] = {
            "job": job,
            "task": task,
            "step": step,
            "total_steps": total_steps,
            "last_update": time.time(),  # Use server time for consistency
        }

    pct = (step / total_steps * 100) if total_steps else 0
    print(f"[{worker_id}] {job}: {step}/{total_steps} ({pct:.1f}%)", flush=True)

    return jsonify(ok=True), 200


def check_stale_workers():
    """Background thread to requeue jobs from stale workers."""
    while True:
        time.sleep(60)  # Check every minute
        now = time.time()
        stale = []

        with lock:
            for worker_id, info in list(active_workers.items()):
                if now - info.get('last_update', 0) > STALE_TIMEOUT:
                    stale.append((worker_id, info))
                    del active_workers[worker_id]

        for worker_id, info in stale:
            task = info.get('task')
            if task:
                with lock:
                    job_key = get_job_key(task)
                    if job_key not in completed_jobs:
                        task_queue.put(task)
                        print(f"[STALE] Re-queued {info['job']} from {worker_id}", flush=True)


@app.route('/requeue_failed', methods=['POST'])
def requeue_failed():
    """Re-add all failed jobs to the queue."""
    with lock:
        count = 0
        for job in failed_jobs:
            parts = job['job_key'].split(':')
            if len(parts) == 3:
                task = {
                    "building": parts[0],
                    "weather": parts[1],
                    "policy": parts[2],
                }
                task_queue.put(task)
                count += 1
        failed_jobs.clear()
    return jsonify(ok=True, requeued=count), 200


@app.route('/upload_results', methods=['POST'])
def upload_results():
    """Worker uploads result files. Expects JSON with base64-encoded files."""
    data = request.get_json(silent=True)

    if not data or not isinstance(data, dict):
        return jsonify(error="expected JSON object"), 400

    building = data.get('building')
    weather = data.get('weather')
    policy = data.get('policy')
    files = data.get('files', {})

    if not all([building, weather, policy]):
        return jsonify(error="missing building, weather, or policy"), 400

    # Create output directory
    policy_dir = Path(RESULTS_DIR) / building / weather / policy
    policy_dir.mkdir(parents=True, exist_ok=True)

    saved = []
    for filename, content_b64 in files.items():
        try:
            content = base64.b64decode(content_b64)
            filepath = policy_dir / filename
            with open(filepath, 'wb') as f:
                f.write(content)
            saved.append(filename)
        except Exception as e:
            app.logger.error(f"Failed to save {filename}: {e}")

    app.logger.info(f"Uploaded {len(saved)} files for {building}/{weather}/{policy}")
    return jsonify(ok=True, saved=saved), 200


if __name__ == '__main__':
    print(f"\nSmart Buildings Task Queue Server")
    print(f"API Key: {VALID_API_KEY[:4]}...{VALID_API_KEY[-4:]}")
    print(f"Buildings dir: {BUILDINGS_DIR}")
    print(f"Results dir: {RESULTS_DIR}")
    print(f"Stale timeout: {STALE_TIMEOUT}s")
    print()

    # Start background thread to check for stale workers
    stale_checker = threading.Thread(target=check_stale_workers, daemon=True)
    stale_checker.start()

    # Listen on all interfaces so workers can connect
    app.run(host='0.0.0.0', port=5000)

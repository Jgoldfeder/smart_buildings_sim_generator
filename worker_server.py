#!/usr/bin/env python3
"""Worker client for distributed smart buildings baselines.

Usage:
    # Run with 4 worker threads connecting to local server
    python worker_server.py 4

    # Connect to remote server
    SERVER=http://remote-server:5000 python worker_server.py 8
"""

import threading
import time
import requests
import sys
import json
import os
import base64
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for threading
import matplotlib.pyplot as plt

# Configuration
API_KEY = os.environ.get("API_KEY", "smart-buildings-key-123")
SERVER = os.environ.get("SERVER", "http://127.0.0.1:5000")
HEADERS = {"X-API-KEY": API_KEY}
RESULTS_DIR = os.environ.get("RESULTS_DIR", "distributed_results")
STEPS = int(os.environ.get("STEPS", 576))  # Default: 2 days
FPS = int(os.environ.get("FPS", 10))
VIDEO_FORMAT = os.environ.get("VIDEO_FORMAT", "gif")


def get_task():
    """Fetch a task from the server."""
    resp = requests.get(f"{SERVER}/get_task", headers=HEADERS)
    if resp.status_code == 204:
        return None
    resp.raise_for_status()
    return resp.json().get("task")


def send_report(report_dict):
    """Send completion report to server."""
    resp = requests.post(f"{SERVER}/report", json=report_dict, headers=HEADERS)
    resp.raise_for_status()
    return resp.json()


def upload_results(building_name, weather, policy, policy_dir):
    """Upload result files to central server."""
    files = {}
    for filepath in policy_dir.iterdir():
        if filepath.is_file():
            with open(filepath, 'rb') as f:
                files[filepath.name] = base64.b64encode(f.read()).decode('utf-8')

    payload = {
        "building": building_name,
        "weather": weather,
        "policy": policy,
        "files": files,
    }
    resp = requests.post(f"{SERVER}/upload_results", json=payload, headers=HEADERS)
    resp.raise_for_status()
    return resp.json()


def get_action_indices(env):
    """Get indices for each action type in continuous_action."""
    temp_indices = []
    pressure_indices = []
    water_temp_indices = []

    idx = 0
    for device, action_name in env._device_action_tuples:
        if "run_command" in action_name:
            continue
        if "supply_air_temperature" in action_name:
            temp_indices.append(idx)
        elif "supply_water_setpoint" in action_name:
            water_temp_indices.append(idx)
        else:
            pressure_indices.append(idx)
        idx += 1

    return temp_indices, pressure_indices, water_temp_indices


def make_action(env, discrete_value, temp_value, pressure_value, water_value):
    """Create action dict with proper values per action type."""
    n_discrete = env.action_spec()["discrete_action"].shape[0]
    n_continuous = env.action_spec()["continuous_action"].shape[0]

    temp_idx, pressure_idx, water_idx = get_action_indices(env)

    continuous = [0.0] * n_continuous
    for i in temp_idx:
        continuous[i] = temp_value
    for i in pressure_idx:
        continuous[i] = pressure_value
    for i in water_idx:
        continuous[i] = water_value

    return {
        "discrete_action": [discrete_value] * n_discrete,
        "continuous_action": continuous,
    }


def get_avg_zone_temp(env):
    """Get average zone temperature from environment."""
    temps = [vav.zone_air_temperature for vav in env.building.simulator._hvac._vavs.values()]
    return np.mean(temps)


def get_current_setpoints(env):
    """Get current (heating, cooling) setpoints based on schedule."""
    schedule = env.building.simulator._hvac.schedule
    timestamp = env.building.current_timestamp
    return schedule.get_temperature_window(timestamp)


def run_policy(env, tracker, policy, steps):
    """Run a baseline policy."""
    tracker.reset()

    for _ in range(steps):
        if policy == "all_off":
            action = make_action(env, discrete_value=0, temp_value=-1, pressure_value=-1, water_value=-1)
        elif policy == "all_on":
            action = make_action(env, discrete_value=1, temp_value=-1, pressure_value=1, water_value=1)
        elif policy == "bang_bang":
            zone_temp = get_avg_zone_temp(env)
            heating_setpoint, cooling_setpoint = get_current_setpoints(env)
            if zone_temp < heating_setpoint:
                action = make_action(env, discrete_value=1, temp_value=1, pressure_value=1, water_value=1)
            elif zone_temp > cooling_setpoint:
                action = make_action(env, discrete_value=1, temp_value=-1, pressure_value=1, water_value=1)
            else:
                action = make_action(env, discrete_value=0, temp_value=-1, pressure_value=-1, water_value=-1)
        else:
            raise ValueError(f"Unknown policy: {policy}")
        tracker.step(action)


def run_task(task, thread_id):
    """Run a single task (building + weather + policy)."""
    from sbsim.smart_control.utils.scenario_generator import (
        load_scenario_from_parts,
        generate_scenario_from_config,
        get_env,
        SimulationTracker,
    )

    building_path = task['building']
    weather = task['weather']
    policy = task['policy']
    building_name = Path(building_path).stem

    job_id = f"{building_name}/{weather}/{policy}"
    print(f"[Thread {thread_id}] Starting: {job_id}")

    try:
        # Generate scenario
        scenario_config = load_scenario_from_parts(building_path, weather, RESULTS_DIR)
        result = generate_scenario_from_config(scenario_config)

        env = get_env(result)
        tracker = SimulationTracker(env, vmin=280, vmax=310)

        # Run policy
        run_policy(env, tracker, policy, STEPS)

        # Create output directory
        policy_dir = Path(RESULTS_DIR) / building_name / weather / policy
        policy_dir.mkdir(parents=True, exist_ok=True)

        # Save video
        video_path = policy_dir / f"simulation.{VIDEO_FORMAT}"
        tracker.create_video(str(video_path), fps=FPS)

        # Save charts
        tracker.plot_overview(color_by_ahu=True)
        plt.savefig(policy_dir / "overview.png", dpi=150, bbox_inches="tight")
        plt.close()

        tracker.plot_rewards()
        plt.savefig(policy_dir / "rewards.png", dpi=150, bbox_inches="tight")
        plt.close()

        tracker.plot_ahu()
        plt.savefig(policy_dir / "ahu.png", dpi=150, bbox_inches="tight")
        plt.close()

        # Save timeseries data
        df = pd.DataFrame({
            "timestamp": [ts.isoformat() for ts in tracker.timestamps],
            "reward": tracker.rewards,
            "occupancy": tracker.occupancy,
        })
        zone_temps_arr = np.array(tracker.zone_temps)
        for i in range(zone_temps_arr.shape[1]):
            df[f"zone_{i}_temp"] = zone_temps_arr[:, i]
        df.to_csv(policy_dir / "timeseries.csv", index=False)

        # Save reward components
        if tracker.reward_components:
            rc_df = pd.DataFrame(tracker.reward_components)
            rc_df["timestamp"] = [ts.isoformat() for ts in tracker.timestamps]
            rc_df.to_csv(policy_dir / "reward_components.csv", index=False)

        # Save summary
        summary = {
            "building": building_name,
            "weather": weather,
            "policy": policy,
            "total_reward": sum(tracker.rewards),
            "total_steps": len(tracker.rewards),
            "avg_reward": np.mean(tracker.rewards),
            "avg_occupancy": np.mean(tracker.occupancy),
            "avg_zone_temp": np.mean([np.mean(t) for t in tracker.zone_temps]),
            "num_rooms": result['num_rooms'],
            "num_ahus": result['num_ahus'],
        }

        if tracker.reward_components:
            summary["total_productivity"] = sum(r.get("productivity", 0) for r in tracker.reward_components)
            summary["total_energy_cost"] = sum(r.get("energy_cost", 0) for r in tracker.reward_components)
            summary["total_carbon_cost"] = sum(r.get("carbon_cost", 0) for r in tracker.reward_components)

        with open(policy_dir / "summary.json", "w") as f:
            json.dump(summary, f, indent=2)

        # Upload results to central server
        print(f"[Thread {thread_id}] Uploading: {job_id}")
        upload_results(building_name, weather, policy, policy_dir)

        print(f"[Thread {thread_id}] Done: {job_id} -> reward={summary['total_reward']:.2f}")

        # Report success to server
        report = {
            "building": building_path,
            "weather": weather,
            "policy": policy,
            "status": "success",
            "total_reward": summary["total_reward"],
        }
        send_report(report)

    except Exception as e:
        print(f"[Thread {thread_id}] Error: {job_id} - {e}")

        # Report failure to server
        report = {
            "building": building_path,
            "weather": weather,
            "policy": policy,
            "status": "error",
            "error": str(e),
        }
        try:
            send_report(report)
        except:
            pass  # Server might be down


def thread_main(thread_id):
    """Main loop for each worker thread."""
    while True:
        try:
            task = get_task()
            if task is None:
                # No work: back off for a bit
                time.sleep(5)
                continue

            run_task(task, thread_id)

        except requests.exceptions.ConnectionError:
            print(f"[Thread {thread_id}] Cannot connect to server, retrying in 10s...")
            time.sleep(10)
        except Exception as e:
            print(f"[Thread {thread_id}] Unexpected error: {e}")
            time.sleep(5)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python worker_server.py <num_threads>")
        print("  Example: python worker_server.py 4")
        print()
        print("Environment variables:")
        print(f"  SERVER={SERVER}")
        print(f"  API_KEY={API_KEY[:4]}...{API_KEY[-4:]}")
        print(f"  RESULTS_DIR={RESULTS_DIR}")
        print(f"  STEPS={STEPS}")
        print(f"  FPS={FPS}")
        print(f"  VIDEO_FORMAT={VIDEO_FORMAT}")
        sys.exit(1)

    NUM_THREADS = int(sys.argv[1])

    print(f"Smart Buildings Worker")
    print(f"Server: {SERVER}")
    print(f"Results dir: {RESULTS_DIR}")
    print(f"Steps: {STEPS}")
    print(f"Video format: {VIDEO_FORMAT}")
    print(f"Threads: {NUM_THREADS}")
    print()

    # Preload data and models to avoid race conditions
    print("Preloading datasets and models...")
    from sbsim.smart_control.utils import classification_util
    # Add any preloading here if needed

    threads = []
    for i in range(NUM_THREADS):
        t = threading.Thread(target=thread_main, args=(i,), daemon=True)
        t.start()
        threads.append(t)
        print(f"Started thread {i}")

    # Keep main alive while threads run
    try:
        while True:
            time.sleep(10)
    except KeyboardInterrupt:
        print("\nShutting down workers...")

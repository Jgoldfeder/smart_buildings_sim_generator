#!/usr/bin/env python3
"""Run baseline policies on multiple buildings in parallel.

Usage:
    # Run all buildings in a directory with all weathers and policies
    python run_baselines_parallel.py --buildings buildings/*.yaml --threads 8

    # Specific buildings and weathers
    python run_baselines_parallel.py --buildings buildings/env_*.yaml --weather hot_summer cold_winter --threads 4

    # Custom policies
    python run_baselines_parallel.py --buildings buildings/*.yaml --policies bang_bang --threads 12
"""
import argparse
import glob
import json
import os
from pathlib import Path
from datetime import datetime
from multiprocessing import Pool, cpu_count
from functools import partial

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for multiprocessing
import matplotlib.pyplot as plt


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


def make_action(env, discrete_value: int, temp_value: float, pressure_value: float, water_value: float):
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


def run_single_job(job, steps, fps, video_format):
    """Run a single (building, weather, policy) combination.

    This function runs in a worker process.
    """
    building_path, weather_name, policy, output_dir = job

    # Import here to avoid issues with multiprocessing
    from sbsim.smart_control.utils.scenario_generator import (
        load_scenario_from_parts,
        generate_scenario_from_config,
        get_env,
        SimulationTracker,
    )

    building_name = Path(building_path).stem
    job_id = f"{building_name}/{weather_name}/{policy}"

    try:
        print(f"[START] {job_id}")

        # Generate scenario
        scenario_config = load_scenario_from_parts(building_path, weather_name, str(output_dir))
        result = generate_scenario_from_config(scenario_config)

        env = get_env(result)
        tracker = SimulationTracker(env, vmin=280, vmax=310)

        # Run policy
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

        # Create output directory
        policy_dir = Path(output_dir) / building_name / weather_name / policy
        policy_dir.mkdir(parents=True, exist_ok=True)

        # Save video
        video_path = policy_dir / f"simulation.{video_format}"
        tracker.create_video(str(video_path), fps=fps)

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
            "weather": weather_name,
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

        print(f"[DONE]  {job_id} -> reward={summary['total_reward']:.2f}")
        return {"status": "success", "job_id": job_id, "summary": summary}

    except Exception as e:
        print(f"[ERROR] {job_id}: {e}")
        return {"status": "error", "job_id": job_id, "error": str(e)}


def main():
    from sbsim.smart_control.utils.scenario_generator import list_weather_presets

    parser = argparse.ArgumentParser(
        description="Run baseline policies on multiple buildings in parallel",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=f"""
Examples:
  # Run all buildings with all weathers (6) and all policies (3)
  python run_baselines_parallel.py --buildings buildings/*.yaml --threads 8

  # Specific weathers only
  python run_baselines_parallel.py --buildings buildings/*.yaml --weather hot_summer cold_winter --threads 4

  # Single policy across all
  python run_baselines_parallel.py --buildings buildings/*.yaml --policies bang_bang --threads 12

Available weather presets: {', '.join(list_weather_presets())}
""")
    parser.add_argument("--buildings", type=str, nargs="+", required=True,
                        help="Building config paths (supports glob patterns)")
    parser.add_argument("--weather", type=str, nargs="*",
                        help="Weather preset(s). Default: all presets")
    parser.add_argument("--policies", type=str, nargs="+", default=["all_off", "all_on", "bang_bang"],
                        help="Policies to run")
    parser.add_argument("--threads", type=int, default=cpu_count(),
                        help=f"Number of parallel workers (default: {cpu_count()})")
    parser.add_argument("--output-dir", type=str, default="results",
                        help="Output directory")
    parser.add_argument("--steps", type=int, default=12*24*2,
                        help="Steps per policy (default: 576 = 2 days)")
    parser.add_argument("--fps", type=int, default=10, help="Video FPS")
    parser.add_argument("--video-format", type=str, default="gif", choices=["mp4", "gif"],
                        help="Video format (default: gif, mp4 requires imageio)")

    args = parser.parse_args()

    # Expand glob patterns
    building_paths = []
    for pattern in args.buildings:
        matches = glob.glob(pattern)
        if matches:
            building_paths.extend(matches)
        else:
            # Not a glob, treat as literal path
            if os.path.exists(pattern):
                building_paths.append(pattern)
            else:
                print(f"Warning: {pattern} not found")

    building_paths = sorted(set(building_paths))  # dedupe and sort

    if not building_paths:
        print("Error: No building configs found")
        return

    # Get weather presets
    weathers = args.weather if args.weather else list_weather_presets()

    # Create output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(args.output_dir) / f"parallel_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Build job list
    jobs = []
    for building in building_paths:
        for weather in weathers:
            for policy in args.policies:
                jobs.append((building, weather, policy, str(output_dir)))

    n_buildings = len(building_paths)
    n_weathers = len(weathers)
    n_policies = len(args.policies)
    n_jobs = len(jobs)

    print(f"Buildings: {n_buildings}")
    print(f"Weathers: {n_weathers} ({', '.join(weathers)})")
    print(f"Policies: {n_policies} ({', '.join(args.policies)})")
    print(f"Total jobs: {n_jobs} ({n_buildings} x {n_weathers} x {n_policies})")
    print(f"Threads: {args.threads}")
    print(f"Output: {output_dir}")
    print(f"Steps per job: {args.steps}")
    print()

    # Run in parallel
    worker_fn = partial(run_single_job, steps=args.steps, fps=args.fps, video_format=args.video_format)

    with Pool(args.threads) as pool:
        results = pool.map(worker_fn, jobs)

    # Aggregate results
    successes = [r for r in results if r["status"] == "success"]
    errors = [r for r in results if r["status"] == "error"]

    print(f"\n{'='*60}")
    print(f"COMPLETED: {len(successes)}/{n_jobs} jobs")
    if errors:
        print(f"ERRORS: {len(errors)}")
        for e in errors:
            print(f"  {e['job_id']}: {e['error']}")
    print("="*60)

    # Save aggregated results
    all_summaries = [r["summary"] for r in successes]
    with open(output_dir / "all_results.json", "w") as f:
        json.dump(all_summaries, f, indent=2)

    # Create summary CSV
    if all_summaries:
        summary_df = pd.DataFrame(all_summaries)
        summary_df.to_csv(output_dir / "summary.csv", index=False)

        # Print pivot table
        print("\nReward summary (building x weather, bang_bang policy):")
        bang_bang = summary_df[summary_df["policy"] == "bang_bang"]
        if not bang_bang.empty:
            pivot = bang_bang.pivot(index="building", columns="weather", values="total_reward")
            print(pivot.to_string())

    print(f"\nAll results saved to: {output_dir}")


if __name__ == "__main__":
    main()

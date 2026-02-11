#!/usr/bin/env python3
"""Run baseline policies on a scenario and save results.

Usage:
    # Single config file (legacy)
    python run_baselines.py --config scenarios/example_scenario.yaml

    # Separate building and weather configs
    python run_baselines.py --building buildings/office_3floor.yaml --weather hot_summer
    python run_baselines.py --building buildings/office_3floor.yaml --weather weather/custom.yaml

    # With options
    python run_baselines.py --building buildings/office.yaml --weather cold_winter --steps 576 --fps 15
"""
import argparse
import json
import os
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sbsim.smart_control.utils.scenario_generator import (
    generate_scenario,
    generate_scenario_from_config,
    load_scenario_from_parts,
    list_weather_presets,
    get_env,
    SimulationTracker,
)


def get_action_indices(env):
    """Get indices for each action type in continuous_action."""
    temp_indices = []
    pressure_indices = []
    water_temp_indices = []

    idx = 0
    for device, action_name in env._device_action_tuples:
        if "run_command" in action_name:
            continue  # discrete
        if "supply_air_temperature" in action_name:
            temp_indices.append(idx)
        elif "supply_water_setpoint" in action_name:
            water_temp_indices.append(idx)
        else:  # pressure setpoints
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


def run_baseline(env, tracker, policy: str, steps: int):
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


def save_data(tracker, output_dir: Path, policy: str):
    """Save all tracker data to files."""
    policy_dir = output_dir / policy
    policy_dir.mkdir(parents=True, exist_ok=True)

    # Save timeseries data as CSV
    df = pd.DataFrame({
        "timestamp": [ts.isoformat() for ts in tracker.timestamps],
        "reward": tracker.rewards,
        "occupancy": tracker.occupancy,
    })

    # Add zone temps (one column per zone)
    zone_temps_arr = np.array(tracker.zone_temps)
    for i in range(zone_temps_arr.shape[1]):
        df[f"zone_{i}_temp"] = zone_temps_arr[:, i]

    df.to_csv(policy_dir / "timeseries.csv", index=False)

    # Save reward components as CSV
    if tracker.reward_components:
        rc_df = pd.DataFrame(tracker.reward_components)
        rc_df["timestamp"] = [ts.isoformat() for ts in tracker.timestamps]
        rc_df.to_csv(policy_dir / "reward_components.csv", index=False)

    # Save AHU log as CSV
    if tracker.ahu_log:
        ahu_df = pd.DataFrame(tracker.ahu_log)
        # Flatten list columns
        for col in ahu_df.columns:
            if isinstance(ahu_df[col].iloc[0], list):
                list_len = len(ahu_df[col].iloc[0])
                for i in range(list_len):
                    ahu_df[f"{col}_{i}"] = ahu_df[col].apply(lambda x: x[i] if i < len(x) else None)
                ahu_df = ahu_df.drop(columns=[col])
        ahu_df.to_csv(policy_dir / "ahu_log.csv", index=False)

    # Save summary stats as JSON
    summary = {
        "policy": policy,
        "total_reward": sum(tracker.rewards),
        "total_steps": len(tracker.rewards),
        "avg_reward": np.mean(tracker.rewards),
        "avg_occupancy": np.mean(tracker.occupancy),
        "avg_zone_temp": np.mean([np.mean(t) for t in tracker.zone_temps]),
        "productivity_weight": tracker.productivity_weight,
        "energy_weight": tracker.energy_weight,
        "carbon_weight": tracker.carbon_weight,
    }

    if tracker.reward_components:
        summary["total_productivity"] = sum(r.get("productivity", 0) for r in tracker.reward_components)
        summary["total_energy_cost"] = sum(r.get("energy_cost", 0) for r in tracker.reward_components)
        summary["total_carbon_cost"] = sum(r.get("carbon_cost", 0) for r in tracker.reward_components)

    with open(policy_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    return summary


def save_charts(tracker, output_dir: Path, policy: str):
    """Save charts as PNG files."""
    import matplotlib
    matplotlib.use("Agg")  # Non-interactive backend

    policy_dir = output_dir / policy
    policy_dir.mkdir(parents=True, exist_ok=True)

    # Temporarily override plt.show to save instead
    original_show = plt.show

    def save_show():
        pass  # Don't show, we'll save manually

    plt.show = save_show

    try:
        # Plot overview
        tracker.plot_overview(color_by_ahu=True)
        plt.savefig(policy_dir / "overview.png", dpi=150, bbox_inches="tight")
        plt.close()

        # Plot rewards
        tracker.plot_rewards()
        plt.savefig(policy_dir / "rewards.png", dpi=150, bbox_inches="tight")
        plt.close()

        # Plot AHU
        tracker.plot_ahu()
        plt.savefig(policy_dir / "ahu.png", dpi=150, bbox_inches="tight")
        plt.close()

    finally:
        plt.show = original_show


def run_scenario(building_path, weather_name, output_dir, args):
    """Run all policies for a single building+weather combination."""
    from sbsim.smart_control.utils.scenario_generator import (
        load_scenario_from_parts,
        generate_scenario_from_config,
        get_env,
        SimulationTracker,
    )

    print(f"\n{'#'*60}")
    print(f"# Building: {building_path}")
    print(f"# Weather: {weather_name}")
    print(f"{'#'*60}\n")

    # Generate scenario
    print("Generating scenario...")
    scenario_config = load_scenario_from_parts(building_path, weather_name, str(output_dir.parent))
    result = generate_scenario_from_config(scenario_config)
    print(f"Generated {result['num_rooms']} rooms with {result['num_ahus']} AHUs")

    env = get_env(result)
    tracker = SimulationTracker(env, vmin=280, vmax=310)

    print(f"Reward weights: {env.reward_function.weights}")
    print()

    # Run each policy
    all_summaries = {}
    for policy in args.policies:
        print(f"{'='*60}")
        print(f"Running policy: {policy}")
        print("="*60)

        run_baseline(env, tracker, policy, args.steps)

        # Save video
        video_path = output_dir / policy / f"simulation.{args.video_format}"
        video_path.parent.mkdir(parents=True, exist_ok=True)
        print(f"Creating video: {video_path}")
        tracker.create_video(str(video_path), fps=args.fps)

        # Save charts
        print("Saving charts...")
        save_charts(tracker, output_dir, policy)

        # Save data
        print("Saving data...")
        summary = save_data(tracker, output_dir, policy)
        all_summaries[policy] = summary

        print(f"Total reward: {summary['total_reward']:.2f}")
        print()

    # Save comparison summary for this weather
    with open(output_dir / "comparison.json", "w") as f:
        json.dump(all_summaries, f, indent=2)

    return all_summaries


def main():
    parser = argparse.ArgumentParser(
        description="Run baseline policies on a scenario",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=f"""
Examples:
  # Single config file (legacy)
  python run_baselines.py --config scenarios/example_scenario.yaml

  # Run building with ALL weather presets (default)
  python run_baselines.py --building buildings/office_3floor.yaml

  # Run building with specific weather presets
  python run_baselines.py --building buildings/office.yaml --weather hot_summer cold_winter

Available weather presets: {', '.join(list_weather_presets())}
""")
    # Config options
    config_group = parser.add_argument_group("Configuration")
    config_group.add_argument("--config", type=str, help="Path to combined scenario YAML config (legacy)")
    config_group.add_argument("--building", type=str, help="Path to building YAML config")
    config_group.add_argument("--weather", type=str, nargs="*",
                              help="Weather preset(s) or path(s). If omitted with --building, runs ALL presets")

    # Run options
    parser.add_argument("--output-dir", type=str, default="results", help="Output directory")
    parser.add_argument("--steps", type=int, default=12*24*2, help="Steps per policy (default: 576 = 2 days)")
    parser.add_argument("--fps", type=int, default=10, help="Video FPS")
    parser.add_argument("--video-format", type=str, default="mp4", choices=["mp4", "gif"], help="Video format")
    parser.add_argument("--policies", type=str, nargs="+", default=["all_off", "all_on", "bang_bang"],
                        help="Policies to run")
    args = parser.parse_args()

    # Validate args
    if args.config and args.building:
        parser.error("Use either --config OR --building, not both")
    if not args.config and not args.building:
        parser.error("Must specify either --config or --building")

    # Determine weather configs to run
    if args.config:
        # Legacy single-config mode
        weathers = [None]  # placeholder
    elif args.weather:
        # Specific weather(s) requested
        weathers = args.weather
    else:
        # Default: all weather presets
        weathers = list_weather_presets()

    # Determine building name
    if args.config:
        config_name = Path(args.config).stem
    else:
        config_name = Path(args.building).stem

    # Create timestamped base output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_output_dir = Path(args.output_dir) / f"{config_name}_{timestamp}"
    base_output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Building: {args.building or args.config}")
    print(f"Weather configs: {weathers if weathers[0] else ['from config']}")
    print(f"Output: {base_output_dir}")
    print(f"Steps: {args.steps}")
    print(f"Policies: {args.policies}")

    # Run for each weather config
    all_results = {}

    if args.config:
        # Legacy single-config mode
        print("\nGenerating scenario...")
        result = generate_scenario(args.config)
        print(f"Generated {result['num_rooms']} rooms with {result['num_ahus']} AHUs")

        env = get_env(result)
        tracker = SimulationTracker(env, vmin=280, vmax=310)

        print(f"Reward weights: {env.reward_function.weights}")
        print()

        # Run each policy
        all_summaries = {}
        for policy in args.policies:
            print(f"{'='*60}")
            print(f"Running policy: {policy}")
            print("="*60)

            run_baseline(env, tracker, policy, args.steps)

            video_path = base_output_dir / policy / f"simulation.{args.video_format}"
            video_path.parent.mkdir(parents=True, exist_ok=True)
            print(f"Creating video: {video_path}")
            tracker.create_video(str(video_path), fps=args.fps)

            print("Saving charts...")
            save_charts(tracker, base_output_dir, policy)

            print("Saving data...")
            summary = save_data(tracker, base_output_dir, policy)
            all_summaries[policy] = summary

            print(f"Total reward: {summary['total_reward']:.2f}")
            print()

        with open(base_output_dir / "comparison.json", "w") as f:
            json.dump(all_summaries, f, indent=2)

        all_results["config"] = all_summaries

    else:
        # Building + weather mode
        for weather in weathers:
            weather_name = Path(weather).stem if weather.endswith('.yaml') else weather
            weather_output_dir = base_output_dir / weather_name
            weather_output_dir.mkdir(parents=True, exist_ok=True)

            summaries = run_scenario(args.building, weather, weather_output_dir, args)
            all_results[weather_name] = summaries

    # Save cross-weather comparison
    with open(base_output_dir / "all_results.json", "w") as f:
        json.dump(all_results, f, indent=2)

    # Print final summary
    print("\n" + "="*60)
    print("FINAL RESULTS SUMMARY")
    print("="*60)
    for weather_name, summaries in all_results.items():
        print(f"\n{weather_name}:")
        for policy, summary in summaries.items():
            print(f"  {policy:12s}: reward={summary['total_reward']:8.2f}")

    print(f"\nAll results saved to: {base_output_dir}")


if __name__ == "__main__":
    main()

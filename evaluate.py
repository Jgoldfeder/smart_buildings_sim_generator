#!/usr/bin/env python3
"""Evaluation script for HVAC control policies.

Usage:
    from evaluate import evaluate, AllOffPolicy, AllOnPolicy, BangBangPolicy

    policy = BangBangPolicy()
    scenarios = [
        ("buildings/building_0001.yaml", "hot_summer"),
        ("buildings/building_0001.yaml", "cold_winter"),
    ]
    results = evaluate(policy, scenarios, num_runs=3, num_steps=864)
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Tuple, Dict, Any
import numpy as np


# =============================================================================
# Policy Base Class and Implementations
# =============================================================================

class Policy(ABC):
    """Base class for HVAC control policies."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Return the policy name."""
        pass

    @abstractmethod
    def get_action(self, env) -> Dict[str, Any]:
        """Return an action given the current environment state.

        Args:
            env: The HVAC environment instance.

        Returns:
            Action dict with 'discrete_action' and 'continuous_action' keys.
        """
        pass

    def reset(self):
        """Reset any internal state. Called at the start of each episode."""
        pass


def _get_action_indices(env):
    """Get indices for different action types (temp, pressure, water)."""
    action_names = env.action_spec()["continuous_action"].action_names
    temp_indices = []
    pressure_indices = []
    water_temp_indices = []

    idx = 0
    for name in action_names:
        if "supply_water_setpoint" in name:
            water_temp_indices.append(idx)
        elif "setpoint" in name:
            temp_indices.append(idx)
        elif "differential_pressure" in name:
            pressure_indices.append(idx)
        idx += 1

    return temp_indices, pressure_indices, water_temp_indices


def _make_action(env, discrete_value: int, temp_value: float,
                 pressure_value: float, water_value: float) -> Dict[str, Any]:
    """Create action dict with proper values per action type."""
    n_discrete = env.action_spec()["discrete_action"].shape[0]
    n_continuous = env.action_spec()["continuous_action"].shape[0]

    temp_idx, pressure_idx, water_idx = _get_action_indices(env)

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


def _get_avg_zone_temp(env) -> float:
    """Get average zone temperature from environment."""
    temps = [vav.zone_air_temperature for vav in env.building.simulator._hvac._vavs.values()]
    return np.mean(temps)


def _get_current_setpoints(env) -> Tuple[float, float]:
    """Get current (heating, cooling) setpoints based on schedule."""
    schedule = env.building.simulator._hvac.schedule
    timestamp = env.building.current_timestamp
    return schedule.get_temperature_window(timestamp)


class AllOffPolicy(Policy):
    """Policy that turns off all HVAC equipment."""

    @property
    def name(self) -> str:
        return "all_off"

    def get_action(self, env) -> Dict[str, Any]:
        return _make_action(env, discrete_value=0, temp_value=-1,
                           pressure_value=-1, water_value=-1)


class AllOnPolicy(Policy):
    """Policy that turns on all HVAC equipment at max."""

    @property
    def name(self) -> str:
        return "all_on"

    def get_action(self, env) -> Dict[str, Any]:
        return _make_action(env, discrete_value=1, temp_value=-1,
                           pressure_value=1, water_value=1)


class BangBangPolicy(Policy):
    """Bang-bang controller that heats when cold, cools when hot."""

    @property
    def name(self) -> str:
        return "bang_bang"

    def get_action(self, env) -> Dict[str, Any]:
        zone_temp = _get_avg_zone_temp(env)
        heating_setpoint, cooling_setpoint = _get_current_setpoints(env)

        if zone_temp < heating_setpoint:
            # Too cold - heat
            return _make_action(env, discrete_value=1, temp_value=1,
                               pressure_value=1, water_value=1)
        elif zone_temp > cooling_setpoint:
            # Too hot - cool
            return _make_action(env, discrete_value=1, temp_value=-1,
                               pressure_value=1, water_value=1)
        else:
            # In comfort range - off
            return _make_action(env, discrete_value=0, temp_value=-1,
                               pressure_value=-1, water_value=-1)


# =============================================================================
# Evaluation Function
# =============================================================================

@dataclass
class EvaluationResult:
    """Results from evaluating a policy on multiple scenarios."""
    policy_name: str
    scenarios: List[Tuple[str, str]]
    num_runs: int
    num_steps: int

    # Returns for each (scenario_idx, run_idx)
    returns: List[List[float]]

    # Average return per scenario
    scenario_averages: List[float]

    # Overall average return
    average_return: float

    # Standard deviation of returns
    std_return: float

    def __repr__(self) -> str:
        lines = [
            f"EvaluationResult(policy={self.policy_name})",
            f"  Scenarios: {len(self.scenarios)}",
            f"  Runs per scenario: {self.num_runs}",
            f"  Steps per episode: {self.num_steps}",
            f"  Average return: {self.average_return:.2f} +/- {self.std_return:.2f}",
        ]
        for i, (building, weather) in enumerate(self.scenarios):
            building_name = building.split("/")[-1].replace(".yaml", "")
            lines.append(f"    {building_name}/{weather}: {self.scenario_averages[i]:.2f}")
        return "\n".join(lines)


def evaluate(
    policy: Policy,
    scenarios: List[Tuple[str, str]],
    num_runs: int = 3,
    num_steps: int = None,
    verbose: bool = True,
) -> EvaluationResult:
    """Evaluate a policy on multiple scenarios.

    Args:
        policy: A Policy instance to evaluate.
        scenarios: List of (building_config_path, weather_name) tuples.
        num_runs: Number of times to run each scenario (default: 3).
        num_steps: Steps per episode. If None, defaults to 3 days (864 steps at 5min intervals).
        verbose: Whether to print progress.

    Returns:
        EvaluationResult with returns for each run and averages.
    """
    from sbsim.smart_control.utils.scenario_generator import (
        load_scenario_from_parts,
        generate_scenario_from_config,
        get_env,
    )

    # Default to 3 days at 5-minute intervals
    if num_steps is None:
        num_steps = 3 * 24 * 12  # 864 steps

    all_returns = []  # List of lists: [scenario][run]

    for scenario_idx, (building_path, weather_name) in enumerate(scenarios):
        if verbose:
            print(f"\nScenario {scenario_idx + 1}/{len(scenarios)}: {building_path} / {weather_name}")

        scenario_returns = []

        for run_idx in range(num_runs):
            if verbose:
                print(f"  Run {run_idx + 1}/{num_runs}...", end=" ", flush=True)

            # Generate scenario (re-generate each run for different random seeds if applicable)
            scenario_config = load_scenario_from_parts(
                building_path, weather_name,
                output_base_dir=f"/tmp/eval_{scenario_idx}_{run_idx}"
            )
            result = generate_scenario_from_config(scenario_config)
            env = get_env(result)

            # Reset policy
            policy.reset()

            # Run episode
            total_reward = 0.0
            env.reset()

            for step in range(num_steps):
                action = policy.get_action(env)
                ts = env.step(action)
                total_reward += ts.reward

            scenario_returns.append(total_reward)

            if verbose:
                print(f"return = {total_reward:.2f}")

        all_returns.append(scenario_returns)

    # Compute statistics
    flat_returns = [r for scenario in all_returns for r in scenario]
    scenario_averages = [np.mean(scenario) for scenario in all_returns]
    average_return = np.mean(flat_returns)
    std_return = np.std(flat_returns)

    result = EvaluationResult(
        policy_name=policy.name,
        scenarios=scenarios,
        num_runs=num_runs,
        num_steps=num_steps,
        returns=all_returns,
        scenario_averages=scenario_averages,
        average_return=average_return,
        std_return=std_return,
    )

    if verbose:
        print(f"\n{result}")

    return result


# =============================================================================
# CLI
# =============================================================================

def main():
    """Example usage from command line."""
    import argparse

    parser = argparse.ArgumentParser(description="Evaluate baseline policies")
    parser.add_argument("--policy", type=str, default="bang_bang",
                       choices=["all_off", "all_on", "bang_bang"],
                       help="Policy to evaluate")
    parser.add_argument("--buildings", type=str, nargs="+",
                       default=["buildings/building_0001.yaml"],
                       help="Building config files")
    parser.add_argument("--weathers", type=str, nargs="+",
                       default=["temperate"],
                       help="Weather presets")
    parser.add_argument("--runs", type=int, default=3,
                       help="Number of runs per scenario")
    parser.add_argument("--steps", type=int, default=864,
                       help="Steps per episode (default: 864 = 3 days)")
    args = parser.parse_args()

    # Create policy
    policy_map = {
        "all_off": AllOffPolicy(),
        "all_on": AllOnPolicy(),
        "bang_bang": BangBangPolicy(),
    }
    policy = policy_map[args.policy]

    # Create scenarios (all combinations of buildings x weathers)
    scenarios = [(b, w) for b in args.buildings for w in args.weathers]

    # Run evaluation
    result = evaluate(policy, scenarios, num_runs=args.runs, num_steps=args.steps)

    print(f"\nFinal Result: {result.average_return:.2f} +/- {result.std_return:.2f}")


if __name__ == "__main__":
    main()

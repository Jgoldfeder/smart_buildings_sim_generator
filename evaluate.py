#!/usr/bin/env python3
"""Evaluation script for HVAC control policies.

Usage:
    from evaluate import evaluate, AllOffPolicy, AllOnPolicy, BangBangPolicy, PolicyConfig

    # Simple: pass policy class, evaluate() handles config creation
    scenarios = [
        ("buildings/building_0001.yaml", "hot_summer"),
        ("buildings/building_0001.yaml", "cold_winter"),
    ]
    result = evaluate(BangBangPolicy, scenarios, num_runs=3, num_steps=864)

    # With custom policy kwargs
    result = evaluate(BangBangPolicy, scenarios,
                      policy_kwargs={"heating_setpoint": 294.0})

    # Manual config creation (for custom policies)
    config = PolicyConfig.from_env(env, floor_plan=fp, zone_map=zm)
    policy = MyCustomPolicy(config)
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List, Tuple, Dict, Any, Optional
import numpy as np


# =============================================================================
# Policy Configuration
# =============================================================================

@dataclass
class PolicyConfig:
    """Configuration for HVAC control policies.

    Contains all information a policy needs to generate actions,
    independent of a specific environment instance.
    """

    # Action space configuration
    n_discrete: int
    n_continuous: int
    temp_indices: List[int]
    pressure_indices: List[int]
    water_indices: List[int]

    # Observation space configuration
    zone_temp_indices: List[int]
    observation_field_names: List[str] = field(default_factory=list)

    # Building layout (optional, for advanced policies)
    floor_plan: Optional[np.ndarray] = None
    zone_map: Optional[np.ndarray] = None
    ahu_zone_assignments: Optional[Dict[str, List[int]]] = None
    num_floors: int = 1

    @classmethod
    def from_env(cls, env, floor_plan: Optional[np.ndarray] = None,
                 zone_map: Optional[np.ndarray] = None,
                 ahu_zone_assignments: Optional[Dict[str, List[int]]] = None,
                 num_floors: int = 1) -> "PolicyConfig":
        """Create PolicyConfig from an environment instance.

        Args:
            env: HVAC environment instance.
            floor_plan: Optional floor plan array (H x W, 0=wall, >0=room_id).
            zone_map: Optional zone map array (H x W, zone indices).
            ahu_zone_assignments: Optional dict mapping AHU id -> list of zone indices.
            num_floors: Number of floors in the building.

        Returns:
            PolicyConfig instance.
        """
        action_spec = env.action_spec()
        n_discrete = action_spec["discrete_action"].shape[0]
        n_continuous = action_spec["continuous_action"].shape[0]

        # Get action indices for different types
        action_names = action_spec["continuous_action"].action_names
        temp_indices = []
        pressure_indices = []
        water_indices = []

        for idx, name in enumerate(action_names):
            if "supply_water_setpoint" in name:
                water_indices.append(idx)
            elif "setpoint" in name:
                temp_indices.append(idx)
            elif "differential_pressure" in name:
                pressure_indices.append(idx)

        # Get observation field names for extracting zone temps
        field_names = list(env.field_names)
        zone_temp_indices = [
            i for i, name in enumerate(field_names)
            if "zone_air_temperature" in name
        ]

        return cls(
            n_discrete=n_discrete,
            n_continuous=n_continuous,
            temp_indices=temp_indices,
            pressure_indices=pressure_indices,
            water_indices=water_indices,
            zone_temp_indices=zone_temp_indices,
            observation_field_names=field_names,
            floor_plan=floor_plan,
            zone_map=zone_map,
            ahu_zone_assignments=ahu_zone_assignments,
            num_floors=num_floors,
        )


# =============================================================================
# Policy Base Class and Implementations
# =============================================================================

class Policy(ABC):
    """Base class for HVAC control policies.

    Policies are initialized with a PolicyConfig containing action/observation specs,
    then get_action() takes only the observation array.
    """

    def __init__(self, config: PolicyConfig):
        """Initialize policy with configuration.

        Args:
            config: PolicyConfig with action/observation specs and optional building layout.
        """
        self.config = config

    @property
    @abstractmethod
    def name(self) -> str:
        """Return the policy name."""
        pass

    @abstractmethod
    def get_action(self, obs: np.ndarray) -> Dict[str, Any]:
        """Return an action given the current observation.

        Args:
            obs: Observation numpy array from env.step() or env.reset().

        Returns:
            Action dict with 'discrete_action' and 'continuous_action' keys.
        """
        pass

    def reset(self):
        """Reset any internal state. Called at the start of each episode."""
        pass

    def _make_action(self, discrete_value: int, temp_value: float,
                     pressure_value: float, water_value: float) -> Dict[str, Any]:
        """Create action dict with proper values per action type."""
        continuous = [0.0] * self.config.n_continuous
        for i in self.config.temp_indices:
            continuous[i] = temp_value
        for i in self.config.pressure_indices:
            continuous[i] = pressure_value
        for i in self.config.water_indices:
            continuous[i] = water_value

        return {
            "discrete_action": [discrete_value] * self.config.n_discrete,
            "continuous_action": continuous,
        }

    def _get_zone_temps(self, obs: np.ndarray) -> np.ndarray:
        """Extract zone temperatures from observation."""
        return obs[self.config.zone_temp_indices]


class AllOffPolicy(Policy):
    """Policy that turns off all HVAC equipment."""

    @property
    def name(self) -> str:
        return "all_off"

    def get_action(self, obs: np.ndarray) -> Dict[str, Any]:
        return self._make_action(discrete_value=0, temp_value=-1,
                                 pressure_value=-1, water_value=-1)


class AllOnPolicy(Policy):
    """Policy that turns on all HVAC equipment at max cooling."""

    @property
    def name(self) -> str:
        return "all_on"

    def get_action(self, obs: np.ndarray) -> Dict[str, Any]:
        return self._make_action(discrete_value=1, temp_value=-1,
                                 pressure_value=1, water_value=1)


class BangBangPolicy(Policy):
    """Bang-bang controller that heats when cold, cools when hot.

    Uses fixed setpoints (can be overridden in constructor).
    """

    def __init__(self, config: PolicyConfig, heating_setpoint: float = 293.0,
                 cooling_setpoint: float = 297.0):
        """Initialize bang-bang policy.

        Args:
            config: PolicyConfig with action/observation specs.
            heating_setpoint: Temperature (K) below which to heat (default: 293K = 20°C).
            cooling_setpoint: Temperature (K) above which to cool (default: 297K = 24°C).
        """
        super().__init__(config)
        self.heating_setpoint = heating_setpoint
        self.cooling_setpoint = cooling_setpoint

    @property
    def name(self) -> str:
        return "bang_bang"

    def get_action(self, obs: np.ndarray) -> Dict[str, Any]:
        zone_temps = self._get_zone_temps(obs)
        avg_temp = np.mean(zone_temps)

        if avg_temp < self.heating_setpoint:
            # Too cold - heat
            return self._make_action(discrete_value=1, temp_value=1,
                                     pressure_value=1, water_value=1)
        elif avg_temp > self.cooling_setpoint:
            # Too hot - cool
            return self._make_action(discrete_value=1, temp_value=-1,
                                     pressure_value=1, water_value=1)
        else:
            # In comfort range - off
            return self._make_action(discrete_value=0, temp_value=-1,
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
    policy_class: type,
    scenarios: List[Tuple[str, str]],
    num_runs: int = 3,
    num_steps: Optional[int] = None,
    policy_kwargs: Optional[Dict[str, Any]] = None,
    verbose: bool = True,
) -> EvaluationResult:
    """Evaluate a policy on multiple scenarios.

    Args:
        policy_class: A Policy class (not instance) to evaluate.
        scenarios: List of (building_config_path, weather_name) tuples.
        num_runs: Number of times to run each scenario (default: 3).
        num_steps: Steps per episode. If None, defaults to 3 days (864 steps at 5min intervals).
        policy_kwargs: Additional kwargs to pass to policy constructor (after config).
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

    if policy_kwargs is None:
        policy_kwargs = {}

    all_returns = []  # List of lists: [scenario][run]
    policy_name = None

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

            # Create PolicyConfig from env and scenario result
            config = PolicyConfig.from_env(
                env,
                floor_plan=result.get("floor_plan"),
                zone_map=result.get("zone_map"),
                ahu_zone_assignments=result.get("ahu_zone_assignments"),
                num_floors=result.get("num_floors", 1),
            )

            # Create policy instance with config
            policy = policy_class(config, **policy_kwargs)
            policy_name = policy.name

            # Reset policy
            policy.reset()

            # Run episode
            total_reward = 0.0
            ts = env.reset()
            obs = ts.observation

            for step in range(num_steps):
                action = policy.get_action(obs)
                ts = env.step(action)
                obs = ts.observation
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
        policy_name=policy_name or "unknown",
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

    # Map policy names to classes
    policy_map = {
        "all_off": AllOffPolicy,
        "all_on": AllOnPolicy,
        "bang_bang": BangBangPolicy,
    }
    policy_class = policy_map[args.policy]

    # Create scenarios (all combinations of buildings x weathers)
    scenarios = [(b, w) for b in args.buildings for w in args.weathers]

    # Run evaluation
    result = evaluate(policy_class, scenarios, num_runs=args.runs, num_steps=args.steps)

    print(f"\nFinal Result: {result.average_return:.2f} +/- {result.std_return:.2f}")


if __name__ == "__main__":
    main()

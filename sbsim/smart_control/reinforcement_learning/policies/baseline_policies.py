"""Baseline policies for HVAC control benchmarking."""

from typing import Optional, List
import numpy as np
import tensorflow as tf
from tf_agents.policies import tf_policy
from tf_agents.trajectories import policy_step
from tf_agents.typing import types


class ConstantPolicy(tf_policy.TFPolicy):
    """Policy that always returns a constant action value."""

    def __init__(
        self,
        time_step_spec,
        action_spec: types.NestedTensorSpec,
        action_value: float = 0.0,
        name: Optional[str] = None,
    ):
        """Initialize constant policy.

        Args:
            time_step_spec: Spec for time steps.
            action_spec: Spec for actions.
            action_value: Constant value to use for all actions.
                          Actions are normalized to [-1, 1].
                          -1 = minimum (off), 1 = maximum (full on).
            name: Optional policy name.
        """
        self._action_value = action_value
        self._action_size = action_spec.shape[0]

        super().__init__(
            time_step_spec=time_step_spec,
            action_spec=action_spec,
            policy_state_spec=(),
            info_spec=(),
            clip=False,
            name=name,
        )

    def _action(self, time_step, policy_state, seed):
        del seed, policy_state

        # Create constant action array with batch dimension
        action_array = np.full(
            (1, self._action_size),
            self._action_value,
            dtype=np.float32
        )

        return policy_step.PolicyStep(
            tf.convert_to_tensor(action_array), (), ()
        )


class AllOffPolicy(ConstantPolicy):
    """Policy that keeps all HVAC systems off (minimum action)."""

    def __init__(
        self,
        time_step_spec,
        action_spec: types.NestedTensorSpec,
        name: Optional[str] = "all_off_policy",
    ):
        super().__init__(
            time_step_spec=time_step_spec,
            action_spec=action_spec,
            action_value=-1.0,  # Minimum = off
            name=name,
        )


class AllOnPolicy(ConstantPolicy):
    """Policy that keeps all HVAC systems at maximum."""

    def __init__(
        self,
        time_step_spec,
        action_spec: types.NestedTensorSpec,
        name: Optional[str] = "all_on_policy",
    ):
        super().__init__(
            time_step_spec=time_step_spec,
            action_spec=action_spec,
            action_value=1.0,  # Maximum = full on
            name=name,
        )


class BangBangPolicy(tf_policy.TFPolicy):
    """Simple bang-bang (on/off) thermostat controller.

    Compares zone temperatures against a setpoint and applies
    full heating/cooling when outside the deadband.
    """

    def __init__(
        self,
        time_step_spec,
        action_spec: types.NestedTensorSpec,
        field_names: List[str],
        heating_setpoint: float = 294.0,  # ~21C / 70F
        cooling_setpoint: float = 297.0,  # ~24C / 75F
        name: Optional[str] = "bang_bang_policy",
    ):
        """Initialize bang-bang policy.

        Args:
            time_step_spec: Spec for time steps.
            action_spec: Spec for actions.
            field_names: List of observation field names from env.field_names.
            heating_setpoint: Temperature (K) below which to heat.
            cooling_setpoint: Temperature (K) above which to cool.
            name: Optional policy name.
        """
        self._action_size = action_spec.shape[0]
        self._field_names = field_names
        self._heating_setpoint = heating_setpoint
        self._cooling_setpoint = cooling_setpoint

        # Find indices of zone temperature observations
        self._temp_indices = [
            i for i, name in enumerate(field_names)
            if 'zone_air_temperature' in name
        ]

        super().__init__(
            time_step_spec=time_step_spec,
            action_spec=action_spec,
            policy_state_spec=(),
            info_spec=(),
            clip=False,
            name=name,
        )

    def _action(self, time_step, policy_state, seed):
        del seed, policy_state

        observation = time_step.observation

        # Get average zone temperature
        if self._temp_indices:
            temps = [observation[0][i].numpy() for i in self._temp_indices]
            avg_temp = np.mean(temps)
        else:
            # No temperature obs found, default to neutral
            avg_temp = (self._heating_setpoint + self._cooling_setpoint) / 2

        # Bang-bang logic
        if avg_temp < self._heating_setpoint:
            # Too cold - full heating
            action_value = 1.0
        elif avg_temp > self._cooling_setpoint:
            # Too hot - full cooling (which is also max action for cooling setpoint)
            action_value = -1.0
        else:
            # In deadband - maintain current (neutral)
            action_value = 0.0

        action_array = np.full(
            (1, self._action_size),
            action_value,
            dtype=np.float32
        )

        return policy_step.PolicyStep(
            tf.convert_to_tensor(action_array), (), ()
        )


def create_all_off_policy(tf_env):
    """Create an all-off policy from a TF environment."""
    from tf_agents.train.utils import spec_utils
    _, action_spec, time_step_spec = spec_utils.get_tensor_specs(tf_env)
    return AllOffPolicy(time_step_spec, action_spec)


def create_all_on_policy(tf_env):
    """Create an all-on policy from a TF environment."""
    from tf_agents.train.utils import spec_utils
    _, action_spec, time_step_spec = spec_utils.get_tensor_specs(tf_env)
    return AllOnPolicy(time_step_spec, action_spec)


def create_bang_bang_policy(
    tf_env,
    heating_setpoint: float = 294.0,
    cooling_setpoint: float = 297.0,
):
    """Create a bang-bang policy from a TF environment.

    Args:
        tf_env: TFPyEnvironment instance.
        heating_setpoint: Temperature (K) below which to heat. Default 294K (~21C).
        cooling_setpoint: Temperature (K) above which to cool. Default 297K (~24C).
    """
    from tf_agents.train.utils import spec_utils
    _, action_spec, time_step_spec = spec_utils.get_tensor_specs(tf_env)

    env = tf_env.pyenv.envs[0]
    field_names = env.field_names

    return BangBangPolicy(
        time_step_spec=time_step_spec,
        action_spec=action_spec,
        field_names=field_names,
        heating_setpoint=heating_setpoint,
        cooling_setpoint=cooling_setpoint,
    )

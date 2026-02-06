"""Blend-toward-mean convection simulator.

Simulates air mixing within rooms by blending each CV's temperature toward
the room-average temperature. This is O(n) per room, preserves spatial
gradients (wall-adjacent CVs stay relatively warmer/cooler), and avoids
the boundary-layer destruction caused by random shuffling.

  T_new[cv] = (1 - alpha) * T[cv] + alpha * T_room_avg
"""

from typing import MutableSequence, Optional

import gin
import numpy as np

from smart_buildings.smart_control.simulator import base_convection_simulator


@gin.configurable
class BlendConvectionSimulator(
    base_convection_simulator.BaseConvectionSimulator
):
  """Simulates convection by blending toward the room mean temperature.

  Attributes:
    _alpha: blending strength in [0, 1]. 0 = no mixing, 1 = perfect mixing.
  """

  def __init__(self, alpha: float = 0.5):
    """Initializes the blend convection simulator.

    Args:
      alpha: blending strength. 0 = no mixing, 1 = instant full mixing.
    """
    if not 0.0 <= alpha <= 1.0:
      raise ValueError(f"alpha must be in [0, 1], got {alpha}")
    self._alpha = alpha

  def apply_convection(
      self,
      room_dict: dict[str, MutableSequence[tuple[int, int]]],
      temp: np.ndarray,
  ) -> None:
    """Applies convection by blending each room's CVs toward the room mean.

    Args:
      room_dict: A dictionary mapping of room coordinates.
      temp: An array of temperatures (modified in place).
    """
    if self._alpha == 0.0:
      return

    for room_name, cvs in room_dict.items():
      if room_name == "exterior_space":
        continue
      if room_name == "interior_wall":
        continue
      if not cvs:
        continue

      # Compute room mean temperature — one pass.
      total = 0.0
      for i, j in cvs:
        total += temp[i, j]
      room_avg = total / len(cvs)

      # Blend each CV toward the mean — second pass.
      keep = 1.0 - self._alpha
      blend = self._alpha * room_avg
      for i, j in cvs:
        temp[i, j] = keep * temp[i, j] + blend

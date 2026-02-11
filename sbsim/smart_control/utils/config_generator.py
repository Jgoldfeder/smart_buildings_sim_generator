"""Config generator for creating varied simulation scenarios."""

import os
import random
from dataclasses import dataclass, field, asdict
from typing import List, Optional, Tuple
import yaml


BUILDING_SHAPES = [
    "rectangle", "trapezoid", "h_shape", "t_shape", "pentagon",
    "oval", "u_shape", "parallelogram", "semicircle", "triangle"
]

# Climate presets: (low_temp_K, high_temp_K, name)
CLIMATES = {
    "hot_summer": (295.0, 313.0),      # 22C-40C
    "mild_summer": (288.0, 303.0),     # 15C-30C
    "cold_winter": (263.0, 278.0),     # -10C-5C
    "mild_winter": (273.0, 288.0),     # 0C-15C
    "tropical": (297.0, 308.0),        # 24C-35C
    "temperate": (283.0, 298.0),       # 10C-25C
}


@dataclass
class ConfigRange:
    """Defines a range for random sampling."""
    min_val: float
    max_val: float

    def sample(self) -> float:
        return random.uniform(self.min_val, self.max_val)

    def sample_int(self) -> int:
        return random.randint(int(self.min_val), int(self.max_val))


@dataclass
class DatasetConfig:
    """Configuration for generating a dataset of scenarios."""
    # Output
    output_dir: str = "dataset"
    num_configs: int = 100

    # Building variations
    shapes: List[str] = field(default_factory=lambda: BUILDING_SHAPES)
    width_range: Tuple[int, int] = (200, 800)
    height_range: Tuple[int, int] = (150, 500)
    num_floors_range: Tuple[int, int] = (1, 5)
    min_room_size_range: Tuple[int, int] = (30, 60)
    shape_ratio_range: Tuple[float, float] = (0.2, 0.45)  # For H/T/U shapes
    composite_coverage_range: Tuple[float, float] = (0.4, 0.8)  # For composite

    # AHU variations
    num_ahus_range: Tuple[int, int] = (1, 6)

    # Climate variations
    climates: List[str] = field(default_factory=lambda: list(CLIMATES.keys()))

    # Occupancy variations
    occupants_per_zone_range: Tuple[int, int] = (1, 10)

    # Simulation
    num_days_range: Tuple[int, int] = (7, 30)

    # Seeds
    base_seed: int = 0


def _sample(val, rng):
    """Sample from value - if tuple, treat as range; if list, choose random; else return as-is."""
    if isinstance(val, tuple) and len(val) == 2:
        if isinstance(val[0], int) and isinstance(val[1], int):
            return rng.randint(val[0], val[1])
        return rng.uniform(val[0], val[1])
    elif isinstance(val, list):
        return rng.choice(val)
    return val


def generate_config(
    name: str,
    seed: int,
    # Building - can be value, tuple (min, max), or list [choices]
    building_shape = "rectangle",
    width = 400,
    height = 300,
    num_floors = 2,
    min_room_size = 40,
    # Shape parameters (for h_shape, t_shape, u_shape, composite)
    shape_ratio = 0.33,          # Controls thick vs thin parts (0.1-0.5)
    composite_coverage = 0.6,    # For composite shape: min canvas coverage (0.0-1.0)
    # AHU
    num_ahus = 2,
    # Weather
    climate = "mild_summer",
    low_temp: Optional[float] = None,
    high_temp: Optional[float] = None,
    # Occupancy
    occupants_per_zone = 1,
    # Simulation
    num_days_in_episode = 21,
    start_timestamp: str = "2023-07-10 06:00",
    output_base_dir: str = "scenarios",
) -> dict:
    """Generate a single config dict.

    All numeric params can be:
        - A single value: width=400
        - A tuple range: width=(300, 600) -> samples random int
        - A list of choices: building_shape=["rectangle", "h_shape"] -> picks one

    Shape parameters:
        - shape_ratio: For H/T/U shapes, controls thick vs thin parts (0.1-0.5)
        - composite_coverage: For composite shape, minimum canvas coverage (0.0-1.0)
    """
    rng = random.Random(seed)

    # Sample all values
    building_shape = _sample(building_shape, rng)
    width = _sample(width, rng)
    height = _sample(height, rng)
    num_floors = _sample(num_floors, rng)
    min_room_size = _sample(min_room_size, rng)
    shape_ratio = _sample(shape_ratio, rng)
    composite_coverage = _sample(composite_coverage, rng)
    num_ahus = _sample(num_ahus, rng)
    climate = _sample(climate, rng)
    occupants_per_zone = _sample(occupants_per_zone, rng)
    num_days_in_episode = _sample(num_days_in_episode, rng)

    # Get climate temps if not specified
    if low_temp is None or high_temp is None:
        low_temp, high_temp = CLIMATES.get(climate, CLIMATES["mild_summer"])

    config = {
        "name": name,
        "seed": seed,
        "output_base_dir": output_base_dir,

        "floor_plan": {
            "width": width,
            "height": height,
            "num_floors": num_floors,
            "min_room_size": min_room_size,
            "max_room_size": min_room_size + 20,
            "building_shape": building_shape,
            "shape_ratio": shape_ratio,
            "composite_coverage": composite_coverage,
        },

        "ahu": {
            "num_ahus": num_ahus,
        },

        "weather": {
            "high_temp": high_temp,
            "low_temp": low_temp,
        },

        "simulation": {
            "start_timestamp": start_timestamp,
            "num_days_in_episode": num_days_in_episode,
            "occupants_per_zone": occupants_per_zone,
        },
    }

    return config


def generate_random_config(
    index: int,
    dataset_config: DatasetConfig,
) -> dict:
    """Generate a random config based on dataset config ranges."""

    seed = dataset_config.base_seed + index
    random.seed(seed)

    # Sample random values
    shape = random.choice(dataset_config.shapes)
    width = random.randint(*dataset_config.width_range)
    height = random.randint(*dataset_config.height_range)
    num_floors = random.randint(*dataset_config.num_floors_range)
    min_room_size = random.randint(*dataset_config.min_room_size_range)
    shape_ratio = random.uniform(*dataset_config.shape_ratio_range)
    composite_coverage = random.uniform(*dataset_config.composite_coverage_range)
    num_ahus = random.randint(*dataset_config.num_ahus_range)
    climate = random.choice(dataset_config.climates)
    occupants = random.randint(*dataset_config.occupants_per_zone_range)
    num_days = random.randint(*dataset_config.num_days_range)

    name = f"env_{index:04d}_{shape}_{climate}"

    return generate_config(
        name=name,
        seed=seed,
        building_shape=shape,
        width=width,
        height=height,
        num_floors=num_floors,
        min_room_size=min_room_size,
        shape_ratio=shape_ratio,
        composite_coverage=composite_coverage,
        num_ahus=num_ahus,
        climate=climate,
        occupants_per_zone=occupants,
        num_days_in_episode=num_days,
        output_base_dir=dataset_config.output_dir,
    )


def generate_dataset_configs(
    dataset_config: Optional[DatasetConfig] = None,
    **kwargs
) -> List[dict]:
    """Generate a list of varied configs for dataset creation.

    Args:
        dataset_config: DatasetConfig instance, or None to use defaults
        **kwargs: Override any DatasetConfig fields

    Returns:
        List of config dicts
    """
    if dataset_config is None:
        dataset_config = DatasetConfig(**kwargs)

    configs = []
    for i in range(dataset_config.num_configs):
        config = generate_random_config(i, dataset_config)
        configs.append(config)

    return configs


def save_configs(
    configs: List[dict],
    output_dir: str = "dataset/configs",
) -> List[str]:
    """Save configs to YAML files.

    Returns:
        List of saved file paths
    """
    os.makedirs(output_dir, exist_ok=True)

    paths = []
    for config in configs:
        name = config["name"]
        path = os.path.join(output_dir, f"{name}.yaml")
        with open(path, "w") as f:
            yaml.dump(config, f, default_flow_style=False, sort_keys=False)
        paths.append(path)

    return paths


def generate_and_save_dataset(
    output_dir: str = "dataset",
    num_configs: int = 100,
    **kwargs
) -> List[str]:
    """Generate and save a dataset of config files.

    Args:
        output_dir: Directory to save configs
        num_configs: Number of configs to generate
        **kwargs: Additional DatasetConfig parameters

    Returns:
        List of saved config file paths
    """
    dataset_config = DatasetConfig(
        output_dir=output_dir,
        num_configs=num_configs,
        **kwargs
    )

    configs = generate_dataset_configs(dataset_config)
    config_dir = os.path.join(output_dir, "configs")
    paths = save_configs(configs, config_dir)

    print(f"Generated {len(paths)} configs in {config_dir}")
    return paths


# Convenience function for quick generation
def quick_dataset(n: int = 10, output_dir: str = "dataset") -> List[str]:
    """Quick way to generate a small dataset.

    Example:
        paths = quick_dataset(10)
        for path in paths:
            env = get_env_from_config(path)
    """
    return generate_and_save_dataset(output_dir=output_dir, num_configs=n)


def visualize_config(config: dict):
    """Generate scenario from config dict and visualize AHU zones.

    Args:
        config: Config dict from generate_config()
    """
    from sbsim.smart_control.utils.scenario_generator import generate_scenario_from_dict
    from sbsim.smart_control.utils.floor_generator import visualize_ahu_zones

    # Generate scenario from config dict
    result = generate_scenario_from_dict(config)

    print(f"Generated {result['num_rooms']} rooms with {result['num_ahus']} AHUs")
    print(f"Shape: {config['floor_plan']['building_shape']}")

    visualize_ahu_zones(result['bsp'], result['ahu_assignments'],
                        title=f"{config['name']} - AHU Zones")

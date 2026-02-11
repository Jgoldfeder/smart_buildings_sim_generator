"""Scenario generator for HVAC simulations.

Generates complete simulation scenarios from YAML configuration files.
Each scenario includes floor plans, zone maps, and gin configs in a named directory.

Example usage:
    from smart_buildings.smart_control.utils.scenario_generator import (
        generate_scenario, get_env, print_action_spec, SimulationTracker
    )

    # Generate scenario and create env
    result = generate_scenario("path/to/config.yaml")
    env = get_env(result)

    # See action spec
    action = print_action_spec(env)

    # Run simulation with tracking
    tracker = SimulationTracker(env)

    for i in range(1000):
        tracker.step(action)

        if i % 100 == 0:
            tracker.render()      # Show building visualization
            tracker.plot()        # Show all charts

    # Render a specific past timestep
    tracker.render(step=500)

    # Get summary
    tracker.summary()
"""

import os
import random
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Any

import numpy as np
import yaml

from smart_buildings.smart_control.utils.floor_generator import (
    BSPFloorPlan,
    MultiFloorPlan,
    assign_rooms_to_ahus,
    OBSERVATION_NORMALIZER_CONFIG,
)

# Lock to prevent race conditions when multiple threads generate floor plans
# with seeded random number generators. Without this lock, concurrent threads
# can interleave their random calls, causing different floor plans to be
# generated even with the same seed.
_floor_plan_generation_lock = threading.Lock()


@dataclass
class FloorPlanConfig:
    """Floor plan generation parameters."""
    width: int = 100
    height: int = 100
    num_floors: int = 1
    min_room_size: int = 20
    max_room_size: int = 40
    wall_thickness: int = 1
    door_width: int = 2
    split_variance: float = 0.3
    building_shape: str = "rectangle"
    shape_ratio: float = 0.33
    composite_coverage: float = 0.5


@dataclass
class AHUConfig:
    """AHU (Air Handling Unit) configuration."""
    num_ahus: int = 2
    max_air_flow_rate: float = 8.67
    recirculation_ratio: float = 0.3
    heating_setpoint: float = 285.0
    cooling_setpoint: float = 298.0
    fan_static_pressure: float = 10000.0
    fan_efficiency: float = 0.9


@dataclass
class WeatherConfig:
    """Weather/ambient temperature configuration."""
    high_temp: float = 305.0  # Kelvin
    low_temp: float = 280.0   # Kelvin
    convection_coefficient: float = 100.0


@dataclass
class HotWaterSystemConfig:
    """Hot water system configuration."""
    pump_differential_head: float = 6.0
    pump_efficiency: float = 0.98
    reheat_water_setpoint: float = 360.0
    boiler_heating_rate: float = 0.5
    boiler_cooling_rate: float = 0.1


@dataclass
class SetpointScheduleConfig:
    """Temperature setpoint schedule configuration."""
    morning_start_hour: int = 6
    evening_start_hour: int = 19
    heating_setpoint_day: float = 294.0
    cooling_setpoint_day: float = 297.0
    heating_setpoint_night: float = 289.0
    cooling_setpoint_night: float = 298.0
    time_zone: str = "US/Pacific"


@dataclass
class RewardConfig:
    """Reward function configuration."""
    max_productivity_personhour_usd: float = 300.0
    min_productivity_personhour_usd: float = 100.0
    productivity_midpoint_delta: float = 0.5
    productivity_decay_stiffness: float = 4.3
    max_electricity_rate: float = 160000.0
    max_natural_gas_rate: float = 400000.0
    productivity_weight: float = 0.6
    energy_cost_weight: float = 0.2
    carbon_emission_weight: float = 0.2


@dataclass
class SimulationConfig:
    """Simulation timing configuration."""
    time_step_sec: int = 300
    convergence_threshold: float = 0.01
    iteration_limit: int = 100
    iteration_warning: int = 20
    start_timestamp: str = "2023-07-10 06:00"  # Monday
    num_days_in_episode: int = 21
    discount_factor: float = 0.9
    occupants_per_zone: int = 1  # Number of occupants per zone


@dataclass
class ScenarioConfig:
    """Complete scenario configuration."""
    name: str = "default_scenario"
    seed: Optional[int] = None
    output_base_dir: str = "scenarios"

    floor_plan: FloorPlanConfig = field(default_factory=FloorPlanConfig)
    ahu: AHUConfig = field(default_factory=AHUConfig)
    weather: WeatherConfig = field(default_factory=WeatherConfig)
    hot_water_system: HotWaterSystemConfig = field(default_factory=HotWaterSystemConfig)
    setpoint_schedule: SetpointScheduleConfig = field(default_factory=SetpointScheduleConfig)
    reward: RewardConfig = field(default_factory=RewardConfig)
    simulation: SimulationConfig = field(default_factory=SimulationConfig)


def load_config(config_path: str) -> ScenarioConfig:
    """Load scenario configuration from a YAML file.

    Args:
        config_path: Path to the YAML configuration file.

    Returns:
        ScenarioConfig object with all parameters.
    """
    with open(config_path, 'r') as f:
        raw = yaml.safe_load(f)

    config = ScenarioConfig()

    # Top-level fields
    if 'name' in raw:
        config.name = raw['name']
    if 'seed' in raw:
        config.seed = raw['seed']
    if 'output_base_dir' in raw:
        config.output_base_dir = raw['output_base_dir']

    # Nested configs
    if 'floor_plan' in raw:
        config.floor_plan = FloorPlanConfig(**raw['floor_plan'])
    if 'ahu' in raw:
        config.ahu = AHUConfig(**raw['ahu'])
    if 'weather' in raw:
        config.weather = WeatherConfig(**raw['weather'])
    if 'hot_water_system' in raw:
        config.hot_water_system = HotWaterSystemConfig(**raw['hot_water_system'])
    if 'setpoint_schedule' in raw:
        config.setpoint_schedule = SetpointScheduleConfig(**raw['setpoint_schedule'])
    if 'reward' in raw:
        config.reward = RewardConfig(**raw['reward'])
    if 'simulation' in raw:
        config.simulation = SimulationConfig(**raw['simulation'])

    return config


# Weather preset directory (relative to package root)
WEATHER_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))), "weather")


def load_weather_config(weather_path_or_name: str) -> WeatherConfig:
    """Load weather configuration from a file or preset name.

    Args:
        weather_path_or_name: Either a path to a YAML file, or a preset name
            (e.g., "hot_summer", "cold_winter").

    Returns:
        WeatherConfig object.
    """
    # Check if it's a preset name
    if not weather_path_or_name.endswith('.yaml') and not os.path.exists(weather_path_or_name):
        # Try as preset name
        preset_path = os.path.join(WEATHER_DIR, f"{weather_path_or_name}.yaml")
        if os.path.exists(preset_path):
            weather_path_or_name = preset_path
        else:
            raise ValueError(f"Weather preset '{weather_path_or_name}' not found. "
                           f"Available presets: {list_weather_presets()}")

    with open(weather_path_or_name, 'r') as f:
        raw = yaml.safe_load(f)

    return WeatherConfig(
        high_temp=raw.get('high_temp', 305.0),
        low_temp=raw.get('low_temp', 280.0),
        convection_coefficient=raw.get('convection_coefficient', 100.0),
    )


def list_weather_presets() -> List[str]:
    """List available weather presets."""
    if not os.path.exists(WEATHER_DIR):
        return []
    return [f.replace('.yaml', '') for f in os.listdir(WEATHER_DIR) if f.endswith('.yaml')]


def load_building_config(building_path: str) -> ScenarioConfig:
    """Load building configuration (without weather) from a YAML file.

    Args:
        building_path: Path to the building YAML configuration file.

    Returns:
        ScenarioConfig object (weather will use defaults).
    """
    return load_config(building_path)


def load_scenario_from_parts(
    building_path: str,
    weather_path_or_name: str,
    output_base_dir: Optional[str] = None,
    policy: Optional[str] = None,
) -> ScenarioConfig:
    """Load a scenario from separate building and weather configs.

    Args:
        building_path: Path to the building YAML configuration file.
        weather_path_or_name: Path to weather YAML or preset name (e.g., "hot_summer").
        output_base_dir: Override for output directory (optional).
        policy: Policy name to include in output path (optional, prevents race conditions).

    Returns:
        ScenarioConfig with building + weather merged.
    """
    # Load building config
    config = load_building_config(building_path)

    # Load and apply weather config
    weather = load_weather_config(weather_path_or_name)
    config.weather = weather

    # Extract weather name for unique output directory
    weather_name = weather_path_or_name
    if weather_path_or_name.endswith('.yaml'):
        weather_name = os.path.basename(weather_path_or_name).replace('.yaml', '')
    elif os.path.sep in weather_path_or_name:
        weather_name = os.path.basename(weather_path_or_name)

    # Include weather (and policy if provided) in output path to avoid race conditions
    if policy:
        config.name = f"{config.name}/{weather_name}/{policy}"
    else:
        config.name = f"{config.name}/{weather_name}"

    # Override output dir if specified
    if output_base_dir:
        config.output_base_dir = output_base_dir

    return config


def _generate_gin_config(
    floor_plan_path: str,
    zone_map_path: str,
    ahu_assignments: List[List[str]],
    config: ScenarioConfig,
) -> str:
    """Generate a complete gin configuration file.

    Args:
        floor_plan_path: Path to the floor plan .npy file.
        zone_map_path: Path to the zone map .npy file.
        ahu_assignments: List of room assignments per AHU.
        config: Full scenario configuration.

    Returns:
        Gin configuration as a string.
    """
    num_ahus = len(ahu_assignments)
    fp = config.floor_plan
    ahu = config.ahu
    weather = config.weather
    hws = config.hot_water_system
    schedule = config.setpoint_schedule
    reward = config.reward
    sim = config.simulation

    # Room lists for each AHU
    room_lists = []
    for i, rooms in enumerate(ahu_assignments, 1):
        room_strs = ", ".join([f"'{r}'" for r in rooms])
        room_lists.append(f"ahu_{i}_rooms = [{room_strs}]")
    room_lists_str = "\n".join(room_lists)

    # AHU definitions
    ahu_defs = []
    for i in range(1, num_ahus + 1):
        ahu_defs.append(f"""ahu_{i}/AirHandler:
  recirculation = %air_handler_recirculation_ratio
  heating_air_temp_setpoint = %air_handler_heating_setpoint
  cooling_air_temp_setpoint = %air_handler_cooling_setpoint
  fan_static_pressure = %fan_static_pressure
  fan_efficiency = %fan_efficiency
  max_air_flow_rate = {ahu.max_air_flow_rate}
  sim_weather_controller = %weather_controller
  device_id = 'ahu_{i}'""")
    ahu_defs_str = "\n\n".join(ahu_defs)

    # AHU mappings
    ahu_mappings = ",\n    ".join(
        [f"@ahu_{i}/AirHandler(): %ahu_{i}_rooms" for i in range(1, num_ahus + 1)]
    )

    # Default actions
    ahu_default_actions = []
    for i in range(1, num_ahus + 1):
        ahu_default_actions.extend([
            f"'ahu_ahu_{i}_supply_air_temperature_setpoint': 293.0",
            f"'ahu_ahu_{i}_static_pressure_setpoint': 20000.0",
            f"'ahu_ahu_{i}_supervisor_run_command': 1.0"
        ])
    ahu_default_actions_str = ",\n        ".join(ahu_default_actions)

    # Action normalizers
    ahu_normalizers = []
    for i in range(1, num_ahus + 1):
        ahu_normalizers.extend([
            f"'ahu_{i}_supply_air_temperature_setpoint': @supply_air_temperature_setpoint/set_action_normalization_constants()",
            f"'ahu_{i}_static_pressure_setpoint': @static_pressure_setpoint/set_action_normalization_constants()",
            f"'ahu_{i}_supervisor_run_command': @run_command/set_action_normalization_constants()"
        ])
    ahu_normalizers_str = ",\n        ".join(ahu_normalizers)

    # Device action tuples
    device_action_tuples = []
    for i in range(1, num_ahus + 1):
        device_action_tuples.extend([
            f"('ahu', 'ahu_{i}_supervisor_run_command')",
            f"('ahu', 'ahu_{i}_supply_air_temperature_setpoint')",
            f"('ahu', 'ahu_{i}_static_pressure_setpoint')"
        ])
    device_action_tuples.extend([
        "('hws', 'supervisor_run_command')",
        "('hws', 'supply_water_setpoint')",
        "('hws', 'differential_pressure')"
    ])
    device_action_tuples_str = ",\n        ".join(device_action_tuples)

    gin_config = f'''###############################################################
# Scenario: {config.name}
# Generated by scenario_generator.py
###############################################################

##########################
### WEATHER CONTROLLER
##########################

convection_coefficient = {weather.convection_coefficient}
ambient_high_temp = {weather.high_temp}
ambient_low_temp = {weather.low_temp}

sim/WeatherController:
  default_low_temp = %ambient_low_temp
  default_high_temp = %ambient_high_temp
  convection_coefficient = %convection_coefficient

weather_controller = @sim/WeatherController()

##########################
### BUILDING
##########################

initial_temp = 294.0
control_volume_cm = 10
floor_height_cm = 300.0

floor_plan_filepath = "{floor_plan_path}"
zone_map_filepath = "{zone_map_path}"

exterior_cv_conductivity = 5.5
exterior_cv_density = 1.0
exterior_cv_heat_capacity = 700.0
interior_wall_cv_conductivity = 50.0
interior_wall_cv_density = 1.0
interior_wall_cv_heat_capacity = 700.0
interior_cv_conductivity = 50.0
interior_cv_density = 0.1
interior_cv_heat_capacity = 700

inside_air_properties/MaterialProperties:
  conductivity = %interior_cv_conductivity
  heat_capacity = %interior_cv_heat_capacity
  density = %interior_cv_density

inside_wall_properties/MaterialProperties:
  conductivity = %interior_wall_cv_conductivity
  heat_capacity = %interior_wall_cv_heat_capacity
  density = %interior_wall_cv_density

building_exterior_properties/MaterialProperties:
  conductivity = %exterior_cv_conductivity
  heat_capacity = %exterior_cv_heat_capacity
  density = %exterior_cv_density

sim/BlendConvectionSimulator:
  alpha = 0.1

sim/FloorPlanBasedBuilding:
  cv_size_cm = %control_volume_cm
  floor_height_cm = %floor_height_cm
  initial_temp = %initial_temp
  inside_air_properties = @inside_air_properties/MaterialProperties()
  inside_wall_properties = @inside_wall_properties/MaterialProperties()
  building_exterior_properties = @building_exterior_properties/MaterialProperties()
  floor_plan_filepath = %floor_plan_filepath
  zone_map_filepath = %zone_map_filepath
  convection_simulator = @sim/BlendConvectionSimulator()

##########################
### SETPOINT SCHEDULE
##########################

morning_start_hour = {schedule.morning_start_hour}
evening_start_hour = {schedule.evening_start_hour}
heating_setpoint_day = {schedule.heating_setpoint_day}
cooling_setpoint_day = {schedule.cooling_setpoint_day}
heating_setpoint_night = {schedule.heating_setpoint_night}
cooling_setpoint_night = {schedule.cooling_setpoint_night}
time_zone = "{schedule.time_zone}"

hvac/SetpointSchedule:
  morning_start_hour = %morning_start_hour
  evening_start_hour = %evening_start_hour
  comfort_temp_window = (%heating_setpoint_day, %cooling_setpoint_day)
  eco_temp_window = (%heating_setpoint_night, %cooling_setpoint_night)
  time_zone = %time_zone

##########################
### HVAC SYSTEM
##########################

water_pump_differential_head = {hws.pump_differential_head}
water_pump_efficiency = {hws.pump_efficiency}
reheat_water_setpoint = {hws.reheat_water_setpoint}
boiler_heating_rate = {hws.boiler_heating_rate}
boiler_cooling_rate = {hws.boiler_cooling_rate}
fan_static_pressure = {ahu.fan_static_pressure}
fan_efficiency = {ahu.fan_efficiency}
air_handler_heating_setpoint = {ahu.heating_setpoint}
air_handler_cooling_setpoint = {ahu.cooling_setpoint}
air_handler_recirculation_ratio = {ahu.recirculation_ratio}
vav_max_air_flowrate = 2.0
vav_reheat_water_flowrate = 0.03

{room_lists_str}

{ahu_defs_str}

hvac/AirHandlerSystem:
  ahus = {{
    {ahu_mappings}
  }}
  device_id = 'ahu'

hvac/WaterPump:
  water_pump_differential_head = %water_pump_differential_head
  water_pump_efficiency = %water_pump_efficiency

hvac/Boiler:
  reheat_water_setpoint = %reheat_water_setpoint
  heating_rate = %boiler_heating_rate
  cooling_rate = %boiler_cooling_rate

hvac/HotWaterSystem:
  pump = @hvac/WaterPump()
  boiler = @hvac/Boiler()
  device_id = 'hws'

sim/FloorPlanBasedHvac:
  air_handler = @hvac/AirHandlerSystem()
  hot_water_system = @hvac/HotWaterSystem()
  schedule = @hvac/SetpointSchedule()
  vav_max_air_flow_rate = %vav_max_air_flowrate
  vav_reheat_max_water_flow_factor = %vav_reheat_water_flowrate

##########################
### SIMULATOR
##########################

time_step_sec = {sim.time_step_sec}
convergence_threshold = {sim.convergence_threshold}
iteration_limit = {sim.iteration_limit}
iteration_warning = {sim.iteration_warning}
start_timestamp = '{sim.start_timestamp}'

sim/to_timestamp.date_str = %start_timestamp

sim_building/TFSimulator:
  building = @sim/FloorPlanBasedBuilding()
  hvac = @sim/FloorPlanBasedHvac()
  weather_controller = %weather_controller
  time_step_sec = %time_step_sec
  convergence_threshold = %convergence_threshold
  iteration_limit = %iteration_limit
  iteration_warning = %iteration_warning
  start_timestamp = @sim/to_timestamp()

##########################
### OCCUPANCY
##########################

work_occupancy = {sim.occupants_per_zone}
nonwork_occupancy = 0.1

randomized_occupancy/RandomizedArrivalDepartureOccupancy:
  zone_assignment = %work_occupancy
  earliest_expected_arrival_hour = 3
  latest_expected_arrival_hour = 12
  earliest_expected_departure_hour = 13
  latest_expected_departure_hour = 23
  time_step_sec = %time_step_sec
  min_occupancy = 0.1

SimulatorBuilding.simulator = @sim_building/TFSimulator()
SimulatorBuilding.occupancy = @randomized_occupancy/RandomizedArrivalDepartureOccupancy()

##########################
### REWARD FUNCTION
##########################

max_productivity_personhour_usd = {reward.max_productivity_personhour_usd}
min_productivity_personhour_usd = {reward.min_productivity_personhour_usd}
productivity_midpoint_delta = {reward.productivity_midpoint_delta}
productivity_decay_stiffness = {reward.productivity_decay_stiffness}
max_electricity_rate = {reward.max_electricity_rate}
max_natural_gas_rate = {reward.max_natural_gas_rate}
productivity_weight = {reward.productivity_weight}
energy_cost_weight = {reward.energy_cost_weight}
carbon_emission_weight = {reward.carbon_emission_weight}

SetpointEnergyCarbonRegretFunction.max_productivity_personhour_usd = %max_productivity_personhour_usd
SetpointEnergyCarbonRegretFunction.min_productivity_personhour_usd = %min_productivity_personhour_usd
SetpointEnergyCarbonRegretFunction.max_electricity_rate = %max_electricity_rate
SetpointEnergyCarbonRegretFunction.max_natural_gas_rate = %max_natural_gas_rate
SetpointEnergyCarbonRegretFunction.productivity_decay_stiffness = %productivity_decay_stiffness
SetpointEnergyCarbonRegretFunction.productivity_midpoint_delta = %productivity_midpoint_delta
SetpointEnergyCarbonRegretFunction.electricity_energy_cost = @ElectricityEnergyCost()
SetpointEnergyCarbonRegretFunction.natural_gas_energy_cost = @NaturalGasEnergyCost()
SetpointEnergyCarbonRegretFunction.productivity_weight = %productivity_weight
SetpointEnergyCarbonRegretFunction.energy_cost_weight = %energy_cost_weight
SetpointEnergyCarbonRegretFunction.carbon_emission_weight = %carbon_emission_weight

##########################
### ACTION NORMALIZATION
##########################

supply_water_bounded_action_normalizer/set_action_normalization_constants.min_normalized_value = -1.
supply_water_bounded_action_normalizer/set_action_normalization_constants.max_normalized_value = 1.0
supply_water_bounded_action_normalizer/set_action_normalization_constants.min_native_value = 310
supply_water_bounded_action_normalizer/set_action_normalization_constants.max_native_value = 350.0

supply_air_temperature_setpoint/set_action_normalization_constants.min_normalized_value = -1.
supply_air_temperature_setpoint/set_action_normalization_constants.max_normalized_value = 1.
supply_air_temperature_setpoint/set_action_normalization_constants.min_native_value = 285
supply_air_temperature_setpoint/set_action_normalization_constants.max_native_value = 305.0

differential_pressure_setpoint/set_action_normalization_constants.min_normalized_value = -1.
differential_pressure_setpoint/set_action_normalization_constants.max_normalized_value = 1.
differential_pressure_setpoint/set_action_normalization_constants.min_native_value = 0
differential_pressure_setpoint/set_action_normalization_constants.max_native_value = 20.0

static_pressure_setpoint/set_action_normalization_constants.min_normalized_value = -1.
static_pressure_setpoint/set_action_normalization_constants.max_normalized_value = 1.
static_pressure_setpoint/set_action_normalization_constants.min_native_value = 0
static_pressure_setpoint/set_action_normalization_constants.max_native_value = 20000.0

run_command/set_action_normalization_constants.min_normalized_value = -1.
run_command/set_action_normalization_constants.max_normalized_value = 1.
run_command/set_action_normalization_constants.min_native_value = 0.0
run_command/set_action_normalization_constants.max_native_value = 1.0

action_normalizer_map = {{
        'supply_water_setpoint': @supply_water_bounded_action_normalizer/set_action_normalization_constants(),
        'differential_pressure': @differential_pressure_setpoint/set_action_normalization_constants(),
        {ahu_normalizers_str},
        'supervisor_run_command': @run_command/set_action_normalization_constants(),
    }}

ActionConfig:
    action_normalizers = %action_normalizer_map

default_actions = {{
        'hws_supply_water_setpoint': 340.0,
        'hws_differential_pressure': 20.0,
        {ahu_default_actions_str},
        'hws_supervisor_run_command': 1.0,
    }}

{OBSERVATION_NORMALIZER_CONFIG}

##########################
### ENVIRONMENT
##########################

discount_factor = {sim.discount_factor}
num_days_in_episode = {sim.num_days_in_episode}
metrics_reporting_interval = 10
label = '{config.name}'
num_hod_features = 12
num_dow_features = 12

Environment.building = @SimulatorBuilding()
Environment.reward_function = @SetpointEnergyCarbonRegretFunction()
Environment.observation_normalizer = @StandardScoreObservationNormalizer()
Environment.action_config = @ActionConfig()
Environment.metrics_reporting_interval = %metrics_reporting_interval
Environment.discount_factor = %discount_factor
Environment.label = %label
Environment.num_days_in_episode = %num_days_in_episode
Environment.default_actions = %default_actions
Environment.num_hod_features = %num_hod_features
Environment.num_dow_features = %num_dow_features

HybridActionEnvironment.building = @SimulatorBuilding()
HybridActionEnvironment.reward_function = @SetpointEnergyCarbonRegretFunction()
HybridActionEnvironment.observation_normalizer = @StandardScoreObservationNormalizer()
HybridActionEnvironment.action_config = @ActionConfig()
HybridActionEnvironment.metrics_reporting_interval = %metrics_reporting_interval
HybridActionEnvironment.discount_factor = %discount_factor
HybridActionEnvironment.label = %label
HybridActionEnvironment.num_days_in_episode = %num_days_in_episode
HybridActionEnvironment.default_actions = %default_actions
HybridActionEnvironment.num_hod_features = %num_hod_features
HybridActionEnvironment.num_dow_features = %num_dow_features
HybridActionEnvironment.device_action_tuples = [
        {device_action_tuples_str},
    ]
'''
    return gin_config


def generate_scenario(config_path: str) -> Dict[str, Any]:
    """Generate a complete simulation scenario from a YAML config file.

    Creates a directory with the scenario name containing:
    - floor_plan.npy: Floor plan for the simulator
    - zone_map.npy: Zone map for room detection
    - config.gin: Complete gin configuration
    - config.yaml: Copy of the input config

    Args:
        config_path: Path to the YAML configuration file.

    Returns:
        Dictionary with:
            - name: Scenario name
            - output_dir: Path to output directory
            - floor_plan_path: Path to floor plan file
            - zone_map_path: Path to zone map file
            - gin_config_path: Path to gin config file
            - num_rooms: Number of rooms generated
            - num_ahus: Number of AHUs
            - ahu_assignments: Room assignments per AHU
    """
    # Load configuration
    config = load_config(config_path)

    # Create output directory
    output_dir = os.path.join(config.output_base_dir, config.name)
    os.makedirs(output_dir, exist_ok=True)

    # Use lock to prevent race conditions with seeded random number generators.
    # Without this, concurrent threads can interleave random calls, causing
    # different floor plans to be generated even with the same seed.
    with _floor_plan_generation_lock:
        # Set random seed if specified (must be inside lock)
        if config.seed is not None:
            random.seed(config.seed)
            np.random.seed(config.seed)

        # Generate floor plan
        fp = config.floor_plan
        if fp.num_floors > 1:
            generator = MultiFloorPlan(
                width=fp.width,
                height=fp.height,
                num_floors=fp.num_floors,
                min_room_size=fp.min_room_size,
                max_room_size=fp.max_room_size,
                wall_thickness=fp.wall_thickness,
                door_width=fp.door_width,
                split_variance=fp.split_variance,
                building_shape=fp.building_shape,
                shape_ratio=fp.shape_ratio,
                composite_coverage=fp.composite_coverage,
            )
        else:
            generator = BSPFloorPlan(
                width=fp.width,
                height=fp.height,
                min_room_size=fp.min_room_size,
                max_room_size=fp.max_room_size,
                wall_thickness=fp.wall_thickness,
                door_width=fp.door_width,
                split_variance=fp.split_variance,
                building_shape=fp.building_shape,
                shape_ratio=fp.shape_ratio,
                composite_coverage=fp.composite_coverage,
            )

        generator.generate()

        # Save floor plan and zone map (still inside lock to prevent partial writes)
        floor_plan_path = os.path.join(output_dir, "floor_plan.npy")
        zone_map_path = os.path.join(output_dir, "zone_map.npy")
        generator.save_for_simulator(floor_plan_path, zone_map_path)

        # Get room count and assign to AHUs (uses seeded random, so keep in lock)
        num_rooms = generator.get_num_rooms()
        ahu_assignments = assign_rooms_to_ahus(
            num_rooms,
            config.ahu.num_ahus,
            seed=config.seed
        )

    # Generate and save gin config
    gin_config = _generate_gin_config(
        floor_plan_path,
        zone_map_path,
        ahu_assignments,
        config,
    )
    gin_config_path = os.path.join(output_dir, "config.gin")
    with open(gin_config_path, 'w') as f:
        f.write(gin_config)

    # Copy input config to output directory
    import shutil
    config_copy_path = os.path.join(output_dir, "config.yaml")
    shutil.copy2(config_path, config_copy_path)

    return {
        'name': config.name,
        'output_dir': output_dir,
        'floor_plan_path': floor_plan_path,
        'zone_map_path': zone_map_path,
        'gin_config_path': gin_config_path,
        'num_rooms': num_rooms,
        'num_ahus': config.ahu.num_ahus,
        'ahu_assignments': ahu_assignments,
        'bsp': generator,
        'config': config,
    }


def generate_scenario_from_config(config: ScenarioConfig) -> Dict[str, Any]:
    """Generate a complete simulation scenario from a ScenarioConfig object.

    Same as generate_scenario but accepts a ScenarioConfig object directly.
    Useful when building/weather configs are loaded separately.

    Args:
        config: ScenarioConfig object with all parameters.

    Returns:
        Same as generate_scenario().
    """
    # Create output directory
    output_dir = os.path.join(config.output_base_dir, config.name)
    os.makedirs(output_dir, exist_ok=True)

    # Use lock to prevent race conditions with seeded random number generators.
    # Without this, concurrent threads can interleave random calls, causing
    # different floor plans to be generated even with the same seed.
    with _floor_plan_generation_lock:
        # Set random seed if specified (must be inside lock)
        if config.seed is not None:
            random.seed(config.seed)
            np.random.seed(config.seed)

        # Generate floor plan
        fp = config.floor_plan
        if fp.num_floors > 1:
            generator = MultiFloorPlan(
                width=fp.width,
                height=fp.height,
                num_floors=fp.num_floors,
                min_room_size=fp.min_room_size,
                max_room_size=fp.max_room_size,
                wall_thickness=fp.wall_thickness,
                door_width=fp.door_width,
                split_variance=fp.split_variance,
                building_shape=fp.building_shape,
                shape_ratio=fp.shape_ratio,
                composite_coverage=fp.composite_coverage,
            )
        else:
            generator = BSPFloorPlan(
                width=fp.width,
                height=fp.height,
                min_room_size=fp.min_room_size,
                max_room_size=fp.max_room_size,
                wall_thickness=fp.wall_thickness,
                door_width=fp.door_width,
                split_variance=fp.split_variance,
                building_shape=fp.building_shape,
                shape_ratio=fp.shape_ratio,
                composite_coverage=fp.composite_coverage,
            )

        generator.generate()

        # Save floor plan and zone map (still inside lock to prevent partial writes)
        floor_plan_path = os.path.join(output_dir, "floor_plan.npy")
        zone_map_path = os.path.join(output_dir, "zone_map.npy")
        generator.save_for_simulator(floor_plan_path, zone_map_path)

        # Get room count and assign to AHUs (uses seeded random, so keep in lock)
        num_rooms = generator.get_num_rooms()
        ahu_assignments = assign_rooms_to_ahus(
            num_rooms,
            config.ahu.num_ahus,
            seed=config.seed
        )

    # Generate and save gin config
    gin_config = _generate_gin_config(
        floor_plan_path,
        zone_map_path,
        ahu_assignments,
        config,
    )
    gin_config_path = os.path.join(output_dir, "config.gin")
    with open(gin_config_path, 'w') as f:
        f.write(gin_config)

    # Save merged config to output directory
    import dataclasses
    config_copy_path = os.path.join(output_dir, "config.yaml")
    with open(config_copy_path, 'w') as f:
        yaml.dump(dataclasses.asdict(config), f, default_flow_style=False)

    return {
        'name': config.name,
        'output_dir': output_dir,
        'floor_plan_path': floor_plan_path,
        'zone_map_path': zone_map_path,
        'gin_config_path': gin_config_path,
        'num_rooms': num_rooms,
        'num_ahus': config.ahu.num_ahus,
        'ahu_assignments': ahu_assignments,
        'bsp': generator,
        'config': config,
    }


def generate_scenario_from_dict(config_dict: Dict[str, Any]) -> Dict[str, Any]:
    """Generate a scenario from a configuration dictionary.

    Same as generate_scenario but accepts a dict instead of a file path.
    Useful for programmatic scenario generation.

    Args:
        config_dict: Configuration dictionary matching YAML structure.

    Returns:
        Same as generate_scenario().
    """
    import tempfile

    # Write dict to temp file and use generate_scenario
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
        yaml.dump(config_dict, f)
        temp_path = f.name

    try:
        result = generate_scenario(temp_path)
    finally:
        os.unlink(temp_path)

    return result


def print_scenario_summary(result: Dict[str, Any]) -> None:
    """Print a summary of a generated scenario."""
    print(f"\n{'='*60}")
    print(f"Scenario: {result['name']}")
    print(f"{'='*60}")
    print(f"Output directory: {result['output_dir']}")
    print(f"Number of rooms: {result['num_rooms']}")
    print(f"Number of AHUs: {result['num_ahus']}")
    print(f"\nAHU Assignments:")
    for i, rooms in enumerate(result['ahu_assignments'], 1):
        print(f"  AHU {i}: {len(rooms)} rooms")
    print(f"\nGenerated files:")
    print(f"  - {result['floor_plan_path']}")
    print(f"  - {result['zone_map_path']}")
    print(f"  - {result['gin_config_path']}")
    print(f"{'='*60}\n")


def get_env(result: Dict[str, Any], reset: bool = True):
    """Create an RL environment from a generated scenario.

    Args:
        result: Result dictionary from generate_scenario() or generate_scenario_from_dict().
        reset: Whether to call env.reset() before returning (default: True).

    Returns:
        HybridActionEnvironment configured for the scenario.

    Example:
        result = generate_scenario("scenarios/my_config.yaml")
        env = get_env(result)
        obs = env.reset()
        action = {'discrete_action': [1, 1, 0], 'continuous_action': [-1, 1, -1, 1, 1.0, -1]}
        time_step = env.step(action)
    """
    import gin

    # Import all modules that register gin configurables
    # These must be imported BEFORE parsing the gin config
    from smart_buildings.smart_control.environment import hybrid_action_environment
    from smart_buildings.smart_control.reward import setpoint_energy_carbon_regret
    from smart_buildings.smart_control.reward import electricity_energy_cost
    from smart_buildings.smart_control.reward import natural_gas_energy_cost
    from smart_buildings.smart_control.simulator import building
    from smart_buildings.smart_control.simulator import hvac
    from smart_buildings.smart_control.simulator import hvac_floorplan_based
    from smart_buildings.smart_control.simulator import tf_simulator
    from smart_buildings.smart_control.simulator import randomized_arrival_departure_occupancy
    from smart_buildings.smart_control.simulator import simulator_building
    from smart_buildings.smart_control.simulator import blend_convection_simulator
    from smart_buildings.smart_control.utils import observation_normalizer
    from smart_buildings.smart_control.utils import environment_utils

    # Clear any existing gin config and parse the scenario's config
    gin.clear_config()
    gin.parse_config_file(result['gin_config_path'])

    # Create the environment
    env = hybrid_action_environment.HybridActionEnvironment()

    if reset:
        env.reset()

    return env


def get_env_from_config(config_path: str, reset: bool = True):
    """Generate a scenario and create an RL environment in one step.

    Convenience function that combines generate_scenario() and get_env().

    Args:
        config_path: Path to the YAML configuration file.
        reset: Whether to call env.reset() before returning (default: True).

    Returns:
        Tuple of (env, result) where env is the HybridActionEnvironment
        and result is the scenario generation result dict.

    Example:
        env, result = get_env_from_config("scenarios/my_config.yaml")
        print(f"Created env with {result['num_rooms']} rooms")
    """
    result = generate_scenario(config_path)
    env = get_env(result, reset=reset)
    return env, result


def print_action_spec(env) -> Dict[str, Any]:
    """Print the action specification for an environment and return an example action.

    Displays the structure of discrete and continuous actions with their
    indices, names, and value ranges. Returns a valid example action dict.

    Args:
        env: HybridActionEnvironment instance.

    Returns:
        Dict with 'discrete_action' and 'continuous_action' lists that can
        be passed directly to env.step().

    Example:
        env = get_env(result)
        example_action = print_action_spec(env)
        time_step = env.step(example_action)
    """
    print("=== ACTION SPEC ===\n")

    print("action = {")
    print(f"    'discrete_action': [  # shape: {env.action_spec()['discrete_action'].shape}")
    discrete_actions = []
    for device, action_name in env._device_action_tuples:
        if 'run_command' in action_name:
            print(f"        # [{len(discrete_actions)}] {device}_{action_name} -> Off=0, On=1")
            discrete_actions.append(1)  # default: On

    print(f"    ],")
    print(f"    'continuous_action': [  # shape: {env.action_spec()['continuous_action'].shape}")
    continuous_actions = []
    for device, action_name in env._device_action_tuples:
        if 'run_command' not in action_name:
            key = f"{device}_{action_name}"
            norm = env.action_normalizers.get(key)
            if norm:
                min_v, max_v = norm._min_native_value, norm._max_native_value
                unit = "K" if 'temperature' in action_name else "Pa"
                print(f"        # [{len(continuous_actions)}] {key} -> [{min_v}, {max_v}] {unit}")
            continuous_actions.append(0.0)  # default: midpoint
    print(f"    ],")
    print("}")

    example_action = {
        'discrete_action': discrete_actions,
        'continuous_action': continuous_actions,
    }
    print(f"\n# Example action (all On, all midpoint):")
    print(f"example_action = {example_action}")

    return example_action


class SimulationTracker:
    """Simple tracker for logging and visualizing simulation runs.

    Example:
        env = get_env(result)
        tracker = SimulationTracker(env)

        for i in range(1000):
            tracker.step(action)
            if i % 100 == 0:
                tracker.render()
                tracker.plot()

        # Render specific timestep
        tracker.render(step=500)

        # Get summary
        tracker.summary()
    """

    def __init__(self, env, vmin: float = 280, vmax: float = 310):
        """Initialize tracker.

        Args:
            env: HybridActionEnvironment instance.
            vmin: Min temperature for rendering colormap.
            vmax: Max temperature for rendering colormap.
        """
        import pandas as pd
        from smart_buildings.smart_control.utils import building_renderer

        self.env = env
        self.vmin = vmin
        self.vmax = vmax

        # Create renderer
        building_layout = env.building.simulator.building.floor_plan
        self.renderer = building_renderer.BuildingRenderer(building_layout, 1)

        # Get env info
        self.ahus = env.building.simulator._hvac.air_handler._ahus
        self.vavs = env.building.simulator._hvac._vavs
        self.zone_ids = list(self.vavs.keys())
        self.time_step_sec = env.building.simulator._time_step_sec
        self.num_ahus = len(self.ahus)

        # Build zone-to-AHU index mapping for coloring
        self.zone_ahu_indices = []
        for zone_id in self.zone_ids:
            vav = self.vavs[zone_id]
            ahu_obj = vav._air_handler
            try:
                ahu_idx = self.ahus.index(ahu_obj)
            except ValueError:
                ahu_idx = 0  # fallback
            self.zone_ahu_indices.append(ahu_idx)

        # Setpoint schedule for temperature range overlay
        self.setpoint_schedule = env.building.simulator._hvac.schedule

        # Reward weights
        reward_fn = env.reward_function
        self.productivity_weight = reward_fn._productivity_weight
        self.energy_weight = reward_fn._energy_cost_weight
        self.carbon_weight = reward_fn._carbon_emission_weight

        # Logs
        self.rewards = []
        self.occupancy = []
        self.timestamps = []
        self.temp_snapshots = []  # For rendering past states
        self.zone_temps = []
        self.ahu_log = []
        self.reward_components = []

        self._step_count = 0

    def reset(self):
        """Reset tracker state and environment."""
        self.env.reset()
        self.rewards = []
        self.occupancy = []
        self.timestamps = []
        self.temp_snapshots = []
        self.zone_temps = []
        self.ahu_log = []
        self.reward_components = []
        self._step_count = 0

    def step(self, action: Dict[str, Any]):
        """Step environment and log all data.

        Args:
            action: Action dict with 'discrete_action' and 'continuous_action'.

        Returns:
            TimeStep from env.step().
        """
        import pandas as pd

        time_step = self.env.step(action)
        reward = float(time_step.reward)
        timestamp = self.env.building.current_timestamp

        # Store temp snapshot for rendering
        temps_array = self.env.building.simulator.building.temp.copy()
        self.temp_snapshots.append(temps_array)

        # Basic logs
        self.rewards.append(reward)
        self.timestamps.append(timestamp)

        # Reward components
        _, reward_response = self.env.get_reward_info_and_response()
        self.reward_components.append({
            'timestamp': timestamp,
            'agent_reward': reward_response.agent_reward_value,
            'productivity_reward': reward_response.productivity_reward,
            'electricity_energy_cost': reward_response.electricity_energy_cost,
            'natural_gas_energy_cost': reward_response.natural_gas_energy_cost,
            'carbon_emitted': reward_response.carbon_emitted,
            'normalized_productivity_regret': reward_response.normalized_productivity_regret,
            'normalized_energy_cost': reward_response.normalized_energy_cost,
            'normalized_carbon_emission': reward_response.normalized_carbon_emission,
            'total_occupancy': reward_response.total_occupancy,
        })

        # Occupancy
        total_occupancy = 0.0
        end_time = timestamp + pd.Timedelta(self.time_step_sec, unit='s')
        for zone_id in self.zone_ids:
            total_occupancy += self.env.building.occupancy.average_zone_occupancy(
                zone_id, timestamp, end_time
            )
        self.occupancy.append(total_occupancy)

        # Zone temps
        temps = [vav.zone_air_temperature for vav in self.vavs.values()]
        self.zone_temps.append(temps)

        # AHU data
        avg_temps_dict = self.env.building.simulator.building.get_zone_average_temps()
        recirculation_temp = sum(avg_temps_dict.values()) / len(avg_temps_dict)
        outside_air = self.ahus[0].outside_air_temperature_sensor

        self.ahu_log.append({
            'timestamp': timestamp,
            'outside_air': outside_air,
            'recirculation_temp': recirculation_temp,
            'ahu_supply_temps': [ahu.get_supply_air_temp(recirculation_temp, outside_air) for ahu in self.ahus],
            'ahu_run_cmds': [int(ahu.run_command) for ahu in self.ahus],
            'ahu_flow_rates': [ahu.air_flow_rate for ahu in self.ahus],
        })

        self._step_count += 1
        return time_step

    def render(self, step: Optional[int] = None):
        """Render building temperature visualization.

        Args:
            step: Step index to render. If None, renders current state.

        Returns:
            PIL Image.
        """
        from IPython.display import display

        if step is not None:
            if step < 0 or step >= len(self.temp_snapshots):
                raise ValueError(f"Step {step} out of range [0, {len(self.temp_snapshots)-1}]")
            temps = self.temp_snapshots[step]
        else:
            temps = self.env.building.simulator.building.temp

        image = self.renderer.render(temps, cmap='bwr', vmin=self.vmin, vmax=self.vmax).convert('RGB')
        display(image)
        return image

    def create_video(
        self,
        output_path: str = "simulation.mp4",
        fps: int = 10,
        show_progress: bool = True,
    ) -> str:
        """Create a video from temperature snapshots.

        Args:
            output_path: Output file path. Supports .mp4, .gif, .avi.
            fps: Frames per second.
            show_progress: Whether to show progress bar.

        Returns:
            Path to the created video file.
        """
        import numpy as np
        from PIL import Image, ImageDraw, ImageFont

        if not self.temp_snapshots:
            raise ValueError("No snapshots to create video from. Run simulation first.")

        # Determine output format
        ext = output_path.lower().split('.')[-1]
        if ext not in ('mp4', 'gif', 'avi'):
            raise ValueError(f"Unsupported format: {ext}. Use mp4, gif, or avi.")

        frames = []
        n_frames = len(self.temp_snapshots)

        iterator = range(n_frames)
        if show_progress:
            try:
                from tqdm import tqdm
                iterator = tqdm(iterator, desc="Rendering frames")
            except ImportError:
                pass

        for i in iterator:
            temps = self.temp_snapshots[i]
            frame = self.renderer.render(temps, cmap='bwr', vmin=self.vmin, vmax=self.vmax).convert('RGB')

            # Add timestamp overlay if available
            if i < len(self.timestamps):
                draw = ImageDraw.Draw(frame)
                ts_str = self.timestamps[i].strftime('%Y-%m-%d %H:%M')
                # Try to use a monospace font, fall back to default
                try:
                    font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf", 16)
                except:
                    font = ImageFont.load_default()
                draw.rectangle([5, 5, 180, 25], fill='white')
                draw.text((10, 7), ts_str, fill='black', font=font)

            frames.append(np.array(frame))

        # Write video
        if ext == 'gif':
            # Use PIL for GIF
            pil_frames = [Image.fromarray(f) for f in frames]
            pil_frames[0].save(
                output_path,
                save_all=True,
                append_images=pil_frames[1:],
                duration=int(1000 / fps),
                loop=0,
            )
        else:
            # Use imageio for mp4/avi
            try:
                import imageio
            except ImportError:
                raise ImportError("imageio required for mp4/avi. Install with: pip install imageio[ffmpeg]")

            writer = imageio.get_writer(output_path, fps=fps)
            for frame in frames:
                writer.append_data(frame)
            writer.close()

        print(f"Video saved to {output_path} ({n_frames} frames, {fps} fps)")
        return output_path

    def plot(self, color_by_ahu: bool = False):
        """Plot all charts (temps, rewards, occupancy, AHU diagnostics, reward components).

        Args:
            color_by_ahu: If True, color zone temps by their AHU assignment.
        """
        self.plot_overview(color_by_ahu=color_by_ahu)
        self.plot_ahu()
        self.plot_rewards()

    def plot_overview(self, color_by_ahu: bool = False):
        """Plot temperature, reward, and occupancy overview.

        Args:
            color_by_ahu: If True, color zone temps by their AHU assignment.
        """
        import matplotlib.pyplot as plt
        import matplotlib.dates as mdates

        if not self.timestamps:
            print("No data to plot.")
            return

        mpl_ts = [ts.to_pydatetime() for ts in self.timestamps]
        n_zones = len(self.zone_temps[0]) if self.zone_temps else 0

        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 10), sharex=True,
                                             gridspec_kw={'height_ratios': [2, 1, 1]})

        # Panel 1: Temperatures with setpoint range overlay
        outside_air = [d['outside_air'] for d in self.ahu_log]

        # Get setpoint schedule data for the time range
        setpoint_data = self.setpoint_schedule.get_plot_data(
            self.timestamps[0], self.timestamps[-1]
        )

        # Draw setpoint ranges as shaded regions (behind zone temps)
        for _, row in setpoint_data.iterrows():
            start = row['start_time'].to_pydatetime()
            end = row['end_time'].to_pydatetime()
            heating = row['heating_setpoint']
            cooling = row['cooling_setpoint']
            color = '#90EE90' if row['comfort_mode'] else '#ADD8E6'  # green=day, blue=night
            ax1.fill_between([start, end], heating, cooling, alpha=0.3, color=color,
                           label='_nolegend_')
            ax1.plot([start, end], [heating, heating], color=color, linewidth=1, linestyle='--')
            ax1.plot([start, end], [cooling, cooling], color=color, linewidth=1, linestyle='--')

        # Add legend entries for setpoint ranges
        ax1.fill_between([], [], [], alpha=0.3, color='#90EE90', label='Comfort Range (Day)')
        ax1.fill_between([], [], [], alpha=0.3, color='#ADD8E6', label='Eco Range (Night)')

        ax1.plot(mpl_ts, outside_air, 'b-', linewidth=2, label='Outside Air')

        # AHU colors for color_by_ahu mode
        ahu_colors = ['red', 'blue', 'green', 'orange', 'purple', 'brown', 'pink', 'cyan']

        if color_by_ahu:
            # Plot zones colored by their AHU assignment
            ahu_plotted = set()
            for i in range(n_zones):
                zone_trace = [t[i] for t in self.zone_temps]
                ahu_idx = self.zone_ahu_indices[i]
                color = ahu_colors[ahu_idx % len(ahu_colors)]
                if ahu_idx not in ahu_plotted:
                    label = f'AHU {ahu_idx + 1} zones'
                    ahu_plotted.add(ahu_idx)
                else:
                    label = "_nolegend_"
                ax1.plot(mpl_ts, zone_trace, color=color, linewidth=1, alpha=0.5, label=label)
        else:
            # Single color for all zones
            for i in range(n_zones):
                zone_trace = [t[i] for t in self.zone_temps]
                label = f'Zone Temps ({n_zones} zones)' if i == 0 else "_nolegend_"
                ax1.plot(mpl_ts, zone_trace, color='#FFD700', linewidth=1, alpha=0.5, label=label)

        ax1.set_ylabel('Temperature (K)')
        ax1.set_title('Temperature Overview')
        ax1.legend(loc='upper left')
        ax1.grid(True, linestyle='--', alpha=0.6)

        # Panel 2: Rewards
        ax2.plot(mpl_ts, self.rewards, 'purple', linewidth=2)
        ax2.fill_between(mpl_ts, self.rewards, alpha=0.3, color='purple')
        ax2.axhline(0, color='gray', linestyle='--', alpha=0.5)
        ax2.set_ylabel('Reward')
        ax2.set_title(f'Reward (cumulative: {sum(self.rewards):.2f})')
        ax2.grid(True, linestyle='--', alpha=0.6)

        # Panel 3: Occupancy
        ax3.plot(mpl_ts, self.occupancy, 'orange', linewidth=2)
        ax3.fill_between(mpl_ts, self.occupancy, alpha=0.3, color='orange')
        ax3.set_ylabel('Occupancy')
        ax3.set_xlabel('Time')
        ax3.set_title(f'Occupancy (avg: {sum(self.occupancy)/len(self.occupancy):.1f})')
        ax3.grid(True, linestyle='--', alpha=0.6)

        ax3.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d %H:%M'))
        fig.autofmt_xdate()
        plt.tight_layout()
        plt.show()

    def plot_ahu(self):
        """Plot AHU diagnostics."""
        import matplotlib.pyplot as plt
        import matplotlib.dates as mdates

        if not self.ahu_log:
            return

        mpl_ts = [d['timestamp'].to_pydatetime() for d in self.ahu_log]
        colors = ['red', 'blue', 'green', 'orange', 'purple', 'brown', 'pink', 'cyan']

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 6), sharex=True)

        # Supply temps
        for i in range(self.num_ahus):
            temps = [d['ahu_supply_temps'][i] for d in self.ahu_log]
            ax1.plot(mpl_ts, temps, color=colors[i % len(colors)], linewidth=2,
                     label=f'AHU {i+1} ({temps[-1]:.1f}K)')
        ax1.set_ylabel('Supply Air Temp (K)')
        ax1.set_title('AHU Supply Air Temperatures')
        ax1.legend(loc='upper right')
        ax1.grid(True, linestyle='--', alpha=0.6)

        # Flow rates
        for i in range(self.num_ahus):
            flows = [d['ahu_flow_rates'][i] for d in self.ahu_log]
            ax2.plot(mpl_ts, flows, color=colors[i % len(colors)], linewidth=2,
                     label=f'AHU {i+1} ({flows[-1]:.2f} m³/s)')
        ax2.set_ylabel('Flow Rate (m³/s)')
        ax2.set_xlabel('Time')
        ax2.set_title('AHU Flow Rates')
        ax2.legend(loc='upper right')
        ax2.grid(True, linestyle='--', alpha=0.6)

        ax2.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d %H:%M'))
        fig.autofmt_xdate()
        plt.tight_layout()
        plt.show()

    def plot_rewards(self):
        """Plot reward component breakdown."""
        import matplotlib.pyplot as plt
        import matplotlib.dates as mdates

        if not self.reward_components:
            return

        mpl_ts = [d['timestamp'].to_pydatetime() for d in self.reward_components]
        total_weight = self.productivity_weight + self.energy_weight + self.carbon_weight

        prod = [d['normalized_productivity_regret'] * self.productivity_weight / total_weight
                for d in self.reward_components]
        energy = [-d['normalized_energy_cost'] * self.energy_weight / total_weight
                  for d in self.reward_components]
        carbon = [-d['normalized_carbon_emission'] * self.carbon_weight / total_weight
                  for d in self.reward_components]
        total = [d['agent_reward'] for d in self.reward_components]

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

        # Weighted components
        ax1.plot(mpl_ts, prod, 'g-', linewidth=2, label=f'Productivity ({self.productivity_weight:.0%})')
        ax1.plot(mpl_ts, energy, 'r-', linewidth=2, label=f'Energy ({self.energy_weight:.0%})')
        ax1.plot(mpl_ts, carbon, 'orange', linewidth=2, label=f'Carbon ({self.carbon_weight:.0%})')
        ax1.plot(mpl_ts, total, 'purple', linewidth=2, linestyle='--', label='Total')
        ax1.axhline(0, color='gray', linestyle='--', alpha=0.5)
        ax1.set_ylabel('Reward Contribution')
        ax1.set_title('Weighted Reward Components')
        ax1.legend(loc='upper right')
        ax1.grid(True, linestyle='--', alpha=0.6)

        # Raw costs
        elec = [d['electricity_energy_cost'] for d in self.reward_components]
        gas = [d['natural_gas_energy_cost'] for d in self.reward_components]
        productivity = [d['productivity_reward'] for d in self.reward_components]

        ax2.plot(mpl_ts, elec, 'r-', linewidth=2, label=f'Electricity (${elec[-1]:.2f})')
        ax2.plot(mpl_ts, gas, 'orange', linewidth=2, label=f'Gas (${gas[-1]:.2f})')
        ax2.plot(mpl_ts, productivity, 'g-', linewidth=2, label=f'Productivity (${productivity[-1]:.2f})')
        ax2.set_ylabel('Cost ($)')
        ax2.set_xlabel('Time')
        ax2.set_title('Raw Costs per Step')
        ax2.legend(loc='upper right')
        ax2.grid(True, linestyle='--', alpha=0.6)

        ax2.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
        fig.autofmt_xdate()
        plt.tight_layout()
        plt.show()

    def summary(self):
        """Print simulation summary."""
        if not self.rewards:
            print("No data yet.")
            return

        print(f"\n{'='*50}")
        print(f"Simulation Summary ({self._step_count} steps)")
        print(f"{'='*50}")
        print(f"Cumulative Reward: {sum(self.rewards):.4f}")
        print(f"Average Reward: {sum(self.rewards)/len(self.rewards):.4f}")
        print(f"Min/Max Reward: {min(self.rewards):.4f} / {max(self.rewards):.4f}")
        print(f"Average Occupancy: {sum(self.occupancy)/len(self.occupancy):.2f}")
        if self.reward_components:
            rc = self.reward_components[-1]
            print(f"Last step costs: Elec=${rc['electricity_energy_cost']:.2f}, "
                  f"Gas=${rc['natural_gas_energy_cost']:.2f}, "
                  f"Carbon={rc['carbon_emitted']:.2f}kg")
        print(f"{'='*50}\n")


# CLI entry point
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Generate HVAC simulation scenarios from YAML configs"
    )
    parser.add_argument(
        "config_path",
        type=str,
        help="Path to YAML configuration file"
    )
    parser.add_argument(
        "--quiet", "-q",
        action="store_true",
        help="Suppress output summary"
    )

    args = parser.parse_args()

    result = generate_scenario(args.config_path)

    if not args.quiet:
        print_scenario_summary(result)

# Smart Buildings Simulation Generator

A procedural HVAC building simulation environment for reinforcement learning research. Generate configurable multi-floor buildings with realistic thermal dynamics, occupancy patterns, and energy costs.

## Installation

```bash
git clone git@github.com:Jgoldfeder/smart_buildings_sim_generator.git
cd smart_buildings_sim_generator
pip install -e .
```

## Quick Start

```python
from smart_buildings.smart_control.utils.scenario_generator import (
    generate_scenario,
    get_env_from_config,
    print_action_spec,
    SimulationTracker
)

# Generate a scenario from YAML config
result = generate_scenario("scenarios/example_scenario.yaml")

# Or create an RL environment directly
env = get_env_from_config("scenarios/example_scenario.yaml")

# View the action space
example_action = print_action_spec(env)

# Run simulation with tracking
tracker = SimulationTracker(env)
for _ in range(100):
    ts = tracker.step(example_action)

tracker.plot()  # Visualize results
tracker.summary()  # Print statistics
```

## Configuration

Create a YAML config file to customize your scenario:

```yaml
name: "my_scenario"
seed: 42

floor_plan:
  num_floors: 2
  num_rooms_per_floor: 8
  building_size: [100, 100]

simulation:
  start_timestamp: "2023-07-10 06:00"
  num_days_in_episode: 7
  occupants_per_zone: 3

weather:
  temp_min: 280
  temp_max: 310
```

See `scenarios/example_scenario.yaml` for all available options.

## License

Apache 2.0

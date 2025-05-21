# OpenRobotics

[![PyPI version](https://badge.fury.io/py/openrobotics.svg)](https://badge.fury.io/py/openrobotics)
[![Documentation Status](https://readthedocs.org/projects/openrobotics/badge/?version=latest)](https://openrobotics.readthedocs.io/en/latest/?badge=latest)
[![CI](https://github.com/llamasearchai/OpenRobotics/workflows/CI/badge.svg)](https://github.com/llamasearchai/OpenRobotics/actions)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A modular and extensible robotics framework that integrates modern machine learning capabilities with hardware control.

## Features

- **Hardware Abstraction**: Support for multiple robot platforms (Arduino, Raspberry Pi, ESP32, ROS)
- **Simulation Environment**: Built-in 2D robot simulation with visualization
- **MLX Integration**: Efficient neural networks running on Apple Silicon
- **LangChain Integration**: Connect LLMs to your robots
- **Modular Design**: Easy to extend with new hardware adapters or capabilities

## Installation

### Prerequisites

- Python 3.9+
- For MLX integration: macOS with Apple Silicon
- For hardware control: Appropriate hardware and drivers

```bash
# Clone the repository
git clone https://github.com/llamasearchai/OpenRobotics.git
cd OpenRobotics

# Install the package with dependencies
pip install -e .

# For development with all extras
pip install -e ".[dev,docs,test]"
```

## Project Structure

## Quick Start

### Simulation Example

```python
from openrobotics.robot_control import Robot, Simulation

# Create a robot
robot = Robot(config={...})

# Create simulation
sim = Simulation()

# Add robot to simulation
robot_id = sim.add_robot(robot)

# Start simulation
sim.start()

# Control your robot
sim.set_robot_velocity(robot_id, linear=0.5, angular=0.2)
```

### Hardware Example

```python
from openrobotics.robot_control import Robot

# Create robot with hardware configuration
robot = Robot(config={
    "hardware_adapter": {
        "type": "arduino",
        "port": "/dev/ttyACM0"
    }
})

# Connect to hardware
robot.connect()

# Move the robot
robot.move(linear_speed=0.5, angular_speed=0.0)
```

## Documentation

Detailed documentation is available in the `docs/` directory.

## Examples

- `examples/robot_simulation_example.py`: Demonstrates the simulation environment
- `examples/physical_robot_example.py`: Shows how to work with physical hardware
- `examples/mlx_robot_control_example.py`: Demonstrates MLX neural network integration

## License

MIT License. See LICENSE file for details.

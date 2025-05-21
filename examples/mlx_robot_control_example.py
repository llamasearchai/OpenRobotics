"""
Example demonstrating integration of MLX with robot control.
"""

import os
import time
import logging
import numpy as np
from typing import Dict, Any, List, Tuple
import matplotlib.pyplot as plt

from openrobotics.robot_control import Robot, Simulation
from openrobotics.mlx_integration import MLXModel, VisionModel
try:
    import mlx.core as mx
except ImportError:
    logging.warning("MLX not available. This example requires MLX to be installed.")
    exit(1)

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MLXRobotController:
    """Robot controller using MLX for perception and decision making."""
    
    def __init__(self, robot: Robot, simulation: Simulation = None, robot_id: str = None):
        """
        Initialize MLX-based robot controller.
        
        Args:
            robot: Robot instance to control
            simulation: Optional simulation environment
            robot_id: Robot ID in simulation
        """
        self.robot = robot
        self.simulation = simulation
        self.robot_id = robot_id
        
        # Initialize MLX vision model for obstacle detection
        self.vision_model = self._setup_vision_model()
        
        # Initialize simple control policy network
        self.control_model = self._setup_control_model()
        
        # Initialize state for model inference
        self.state_history = []
        self.max_history_len = 10
    
    def _setup_vision_model(self) -> VisionModel:
        """Set up MLX vision model for perception."""
        # This would typically load a pre-trained vision model
        # For this example, we'll create a simple model that processes distance sensor data
        
        # Define a simple convolutional network for processing sensor data
        def create_vision_network():
            import mlx.nn as nn
            
            model = nn.Sequential(
                nn.Linear(8, 64),   # 8 distance sensors as input
                nn.ReLU(),
                nn.Linear(64, 32),
                nn.ReLU(),
                nn.Linear(32, 16),
                nn.ReLU(),
                nn.Linear(16, 4)    # Output features representing environment perception
            )
            return model
        
        # Create vision model
        vision_model = VisionModel(
            model=create_vision_network(),
            input_shape=(8,),
            config={"normalize": True}
        )
        
        logger.info("MLX vision model initialized")
        return vision_model
    
    def _setup_control_model(self) -> MLXModel:
        """Set up MLX model for control policy."""
        # Define a simple control network
        def create_control_network():
            import mlx.nn as nn
            
            model = nn.Sequential(
                nn.Linear(4 + 3, 32),  # 4 vision features + 3 robot state values
                nn.ReLU(),
                nn.Linear(32, 16),
                nn.ReLU(),
                nn.Linear(16, 2)       # Output: [linear_velocity, angular_velocity]
            )
            return model
        
        # Create control model
        control_model = MLXModel(
            model=create_control_network(),
            input_shape=(7,),
            config={"output_scale": 1.0}
        )
        
        logger.info("MLX control model initialized")
        return control_model
    
    def process_sensor_data(self, sensor_data: Dict[str, Any]) -> mx.array:
        """
        Process sensor data into input for MLX models.
        
        Args:
            sensor_data: Sensor readings from robot
            
        Returns:
            MLX array of processed sensor data
        """
        # Extract distance sensors data
        distance_sensors = sensor_data.get("distance_sensors", {})
        distances = distance_sensors.get("distances", [])
        
        # If we don't have enough distance readings, pad with max range
        while len(distances) < 8:
            distances.append(5.0)
        
        # Truncate if too many
        distances = distances[:8]
        
        # Normalize distances (closer obstacles = higher values)
        normalized_distances = [1.0 - min(d, 5.0) / 5.0 for d in distances]
        
        # Convert to MLX array
        return mx.array(normalized_distances, dtype=mx.float32)
    
    def control_loop(self):
        """Run the MLX-based control loop."""
        # Connect robot
        self.robot.connect()
        
        # Control loop
        try:
            while True:
                # Read sensor data
                sensor_data = self.robot.read_sensors()
                
                # Process sensor data for vision model
                sensor_features = self.process_sensor_data(sensor_data)
                
                # Get robot state
                robot_state = self.robot.get_state()
                position = robot_state["position"]
                velocity = robot_state["velocity"]
                
                # Create state array with position and velocity
                state_array = mx.array([
                    position["x"] / 10.0,  # Normalize by world size
                    position["y"] / 10.0,
                    position["theta"] / (2 * np.pi)
                ], dtype=mx.float32)
                
                # Run vision model to extract environment features
                with mx.capture_gradients():
                    env_features = self.vision_model(sensor_features)
                
                # Combine with state for control model input
                control_input = mx.concatenate([env_features, state_array])
                
                # Run control model to get velocity commands
                with mx.capture_gradients():
                    velocity_output = self.control_model(control_input)
                
                # Extract linear and angular velocity
                linear_vel = float(velocity_output[0].item()) 
                angular_vel = float(velocity_output[1].item())
                
                # Apply to robot in simulation or hardware
                if self.simulation and self.robot_id:
                    self.simulation.set_robot_velocity(self.robot_id, linear_vel, angular_vel)
                else:
                    self.robot.move(linear_vel, angular_vel)
                
                logger.info(f"Velocities: linear={linear_vel:.2f}, angular={angular_vel:.2f}")
                
                # Store state for potential learning updates
                self.state_history.append({
                    "sensor_data": sensor_features,
                    "state": state_array,
                    "action": velocity_output,
                    "time": time.time()
                })
                
                # Keep history limited to recent states
                if len(self.state_history) > self.max_history_len:
                    self.state_history.pop(0)
                
                # Short delay
                time.sleep(0.1)
        
        except KeyboardInterrupt:
            logger.info("Control loop interrupted by user")
            
            if self.simulation and self.robot_id:
                self.simulation.set_robot_velocity(self.robot_id, 0.0, 0.0)
            else:
                self.robot.stop()

def create_robot_with_sensors():
    """Create a robot instance with appropriate sensors for the MLX controller."""
    # Configure robot
    config = {
        "name": "mlx_bot",
        "sensors": {
            "front_distance": {
                "type": "distance",
                "position": {"x": 0.2, "y": 0.0, "z": 0.1},
                "direction": {"x": 1.0, "y": 0.0, "z": 0.0},
                "range": 5.0
            },
            "front_left_distance": {
                "type": "distance",
                "position": {"x": 0.15, "y": 0.15, "z": 0.1},
                "direction": {"x": 0.7, "y": 0.7, "z": 0.0},
                "range": 5.0
            },
            "front_right_distance": {
                "type": "distance",
                "position": {"x": 0.15, "y": -0.15, "z": 0.1},
                "direction": {"x": 0.7, "y": -0.7, "z": 0.0},
                "range": 5.0
            },
            "left_distance": {
                "type": "distance",
                "position": {"x": 0.0, "y": 0.2, "z": 0.1},
                "direction": {"x": 0.0, "y": 1.0, "z": 0.0},
                "range": 5.0
            },
            "right_distance": {
                "type": "distance",
                "position": {"x": 0.0, "y": -0.2, "z": 0.1},
                "direction": {"x": 0.0, "y": -1.0, "z": 0.0},
                "range": 5.0
            }
        },
        "actuators": {
            "left_motor": {
                "type": "motor",
                "max_speed": 1.0
            },
            "right_motor": {
                "type": "motor",
                "max_speed": 1.0
            }
        }
    }
    
    return Robot(config)

def run_mlx_controller_demo():
    """Run demo with MLX-based robot control in simulation."""
    # Create simulation environment
    sim_config = {
        "world_size": (15.0, 15.0),
        "obstacles": [
            {"type": "box", "x": 5.0, "y": 5.0, "width": 2.0, "height": 2.0},
            {"type": "circle", "x": 10.0, "y": 8.0, "radius": 1.0},
            {"type": "circle", "x": 3.0, "y": 12.0, "radius": 1.5},
            {"type": "box", "x": 12.0, "y": 12.0, "width": 3.0, "height": 1.0}
        ],
        "update_rate": 60,
        "visualization": {
            "enabled": True,
            "live_update": True
        }
    }
    
    # Create simulation
    sim = Simulation(sim_config)
    
    # Create robot
    robot = create_robot_with_sensors()
    
    # Add robot to simulation at starting position
    robot_id = sim.add_robot(robot, pose={"x": 2.0, "y": 2.0, "theta": 0.0})
    
    # Start simulation
    sim.start()
    
    # Create MLX controller
    controller = MLXRobotController(robot, sim, robot_id)
    
    try:
        # Run controller
        controller.control_loop()
    except KeyboardInterrupt:
        pass
    finally:
        # Stop simulation
        sim.stop()
        logger.info("Simulation stopped")

if __name__ == "__main__":
    try:
        # Run MLX controller demo
        run_mlx_controller_demo()
    except Exception as e:
        logger.exception(f"Error in MLX controller demo: {e}") 
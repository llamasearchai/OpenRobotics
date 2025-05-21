"""
Example demonstrating robot simulation in OpenRobotics.
"""

import time
import logging
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, Any

from openrobotics.robot_control import Robot, Simulation

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def create_test_robot():
    """Create a robot instance for simulation."""
    # Configure robot
    config = {
        "name": "explorer",
        "sensors": {
            "front_distance": {
                "type": "distance",
                "position": {"x": 0.2, "y": 0.0, "z": 0.1},
                "direction": {"x": 1.0, "y": 0.0, "z": 0.0},
                "range": 5.0
            },
            "left_distance": {
                "type": "distance",
                "position": {"x": 0.1, "y": 0.1, "z": 0.1},
                "direction": {"x": 0.0, "y": 1.0, "z": 0.0},
                "range": 5.0
            },
            "right_distance": {
                "type": "distance",
                "position": {"x": 0.1, "y": -0.1, "z": 0.1},
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
            },
            "camera_servo": {
                "type": "servo",
                "min_angle": -90.0,
                "max_angle": 90.0
            }
        },
        "parameters": {
            "radius": 0.2,
            "max_speed": 1.0,
            "max_angular_speed": 1.5
        }
    }
    
    return Robot(config)

def obstacle_avoidance_controller(robot: Robot, simulation: Simulation, robot_id: str):
    """Simple obstacle avoidance controller."""
    # Connect robot
    robot.connect()
    
    # Main control loop
    try:
        while True:
            # Read sensor data from simulation
            sensor_data = robot.read_sensors()
            distance_sensors = sensor_data.get("distance_sensors", {})
            
            if "distances" in distance_sensors:
                distances = distance_sensors["distances"]
                
                # Extract front, left and right distances
                front_distance = distances[0] if len(distances) > 0 else 5.0
                left_distance = distances[2] if len(distances) > 2 else 5.0
                right_distance = distances[6] if len(distances) > 6 else 5.0
                
                logger.info(f"Distances - Front: {front_distance:.2f}, Left: {left_distance:.2f}, Right: {right_distance:.2f}")
                
                # Simple obstacle avoidance logic
                if front_distance < 1.0:
                    # Obstacle ahead, turn to the side with more space
                    if left_distance > right_distance:
                        # Turn left
                        linear_speed = 0.0
                        angular_speed = 0.8
                        logger.info("Turning left to avoid obstacle")
                    else:
                        # Turn right
                        linear_speed = 0.0
                        angular_speed = -0.8
                        logger.info("Turning right to avoid obstacle")
                else:
                    # No obstacle, go forward with small adjustments
                    linear_speed = 0.5
                    if left_distance < 1.0:
                        # Obstacle on left, turn slightly right
                        angular_speed = -0.3
                        logger.info("Adjusting right to avoid left obstacle")
                    elif right_distance < 1.0:
                        # Obstacle on right, turn slightly left
                        angular_speed = 0.3
                        logger.info("Adjusting left to avoid right obstacle")
                    else:
                        # No obstacles nearby, go straight
                        angular_speed = 0.0
                
                # Update robot velocity in simulation
                simulation.set_robot_velocity(robot_id, linear_speed, angular_speed)
            
            # Pause to simulate realistic control loop
            time.sleep(0.1)
    
    except KeyboardInterrupt:
        # Stop robot when interrupted
        simulation.set_robot_velocity(robot_id, 0.0, 0.0)
        logger.info("Robot stopped by user")

def run_obstacle_avoidance_demo():
    """Run obstacle avoidance demo in simulation."""
    # Create simulation environment
    sim_config = {
        "world_size": (15.0, 15.0),
        "obstacles": [
            {"type": "box", "x": 5.0, "y": 5.0, "width": 2.0, "height": 2.0},
            {"type": "circle", "x": 10.0, "y": 8.0, "radius": 1.0},
            {"type": "circle", "x": 3.0, "y": 12.0, "radius": 1.5},
            {"type": "box", "x": 12.0, "y": 12.0, "width": 3.0, "height": 1.0},
            {"type": "box", "x": 8.0, "y": 2.0, "width": 1.0, "height": 4.0}
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
    robot = create_test_robot()
    
    # Add robot to simulation at starting position
    robot_id = sim.add_robot(robot, pose={"x": 2.0, "y": 2.0, "theta": 0.0})
    
    # Start simulation
    sim.start()
    
    try:
        # Run obstacle avoidance controller
        obstacle_avoidance_controller(robot, sim, robot_id)
    except KeyboardInterrupt:
        pass
    finally:
        # Stop simulation
        sim.stop()
        logger.info("Simulation stopped")

def run_multiple_robots_demo():
    """Run demo with multiple robots in simulation."""
    # Create simulation environment
    sim_config = {
        "world_size": (20.0, 20.0),
        "obstacles": [
            {"type": "box", "x": 10.0, "y": 10.0, "width": 4.0, "height": 4.0},
            {"type": "circle", "x": 5.0, "y": 15.0, "radius": 2.0},
            {"type": "circle", "x": 15.0, "y": 5.0, "radius": 2.0}
        ],
        "update_rate": 60,
        "visualization": {
            "enabled": True,
            "live_update": True
        }
    }
    
    # Create simulation
    sim = Simulation(sim_config)
    
    # Create robots
    robots = []
    robot_ids = []
    
    # Add 3 robots at different positions
    for i in range(3):
        robot = create_test_robot()
        robots.append(robot)
        
        # Position robots in different corners
        positions = [
            {"x": 2.0, "y": 2.0, "theta": np.pi/4},
            {"x": 18.0, "y": 2.0, "theta": np.pi*3/4},
            {"x": 2.0, "y": 18.0, "theta": -np.pi/4}
        ]
        
        robot_id = sim.add_robot(robot, name=f"robot_{i}", pose=positions[i])
        robot_ids.append(robot_id)
    
    # Start simulation
    sim.start()
    
    # Connect all robots
    for robot in robots:
        robot.connect()
    
    try:
        # Simple motion demo - make robots move in a circular pattern
        for _ in range(300):  # Run for 30 seconds at 10 Hz
            for i, robot_id in enumerate(robot_ids):
                # Calculate circular motion parameters
                t = time.time() * 0.5
                # Each robot follows a different circular path
                if i == 0:
                    # First robot: clockwise circle
                    linear = 0.5
                    angular = -0.5
                elif i == 1:
                    # Second robot: counter-clockwise circle
                    linear = 0.5
                    angular = 0.5
                else:
                    # Third robot: figure-eight pattern
                    linear = 0.5
                    angular = np.sin(t) * 0.8
                
                # Update robot velocity
                sim.set_robot_velocity(robot_id, linear, angular)
            
            # Sleep to maintain control rate
            time.sleep(0.1)
        
        # Stop all robots
        for robot_id in robot_ids:
            sim.set_robot_velocity(robot_id, 0.0, 0.0)
    
    except KeyboardInterrupt:
        # Stop all robots when interrupted
        for robot_id in robot_ids:
            sim.set_robot_velocity(robot_id, 0.0, 0.0)
        logger.info("Robots stopped by user")
    
    finally:
        # Stop simulation
        sim.stop()
        logger.info("Simulation stopped")

if __name__ == "__main__":
    try:
        # Choose which demo to run
        # Uncomment one of the following:
        
        # Simple obstacle avoidance with a single robot
        run_obstacle_avoidance_demo()
        
        # Multiple robots demo
        # run_multiple_robots_demo()
        
    except KeyboardInterrupt:
        logger.info("Example terminated by user")
    except Exception as e:
        logger.exception(f"Error in simulation example: {e}") 
"""
Example demonstrating control of physical robots using hardware adapters.
"""

import time
import logging
from typing import Dict, Any

from openrobotics.robot_control import Robot
from openrobotics.robot_control.hardware import create_adapter

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def create_arduino_robot():
    """Create a robot instance using Arduino hardware."""
    # Configure robot with Arduino hardware adapter
    config = {
        "name": "arduino_bot",
        "hardware_adapter": {
            "type": "arduino",
            "port": "/dev/ttyACM0",  # Update with your actual port
            "baud_rate": 115200,
            "protocol": "json"
        },
        "sensors": {
            "ultrasonic": {
                "type": "distance",
                "pin": 7,  # Trigger pin
                "echo_pin": 8  # Echo pin
            },
            "ir_left": {
                "type": "infrared",
                "pin": "A0"
            },
            "ir_right": {
                "type": "infrared",
                "pin": "A1"
            }
        },
        "actuators": {
            "left_motor": {
                "type": "motor",
                "pins": {"forward": 5, "backward": 6},
                "max_speed": 1.0
            },
            "right_motor": {
                "type": "motor",
                "pins": {"forward": 9, "backward": 10},
                "max_speed": 1.0
            },
            "pan_servo": {
                "type": "servo",
                "pin": 11,
                "min_angle": 0,
                "max_angle": 180
            }
        }
    }
    
    return Robot(config)

def create_raspberry_pi_robot():
    """Create a robot instance using Raspberry Pi hardware."""
    # Configure robot with Raspberry Pi hardware adapter
    config = {
        "name": "pi_bot",
        "hardware_adapter": {
            "type": "raspberry_pi",
            "motor_pins": {
                "left": {"forward": 17, "backward": 27},
                "right": {"forward": 22, "backward": 23}
            },
            "pwm_pins": {
                "left": 18,
                "right": 13
            },
            "sensor_pins": {
                "front_distance": {
                    "pin": 24,
                    "echo_pin": 25,
                    "type": "ultrasonic"
                }
            }
        },
        "sensors": {
            "front_distance": {
                "type": "distance",
                "position": {"x": 0.1, "y": 0.0, "z": 0.05},
                "direction": {"x": 1.0, "y": 0.0, "z": 0.0},
                "range": 4.0
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

def run_arduino_demo():
    """Run demo with Arduino-based robot."""
    # Create robot
    robot = create_arduino_robot()
    
    # Connect to hardware
    if not robot.connect():
        logger.error("Failed to connect to Arduino hardware")
        return
    
    logger.info("Connected to Arduino robot")
    
    try:
        # Move servo from left to right
        for angle in range(0, 181, 10):
            logger.info(f"Setting servo angle to {angle} degrees")
            robot.set_actuator("pan_servo", angle)
            time.sleep(0.1)
        
        # Reset servo to center
        robot.set_actuator("pan_servo", 90)
        time.sleep(0.5)
        
        # Simple motion sequence
        logger.info("Moving forward")
        robot.move(0.5, 0.0)  # Move forward at half speed
        time.sleep(2.0)
        
        logger.info("Turning left")
        robot.move(0.0, 0.5)  # Turn left
        time.sleep(1.0)
        
        logger.info("Moving forward")
        robot.move(0.5, 0.0)  # Move forward again
        time.sleep(2.0)
        
        logger.info("Turning right")
        robot.move(0.0, -0.5)  # Turn right
        time.sleep(1.0)
        
        logger.info("Moving backward")
        robot.move(-0.5, 0.0)  # Move backward
        time.sleep(2.0)
        
        logger.info("Stopping")
        robot.stop()
        
        # Read and display sensor data
        for _ in range(5):
            sensor_data = robot.read_sensors()
            logger.info(f"Ultrasonic distance: {sensor_data.get('ultrasonic', {}).get('distance', 'N/A')} cm")
            logger.info(f"Left IR: {sensor_data.get('ir_left', {}).get('value', 'N/A')}")
            logger.info(f"Right IR: {sensor_data.get('ir_right', {}).get('value', 'N/A')}")
            time.sleep(1.0)
    
    except KeyboardInterrupt:
        logger.info("Demo interrupted by user")
    
    finally:
        # Stop and disconnect
        robot.stop()
        robot.disconnect()
        logger.info("Robot disconnected")

def run_raspberry_pi_demo():
    """Run demo with Raspberry Pi-based robot."""
    # Create robot
    robot = create_raspberry_pi_robot()
    
    # Connect to hardware
    if not robot.connect():
        logger.error("Failed to connect to Raspberry Pi hardware")
        return
    
    logger.info("Connected to Raspberry Pi robot")
    
    try:
        # Simple obstacle avoidance loop
        for _ in range(60):  # Run for 30 seconds
            # Read distance sensor
            sensor_data = robot.read_sensors()
            front_distance = sensor_data.get("front_distance", {}).get("distance", 100.0)
            
            logger.info(f"Front distance: {front_distance} cm")
            
            # Basic obstacle avoidance
            if front_distance < 30.0:
                # Obstacle detected, back up and turn
                logger.info("Obstacle detected, backing up and turning")
                robot.move(-0.5, 0.0)  # Back up
                time.sleep(1.0)
                robot.move(0.0, 0.7)  # Turn
                time.sleep(1.5)
            else:
                # No obstacle, move forward
                robot.move(0.5, 0.0)
            
            # Short delay
            time.sleep(0.5)
        
        # Stop robot
        robot.stop()
    
    except KeyboardInterrupt:
        logger.info("Demo interrupted by user")
    
    finally:
        # Stop and disconnect
        robot.stop()
        robot.disconnect()
        logger.info("Robot disconnected")

if __name__ == "__main__":
    try:
        # Choose which demo to run
        # Note: only run these on actual hardware
        
        # Arduino demo (uncomment to run)
        # run_arduino_demo()
        
        # Raspberry Pi demo (uncomment to run)
        # run_raspberry_pi_demo()
        
        # If no hardware is selected, show a message
        logger.info("Please uncomment one of the demo functions to run with actual hardware")
        logger.info("Note: These demos require appropriate hardware connections")
        
    except Exception as e:
        logger.exception(f"Error in hardware demo: {e}") 
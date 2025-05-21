"""
Example demonstrating how to use the hardware adapters in OpenRobotics.
"""

import time
import logging
from typing import Dict, Any

from openrobotics.robot_control.hardware import create_adapter

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def demonstrate_arduino_adapter():
    """Demonstrate the use of ArduinoAdapter."""
    logger.info("Testing Arduino adapter...")
    
    # Create Arduino adapter with sample configuration
    config = {
        "port": "/dev/ttyACM0",  # Replace with your actual port
        "baud_rate": 115200,
        "protocol": "json",
        "arduino_type": "uno",
        "reset_on_connect": True
    }
    
    adapter = create_adapter("arduino", config)
    
    # Connect to Arduino
    if adapter.connect():
        logger.info("Successfully connected to Arduino")
        
        # Setup a servo on pin a9
        if adapter.setup_servo(9):
            logger.info("Servo setup successful")
            
            # Move servo to different positions
            for angle in [0, 45, 90, 135, 180, 90]:
                logger.info(f"Setting servo angle to {angle} degrees")
                adapter.set_servo_angle(9, angle)
                time.sleep(1)
        
        # Set up digital and analog pins
        adapter.set_pin_mode(13, "output")  # LED pin
        adapter.set_pin_mode(2, "input_pullup")  # Button pin
        adapter.set_pin_mode(A0, "analog")  # Analog pin
        
        # Blink LED
        for _ in range(5):
            adapter.digital_write(13, True)
            time.sleep(0.5)
            adapter.digital_write(13, False)
            time.sleep(0.5)
        
        # Read button state
        button_state = adapter.digital_read(2)
        logger.info(f"Button state: {'Pressed' if button_state is False else 'Released'}")
        
        # Read analog value
        analog_value = adapter.analog_read(A0)
        logger.info(f"Analog reading: {analog_value}")
        
        # Disconnect
        adapter.disconnect()
        logger.info("Disconnected from Arduino")
    else:
        logger.error("Failed to connect to Arduino")

def demonstrate_raspberry_pi_adapter():
    """Demonstrate the use of RPiAdapter."""
    logger.info("Testing Raspberry Pi adapter...")
    
    # Create Raspberry Pi adapter with sample configuration
    config = {
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
            },
            "button": {
                "pin": 4,
                "type": "input_pullup"
            }
        }
    }
    
    adapter = create_adapter("raspberry_pi", config)
    
    # Connect to Raspberry Pi GPIO
    if adapter.connect():
        logger.info("Successfully connected to Raspberry Pi GPIO")
        
        # Drive forward
        logger.info("Driving forward for 2 seconds")
        adapter.send_command("move_forward", speed=0.7, duration=2.0)
        
        # Turn left
        logger.info("Turning left for 1 second")
        adapter.send_command("turn_left", speed=0.5, duration=1.0)
        
        # Turn right
        logger.info("Turning right for 1 second")
        adapter.send_command("turn_right", speed=0.5, duration=1.0)
        
        # Stop
        logger.info("Stopping")
        adapter.send_command("stop")
        
        # Read sensors
        sensor_data = adapter.read_sensors()
        logger.info(f"Sensor data: {sensor_data}")
        
        # Disconnect
        adapter.disconnect()
        logger.info("Disconnected from Raspberry Pi GPIO")
    else:
        logger.error("Failed to connect to Raspberry Pi GPIO")

def demonstrate_ros_adapter():
    """Demonstrate the use of ROSAdapter."""
    logger.info("Testing ROS adapter...")
    
    # Create ROS adapter with sample configuration
    config = {
        "ros_master_uri": "http://localhost:11311",
        "node_name": "openrobotics_demo",
        "cmd_vel_topic": "/cmd_vel",
        "odom_topic": "/odom",
        "sensors": {
            "lidar": {
                "topic": "/scan",
                "msg_type": "LaserScan"
            },
            "camera": {
                "topic": "/camera/image_raw",
                "msg_type": "Image"
            }
        }
    }
    
    adapter = create_adapter("ros", config)
    
    # Connect to ROS
    if adapter.connect():
        logger.info("Successfully connected to ROS")
        
        # Drive forward
        logger.info("Driving forward")
        adapter.send_command("move_forward", speed=0.2)
        time.sleep(2)
        
        # Turn left
        logger.info("Turning left")
        adapter.send_command("turn_left", angular_speed=0.5)
        time.sleep(2)
        
        # Stop
        logger.info("Stopping")
        adapter.send_command("stop")
        
        # Read sensors
        sensor_data = adapter.read_sensors()
        logger.info(f"Position: {sensor_data.get('position')}")
        if 'lidar' in sensor_data:
            logger.info(f"LIDAR readings: {len(sensor_data['lidar']['ranges'])} points")
        
        # Disconnect
        adapter.disconnect()
        logger.info("Disconnected from ROS")
    else:
        logger.error("Failed to connect to ROS")

def demonstrate_esp32_adapter():
    """Demonstrate the use of ESP32Adapter."""
    logger.info("Testing ESP32 adapter...")
    
    # Create ESP32 adapter with sample configuration
    config = {
        "port": "/dev/ttyUSB0",  # Replace with your actual port
        "baud_rate": 115200,
        "protocol": "json",
        "reset_on_connect": True,
        "wifi_config": {
            "ssid": "YourWiFiNetwork",
            "password": "YourWiFiPassword"
        }
    }
    
    adapter = create_adapter("esp32", config)
    
    # Connect to ESP32
    if adapter.connect():
        logger.info("Successfully connected to ESP32")
        
        # Configure GPIO pins
        adapter.set_pin_mode(2, "output")  # Built-in LED on most ESP32 boards
        adapter.set_pin_mode(4, "input_pullup")  # Example input pin
        
        # Set up LEDC PWM
        adapter.ledc_setup(channel=0, frequency=5000, resolution=8)
        adapter.ledc_attach_pin(pin=5, channel=0)
        
        # Blink LED
        for _ in range(5):
            adapter.digital_write(2, True)
            time.sleep(0.5)
            adapter.digital_write(2, False)
            time.sleep(0.5)
        
        # PWM fade
        logger.info("Fading LED with PWM")
        for duty in range(0, 256, 5):
            adapter.ledc_write(channel=0, duty=duty)
            time.sleep(0.05)
        for duty in range(255, -1, -5):
            adapter.ledc_write(channel=0, duty=duty)
            time.sleep(0.05)
        
        # Read input pin
        pin_state = adapter.digital_read(4)
        logger.info(f"Pin state: {pin_state}")
        
        # Read ADC
        adc_value = adapter.analog_read(34)  # Common ADC pin on ESP32
        logger.info(f"ADC reading: {adc_value}")
        
        # Disconnect
        adapter.disconnect()
        logger.info("Disconnected from ESP32")
    else:
        logger.error("Failed to connect to ESP32")

if __name__ == "__main__":
    try:
        # Choose which adapter to demonstrate (uncomment as needed)
        # demonstrate_arduino_adapter()
        # demonstrate_raspberry_pi_adapter()
        # demonstrate_ros_adapter()
        demonstrate_esp32_adapter()
        
    except KeyboardInterrupt:
        logger.info("Example terminated by user")
    except Exception as e:
        logger.exception(f"Error in example: {e}") 
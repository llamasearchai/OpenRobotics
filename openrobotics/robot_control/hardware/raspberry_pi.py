"""
Raspberry Pi hardware adapter for direct GPIO control.
"""

import logging
import time
import threading
from typing import Dict, List, Any, Optional, Union

from openrobotics.robot_control.hardware.base import HardwareAdapter

logger = logging.getLogger(__name__)

class RPiAdapter(HardwareAdapter):
    """Adapter for Raspberry Pi GPIO-based robots."""
    
    def __init__(self, config: Dict[str, Any] = None):
        """
        Initialize Raspberry Pi adapter.
        
        Args:
            config: Hardware configuration with the following options:
                - motor_pins: Dict mapping motor names to GPIO pins (e.g. {"left": {"forward": 17, "backward": 27}, "right": {"forward": 22, "backward": 23}})
                - pwm_pins: Dict mapping motor names to PWM pins (e.g. {"left": 18, "right": 13})
                - sensor_pins: Dict mapping sensor names to GPIO pins
                - i2c_sensors: Dict mapping sensor names to I2C configurations
                - pwm_freq: PWM frequency in Hz (default: 1000)
        """
        super().__init__(config)
        
        # Set default configuration
        self.config.setdefault("motor_pins", {})
        self.config.setdefault("pwm_pins", {})
        self.config.setdefault("sensor_pins", {})
        self.config.setdefault("i2c_sensors", {})
        self.config.setdefault("pwm_freq", 1000)
        
        # Internal state
        self.gpio = None
        self.i2c = None
        self.pwm_controllers = {}
        self.sensor_data = {}
        self.connected = False
        self.sensor_thread = None
        self.running = False
        
        # Try to import RPi.GPIO
        try:
            import RPi.GPIO as GPIO
            self.gpio = GPIO
            self.gpio_available = True
        except ImportError:
            logger.warning("RPi.GPIO module not found. RPi adapter will be simulated.")
            self.gpio_available = False
        
        # Try to import smbus for I2C
        try:
            import smbus
            self.i2c = smbus
            self.i2c_available = True
        except ImportError:
            logger.warning("smbus module not found. I2C sensors will be simulated.")
            self.i2c_available = False
    
    def connect(self) -> bool:
        """
        Connect to Raspberry Pi GPIO.
        
        Returns:
            True if connection successful
        """
        if not self.gpio_available:
            logger.warning("GPIO not available. Running in simulation mode.")
            self.connected = True
            return True
            
        try:
            # Set GPIO mode
            self.gpio.setmode(self.gpio.BCM)
            
            # Set up motor pins
            for motor_name, pins in self.config["motor_pins"].items():
                # Set up forward and backward pins
                if "forward" in pins:
                    self.gpio.setup(pins["forward"], self.gpio.OUT)
                    self.gpio.output(pins["forward"], self.gpio.LOW)
                
                if "backward" in pins:
                    self.gpio.setup(pins["backward"], self.gpio.OUT)
                    self.gpio.output(pins["backward"], self.gpio.LOW)
            
            # Set up PWM pins
            for motor_name, pin in self.config["pwm_pins"].items():
                self.gpio.setup(pin, self.gpio.OUT)
                
                # Create PWM controller
                pwm = self.gpio.PWM(pin, self.config["pwm_freq"])
                pwm.start(0)  # Start with 0% duty cycle
                self.pwm_controllers[motor_name] = pwm
            
            # Set up sensor pins
            for sensor_name, pin_config in self.config["sensor_pins"].items():
                pin = pin_config["pin"]
                pin_type = pin_config.get("type", "input")
                
                if pin_type == "input":
                    self.gpio.setup(pin, self.gpio.IN)
                elif pin_type == "input_pullup":
                    self.gpio.setup(pin, self.gpio.IN, pull_up_down=self.gpio.PUD_UP)
                elif pin_type == "input_pulldown":
                    self.gpio.setup(pin, self.gpio.IN, pull_up_down=self.gpio.PUD_DOWN)
            
            # Set up I2C sensors
            if self.i2c_available and self.config["i2c_sensors"]:
                # Initialize I2C bus
                bus_num = self.config.get("i2c_bus", 1)  # Default to bus 1
                self.i2c_bus = self.i2c.SMBus(bus_num)
                
                # Initialize I2C sensors
                for sensor_name, i2c_config in self.config["i2c_sensors"].items():
                    sensor_type = i2c_config.get("type", "")
                    address = i2c_config.get("address", 0)
                    
                    # Initialize sensor based on type
                    if sensor_type == "bme280":
                        self._init_bme280(sensor_name, address)
                    elif sensor_type == "mpu6050":
                        self._init_mpu6050(sensor_name, address)
                    # Add more sensor initializations as needed
            
            # Start sensor reading thread
            self.running = True
            self.sensor_thread = threading.Thread(
                target=self._sensor_reading_loop,
                daemon=True
            )
            self.sensor_thread.start()
            
            # Mark as connected
            self.connected = True
            logger.info("Connected to Raspberry Pi GPIO")
            
            return True
        
        except Exception as e:
            logger.error(f"Error connecting to Raspberry Pi GPIO: {e}")
            
            # Clean up if error
            if self.gpio_available:
                self.gpio.cleanup()
            
            return False
    
    def disconnect(self) -> bool:
        """
        Disconnect from Raspberry Pi GPIO.
        
        Returns:
            True if disconnection successful
        """
        if not self.gpio_available:
            self.connected = False
            return True
            
        try:
            # Stop sensor thread
            self.running = False
            if self.sensor_thread:
                self.sensor_thread.join(timeout=1.0)
            
            # Stop PWM controllers
            for pwm in self.pwm_controllers.values():
                pwm.stop()
            
            # Clean up GPIO
            self.gpio.cleanup()
            
            # Clear state
            self.pwm_controllers = {}
            self.connected = False
            
            logger.info("Disconnected from Raspberry Pi GPIO")
            return True
        
        except Exception as e:
            logger.error(f"Error disconnecting from Raspberry Pi GPIO: {e}")
            return False
    
    def send_command(self, command: str, **params) -> Dict[str, Any]:
        """
        Send command to Raspberry Pi robot.
        
        Args:
            command: Command name (move_forward, move_backward, turn_left, turn_right, stop)
            params: Command parameters (speed, duration)
            
        Returns:
            Command result
        """
        if not self.gpio_available:
            # Simulate command execution
            logger.info(f"Simulating RPi command: {command} with params {params}")
            return {"success": True, "command": command, "params": params}
            
        try:
            speed = params.get("speed", 0.5)  # 0.0 to 1.0
            duration = params.get("duration", 0.0)  # seconds, 0 for continuous
            
            # Convert speed to PWM duty cycle (0-100)
            duty_cycle = int(speed * 100)
            
            if command == "move_forward":
                self._set_motors(duty_cycle, duty_cycle)
            elif command == "move_backward":
                self._set_motors(-duty_cycle, -duty_cycle)
            elif command == "turn_left":
                self._set_motors(-duty_cycle, duty_cycle)
            elif command == "turn_right":
                self._set_motors(duty_cycle, -duty_cycle)
            elif command == "stop":
                self._set_motors(0, 0)
            else:
                return {"success": False, "error": f"Unknown command: {command}"}
            
            # Run for specified duration if provided
            if duration > 0:
                time.sleep(duration)
                self._set_motors(0, 0)
            
            return {
                "success": True,
                "command": command,
                "params": params
            }
        
        except Exception as e:
            logger.error(f"Error sending command to Raspberry Pi: {e}")
            return {"success": False, "error": str(e)}
    
    def read_sensors(self) -> Dict[str, Any]:
        """
        Read sensor data from Raspberry Pi.
        
        Returns:
            Dictionary of sensor data
        """
        # Return latest sensor data
        return self.sensor_data.copy()
    
    def _set_motors(self, left_speed: int, right_speed: int):
        """
        Set motor speeds.
        
        Args:
            left_speed: Left motor speed (-100 to 100)
            right_speed: Right motor speed (-100 to 100)
        """
        # Set left motor
        if "left" in self.config["motor_pins"]:
            pins = self.config["motor_pins"]["left"]
            
            if left_speed > 0:
                # Forward
                if "forward" in pins:
                    self.gpio.output(pins["forward"], self.gpio.HIGH)
                if "backward" in pins:
                    self.gpio.output(pins["backward"], self.gpio.LOW)
            elif left_speed < 0:
                # Backward
                if "forward" in pins:
                    self.gpio.output(pins["forward"], self.gpio.LOW)
                if "backward" in pins:
                    self.gpio.output(pins["backward"], self.gpio.HIGH)
            else:
                # Stop
                if "forward" in pins:
                    self.gpio.output(pins["forward"], self.gpio.LOW)
                if "backward" in pins:
                    self.gpio.output(pins["backward"], self.gpio.LOW)
            
            # Set PWM
            if "left" in self.pwm_controllers:
                self.pwm_controllers["left"].ChangeDutyCycle(abs(left_speed))
        
        # Set right motor
        if "right" in self.config["motor_pins"]:
            pins = self.config["motor_pins"]["right"]
            
            if right_speed > 0:
                # Forward
                if "forward" in pins:
                    self.gpio.output(pins["forward"], self.gpio.HIGH)
                if "backward" in pins:
                    self.gpio.output(pins["backward"], self.gpio.LOW)
            elif right_speed < 0:
                # Backward
                if "forward" in pins:
                    self.gpio.output(pins["forward"], self.gpio.LOW)
                if "backward" in pins:
                    self.gpio.output(pins["backward"], self.gpio.HIGH)
            else:
                # Stop
                if "forward" in pins:
                    self.gpio.output(pins["forward"], self.gpio.LOW)
                if "backward" in pins:
                    self.gpio.output(pins["backward"], self.gpio.LOW)
            
            # Set PWM
            if "right" in self.pwm_controllers:
                self.pwm_controllers["right"].ChangeDutyCycle(abs(right_speed))
    
    def _sensor_reading_loop(self):
        """Background thread for continuous sensor reading."""
        while self.running:
            try:
                # Read digital sensors
                for sensor_name, pin_config in self.config["sensor_pins"].items():
                    pin = pin_config["pin"]
                    sensor_type = pin_config.get("type", "digital")
                    
                    if sensor_type == "digital" or sensor_type.startswith("input"):
                        # Read digital input
                        value = self.gpio.input(pin)
                        self.sensor_data[sensor_name] = {
                            "value": value,
                            "timestamp": time.time()
                        }
                    elif sensor_type == "ultrasonic" and "echo_pin" in pin_config:
                        # Read ultrasonic sensor
                        trig_pin = pin
                        echo_pin = pin_config["echo_pin"]
                        distance = self._read_ultrasonic(trig_pin, echo_pin)
                        self.sensor_data[sensor_name] = {
                            "distance": distance,
                            "timestamp": time.time()
                        }
                
                # Read I2C sensors
                if self.i2c_available:
                    for sensor_name, i2c_config in self.config["i2c_sensors"].items():
                        sensor_type = i2c_config.get("type", "")
                        address = i2c_config.get("address", 0)
                        
                        if sensor_type == "bme280":
                            self._read_bme280(sensor_name, address)
                        elif sensor_type == "mpu6050":
                            self._read_mpu6050(sensor_name, address)
                        # Add more sensor reading as needed
            
            except Exception as e:
                logger.error(f"Error reading sensors: {e}")
            
            # Sleep to avoid CPU overuse
            time.sleep(0.1)
    
    def _read_ultrasonic(self, trig_pin: int, echo_pin: int) -> float:
        """
        Read distance from ultrasonic sensor.
        
        Args:
            trig_pin: Trigger pin
            echo_pin: Echo pin
            
        Returns:
            Distance in centimeters
        """
        # Send trigger pulse
        self.gpio.output(trig_pin, self.gpio.LOW)
        time.sleep(0.00001)  # 10 microseconds
        self.gpio.output(trig_pin, self.gpio.HIGH)
        time.sleep(0.00001)  # 10 microseconds
        self.gpio.output(trig_pin, self.gpio.LOW)
        
        # Measure echo pulse duration
        pulse_start = time.time()
        pulse_end = time.time()
        
        # Wait for echo to go HIGH
        timeout_start = time.time()
        while self.gpio.input(echo_pin) == self.gpio.LOW:
            pulse_start = time.time()
            if pulse_start - timeout_start > 0.1:  # 100ms timeout
                return 0.0  # Timeout, return 0 distance
        
        # Wait for echo to go LOW
        timeout_start = time.time()
        while self.gpio.input(echo_pin) == self.gpio.HIGH:
            pulse_end = time.time()
            if pulse_end - timeout_start > 0.1:  # 100ms timeout
                return 0.0  # Timeout, return 0 distance
        
        # Calculate distance
        pulse_duration = pulse_end - pulse_start
        distance = pulse_duration * 17150  # Speed of sound = 343 m/s = 34300 cm/s
                                          # Distance = (time * speed) / 2
        
        return distance
    
    def _init_bme280(self, sensor_name: str, address: int):
        """Initialize BME280 temperature/humidity/pressure sensor."""
        # This is a simplified initialization that would need to be expanded
        # with actual BME280 register setup for a real implementation
        pass
    
    def _read_bme280(self, sensor_name: str, address: int):
        """Read data from BME280 sensor."""
        # This is a simplified reading that would need to be expanded
        # with actual BME280 register reading for a real implementation
        
        # Simulate readings for now
        self.sensor_data[sensor_name] = {
            "temperature": 25.0 + 2.0 * (0.5 - (time.time() % 60) / 60),
            "humidity": 50.0 + 10.0 * (0.5 - (time.time() % 120) / 120),
            "pressure": 101325.0 + 100.0 * (0.5 - (time.time() % 300) / 300),
            "timestamp": time.time()
        }
    
    def _init_mpu6050(self, sensor_name: str, address: int):
        """Initialize MPU6050 IMU sensor."""
        # This is a simplified initialization that would need to be expanded
        # with actual MPU6050 register setup for a real implementation
        pass
    
    def _read_mpu6050(self, sensor_name: str, address: int):
        """Read data from MPU6050 sensor."""
        # This is a simplified reading that would need to be expanded
        # with actual MPU6050 register reading for a real implementation
        
        # Simulate readings for now
        self.sensor_data[sensor_name] = {
            "accelerometer": {
                "x": 0.1 * (0.5 - (time.time() % 10) / 10),
                "y": 0.1 * (0.5 - (time.time() % 15) / 15),
                "z": 9.81 + 0.05 * (0.5 - (time.time() % 5) / 5)
            },
            "gyroscope": {
                "x": 0.5 * (0.5 - (time.time() % 8) / 8),
                "y": 0.5 * (0.5 - (time.time() % 12) / 12),
                "z": 0.5 * (0.5 - (time.time() % 6) / 6)
            },
            "temperature": 25.0 + 0.5 * (0.5 - (time.time() % 30) / 30),
            "timestamp": time.time()
        } 
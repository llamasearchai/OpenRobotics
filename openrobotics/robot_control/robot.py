"""
Robot control for physical or simulated robots.
"""

import logging
import time
from typing import Dict, List, Any, Optional, Union

from openrobotics.robot_control.sensors import Sensor, SensorArray
from openrobotics.robot_control.actuators import Actuator, Motor, Servo
from openrobotics.robot_control.hardware import HardwareAdapter, create_adapter

logger = logging.getLogger(__name__)

class Robot:
    """Base robot class for controlling physical or simulated robots."""
    
    def __init__(self, config: Dict[str, Any] = None):
        """
        Initialize robot with configuration.
        
        Args:
            config: Robot configuration:
                - name: Robot name
                - hardware_adapter: Hardware adapter configuration
                - sensors: Sensor configurations
                - actuators: Actuator configurations
                - parameters: Robot parameters
        """
        self.config = config or {}
        
        # Set default configuration
        self.config.setdefault("name", "robot")
        self.config.setdefault("hardware_adapter", {"type": "simulation"})
        self.config.setdefault("sensors", {})
        self.config.setdefault("actuators", {})
        self.config.setdefault("parameters", {})
        
        # Internal state
        self.hardware_adapter = None
        self.sensors = {}
        self.actuators = {}
        self.state = {
            "position": {"x": 0.0, "y": 0.0, "theta": 0.0},
            "velocity": {"linear": 0.0, "angular": 0.0},
            "battery": 1.0,
            "status": "initialized",
            "error": None
        }
        self.connected = False
        
        # Initialize hardware adapter
        self._init_hardware_adapter()
        
        # Initialize sensors and actuators
        self._init_sensors()
        self._init_actuators()
    
    def _init_hardware_adapter(self):
        """Initialize hardware adapter based on configuration."""
        adapter_config = self.config["hardware_adapter"]
        adapter_type = adapter_config.get("type", "simulation")
        
        try:
            # Create appropriate hardware adapter
            if adapter_type == "simulation":
                # For simulation, we don't create an actual adapter
                # The simulation environment will handle this
                pass
            else:
                # Create hardware adapter using factory
                self.hardware_adapter = create_adapter(adapter_type, adapter_config)
                logger.info(f"Created {adapter_type} hardware adapter")
        
        except Exception as e:
            logger.error(f"Error initializing hardware adapter: {e}")
            self.state["error"] = f"Hardware adapter initialization failed: {str(e)}"
    
    def _init_sensors(self):
        """Initialize sensors based on configuration."""
        sensor_configs = self.config.get("sensors", {})
        
        for sensor_name, sensor_config in sensor_configs.items():
            try:
                sensor_type = sensor_config.get("type", "generic")
                
                # Create sensor
                sensor = Sensor(sensor_name, sensor_type, sensor_config)
                self.sensors[sensor_name] = sensor
                
                logger.info(f"Initialized sensor '{sensor_name}' of type '{sensor_type}'")
            
            except Exception as e:
                logger.error(f"Error initializing sensor '{sensor_name}': {e}")
    
    def _init_actuators(self):
        """Initialize actuators based on configuration."""
        actuator_configs = self.config.get("actuators", {})
        
        for actuator_name, actuator_config in actuator_configs.items():
            try:
                actuator_type = actuator_config.get("type", "generic")
                
                # Create appropriate actuator type
                if actuator_type == "motor":
                    actuator = Motor(actuator_name, actuator_config)
                elif actuator_type == "servo":
                    actuator = Servo(actuator_name, actuator_config)
                else:
                    actuator = Actuator(actuator_name, actuator_type, actuator_config)
                
                self.actuators[actuator_name] = actuator
                
                logger.info(f"Initialized actuator '{actuator_name}' of type '{actuator_type}'")
            
            except Exception as e:
                logger.error(f"Error initializing actuator '{actuator_name}': {e}")
    
    def connect(self) -> bool:
        """
        Connect to hardware.
        
        Returns:
            True if connection successful
        """
        if self.connected:
            logger.warning("Robot already connected")
            return True
        
        try:
            # If using simulation, there's no need to connect
            if self.hardware_adapter is None:
                self.connected = True
                self.state["status"] = "connected (simulation)"
                return True
            
            # Connect hardware adapter
            if self.hardware_adapter.connect():
                self.connected = True
                self.state["status"] = "connected"
                logger.info(f"Robot '{self.config['name']}' connected to hardware")
                return True
            else:
                self.state["status"] = "connection failed"
                self.state["error"] = "Failed to connect to hardware"
                logger.error(f"Failed to connect robot '{self.config['name']}' to hardware")
                return False
        
        except Exception as e:
            self.state["status"] = "connection error"
            self.state["error"] = str(e)
            logger.error(f"Error connecting robot '{self.config['name']}': {e}")
            return False
    
    def disconnect(self) -> bool:
        """
        Disconnect from hardware.
        
        Returns:
            True if disconnection successful
        """
        if not self.connected:
            return True
        
        try:
            # If using simulation, there's no need to disconnect
            if self.hardware_adapter is None:
                self.connected = False
                self.state["status"] = "disconnected (simulation)"
                return True
            
            # Stop all actuators before disconnecting
            self.stop()
            
            # Disconnect hardware adapter
            if self.hardware_adapter.disconnect():
                self.connected = False
                self.state["status"] = "disconnected"
                logger.info(f"Robot '{self.config['name']}' disconnected from hardware")
                return True
            else:
                self.state["error"] = "Failed to disconnect from hardware"
                logger.error(f"Failed to disconnect robot '{self.config['name']}' from hardware")
                return False
        
        except Exception as e:
            self.state["error"] = str(e)
            logger.error(f"Error disconnecting robot '{self.config['name']}': {e}")
            return False
    
    def move(self, linear_speed: float, angular_speed: float) -> bool:
        """
        Move robot with specified linear and angular speeds.
        
        Args:
            linear_speed: Linear speed in m/s (positive = forward, negative = backward)
            angular_speed: Angular speed in rad/s (positive = left, negative = right)
            
        Returns:
            True if command successful
        """
        if not self.connected:
            logger.error("Cannot move: Robot not connected")
            return False
        
        try:
            # Update state
            self.state["velocity"]["linear"] = linear_speed
            self.state["velocity"]["angular"] = angular_speed
            
            # If using simulation, we just update the state
            if self.hardware_adapter is None:
                return True
            
            # Determine command based on direction
            if linear_speed > 0 and abs(angular_speed) < 0.1:
                command = "move_forward"
                params = {"speed": linear_speed}
            elif linear_speed < 0 and abs(angular_speed) < 0.1:
                command = "move_backward"
                params = {"speed": -linear_speed}
            elif angular_speed > 0 and abs(linear_speed) < 0.1:
                command = "turn_left"
                params = {"angular_speed": angular_speed}
            elif angular_speed < 0 and abs(linear_speed) < 0.1:
                command = "turn_right"
                params = {"angular_speed": -angular_speed}
            else:
                # Combined movement (use raw command)
                command = "move"
                params = {"linear": linear_speed, "angular": angular_speed}
            
            # Send command to hardware adapter
            result = self.hardware_adapter.send_command(command, **params)
            
            if not result.get("success", False):
                error = result.get("error", "Unknown error")
                logger.error(f"Move command failed: {error}")
                return False
            
            return True
        
        except Exception as e:
            logger.error(f"Error executing move command: {e}")
            return False
    
    def stop(self) -> bool:
        """
        Stop robot movement.
        
        Returns:
            True if command successful
        """
        if not self.connected:
            logger.warning("Cannot stop: Robot not connected")
            return False
        
        try:
            # Update state
            self.state["velocity"]["linear"] = 0.0
            self.state["velocity"]["angular"] = 0.0
            
            # If using simulation, we just update the state
            if self.hardware_adapter is None:
                return True
            
            # Send stop command to hardware adapter
            result = self.hardware_adapter.send_command("stop")
            
            if not result.get("success", False):
                error = result.get("error", "Unknown error")
                logger.error(f"Stop command failed: {error}")
                return False
            
            return True
        
        except Exception as e:
            logger.error(f"Error executing stop command: {e}")
            return False
    
    def read_sensors(self) -> Dict[str, Any]:
        """
        Read all sensor data.
        
        Returns:
            Dictionary of sensor readings
        """
        sensor_data = {}
        
        try:
            # If using simulation, we just return the current state
            if self.hardware_adapter is None:
                return {
                    "position": self.state["position"].copy(),
                    "timestamp": time.time()
                }
            
            # Read from hardware adapter
            hardware_data = self.hardware_adapter.read_sensors()
            
            # Update position from hardware data if available
            if "position" in hardware_data:
                self.state["position"] = hardware_data["position"]
            
            # Process data for each sensor
            for sensor_name, sensor in self.sensors.items():
                # Get raw data from hardware for this sensor
                raw_data = hardware_data.get(sensor_name, {})
                
                # Process through sensor object
                processed_data = sensor.process_reading(raw_data)
                
                # Add to sensor data
                sensor_data[sensor_name] = processed_data
            
            # Add position data
            sensor_data["position"] = self.state["position"].copy()
            sensor_data["timestamp"] = time.time()
            
            return sensor_data
        
        except Exception as e:
            logger.error(f"Error reading sensors: {e}")
            return {"error": str(e), "timestamp": time.time()}
    
    def update_state_from_simulation(self, pose: Dict[str, float], sensor_data: Dict[str, Any]) -> None:
        """
        Update robot state from simulation data.
        This method is called by the simulation environment to update the robot's state.
        
        Args:
            pose: Robot pose (x, y, theta)
            sensor_data: Sensor readings
        """
        # Update position
        self.state["position"] = pose.copy()
        
        # Update sensor readings
        for sensor_name, sensor in self.sensors.items():
            if sensor_name in sensor_data:
                sensor.last_reading = sensor_data[sensor_name]
    
    def get_state(self) -> Dict[str, Any]:
        """
        Get robot state.
        
        Returns:
            Robot state
        """
        return self.state.copy()
    
    def set_actuator(self, actuator_name: str, value: Any) -> bool:
        """
        Set actuator value.
        
        Args:
            actuator_name: Actuator name
            value: Actuator value
            
        Returns:
            True if command successful
        """
        if not self.connected:
            logger.error(f"Cannot set actuator '{actuator_name}': Robot not connected")
            return False
        
        if actuator_name not in self.actuators:
            logger.error(f"Actuator '{actuator_name}' not found")
            return False
        
        try:
            actuator = self.actuators[actuator_name]
            
            # Update actuator state
            actuator.set_value(value)
            
            # If using simulation, we just update the actuator state
            if self.hardware_adapter is None:
                return True
            
            # Send command to hardware adapter
            if actuator.type == "motor":
                result = self.hardware_adapter.send_command(
                    "set_motor",
                    name=actuator_name,
                    speed=value
                )
            elif actuator.type == "servo":
                result = self.hardware_adapter.send_command(
                    "set_servo",
                    name=actuator_name,
                    angle=value
                )
            else:
                result = self.hardware_adapter.send_command(
                    "set_actuator",
                    name=actuator_name,
                    value=value
                )
            
            if not result.get("success", False):
                error = result.get("error", "Unknown error")
                logger.error(f"Set actuator command failed: {error}")
                return False
            
            return True
        
        except Exception as e:
            logger.error(f"Error setting actuator '{actuator_name}': {e}")
            return False 
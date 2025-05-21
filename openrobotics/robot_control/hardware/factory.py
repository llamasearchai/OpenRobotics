"""
Factory for creating hardware adapters.
"""

import logging
from typing import Dict, Any, Optional

from openrobotics.robot_control.hardware.base import HardwareAdapter
from openrobotics.robot_control.hardware.ros_adapter import ROSAdapter
from openrobotics.robot_control.hardware.raspberry_pi import RPiAdapter
from openrobotics.robot_control.hardware.serial_adapter import SerialAdapter
from openrobotics.robot_control.hardware.arduino_adapter import ArduinoAdapter
from openrobotics.robot_control.hardware.esp32_adapter import ESP32Adapter

logger = logging.getLogger(__name__)

def create_adapter(adapter_type: str, config: Optional[Dict[str, Any]] = None) -> HardwareAdapter:
    """
    Create a hardware adapter of the specified type.
    
    Args:
        adapter_type: Type of adapter (ros, raspberry_pi, serial, arduino, esp32)
        config: Adapter configuration
        
    Returns:
        Hardware adapter instance
        
    Raises:
        ValueError: If adapter type is unknown
    """
    config = config or {}
    
    if adapter_type == "ros":
        return ROSAdapter(config)
    elif adapter_type == "raspberry_pi":
        return RPiAdapter(config)
    elif adapter_type == "serial":
        return SerialAdapter(config)
    elif adapter_type == "arduino":
        return ArduinoAdapter(config)
    elif adapter_type == "esp32":
        return ESP32Adapter(config)
    else:
        raise ValueError(f"Unknown adapter type: {adapter_type}") 
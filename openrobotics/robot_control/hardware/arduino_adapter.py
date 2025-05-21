"""
Arduino adapter for communicating with Arduino-based robots.
"""

import logging
import time
from typing import Dict, List, Any, Optional, Union

from openrobotics.robot_control.hardware.serial_adapter import SerialAdapter

logger = logging.getLogger(__name__)

class ArduinoAdapter(SerialAdapter):
    """Adapter for communicating with Arduino-based robots."""
    
    def __init__(self, config: Dict[str, Any] = None):
        """
        Initialize Arduino adapter.
        
        Args:
            config: Configuration with the following options:
                - port: Serial port (e.g., /dev/ttyACM0, COM3)
                - baud_rate: Baud rate (default: 115200)
                - timeout: Read timeout in seconds (default: 1.0)
                - protocol: Communication protocol (json, binary, text)
                - arduino_type: Type of Arduino (uno, mega, nano, etc.)
                - reset_on_connect: Reset Arduino on connect (default: True)
        """
        # Set default Arduino configuration
        config = config or {}
        config.setdefault("protocol", "json")
        config.setdefault("arduino_type", "uno")
        config.setdefault("reset_on_connect", True)
        
        # Initialize parent class
        super().__init__(config)
        
        # Initialize Arduino-specific state
        self.firmware_info = None
        self.capabilities = None
        self.servo_pins = {}
    
    def connect(self) -> bool:
        """
        Connect to Arduino device.
        
        Returns:
            True if connection successful
        """
        # Connect using parent's method
        result = super().connect()
        
        if not result or not self.serial_available:
            return result
        
        try:
            # Reset Arduino if requested (using DTR pin toggling)
            if self.config["reset_on_connect"]:
                self.serial.setDTR(False)
                time.sleep(0.1)
                self.serial.setDTR(True)
                time.sleep(2.0)  # Wait for Arduino to boot
            
            # Query firmware information
            response = self.send_command("firmware_info")
            if response.get("success") and "response" in response:
                self.firmware_info = response["response"]
                logger.info(f"Connected to Arduino firmware: {self.firmware_info}")
            
            # Query capabilities
            response = self.send_command("capabilities")
            if response.get("success") and "response" in response:
                self.capabilities = response["response"]
                logger.info(f"Arduino capabilities: {self.capabilities}")
            
            return True
        
        except Exception as e:
            logger.error(f"Error during Arduino initialization: {e}")
            self.disconnect()
            return False
    
    def setup_servo(self, pin: int, min_pulse: int = 544, max_pulse: int = 2400) -> bool:
        """
        Set up a servo motor on the specified pin.
        
        Args:
            pin: Arduino pin number
            min_pulse: Minimum pulse width in microseconds (default: 544)
            max_pulse: Maximum pulse width in microseconds (default: 2400)
            
        Returns:
            True if successful
        """
        response = self.send_command("servo_setup", pin=pin, min_pulse=min_pulse, max_pulse=max_pulse)
        
        if response.get("success") and response.get("response", {}).get("result") == "ok":
            self.servo_pins[pin] = {"min_pulse": min_pulse, "max_pulse": max_pulse}
            return True
        
        return False
    
    def set_servo_angle(self, pin: int, angle: float) -> bool:
        """
        Set servo angle.
        
        Args:
            pin: Arduino pin number
            angle: Angle in degrees (0-180)
            
        Returns:
            True if successful
        """
        if pin not in self.servo_pins:
            logger.error(f"Servo on pin {pin} not set up")
            return False
        
        response = self.send_command("servo_write", pin=pin, angle=angle)
        
        return response.get("success") and response.get("response", {}).get("result") == "ok"
    
    def set_pin_mode(self, pin: int, mode: str) -> bool:
        """
        Set Arduino pin mode.
        
        Args:
            pin: Arduino pin number
            mode: Pin mode (input, output, input_pullup, pwm, analog)
            
        Returns:
            True if successful
        """
        response = self.send_command("pin_mode", pin=pin, mode=mode)
        
        return response.get("success") and response.get("response", {}).get("result") == "ok"
    
    def digital_write(self, pin: int, value: bool) -> bool:
        """
        Write digital value to pin.
        
        Args:
            pin: Arduino pin number
            value: Digital value (True/False)
            
        Returns:
            True if successful
        """
        response = self.send_command("digital_write", pin=pin, value=1 if value else 0)
        
        return response.get("success") and response.get("response", {}).get("result") == "ok"
    
    def analog_write(self, pin: int, value: int) -> bool:
        """
        Write analog (PWM) value to pin.
        
        Args:
            pin: Arduino pin number
            value: Analog value (0-255)
            
        Returns:
            True if successful
        """
        response = self.send_command("analog_write", pin=pin, value=value)
        
        return response.get("success") and response.get("response", {}).get("result") == "ok"
    
    def digital_read(self, pin: int) -> Optional[bool]:
        """
        Read digital value from pin.
        
        Args:
            pin: Arduino pin number
            
        Returns:
            Digital value (True/False) or None if failed
        """
        response = self.send_command("digital_read", pin=pin)
        
        if response.get("success") and "response" in response:
            return response["response"].get("value") == 1
        
        return None
    
    def analog_read(self, pin: int) -> Optional[int]:
        """
        Read analog value from pin.
        
        Args:
            pin: Arduino pin number
            
        Returns:
            Analog value (0-1023) or None if failed
        """
        response = self.send_command("analog_read", pin=pin)
        
        if response.get("success") and "response" in response:
            return response["response"].get("value")
        
        return None 
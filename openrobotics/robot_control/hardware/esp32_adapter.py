"""
ESP32 adapter for communicating with ESP32-based robots.
"""

import logging
import time
import json
from typing import Dict, List, Any, Optional, Union

from openrobotics.robot_control.hardware.serial_adapter import SerialAdapter

logger = logging.getLogger(__name__)

class ESP32Adapter(SerialAdapter):
    """Adapter for communicating with ESP32-based robots."""
    
    def __init__(self, config: Dict[str, Any] = None):
        """
        Initialize ESP32 adapter.
        
        Args:
            config: Configuration with the following options:
                - port: Serial port (e.g., /dev/ttyUSB0, COM3)
                - baud_rate: Baud rate (default: 115200)
                - timeout: Read timeout in seconds (default: 1.0)
                - protocol: Communication protocol (json, binary, text)
                - wifi_config: WiFi configuration for ESP32
                - reset_on_connect: Reset ESP32 on connect (default: True)
                - use_bluetooth: Use Bluetooth instead of serial (default: False)
        """
        # Set default ESP32 configuration
        config = config or {}
        config.setdefault("protocol", "json")
        config.setdefault("baud_rate", 115200)
        config.setdefault("reset_on_connect", True)
        config.setdefault("use_bluetooth", False)
        config.setdefault("wifi_config", {})
        
        # Initialize parent class
        super().__init__(config)
        
        # Initialize ESP32-specific state
        self.firmware_info = None
        self.capabilities = None
        self.gpio_states = {}
        self.wifi_connected = False
        self.bluetooth = None
        
        # Try to import Bluetooth libraries if needed
        if self.config.get("use_bluetooth", False):
            try:
                import bluetooth
                self.bluetooth_module = bluetooth
                self.bluetooth_available = True
            except ImportError:
                logger.warning("Bluetooth module not found. Bluetooth mode will not be available.")
                self.bluetooth_available = False
    
    def connect(self) -> bool:
        """
        Connect to ESP32 device.
        
        Returns:
            True if connection successful
        """
        # If using Bluetooth
        if self.config.get("use_bluetooth", False):
            return self._connect_bluetooth()
        
        # Otherwise use serial connection
        result = super().connect()
        
        if not result or not self.serial_available:
            return result
        
        try:
            # Reset ESP32 if requested (using DTR pin toggling)
            if self.config["reset_on_connect"]:
                self.serial.setDTR(False)
                self.serial.setRTS(True)
                time.sleep(0.1)
                self.serial.setRTS(False)
                self.serial.setDTR(True)
                time.sleep(1.5)  # Wait for ESP32 to boot
            
            # Query firmware information
            response = self.send_command("firmware_info")
            if response.get("success") and "response" in response:
                self.firmware_info = response["response"]
                logger.info(f"Connected to ESP32 firmware: {self.firmware_info}")
            
            # Query capabilities
            response = self.send_command("capabilities")
            if response.get("success") and "response" in response:
                self.capabilities = response["response"]
                logger.info(f"ESP32 capabilities: {self.capabilities}")
            
            # Configure WiFi if needed
            if self.config.get("wifi_config"):
                self._configure_wifi()
            
            return True
        
        except Exception as e:
            logger.error(f"Error during ESP32 initialization: {e}")
            self.disconnect()
            return False
    
    def _connect_bluetooth(self) -> bool:
        """Connect to ESP32 using Bluetooth."""
        if not hasattr(self, 'bluetooth_available') or not self.bluetooth_available:
            logger.error("Bluetooth support not available")
            return False
        
        try:
            import bluetooth
            
            # Get ESP32 Bluetooth address
            bt_addr = self.config.get("bluetooth_address")
            bt_name = self.config.get("bluetooth_name")
            
            if not bt_addr and not bt_name:
                logger.error("Neither Bluetooth address nor name provided")
                return False
            
            # If only name provided, scan for the device
            if not bt_addr and bt_name:
                logger.info(f"Scanning for Bluetooth device with name: {bt_name}")
                devices = bluetooth.discover_devices(lookup_names=True)
                for addr, name in devices:
                    if name == bt_name:
                        bt_addr = addr
                        logger.info(f"Found device {bt_name} with address {bt_addr}")
                        break
                
                if not bt_addr:
                    logger.error(f"Bluetooth device with name {bt_name} not found")
                    return False
            
            # Connect to device
            port = 1  # RFCOMM port
            self.bluetooth = bluetooth.BluetoothSocket(bluetooth.RFCOMM)
            self.bluetooth.connect((bt_addr, port))
            
            # Set timeout
            self.bluetooth.settimeout(self.config.get("timeout", 1.0))
            
            # Start reading thread for Bluetooth
            self.running = True
            self.thread = threading.Thread(
                target=self._bluetooth_read_loop,
                daemon=True
            )
            self.thread.start()
            
            # Mark as connected
            self.connected = True
            logger.info(f"Connected to ESP32 via Bluetooth at {bt_addr}")
            
            return True
        
        except Exception as e:
            logger.error(f"Error connecting to ESP32 via Bluetooth: {e}")
            return False
    
    def _bluetooth_read_loop(self):
        """Background thread for reading from Bluetooth."""
        buffer = b""
        protocol = self.config["protocol"]
        
        while self.running and self.bluetooth:
            try:
                # Read data
                data = self.bluetooth.recv(1024)
                
                if data:
                    # Add to buffer
                    buffer += data
                    
                    # Process data similar to serial processing
                    line_ending = self.config["newline"].encode('utf-8')
                    while line_ending in buffer:
                        line, buffer = buffer.split(line_ending, 1)
                        
                        # Process line
                        try:
                            line_str = line.decode('utf-8').strip()
                            
                            if protocol == "json":
                                data = json.loads(line_str)
                                
                                # Handle different message types
                                if "sensor" in data:
                                    sensor_name = data.pop("sensor")
                                    self.sensor_data[sensor_name] = {
                                        **data,
                                        "timestamp": time.time()
                                    }
                                elif "response" in data:
                                    self.sensor_data["response"] = data
                                else:
                                    self.sensor_data["raw"] = data
                        
                        except Exception as e:
                            logger.error(f"Error processing Bluetooth data: {e}")
            
            except Exception as e:
                logger.error(f"Error reading from Bluetooth: {e}")
                if str(e).startswith("timed out"):
                    # Ignore timeout errors
                    pass
                else:
                    # If connection lost
                    self.connected = False
                    break
            
            # Sleep to avoid CPU overuse
            time.sleep(0.01)
    
    def disconnect(self) -> bool:
        """
        Disconnect from ESP32 device.
        
        Returns:
            True if disconnection successful
        """
        # If using Bluetooth
        if hasattr(self, 'bluetooth') and self.bluetooth:
            try:
                # Stop reading thread
                self.running = False
                if self.thread:
                    self.thread.join(timeout=1.0)
                
                # Close Bluetooth connection
                self.bluetooth.close()
                self.bluetooth = None
                
                # Mark as disconnected
                self.connected = False
                
                logger.info("Disconnected from ESP32 via Bluetooth")
                return True
            
            except Exception as e:
                logger.error(f"Error disconnecting from ESP32 via Bluetooth: {e}")
                return False
        
        # Otherwise use serial disconnection
        return super().disconnect()
    
    def send_command(self, command: str, **params) -> Dict[str, Any]:
        """
        Send command to ESP32 device.
        
        Args:
            command: Command name
            params: Command parameters
            
        Returns:
            Command result
        """
        # If using Bluetooth
        if hasattr(self, 'bluetooth') and self.bluetooth:
            try:
                if not self.connected:
                    return {"success": False, "error": "Not connected to ESP32 via Bluetooth"}
                
                # Prepare command
                protocol = self.config["protocol"]
                
                if protocol == "json":
                    cmd_data = {"cmd": command, **params}
                    cmd_str = json.dumps(cmd_data) + self.config["newline"]
                    data = cmd_str.encode('utf-8')
                else:
                    return {"success": False, "error": "Only JSON protocol supported over Bluetooth"}
                
                # Send command
                self.bluetooth.send(data)
                
                # Wait for response
                start_time = time.time()
                response = None
                
                while time.time() - start_time < 1.0:  # 1 second timeout
                    if "response" in self.sensor_data:
                        response = self.sensor_data.pop("response")
                        break
                    time.sleep(0.01)
                
                return {
                    "success": True,
                    "command": command,
                    "params": params,
                    "response": response
                }
            
            except Exception as e:
                logger.error(f"Error sending command to ESP32 via Bluetooth: {e}")
                return {"success": False, "error": str(e)}
        
        # Otherwise use serial command
        return super().send_command(command, **params)
    
    def _configure_wifi(self) -> bool:
        """Configure WiFi settings on ESP32."""
        wifi_config = self.config.get("wifi_config", {})
        
        if not wifi_config or not isinstance(wifi_config, dict):
            return False
        
        ssid = wifi_config.get("ssid")
        password = wifi_config.get("password")
        
        if not ssid:
            logger.warning("WiFi SSID not provided, skipping WiFi configuration")
            return False
        
        try:
            response = self.send_command(
                "wifi_connect", 
                ssid=ssid, 
                password=password or ""
            )
            
            if response.get("success") and response.get("response", {}).get("result") == "ok":
                self.wifi_connected = True
                logger.info(f"ESP32 connected to WiFi network: {ssid}")
                return True
            
            # If response indicates failure
            error = response.get("response", {}).get("error", "Unknown error")
            logger.error(f"Failed to connect ESP32 to WiFi: {error}")
            return False
        
        except Exception as e:
            logger.error(f"Error configuring WiFi on ESP32: {e}")
            return False
    
    def set_pin_mode(self, pin: int, mode: str) -> bool:
        """
        Set ESP32 GPIO pin mode.
        
        Args:
            pin: GPIO pin number
            mode: Pin mode (input, output, input_pullup, input_pulldown, analog)
            
        Returns:
            True if successful
        """
        response = self.send_command("gpio_mode", pin=pin, mode=mode)
        
        if response.get("success") and response.get("response", {}).get("result") == "ok":
            return True
        
        return False
    
    def digital_write(self, pin: int, value: bool) -> bool:
        """
        Write digital value to GPIO pin.
        
        Args:
            pin: GPIO pin number
            value: Digital value (True/False)
            
        Returns:
            True if successful
        """
        response = self.send_command("gpio_write", pin=pin, value=1 if value else 0)
        
        if response.get("success") and response.get("response", {}).get("result") == "ok":
            self.gpio_states[pin] = bool(value)
            return True
        
        return False
    
    def digital_read(self, pin: int) -> Optional[bool]:
        """
        Read digital value from GPIO pin.
        
        Args:
            pin: GPIO pin number
            
        Returns:
            Digital value (True/False) or None if failed
        """
        response = self.send_command("gpio_read", pin=pin)
        
        if response.get("success") and "response" in response:
            value = response["response"].get("value")
            if value is not None:
                self.gpio_states[pin] = value == 1
                return value == 1
        
        return None
    
    def analog_read(self, pin: int) -> Optional[int]:
        """
        Read analog value from ADC pin.
        
        Args:
            pin: ADC pin number
            
        Returns:
            Analog value (0-4095) or None if failed
        """
        response = self.send_command("adc_read", pin=pin)
        
        if response.get("success") and "response" in response:
            return response["response"].get("value")
        
        return None
    
    def ledc_setup(self, channel: int, frequency: int, resolution: int) -> bool:
        """
        Set up LEDC PWM channel.
        
        Args:
            channel: LEDC channel (0-15)
            frequency: PWM frequency in Hz
            resolution: PWM resolution in bits (1-14)
            
        Returns:
            True if successful
        """
        response = self.send_command(
            "ledc_setup", 
            channel=channel, 
            frequency=frequency, 
            resolution=resolution
        )
        
        return response.get("success") and response.get("response", {}).get("result") == "ok"
    
    def ledc_attach_pin(self, pin: int, channel: int) -> bool:
        """
        Attach GPIO pin to LEDC PWM channel.
        
        Args:
            pin: GPIO pin number
            channel: LEDC channel (0-15)
            
        Returns:
            True if successful
        """
        response = self.send_command("ledc_attach", pin=pin, channel=channel)
        
        return response.get("success") and response.get("response", {}).get("result") == "ok"
    
    def ledc_write(self, channel: int, duty: int) -> bool:
        """
        Write duty cycle to LEDC PWM channel.
        
        Args:
            channel: LEDC channel (0-15)
            duty: Duty cycle (depends on resolution set in ledc_setup)
            
        Returns:
            True if successful
        """
        response = self.send_command("ledc_write", channel=channel, duty=duty)
        
        return response.get("success") and response.get("response", {}).get("result") == "ok" 
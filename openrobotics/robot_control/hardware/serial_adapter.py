"""
Serial adapter for communicating with microcontroller-based robots (Arduino, ESP32, etc).
"""

import logging
import threading
import time
import json
from typing import Dict, List, Any, Optional, Union

from openrobotics.robot_control.hardware.base import HardwareAdapter

logger = logging.getLogger(__name__)

class SerialAdapter(HardwareAdapter):
    """Adapter for serial communication with microcontroller-based robots."""
    
    def __init__(self, config: Dict[str, Any] = None):
        """
        Initialize Serial adapter.
        
        Args:
            config: Serial configuration with the following options:
                - port: Serial port (e.g., /dev/ttyUSB0, COM3)
                - baud_rate: Baud rate (default: 115200)
                - timeout: Read timeout in seconds (default: 1.0)
                - protocol: Communication protocol (json, binary, text)
                - newline: Line ending character(s) (default: \n)
        """
        super().__init__(config)
        
        # Set default configuration
        self.config.setdefault("port", "")
        self.config.setdefault("baud_rate", 115200)
        self.config.setdefault("timeout", 1.0)
        self.config.setdefault("protocol", "json")
        self.config.setdefault("newline", "\n")
        
        # Internal state
        self.serial = None
        self.thread = None
        self.running = False
        self.connected = False
        self.sensor_data = {}
        self.lock = threading.Lock()
        
        # Try to import Serial
        try:
            import serial
            self.serial_module = serial
            self.serial_available = True
        except ImportError:
            logger.warning("pyserial module not found. Serial adapter will be simulated.")
            self.serial_available = False
    
    def connect(self) -> bool:
        """
        Connect to serial device.
        
        Returns:
            True if connection successful
        """
        if not self.serial_available:
            logger.warning("pyserial not available. Running in simulation mode.")
            self.connected = True
            return True
            
        try:
            # Validate port
            if not self.config["port"]:
                logger.error("Serial port not specified")
                return False
            
            # Create serial connection
            self.serial = self.serial_module.Serial(
                port=self.config["port"],
                baudrate=self.config["baud_rate"],
                timeout=self.config["timeout"]
            )
            
            # Flush buffers
            self.serial.reset_input_buffer()
            self.serial.reset_output_buffer()
            
            # Start reading thread
            self.running = True
            self.thread = threading.Thread(
                target=self._serial_read_loop,
                daemon=True
            )
            self.thread.start()
            
            # Mark as connected
            self.connected = True
            logger.info(f"Connected to serial port {self.config['port']} at {self.config['baud_rate']} baud")
            
            return True
        
        except Exception as e:
            logger.error(f"Error connecting to serial port: {e}")
            return False
    
    def disconnect(self) -> bool:
        """
        Disconnect from serial device.
        
        Returns:
            True if disconnection successful
        """
        if not self.serial_available:
            self.connected = False
            return True
            
        try:
            # Stop reading thread
            self.running = False
            if self.thread:
                self.thread.join(timeout=1.0)
            
            # Close serial connection
            if self.serial:
                self.serial.close()
                self.serial = None
            
            # Mark as disconnected
            self.connected = False
            
            logger.info("Disconnected from serial port")
            return True
        
        except Exception as e:
            logger.error(f"Error disconnecting from serial port: {e}")
            return False
    
    def send_command(self, command: str, **params) -> Dict[str, Any]:
        """
        Send command to serial device.
        
        Args:
            command: Command name
            params: Command parameters
            
        Returns:
            Command result
        """
        if not self.serial_available:
            # Simulate command execution
            logger.info(f"Simulating serial command: {command} with params {params}")
            return {"success": True, "command": command, "params": params}
            
        try:
            if not self.connected or not self.serial:
                return {"success": False, "error": "Not connected to serial device"}
            
            # Prepare command based on protocol
            protocol = self.config["protocol"]
            
            if protocol == "json":
                # Format as JSON
                cmd_data = {"cmd": command, **params}
                cmd_str = json.dumps(cmd_data) + self.config["newline"]
                data = cmd_str.encode('utf-8')
            elif protocol == "text":
                # Format as text
                param_str = " ".join(f"{k}:{v}" for k, v in params.items())
                cmd_str = f"{command} {param_str}{self.config['newline']}"
                data = cmd_str.encode('utf-8')
            elif protocol == "binary":
                # Simple binary protocol (would need customization)
                # Format: [CMD_ID][PARAM1][PARAM2]...
                # This is very simplified and would need to be adapted to specific binary protocol
                cmd_id = {"move_forward": 1, "move_backward": 2, "turn_left": 3, "turn_right": 4, "stop": 0}.get(command, 255)
                speed = params.get("speed", 0.5)
                speed_byte = int(speed * 255)
                data = bytes([cmd_id, speed_byte])
            else:
                return {"success": False, "error": f"Unknown protocol: {protocol}"}
            
            # Send command
            with self.lock:
                self.serial.write(data)
                self.serial.flush()
            
            # For simple protocols, wait for acknowledgment
            if protocol in ["json", "text"]:
                # Wait for response (simple timeout-based approach)
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
            else:
                # For binary protocol, assume success without response
                return {
                    "success": True,
                    "command": command,
                    "params": params
                }
        
        except Exception as e:
            logger.error(f"Error sending command to serial device: {e}")
            return {"success": False, "error": str(e)}
    
    def read_sensors(self) -> Dict[str, Any]:
        """
        Read sensor data from serial device.
        
        Returns:
            Dictionary of sensor data
        """
        # Return latest sensor data
        return self.sensor_data.copy()
    
    def _serial_read_loop(self):
        """Background thread for reading from serial port."""
        protocol = self.config["protocol"]
        buffer = b""
        
        while self.running and self.serial:
            try:
                # Read data
                if protocol == "binary":
                    # Binary protocol needs custom implementation
                    self._read_binary_protocol()
                else:
                    # Text-based protocols (JSON, text)
                    data = self.serial.read(1024)  # Read up to 1KB at a time
                    
                    if data:
                        # Add to buffer
                        buffer += data
                        
                        # Look for line ending
                        line_ending = self.config["newline"].encode('utf-8')
                        while line_ending in buffer:
                            # Split at line ending
                            line, buffer = buffer.split(line_ending, 1)
                            
                            # Process line
                            try:
                                line_str = line.decode('utf-8').strip()
                                
                                if protocol == "json":
                                    # Parse JSON
                                    data = json.loads(line_str)
                                    
                                    # Handle different message types
                                    if "sensor" in data:
                                        # Sensor data
                                        sensor_name = data.pop("sensor")
                                        self.sensor_data[sensor_name] = {
                                            **data,
                                            "timestamp": time.time()
                                        }
                                    elif "response" in data:
                                        # Command response
                                        self.sensor_data["response"] = data
                                    else:
                                        # Unknown data, store as raw
                                        self.sensor_data["raw"] = data
                                
                                elif protocol == "text":
                                    # Parse text format (assumed to be key:value pairs)
                                    parts = line_str.split()
                                    
                                    if parts and ":" in parts[0]:
                                        # Get message type
                                        msg_type, msg_name = parts[0].split(":", 1)
                                        
                                        # Parse key:value pairs
                                        values = {}
                                        for part in parts[1:]:
                                            if ":" in part:
                                                k, v = part.split(":", 1)
                                                # Try to convert to number if possible
                                                try:
                                                    values[k] = float(v)
                                                except ValueError:
                                                    values[k] = v
                                        
                                        # Store based on message type
                                        if msg_type == "sensor":
                                            self.sensor_data[msg_name] = {
                                                **values,
                                                "timestamp": time.time()
                                            }
                                        elif msg_type == "response":
                                            self.sensor_data["response"] = {
                                                "id": msg_name,
                                                **values
                                            }
                                    else:
                                        # Raw data
                                        self.sensor_data["raw"] = line_str
                            
                            except Exception as e:
                                logger.error(f"Error processing serial data: {e}")
                                logger.debug(f"Raw data: {line}")
            
            except Exception as e:
                logger.error(f"Error reading from serial port: {e}")
                
                # Check if serial is still open
                if self.serial and not self.serial.is_open:
                    logger.error("Serial port closed unexpectedly")
                    self.connected = False
                    break
            
            # Sleep to avoid CPU overuse
            time.sleep(0.01)
    
    def _read_binary_protocol(self):
        """
        Read and parse binary protocol.
        This is a placeholder and would need to be customized for specific binary protocols.
        """
        try:
            # Read header byte
            header = self.serial.read(1)
            
            if not header:
                return
            
            # Parse packet type
            packet_type = header[0]
            
            if packet_type == 0x01:  # Sensor data packet
                # Read sensor ID
                sensor_id = self.serial.read(1)[0]
                
                # Read value length
                value_len = self.serial.read(1)[0]
                
                # Read values
                values_raw = self.serial.read(value_len)
                
                # Parse values (this would depend on the specific protocol)
                # This is a very simplified example
                if len(values_raw) >= 2:
                    value = (values_raw[0] << 8) | values_raw[1]
                    
                    # Map sensor IDs to names
                    sensor_names = {
                        1: "distance",
                        2: "temperature",
                        3: "humidity",
                        4: "light"
                    }
                    
                    sensor_name = sensor_names.get(sensor_id, f"sensor_{sensor_id}")
                    
                    # Store sensor data
                    self.sensor_data[sensor_name] = {
                        "value": value,
                        "timestamp": time.time()
                    }
            
            elif packet_type == 0x02:  # Response packet
                # Read response code
                response_code = self.serial.read(1)[0]
                
                # Read optional data length
                data_len = self.serial.read(1)[0]
                
                # Read data if any
                data = self.serial.read(data_len) if data_len > 0 else b""
                
                # Store response
                self.sensor_data["response"] = {
                    "code": response_code,
                    "data": data.hex() if data else None
                }
        
        except Exception as e:
            logger.error(f"Error reading binary protocol: {e}") 
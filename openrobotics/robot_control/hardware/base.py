"""
Base classes for hardware adapters.
"""

import logging
from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional, Union

logger = logging.getLogger(__name__)

class HardwareAdapter(ABC):
    """Base class for hardware adapters."""
    
    def __init__(self, config: Dict[str, Any] = None):
        """
        Initialize hardware adapter.
        
        Args:
            config: Hardware configuration
        """
        self.config = config or {}
        self.connected = False
    
    @abstractmethod
    def connect(self) -> bool:
        """
        Connect to hardware.
        
        Returns:
            True if connection successful
        """
        pass
    
    @abstractmethod
    def disconnect(self) -> bool:
        """
        Disconnect from hardware.
        
        Returns:
            True if disconnection successful
        """
        pass
    
    @abstractmethod
    def send_command(self, command: str, **params) -> Dict[str, Any]:
        """
        Send command to hardware.
        
        Args:
            command: Command to send
            params: Command parameters
            
        Returns:
            Command result
        """
        pass
    
    @abstractmethod
    def read_sensors(self) -> Dict[str, Any]:
        """
        Read sensor data from hardware.
        
        Returns:
            Sensor data
        """
        pass
    
    def is_connected(self) -> bool:
        """
        Check if connected to hardware.
        
        Returns:
            True if connected
        """
        return self.connected 
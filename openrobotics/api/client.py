'''
Client for the OpenRobotics API.
'''

import json
import logging
import time
from typing import Dict, List, Optional, Any, Union
from pathlib import Path

try:
    import requests
    HAS_REQUESTS = True
except ImportError:
    HAS_REQUESTS = False
    print("Requests not found. API client will not be available.")

logger = logging.getLogger(__name__)


class RoboticsClient:
    '''
    Client for the OpenRobotics API.
    
    This class provides a simple interface for interacting with
    the OpenRobotics API.
    '''
    
    def __init__(
        self,
        base_url: str = "http://localhost:8000",
        timeout: float = 10.0,
    ):
        '''
        Initialize a robotics client.
        
        Args:
            base_url: Base URL of the OpenRobotics API
            timeout: Timeout for requests in seconds
        '''
        if not HAS_REQUESTS:
            raise ImportError("Requests library is required for RoboticsClient")
            
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
    
    def _request(
        self,
        method: str,
        path: str,
        params: Optional[Dict[str, Any]] = None,
        data: Optional[Any] = None,
    ) -> Any:
        '''
        Make a request to the API.
        
        Args:
            method: HTTP method (GET, POST, PUT, DELETE)
            path: API path
            params: Optional query parameters
            data: Optional request data
            
        Returns:
            Response data
        '''
        url = f"{self.base_url}/{path.lstrip('/')}"
        headers = {"Content-Type": "application/json"}
        
        try:
            if method == "GET":
                response = requests.get(url, headers=headers, params=params, timeout=self.timeout)
            elif method == "POST":
                response = requests.post(url, headers=headers, params=params, json=data, timeout=self.timeout)
            else:
                raise ValueError(f"Unsupported HTTP method: {method}")
            
            # Check for errors
            response.raise_for_status()
            
            # Return response data
            return response.json()
        
        except requests.exceptions.RequestException as e:
            logger.error(f"Request error: {e}")
            
            # Try to get error message from response
            try:
                error_data = e.response.json()
                error_message = error_data.get("detail", str(e))
            except:
                error_message = str(e)
            
            raise RuntimeError(f"API request failed: {error_message}")
    
    def get_robot_position(self) -> Dict[str, Any]:
        '''
        Get the position of the robot.
        
        Returns:
            Robot position
        '''
        return self._request("GET", "/robot/position")
    
    def get_sensor_data(self, sensor_name: Optional[str] = None) -> Dict[str, Any]:
        '''
        Get sensor data from the robot.
        
        Args:
            sensor_name: Optional name of specific sensor to read
            
        Returns:
            Sensor data
        '''
        params = {}
        if sensor_name:
            params["sensor_name"] = sensor_name
        
        return self._request("GET", "/robot/sensor_data", params=params)
    
    def execute_motion(
        self,
        action: str,
        parameters: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        '''
        Execute a motion command on the robot.
        
        Args:
            action: Motion action
            parameters: Optional parameters for the action
            
        Returns:
            Motion execution results
        '''
        data = {
            "action": action,
            "parameters": parameters or {},
        }
        
        return self._request("POST", "/robot/motion", data=data) 
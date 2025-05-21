"""
OpenRobotics API.

This module provides the FastAPI server and client for interacting
with the OpenRobotics system.
"""

API_VERSION = "v1"

# Import key components for easier access
try:
    from .server import app, start_api_server
except ImportError:
    pass

try:
    from .client import RoboticsClient
except ImportError:
    pass 
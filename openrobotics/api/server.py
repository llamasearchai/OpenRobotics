'''
FastAPI server for OpenRobotics.
'''

import os
import logging
import threading
import time
from typing import Dict, List, Optional, Any, Union
from pathlib import Path

try:
    from fastapi import FastAPI, HTTPException, Depends, Query, Body, Path as FastAPIPath # Renamed Path to avoid conflict
    from fastapi.middleware.cors import CORSMiddleware
    from fastapi.responses import JSONResponse
    from pydantic import BaseModel, Field
    import uvicorn
    HAS_FASTAPI = True
except ImportError:
    HAS_FASTAPI = False
    print("FastAPI not installed. API functionality will be unavailable.")

from openrobotics.config import config

logger = logging.getLogger(__name__)

# Initialize FastAPI app if available
if HAS_FASTAPI:
    app = FastAPI(
        title="OpenRobotics API",
        description="API for controlling robots and accessing robotics data",
        version="0.1.0",
    )

    # Add CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    
    # Pydantic models for request and response data
    class MotionCommand(BaseModel):
        action: str
        parameters: Optional[Dict[str, Any]] = None
    
    # Global variables
    _robot = None
    _database = None
    _llm_agent = None
    _start_time = time.time()
    
    # Routes
    @app.get("/", response_model=Dict[str, Any])
    async def get_server_info():
        '''Get server information.'''
        server_info = {
            "version": "0.1.0",
            "uptime": time.time() - _start_time,
        }
        
        return server_info
    
    # Robot control endpoints
    @app.post("/robot/motion", response_model=Dict[str, Any])
    async def execute_motion(command: MotionCommand):
        '''Execute a motion command on the active robot.'''
        if not _robot:
            raise HTTPException(status_code=503, detail="Robot not available")
        
        try:
            # Execute motion on robot
            result = _robot.execute_motion(command.action, **(command.parameters or {}))
            
            # Log command to database if available
            if _database:
                command_dict = {
                    "robot_id": _robot.name,
                    "command": command.action,
                    "parameters": command.parameters,
                    "result": result,
                    "status": "success" if result.get("success", False) else "failure",
                }
                _database.log_command(command_dict)
            
            return result
        except Exception as e:
            logger.error(f"Error executing motion: {e}")
            
            # Log error to database if available
            if _database:
                command_dict = {
                    "robot_id": _robot.name,
                    "command": command.action,
                    "parameters": command.parameters,
                    "result": {"error": str(e)},
                    "status": "error",
                }
                _database.log_command(command_dict)
            
            raise HTTPException(status_code=500, detail=str(e))
    
    @app.get("/robot/position", response_model=Dict[str, Any])
    async def get_robot_position():
        '''Get the position of the active robot.'''
        if not _robot:
            raise HTTPException(status_code=503, detail="Robot not available")
        
        try:
            # Get position
            position = _robot.get_position()
            return position
        except Exception as e:
            logger.error(f"Error getting robot position: {e}")
            raise HTTPException(status_code=500, detail=str(e))
    
    @app.get("/robot/sensor_data", response_model=Dict[str, Any])
    async def get_robot_sensor_data(
        sensor_name: Optional[str] = Query(None, description="Name of specific sensor to read (if None, read all)")
    ):
        '''Get sensor data from the active robot.'''
        if not _robot:
            raise HTTPException(status_code=503, detail="Robot not available")
        
        try:
            # Get sensor data
            sensor_data = _robot.get_sensor_data(sensor_name)
            return sensor_data
        except Exception as e:
            logger.error(f"Error getting sensor data: {e}")
            raise HTTPException(status_code=500, detail=str(e))


def start_api_server(
    robot_config: Optional[Union[str, Path]] = None,
    host: str = "127.0.0.1",
    port: int = 8000,
    db_path: Optional[Union[str, Path]] = None,
    debug: bool = False,
):
    '''
    Start the FastAPI server.
    
    Args:
        robot_config: Optional path to robot configuration file
        host: Host to listen on
        port: Port to listen on
        db_path: Path to SQLite database file
        debug: Whether to enable debug mode
    '''
    if not HAS_FASTAPI:
        logger.error("FastAPI is not installed. Cannot start API server.")
        return
        
    global _robot, _database, _llm_agent
    
    # Configure logging
    logging_level = logging.DEBUG if debug else logging.INFO
    logging.basicConfig(
        level=logging_level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    
    # Initialize database
    if db_path:
        try:
            from openrobotics.data_storage.database import RoboticsDB
            logger.info(f"Initializing database at {db_path}")
            _database = RoboticsDB(db_path)
        except Exception as e:
            logger.error(f"Error initializing database: {e}")
            _database = None
    
    # Initialize robot
    try:
        from openrobotics.robot_control.robot import Robot
        logger.info("Initializing robot")
        if robot_config:
            _robot = Robot.from_config(robot_config)
        else:
            _robot = Robot(name="api_robot", simulation=True)
    except Exception as e:
        logger.error(f"Error initializing robot: {e}")
        _robot = None
    
    # Initialize LLM agent
    try:
        from openrobotics.langchain_integration.agents import LLMAgent
        logger.info("Initializing LLM agent")
        _llm_agent = LLMAgent(robot=_robot, database=_database)
    except Exception as e:
        logger.error(f"Error initializing LLM agent: {e}")
        _llm_agent = None
    
    # Start FastAPI server
    logger.info(f"Starting FastAPI server on {host}:{port}")
    uvicorn.run(app, host=host, port=port, log_level="debug" if debug else "info") 
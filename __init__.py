"""
OpenRobotics: A comprehensive robotics development framework.

This package integrates MLX, LangChain, DSPy, and various other tools to create
a powerful environment for building advanced robotics applications.
"""

__version__ = "0.1.0"

from openrobotics.robot_control.robot import Robot
from openrobotics.robot_control.sensors import SensorArray, Sensor
from openrobotics.robot_control.actuators import Actuator, Motor, Servo

try:
    from openrobotics.langchain_integration.agents import LLMAgent, MultiAgentOrchestrator
except ImportError:
    pass

try:
    from openrobotics.mlx_integration.models import MLXModel, VisionModel, ControlModel
except ImportError:
    pass

from openrobotics.data_storage.database import RoboticsDB

try:
    from openrobotics.api.client import RoboticsClient
except ImportError:
    pass

# Convenience function to set up complete system
def create_robotics_system(
    robot_config: str = None,
    model_name: str = "gpt-4",
    database_path: str = "robotics.db",
    enable_api: bool = True,
    api_port: int = 8000,
):
    """
    Create a complete robotics system with all components configured.
    
    Args:
        robot_config: Path to robot configuration YAML file
        model_name: Name of the LLM model to use
        database_path: Path to SQLite database
        enable_api: Whether to start the FastAPI server
        api_port: Port for the FastAPI server
        
    Returns:
        Tuple containing (Robot, LLMAgent, RoboticsDB, optional API instance)
    """
    from openrobotics.robot_control.robot import Robot
    from openrobotics.data_storage.database import RoboticsDB
    import yaml
    
    # Initialize robot from config
    if robot_config:
        with open(robot_config, 'r') as f:
            config = yaml.safe_load(f)
        robot = Robot.from_config(config)
    else:
        robot = Robot(name="default_robot")
        
    # Initialize database
    db = RoboticsDB(database_path)
    
    # Initialize agent
    try:
        from openrobotics.langchain_integration.agents import LLMAgent
        agent = LLMAgent(robot=robot, model=model_name, database=db)
    except ImportError:
        agent = None
    
    # Start API if requested
    if enable_api:
        try:
            from openrobotics.api.server import start_api_server
            import threading
            
            api_thread = threading.Thread(
                target=start_api_server,
                args=(robot_config, "localhost", api_port, database_path),
                daemon=True
            )
            api_thread.start()
            return robot, agent, db, api_thread
        except ImportError:
            pass
    
    return robot, agent, db
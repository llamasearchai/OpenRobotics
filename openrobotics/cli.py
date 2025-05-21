"""
Command-line interface for OpenRobotics.
"""

import os
import sys
import logging
from pathlib import Path
from typing import Optional, List

import typer
from rich.console import Console
from rich.logging import RichHandler
from rich.progress import Progress

from openrobotics import __version__, create_robotics_system
from openrobotics.config import config
from openrobotics.api.server import start_api_server

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    datefmt="[%X]",
    handlers=[RichHandler(rich_tracebacks=True)]
)
log = logging.getLogger("openrobotics")

# Setup console
console = Console()

# Create Typer app
app = typer.Typer(
    name="openrobotics",
    help="OpenRobotics: A comprehensive robotics development framework",
    add_completion=False,
)


@app.callback()
def callback():
    """OpenRobotics CLI: Control and manage robotics systems."""
    pass


@app.command()
def version():
    """Show the OpenRobotics version."""
    console.print(f"OpenRobotics v{__version__}")


@app.command()
def server(
    host: str = typer.Option(
        "127.0.0.1", "--host", help="API server host"
    ),
    port: int = typer.Option(
        8000, "--port", "-p", help="API server port"
    ),
    robot_config: Optional[Path] = typer.Option(
        None, "--config", "-c", help="Path to robot configuration YAML file"
    ),
    database: Optional[Path] = typer.Option(
        None, "--db", help="Path to SQLite database file"
    ),
    enable_datasette: bool = typer.Option(
        True, "--datasette/--no-datasette", help="Enable Datasette for database exploration"
    ),
    datasette_port: int = typer.Option(
        8001, "--datasette-port", help="Port for Datasette server"
    ),
    debug: bool = typer.Option(
        False, "--debug", help="Enable debug mode"
    ),
):
    """Start the OpenRobotics API server."""
    try:
        from openrobotics.api.server import start_api_server
        
        if debug:
            logging.getLogger().setLevel(logging.DEBUG)
            log.debug("Debug mode enabled")
        
        # Get database path
        db_path = database or config.get("database", "path")
        
        console.print(f"Starting OpenRobotics server on [bold]{host}:{port}[/bold]")
        if enable_datasette:
            console.print(f"Datasette will be available at [bold]http://{host}:{datasette_port}[/bold]")
        
        # Start server (this blocks)
        start_api_server(
            robot_config=robot_config,
            host=host,
            port=port,
            db_path=db_path,
            enable_datasette=enable_datasette,
            datasette_port=datasette_port,
            debug=debug
        )
    except KeyboardInterrupt:
        console.print("\n[bold yellow]Server stopped[/bold yellow]")
    except Exception as e:
        console.print(f"[bold red]Error starting server: {e}[/bold red]")
        if debug:
            import traceback
            console.print(traceback.format_exc())
        sys.exit(1)


@app.command()
def run(
    script: Path = typer.Argument(
        ..., help="Path to Python script to run with OpenRobotics environment"
    ),
    args: Optional[List[str]] = typer.Argument(
        None, help="Arguments to pass to the script"
    ),
    robot_config: Optional[Path] = typer.Option(
        None, "--config", "-c", help="Path to robot configuration YAML file"
    ),
    simulation: bool = typer.Option(
        True, "--simulation/--no-simulation", help="Enable simulation mode"
    ),
):
    """Run a Python script with OpenRobotics environment."""
    try:
        # Set simulation mode
        config.set("robotics", "simulation_enabled", simulation)
        
        # Add script directory to Python path
        script_dir = script.absolute().parent
        sys.path.insert(0, str(script_dir))
        
        # Get script name without extension
        script_name = script.stem
        
        console.print(f"Running [bold]{script}[/bold]")
        if simulation:
            console.print("[bold yellow]Simulation mode enabled[/bold yellow]")
        
        # Set commandline arguments
        if args:
            sys.argv = [str(script)] + args
        else:
            sys.argv = [str(script)]
        
        # Execute script
        with open(script, 'r') as f:
            script_code = f.read()
        
        # Create globals with robot system already set up
        robot, agent, db = create_robotics_system(robot_config=robot_config if robot_config else None)
        
        globals_dict = {
            "__file__": str(script),
            "__name__": "__main__",
            "robot": robot,
            "agent": agent,
            "db": db,
        }
        
        # Execute the script
        exec(script_code, globals_dict)
        
    except KeyboardInterrupt:
        console.print("\n[bold yellow]Script execution interrupted[/bold yellow]")
    except Exception as e:
        console.print(f"[bold red]Error running script: {e}[/bold red]")
        import traceback
        console.print(traceback.format_exc())
        sys.exit(1)


@app.command()
def simulate(
    scenario: str = typer.Argument(
        "basic_navigation", help="Name of the scenario to simulate"
    ),
    duration: int = typer.Option(
        60, "--duration", "-d", help="Duration of simulation in seconds"
    ),
    headless: bool = typer.Option(
        False, "--headless", help="Run simulation without visualization"
    ),
):
    """Run a simulation scenario."""
    try:
        from openrobotics.robot_control.simulation import run_simulation
        
        available_scenarios = [
            "basic_navigation",
            "obstacle_avoidance",
            "multi_agent_coordination",
            "object_manipulation"
        ]
        
        if scenario not in available_scenarios:
            console.print(f"[bold red]Unknown scenario: {scenario}[/bold red]")
            console.print(f"Available scenarios: {', '.join(available_scenarios)}")
            sys.exit(1)
        
        console.print(f"Running simulation scenario: [bold]{scenario}[/bold]")
        console.print(f"Duration: {duration} seconds")
        if headless:
            console.print("[bold yellow]Running in headless mode[/bold yellow]")
        
        # Run simulation
        with Progress() as progress:
            task = progress.add_task(f"Running {scenario}", total=duration)
            
            def progress_callback(current_time):
                progress.update(task, completed=current_time)
            
            run_simulation(
                scenario=scenario,
                duration=duration,
                headless=headless,
                progress_callback=progress_callback
            )
        
        console.print("[bold green]Simulation completed[/bold green]")
        
    except KeyboardInterrupt:
        console.print("\n[bold yellow]Simulation interrupted[/bold yellow]")
    except Exception as e:
        console.print(f"[bold red]Error in simulation: {e}[/bold red]")
        import traceback
        console.print(traceback.format_exc())
        sys.exit(1)


@app.command()
def llm(
    prompt: str = typer.Argument(
        ..., help="Prompt to send to the language model"
    ),
    model: str = typer.Option(
        None, "--model", "-m", help="Language model to use"
    ),
    robot_config: Optional[Path] = typer.Option(
        None, "--config", "-c", help="Path to robot configuration YAML file"
    ),
    json_output: bool = typer.Option(
        False, "--json", help="Output in JSON format"
    ),
):
    """Send a prompt to the language model."""
    try:
        from openrobotics.langchain_integration.agents import LLMAgent
        from openrobotics.robot_control.robot import Robot
        import json
        
        # Get model name
        model_name = model or config.get("llm", "default_model")
        
        # Initialize robot and agent
        if robot_config:
            import yaml
            with open(robot_config, 'r') as f:
                robot_cfg = yaml.safe_load(f)
            robot = Robot.from_config(robot_cfg)
        else:
            robot = Robot(name="cli_robot", simulation=True)
        
        agent = LLMAgent(robot=robot, model=model_name)
        
        # Send prompt to LLM
        console.print(f"Using model: [bold]{model_name}[/bold]")
        with console.status("Thinking..."):
            response = agent.query(prompt)
        
        # Output response
        if json_output:
            console.print_json(json.dumps({
                "prompt": prompt,
                "model": model_name,
                "response": response
            }))
        else:
            console.print("\n[bold green]Response:[/bold green]")
            console.print(response)
        
    except KeyboardInterrupt:
        console.print("\n[bold yellow]Request interrupted[/bold yellow]")
    except Exception as e:
        console.print(f"[bold red]Error: {e}[/bold red]")
        import traceback
        console.print(traceback.format_exc())
        sys.exit(1)


# Main entry point
if __name__ == "__main__":
    app()

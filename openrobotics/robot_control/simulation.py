"""
Robot simulation module for testing and development without physical hardware.
"""

import logging
import time
import threading
from typing import Dict, List, Any, Optional, Union
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

from openrobotics.robot_control.robot import Robot

logger = logging.getLogger(__name__)

class Simulation:
    """Simulation environment for virtual robot testing."""
    
    def __init__(self, config: Dict[str, Any] = None):
        """
        Initialize simulation environment.
        
        Args:
            config: Simulation configuration with options:
                - world_size: Tuple of (width, height) in meters
                - obstacles: List of obstacle descriptions
                - update_rate: Simulation update rate in Hz
                - physics: Physics engine settings
                - visualization: Visualization settings
        """
        self.config = config or {}
        
        # Set default configuration
        self.config.setdefault("world_size", (10.0, 10.0))
        self.config.setdefault("obstacles", [])
        self.config.setdefault("update_rate", 30)
        self.config.setdefault("physics", {"friction": 0.1, "inertia": 0.2})
        self.config.setdefault("visualization", {"enabled": True, "live_update": True})
        
        # Internal state
        self.robots = {}
        self.obstacles = []
        self.running = False
        self.thread = None
        self.time = 0.0
        self.dt = 1.0 / self.config["update_rate"]
        
        # Visualization
        self.fig = None
        self.ax = None
        self.robot_markers = {}
        self.obstacle_patches = []
        self.animation = None
        
        # Initialize obstacles
        self._init_obstacles()
    
    def _init_obstacles(self):
        """Initialize obstacles from configuration."""
        for obs_config in self.config["obstacles"]:
            obs_type = obs_config.get("type", "box")
            
            if obs_type == "box":
                x = obs_config.get("x", 0.0)
                y = obs_config.get("y", 0.0)
                width = obs_config.get("width", 1.0)
                height = obs_config.get("height", 1.0)
                self.obstacles.append({
                    "type": "box",
                    "x": x,
                    "y": y,
                    "width": width,
                    "height": height,
                    "x_min": x - width/2,
                    "x_max": x + width/2,
                    "y_min": y - height/2,
                    "y_max": y + height/2
                })
            
            elif obs_type == "circle":
                x = obs_config.get("x", 0.0)
                y = obs_config.get("y", 0.0)
                radius = obs_config.get("radius", 0.5)
                self.obstacles.append({
                    "type": "circle",
                    "x": x,
                    "y": y,
                    "radius": radius
                })
    
    def add_robot(self, robot: Robot, name: str = None, pose: Dict[str, float] = None) -> str:
        """
        Add robot to simulation.
        
        Args:
            robot: Robot instance
            name: Robot name (optional)
            pose: Initial pose as dict with x, y, theta keys
            
        Returns:
            Robot identifier
        """
        # Generate name if not provided
        if name is None:
            name = f"robot_{len(self.robots)}"
        
        # Set default pose if not provided
        if pose is None:
            pose = {"x": 0.0, "y": 0.0, "theta": 0.0}
        
        # Add to robots dictionary
        self.robots[name] = {
            "robot": robot,
            "pose": pose.copy(),
            "velocity": {"linear": 0.0, "angular": 0.0},
            "sensors": {},
            "actuators": {},
            "config": {
                "radius": 0.2,
                "max_speed": 1.0,
                "max_acceleration": 2.0
            }
        }
        
        logger.info(f"Added robot '{name}' to simulation at position ({pose['x']}, {pose['y']})")
        return name
    
    def remove_robot(self, name: str) -> bool:
        """
        Remove robot from simulation.
        
        Args:
            name: Robot name or ID
            
        Returns:
            True if robot was removed
        """
        if name in self.robots:
            del self.robots[name]
            if name in self.robot_markers:
                del self.robot_markers[name]
            return True
        return False
    
    def start(self) -> bool:
        """
        Start simulation.
        
        Returns:
            True if simulation started successfully
        """
        if self.running:
            logger.warning("Simulation already running")
            return False
        
        # Start simulation thread
        self.running = True
        self.thread = threading.Thread(target=self._simulation_loop)
        self.thread.daemon = True
        self.thread.start()
        
        # Start visualization if enabled
        if self.config["visualization"]["enabled"]:
            self._init_visualization()
        
        logger.info("Simulation started")
        return True
    
    def stop(self) -> bool:
        """
        Stop simulation.
        
        Returns:
            True if simulation stopped successfully
        """
        if not self.running:
            logger.warning("Simulation not running")
            return False
        
        # Stop simulation thread
        self.running = False
        if self.thread:
            self.thread.join(timeout=1.0)
        
        # Stop visualization
        if self.animation:
            self.animation.event_source.stop()
        
        logger.info("Simulation stopped")
        return True
    
    def _simulation_loop(self):
        """Main simulation loop."""
        last_time = time.time()
        
        while self.running:
            # Calculate elapsed time
            current_time = time.time()
            elapsed = current_time - last_time
            last_time = current_time
            
            # Update simulation at fixed rate
            if elapsed < self.dt:
                time.sleep(self.dt - elapsed)
            
            # Update simulation time
            self.time += self.dt
            
            # Update robot states
            self._update_robots(self.dt)
            
            # Check for collisions
            self._check_collisions()
            
            # Update sensors
            self._update_sensors()
    
    def _update_robots(self, dt: float):
        """
        Update robot positions based on their velocities.
        
        Args:
            dt: Time delta in seconds
        """
        for name, robot_data in self.robots.items():
            # Get current pose and velocity
            pose = robot_data["pose"]
            velocity = robot_data["velocity"]
            
            # Update position based on velocity
            pose["x"] += velocity["linear"] * dt * np.cos(pose["theta"])
            pose["y"] += velocity["linear"] * dt * np.sin(pose["theta"])
            pose["theta"] += velocity["angular"] * dt
            
            # Normalize angle to -pi to pi
            pose["theta"] = np.mod(pose["theta"] + np.pi, 2 * np.pi) - np.pi
            
            # Apply world boundaries
            world_size = self.config["world_size"]
            robot_radius = robot_data["config"]["radius"]
            
            pose["x"] = max(robot_radius, min(world_size[0] - robot_radius, pose["x"]))
            pose["y"] = max(robot_radius, min(world_size[1] - robot_radius, pose["y"]))
    
    def _check_collisions(self):
        """Check for collisions between robots and obstacles."""
        # Check robot-obstacle collisions
        for name, robot_data in self.robots.items():
            robot_x = robot_data["pose"]["x"]
            robot_y = robot_data["pose"]["y"]
            robot_radius = robot_data["config"]["radius"]
            
            for obstacle in self.obstacles:
                collision = False
                
                if obstacle["type"] == "box":
                    # Check if robot overlaps with box
                    closest_x = max(obstacle["x_min"], min(robot_x, obstacle["x_max"]))
                    closest_y = max(obstacle["y_min"], min(robot_y, obstacle["y_max"]))
                    
                    distance = np.sqrt((robot_x - closest_x)**2 + (robot_y - closest_y)**2)
                    collision = distance < robot_radius
                
                elif obstacle["type"] == "circle":
                    # Check if robot overlaps with circle
                    obs_x = obstacle["x"]
                    obs_y = obstacle["y"]
                    obs_radius = obstacle["radius"]
                    
                    distance = np.sqrt((robot_x - obs_x)**2 + (robot_y - obs_y)**2)
                    collision = distance < (robot_radius + obs_radius)
                
                if collision:
                    # Simple collision response - stop the robot
                    robot_data["velocity"]["linear"] = 0.0
                    robot_data["velocity"]["angular"] = 0.0
                    
                    # Log collision
                    logger.debug(f"Robot '{name}' collided with obstacle")
                    break
        
        # Check robot-robot collisions (simplified)
        robot_items = list(self.robots.items())
        for i in range(len(robot_items)):
            name_a, robot_a = robot_items[i]
            for j in range(i + 1, len(robot_items)):
                name_b, robot_b = robot_items[j]
                
                # Calculate distance between robots
                dist = np.sqrt((robot_a["pose"]["x"] - robot_b["pose"]["x"])**2 + 
                              (robot_a["pose"]["y"] - robot_b["pose"]["y"])**2)
                
                # Check for collision
                if dist < (robot_a["config"]["radius"] + robot_b["config"]["radius"]):
                    # Simple collision response - stop both robots
                    robot_a["velocity"]["linear"] = 0.0
                    robot_a["velocity"]["angular"] = 0.0
                    robot_b["velocity"]["linear"] = 0.0
                    robot_b["velocity"]["angular"] = 0.0
                    
                    # Log collision
                    logger.debug(f"Robot '{name_a}' collided with robot '{name_b}'")
    
    def _update_sensors(self):
        """Update simulated sensor readings for all robots."""
        for name, robot_data in self.robots.items():
            robot = robot_data["robot"]
            pose = robot_data["pose"]
            
            # Simulate sensor readings based on robot position
            # This would be expanded for specific sensor types
            
            # Example: Distance sensors
            sensor_data = {}
            
            # Simulate distance sensors in different directions
            angles = [0, np.pi/4, np.pi/2, 3*np.pi/4, np.pi, -3*np.pi/4, -np.pi/2, -np.pi/4]
            distances = []
            
            for angle in angles:
                # Sensor direction in world coordinates
                sensor_angle = pose["theta"] + angle
                sensor_dir_x = np.cos(sensor_angle)
                sensor_dir_y = np.sin(sensor_angle)
                
                # Find distance to nearest obstacle
                max_range = 5.0  # Maximum sensor range
                closest_dist = max_range
                
                # Check distance to obstacles
                for obstacle in self.obstacles:
                    if obstacle["type"] == "box":
                        # Calculate distance to box (simplified)
                        # This is a simplified calculation and could be improved
                        closest_x = max(obstacle["x_min"], min(pose["x"], obstacle["x_max"]))
                        closest_y = max(obstacle["y_min"], min(pose["y"], obstacle["y_max"]))
                        
                        dx = closest_x - pose["x"]
                        dy = closest_y - pose["y"]
                        
                        # Project onto sensor direction
                        dot_product = dx * sensor_dir_x + dy * sensor_dir_y
                        
                        if dot_product > 0:  # Only consider obstacles in front of sensor
                            # Distance from robot to point of interest on obstacle
                            dist = np.sqrt(dx**2 + dy**2)
                            closest_dist = min(closest_dist, dist)
                    
                    elif obstacle["type"] == "circle":
                        # Vector from robot to obstacle center
                        dx = obstacle["x"] - pose["x"]
                        dy = obstacle["y"] - pose["y"]
                        
                        # Distance to obstacle center
                        dist_to_center = np.sqrt(dx**2 + dy**2)
                        
                        # Project onto sensor direction
                        dot_product = dx * sensor_dir_x + dy * sensor_dir_y
                        
                        if dot_product > 0:  # Only consider obstacles in front of sensor
                            # Perpendicular distance from sensor line to obstacle center
                            perp_dist = abs(dx * sensor_dir_y - dy * sensor_dir_x)
                            
                            # If perpendicular distance is less than radius, sensor ray intersects obstacle
                            if perp_dist < obstacle["radius"]:
                                # Calculate distance to intersection point (simplified)
                                dist_to_intersection = dot_product - np.sqrt(obstacle["radius"]**2 - perp_dist**2)
                                closest_dist = min(closest_dist, dist_to_intersection)
                
                # Check distance to world boundaries
                world_size = self.config["world_size"]
                
                # Distance to boundaries in sensor direction
                if sensor_dir_x > 0:
                    dist_to_boundary = (world_size[0] - pose["x"]) / sensor_dir_x
                    closest_dist = min(closest_dist, dist_to_boundary)
                elif sensor_dir_x < 0:
                    dist_to_boundary = pose["x"] / -sensor_dir_x
                    closest_dist = min(closest_dist, dist_to_boundary)
                
                if sensor_dir_y > 0:
                    dist_to_boundary = (world_size[1] - pose["y"]) / sensor_dir_y
                    closest_dist = min(closest_dist, dist_to_boundary)
                elif sensor_dir_y < 0:
                    dist_to_boundary = pose["y"] / -sensor_dir_y
                    closest_dist = min(closest_dist, dist_to_boundary)
                
                # Add noise to sensor reading (optional)
                noise = np.random.normal(0, 0.05)  # Gaussian noise
                distance = max(0, min(closest_dist + noise, max_range))
                
                distances.append(distance)
            
            # Store sensor data
            sensor_data["distance_sensors"] = {
                "angles": angles,
                "distances": distances,
                "timestamp": self.time
            }
            
            # Add position data
            sensor_data["position"] = {
                "x": pose["x"],
                "y": pose["y"],
                "theta": pose["theta"],
                "timestamp": self.time
            }
            
            # Update robot sensors
            robot_data["sensors"] = sensor_data
    
    def set_robot_velocity(self, name: str, linear: float, angular: float) -> bool:
        """
        Set robot velocity.
        
        Args:
            name: Robot name
            linear: Linear velocity in m/s
            angular: Angular velocity in rad/s
            
        Returns:
            True if successful
        """
        if name not in self.robots:
            return False
        
        robot_data = self.robots[name]
        
        # Apply velocity limits
        max_speed = robot_data["config"]["max_speed"]
        linear = max(-max_speed, min(max_speed, linear))
        angular = max(-np.pi, min(np.pi, angular))
        
        # Set velocity
        robot_data["velocity"]["linear"] = linear
        robot_data["velocity"]["angular"] = angular
        
        return True
    
    def get_robot_pose(self, name: str) -> Optional[Dict[str, float]]:
        """
        Get robot pose.
        
        Args:
            name: Robot name
            
        Returns:
            Robot pose (x, y, theta) or None if robot doesn't exist
        """
        if name not in self.robots:
            return None
        
        return self.robots[name]["pose"].copy()
    
    def get_robot_sensors(self, name: str) -> Optional[Dict[str, Any]]:
        """
        Get robot sensor data.
        
        Args:
            name: Robot name
            
        Returns:
            Sensor data or None if robot doesn't exist
        """
        if name not in self.robots:
            return None
        
        return self.robots[name]["sensors"].copy()
    
    def _init_visualization(self):
        """Initialize visualization."""
        import matplotlib.pyplot as plt
        from matplotlib.patches import Rectangle, Circle, Arrow
        
        # Create figure and axes
        self.fig, self.ax = plt.subplots(figsize=(10, 10))
        
        # Set world limits
        world_size = self.config["world_size"]
        self.ax.set_xlim(0, world_size[0])
        self.ax.set_ylim(0, world_size[1])
        self.ax.set_aspect('equal')
        self.ax.grid(True)
        self.ax.set_title('Robot Simulation')
        
        # Add obstacles
        for obstacle in self.obstacles:
            if obstacle["type"] == "box":
                rect = Rectangle(
                    (obstacle["x_min"], obstacle["y_min"]),
                    obstacle["width"],
                    obstacle["height"],
                    facecolor='gray',
                    alpha=0.7
                )
                self.obstacle_patches.append(rect)
                self.ax.add_patch(rect)
            
            elif obstacle["type"] == "circle":
                circle = Circle(
                    (obstacle["x"], obstacle["y"]),
                    obstacle["radius"],
                    facecolor='gray',
                    alpha=0.7
                )
                self.obstacle_patches.append(circle)
                self.ax.add_patch(circle)
        
        # Add robots
        for name, robot_data in self.robots.items():
            pose = robot_data["pose"]
            radius = robot_data["config"]["radius"]
            
            # Robot body
            robot_circle = Circle(
                (pose["x"], pose["y"]),
                radius,
                facecolor='blue',
                edgecolor='black',
                alpha=0.7
            )
            
            # Direction indicator
            arrow_length = radius * 0.8
            robot_arrow = Arrow(
                pose["x"],
                pose["y"],
                arrow_length * np.cos(pose["theta"]),
                arrow_length * np.sin(pose["theta"]),
                width=radius * 0.5,
                facecolor='red'
            )
            
            self.robot_markers[name] = {
                "circle": robot_circle,
                "arrow": robot_arrow
            }
            
            self.ax.add_patch(robot_circle)
            self.ax.add_patch(robot_arrow)
        
        # Start animation if live update is enabled
        if self.config["visualization"]["live_update"]:
            self.animation = FuncAnimation(
                self.fig,
                self._update_visualization,
                interval=int(self.dt * 1000),
                blit=False
            )
            
            plt.show(block=False)
    
    def _update_visualization(self, frame):
        """Update visualization with current robot states."""
        # Update robot markers
        for name, robot_data in self.robots.items():
            if name not in self.robot_markers:
                continue
                
            pose = robot_data["pose"]
            markers = self.robot_markers[name]
            
            # Update position
            markers["circle"].center = (pose["x"], pose["y"])
            
            # Remove old arrow and create new one at updated position and orientation
            markers["arrow"].remove()
            
            arrow_length = robot_data["config"]["radius"] * 0.8
            new_arrow = Arrow(
                pose["x"],
                pose["y"],
                arrow_length * np.cos(pose["theta"]),
                arrow_length * np.sin(pose["theta"]),
                width=robot_data["config"]["radius"] * 0.5,
                facecolor='red'
            )
            
            markers["arrow"] = new_arrow
            self.ax.add_patch(new_arrow)
        
        # Update title with simulation time
        self.ax.set_title(f'Robot Simulation - Time: {self.time:.2f}s')
        
        return []  # Return artists that were updated

def run_simulation(config=None, duration=None):
    """
    Run a simulation with the given configuration.
    
    Args:
        config: Simulation configuration
        duration: Simulation duration in seconds (None for indefinite)
        
    Returns:
        Simulation instance
    """
    # Create default configuration if not provided
    if config is None:
        config = {
            "world_size": (10.0, 10.0),
            "obstacles": [
                {"type": "box", "x": 3.0, "y": 3.0, "width": 1.0, "height": 1.0},
                {"type": "circle", "x": 7.0, "y": 7.0, "radius": 0.5}
            ],
            "visualization": {"enabled": True, "live_update": True}
        }
    
    # Create simulation
    sim = Simulation(config)
    
    # Start simulation
    sim.start()
    
    # Run for specified duration
    if duration is not None:
        try:
            time.sleep(duration)
            sim.stop()
        except KeyboardInterrupt:
            logger.info("Simulation interrupted by user")
            sim.stop()
    
    return sim 
"""
ROS adapter for connecting with ROS-based robots.
"""

import logging
import threading
import time
from typing import Dict, List, Any, Optional, Union
import json

from openrobotics.robot_control.hardware.base import HardwareAdapter

logger = logging.getLogger(__name__)

class ROSAdapter(HardwareAdapter):
    """Adapter for Robot Operating System (ROS) integration."""
    
    def __init__(self, config: Dict[str, Any] = None):
        """
        Initialize ROS adapter.
        
        Args:
            config: ROS configuration with the following options:
                - ros_master_uri: ROS master URI (default: http://localhost:11311)
                - node_name: Name for this ROS node (default: openrobotics_adapter)
                - cmd_vel_topic: Topic for publishing velocity commands (default: /cmd_vel)
                - odom_topic: Topic for subscribing to odometry (default: /odom)
                - sensors: Dict mapping sensor names to ROS topics
        """
        super().__init__(config)
        
        # Set default configuration
        self.config.setdefault("ros_master_uri", "http://localhost:11311")
        self.config.setdefault("node_name", "openrobotics_adapter")
        self.config.setdefault("cmd_vel_topic", "/cmd_vel")
        self.config.setdefault("odom_topic", "/odom")
        self.config.setdefault("sensors", {})
        
        # Internal state
        self.ros_node = None
        self.publishers = {}
        self.subscribers = {}
        self.sensor_data = {}
        self.position = {"x": 0.0, "y": 0.0, "theta": 0.0}
        self.connected = False
        
        # Import ROS Python modules
        try:
            import rospy
            self.rospy = rospy
            self.ros_available = True
        except ImportError:
            logger.warning("ROS Python modules not found. ROS adapter will be simulated.")
            self.ros_available = False
    
    def connect(self) -> bool:
        """
        Connect to ROS.
        
        Returns:
            True if connection successful
        """
        if not self.ros_available:
            logger.warning("ROS not available. Running in simulation mode.")
            self.connected = True
            return True
            
        try:
            # Import ROS message types
            from geometry_msgs.msg import Twist
            from nav_msgs.msg import Odometry
            from sensor_msgs.msg import LaserScan, Image, Imu
            from tf.transformations import euler_from_quaternion
            
            # Set ROS master URI
            import os
            os.environ["ROS_MASTER_URI"] = self.config["ros_master_uri"]
            
            # Initialize ROS node
            self.rospy.init_node(self.config["node_name"], anonymous=True)
            
            # Create publisher for cmd_vel
            self.publishers["cmd_vel"] = self.rospy.Publisher(
                self.config["cmd_vel_topic"],
                Twist,
                queue_size=10
            )
            
            # Subscribe to odometry
            def odom_callback(msg):
                # Extract position and orientation
                pos = msg.pose.pose.position
                quat = msg.pose.pose.orientation
                
                # Convert quaternion to Euler angles
                roll, pitch, yaw = euler_from_quaternion(
                    [quat.x, quat.y, quat.z, quat.w]
                )
                
                # Update position
                self.position = {
                    "x": pos.x,
                    "y": pos.y,
                    "theta": yaw
                }
            
            self.subscribers["odom"] = self.rospy.Subscriber(
                self.config["odom_topic"],
                Odometry,
                odom_callback
            )
            
            # Subscribe to sensor topics
            for sensor_name, topic_info in self.config["sensors"].items():
                topic = topic_info["topic"]
                msg_type_str = topic_info.get("msg_type", "")
                
                # Determine message type
                if msg_type_str == "LaserScan" or "/scan" in topic:
                    msg_type = LaserScan
                    
                    def laser_callback(msg, sensor_name=sensor_name):
                        self.sensor_data[sensor_name] = {
                            "ranges": list(msg.ranges),
                            "angle_min": msg.angle_min,
                            "angle_max": msg.angle_max,
                            "angle_increment": msg.angle_increment,
                            "timestamp": msg.header.stamp.secs + msg.header.stamp.nsecs * 1e-9
                        }
                    
                    callback = laser_callback
                
                elif msg_type_str == "Image" or "/image" in topic or "/camera" in topic:
                    msg_type = Image
                    from cv_bridge import CvBridge
                    bridge = CvBridge()
                    
                    def image_callback(msg, sensor_name=sensor_name):
                        try:
                            # Convert to OpenCV image
                            cv_image = bridge.imgmsg_to_cv2(msg, "rgb8")
                            self.sensor_data[sensor_name] = {
                                "image": cv_image,
                                "timestamp": msg.header.stamp.secs + msg.header.stamp.nsecs * 1e-9
                            }
                        except Exception as e:
                            logger.error(f"Error converting image: {e}")
                    
                    callback = image_callback
                
                elif msg_type_str == "Imu" or "/imu" in topic:
                    msg_type = Imu
                    
                    def imu_callback(msg, sensor_name=sensor_name):
                        self.sensor_data[sensor_name] = {
                            "orientation": {
                                "x": msg.orientation.x,
                                "y": msg.orientation.y,
                                "z": msg.orientation.z,
                                "w": msg.orientation.w
                            },
                            "angular_velocity": {
                                "x": msg.angular_velocity.x,
                                "y": msg.angular_velocity.y,
                                "z": msg.angular_velocity.z
                            },
                            "linear_acceleration": {
                                "x": msg.linear_acceleration.x,
                                "y": msg.linear_acceleration.y,
                                "z": msg.linear_acceleration.z
                            },
                            "timestamp": msg.header.stamp.secs + msg.header.stamp.nsecs * 1e-9
                        }
                    
                    callback = imu_callback
                
                else:
                    logger.warning(f"Unknown message type for sensor {sensor_name}")
                    continue
                
                # Create subscriber
                self.subscribers[sensor_name] = self.rospy.Subscriber(
                    topic,
                    msg_type,
                    callback
                )
            
            # Wait for connection to be established
            time.sleep(1.0)
            
            # Mark as connected
            self.connected = True
            logger.info(f"Connected to ROS master at {self.config['ros_master_uri']}")
            
            return True
        
        except Exception as e:
            logger.error(f"Error connecting to ROS: {e}")
            return False
    
    def disconnect(self) -> bool:
        """
        Disconnect from ROS.
        
        Returns:
            True if disconnection successful
        """
        if not self.ros_available:
            self.connected = False
            return True
            
        try:
            # Unregister subscribers
            for name, subscriber in self.subscribers.items():
                subscriber.unregister()
            
            # Shutdown ROS node
            self.rospy.signal_shutdown("Disconnecting ROSAdapter")
            
            # Clear state
            self.publishers = {}
            self.subscribers = {}
            self.connected = False
            
            logger.info("Disconnected from ROS")
            return True
        
        except Exception as e:
            logger.error(f"Error disconnecting from ROS: {e}")
            return False
    
    def send_command(self, command: str, **params) -> Dict[str, Any]:
        """
        Send command to ROS robot.
        
        Args:
            command: Command name (move_forward, move_backward, turn_left, turn_right, stop)
            params: Command parameters (speed, angular_speed)
            
        Returns:
            Command result
        """
        if not self.ros_available:
            # Simulate command execution
            logger.info(f"Simulating ROS command: {command} with params {params}")
            return {"success": True, "command": command, "params": params}
            
        try:
            if command not in ["move_forward", "move_backward", "turn_left", "turn_right", "stop"]:
                return {"success": False, "error": f"Unknown command: {command}"}
            
            if "cmd_vel" not in self.publishers:
                return {"success": False, "error": "cmd_vel publisher not initialized"}
            
            # Import Twist message
            from geometry_msgs.msg import Twist
            
            # Create Twist message
            twist = Twist()
            
            # Set linear and angular velocities based on command
            if command == "move_forward":
                twist.linear.x = params.get("speed", 0.5)
            elif command == "move_backward":
                twist.linear.x = -params.get("speed", 0.5)
            elif command == "turn_left":
                twist.angular.z = params.get("angular_speed", 0.5)
            elif command == "turn_right":
                twist.angular.z = -params.get("angular_speed", 0.5)
            
            # Publish command
            self.publishers["cmd_vel"].publish(twist)
            
            return {
                "success": True,
                "command": command,
                "params": params,
                "position": self.position
            }
        
        except Exception as e:
            logger.error(f"Error sending command to ROS: {e}")
            return {"success": False, "error": str(e)}
    
    def read_sensors(self) -> Dict[str, Any]:
        """
        Read sensor data from ROS topics.
        
        Returns:
            Dictionary of sensor data
        """
        # Return latest sensor data
        sensor_data = self.sensor_data.copy()
        
        # Add position data from odometry
        sensor_data["position"] = self.position.copy()
        
        return sensor_data 
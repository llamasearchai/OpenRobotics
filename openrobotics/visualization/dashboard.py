"""
Web-based visualization dashboard for OpenRobotics.
"""

import os
import logging
import threading
import time
import json
from typing import Dict, List, Any, Optional, Union
from pathlib import Path

import numpy as np
from PIL import Image
import base64
import io

logger = logging.getLogger(__name__)

try:
    from fastapi import FastAPI, WebSocket, WebSocketDisconnect
    from fastapi.responses import HTMLResponse, JSONResponse
    from fastapi.staticfiles import StaticFiles
    import uvicorn
    HAS_FASTAPI = True
except ImportError:
    HAS_FASTAPI = False
    logger.warning("FastAPI not installed. Dashboard functionality will be unavailable.")

class DashboardManager:
    """Manager for robot visualization dashboard."""
    
    def __init__(
        self,
        host: str = "localhost",
        port: int = 8050,
        update_interval: float = 0.1,
        static_dir: Optional[str] = None
    ):
        """
        Initialize dashboard manager.
        
        Args:
            host: Host to bind server
            port: Port to bind server
            update_interval: Dashboard update interval in seconds
            static_dir: Directory for static dashboard files
        """
        if not HAS_FASTAPI:
            raise ImportError("FastAPI is required for dashboard functionality")
        
        self.host = host
        self.port = port
        self.update_interval = update_interval
        
        # Find static files directory
        if static_dir:
            self.static_dir = Path(static_dir)
        else:
            # Default to package directory / visualization / static
            self.static_dir = Path(__file__).parent / "static"
            if not self.static_dir.exists():
                self.static_dir.mkdir(parents=True)
                # Create basic HTML file if it doesn't exist
                self._create_default_dashboard()
        
        # Initialize FastAPI app
        self.app = FastAPI(title="OpenRobotics Dashboard")
        
        # Mount static files
        self.app.mount("/static", StaticFiles(directory=str(self.static_dir)), name="static")
        
        # Initialize WebSocket connection manager
        self.connection_manager = ConnectionManager()
        
        # Set up routes
        self._setup_routes()
        
        # Data storage
        self.robots = {}
        self.simulations = {}
        self.running = False
        self.update_thread = None
    
    def _setup_routes(self):
        """Set up dashboard routes."""
        @self.app.get("/", response_class=HTMLResponse)
        async def get_dashboard():
            """Serve dashboard HTML page."""
            index_path = self.static_dir / "index.html"
            if index_path.exists():
                with open(index_path, "r") as f:
                    return f.read()
            else:
                return "<html><body><h1>OpenRobotics Dashboard</h1><p>Dashboard static files not found.</p></body></html>"
        
        @self.app.websocket("/ws")
        async def websocket_endpoint(websocket: WebSocket):
            """WebSocket endpoint for real-time updates."""
            await self.connection_manager.connect(websocket)
            try:
                while True:
                    # Receive message from client
                    data = await websocket.receive_text()
                    
                    # Process client message
                    try:
                        message = json.loads(data)
                        await self._handle_client_message(websocket, message)
                    except json.JSONDecodeError:
                        logger.error(f"Invalid JSON received: {data}")
            except WebSocketDisconnect:
                self.connection_manager.disconnect(websocket)
        
        @self.app.get("/api/robots", response_class=JSONResponse)
        async def get_robots():
            """Get list of available robots."""
            return {"robots": list(self.robots.keys())}
        
        @self.app.get("/api/robots/{robot_name}", response_class=JSONResponse)
        async def get_robot_data(robot_name: str):
            """Get data for a specific robot."""
            if robot_name in self.robots:
                return self.robots[robot_name].get_dashboard_data()
            return {"error": f"Robot '{robot_name}' not found"}
    
    async def _handle_client_message(self, websocket: WebSocket, message: Dict):
        """Handle message from client."""
        if "type" not in message:
            await websocket.send_json({"error": "Message type not specified"})
            return
        
        msg_type = message["type"]
        
        if msg_type == "subscribe":
            # Subscribe to robot or simulation updates
            if "robot" in message:
                robot_name = message["robot"]
                if robot_name in self.robots:
                    self.connection_manager.subscribe(websocket, f"robot:{robot_name}")
                    await websocket.send_json({"type": "subscribed", "robot": robot_name})
                else:
                    await websocket.send_json({"error": f"Robot '{robot_name}' not found"})
            elif "simulation" in message:
                sim_name = message["simulation"]
                if sim_name in self.simulations:
                    self.connection_manager.subscribe(websocket, f"simulation:{sim_name}")
                    await websocket.send_json({"type": "subscribed", "simulation": sim_name})
                else:
                    await websocket.send_json({"error": f"Simulation '{sim_name}' not found"})
        
        elif msg_type == "command":
            # Send command to robot
            if "robot" in message and "command" in message:
                robot_name = message["robot"]
                command = message["command"]
                parameters = message.get("parameters", {})
                
                if robot_name in self.robots:
                    robot = self.robots[robot_name]
                    # Execute command
                    try:
                        result = robot.execute_motion(command, **parameters)
                        await websocket.send_json({"type": "command_result", "success": True, "result": result})
                    except Exception as e:
                        await websocket.send_json({"type": "command_result", "success": False, "error": str(e)})
                else:
                    await websocket.send_json({"error": f"Robot '{robot_name}' not found"})
    
    def _create_default_dashboard(self):
        """Create default dashboard files if they don't exist."""
        index_html = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>OpenRobotics Dashboard</title>
    <style>
        body { 
            font-family: Arial, sans-serif; 
            margin: 0;
            padding: 0;
            background-color: #f5f5f5;
        }
        .container {
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
        }
        header {
            background-color: #333;
            color: white;
            padding: 10px 20px;
            display: flex;
            justify-content: space-between;
            align-items: center;
        }
        h1 { margin: 0; }
        .robots-container {
            display: flex;
            flex-wrap: wrap;
            gap: 20px;
            margin-top: 20px;
        }
        .robot-card {
            background-color: white;
            border-radius: 8px;
            box-shadow: 0 2px 5px rgba(0,0,0,0.1);
            padding: 15px;
            width: 100%;
            max-width: 500px;
        }
        .robot-header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 15px;
            border-bottom: 1px solid #eee;
            padding-bottom: 10px;
        }
        .robot-status {
            display: inline-block;
            padding: 5px 10px;
            border-radius: 15px;
            font-size: 12px;
            font-weight: bold;
        }
        .status-active {
            background-color: #d4edda;
            color: #155724;
        }
        .status-inactive {
            background-color: #f8d7da;
            color: #721c24;
        }
        .visualization {
            width: 100%;
            height: 300px;
            background-color: #eee;
            border-radius: 5px;
            margin-bottom: 15px;
            overflow: hidden;
            position: relative;
        }
        .visualization canvas {
            width: 100%;
            height: 100%;
        }
        .controls {
            display: flex;
            gap: 10px;
            margin-top: 10px;
        }
        button {
            padding: 8px 15px;
            border: none;
            border-radius: 4px;
            background-color: #4CAF50;
            color: white;
            cursor: pointer;
        }
        button:hover {
            background-color: #45a049;
        }
        .sensor-data {
            font-family: monospace;
            background-color: #f8f9fa;
            padding: 10px;
            border-radius: 4px;
            margin-top: 15px;
            max-height: 150px;
            overflow-y: auto;
        }
        .loading {
            text-align: center;
            padding: 40px;
            font-size: 20px;
            color: #666;
        }
    </style>
</head>
<body>
    <header>
        <h1>OpenRobotics Dashboard</h1>
        <div id="connection-status">
            <span id="status-indicator" style="color: red;">⬤</span>
            <span id="status-text">Disconnected</span>
        </div>
    </header>
    
    <div class="container">
        <div id="robots-container" class="robots-container">
            <div class="loading">Loading robots...</div>
        </div>
    </div>
    
    <script>
        // WebSocket connection
        let ws;
        let robots = {};
        let reconnectAttempts = 0;
        const maxReconnectAttempts = 5;
        
        function connect() {
            ws = new WebSocket(`ws://${window.location.host}/ws`);
            
            ws.onopen = function() {
                document.getElementById('status-indicator').style.color = 'green';
                document.getElementById('status-text').textContent = 'Connected';
                reconnectAttempts = 0;
                
                // Get available robots
                fetch('/api/robots')
                    .then(response => response.json())
                    .then(data => {
                        if (data.robots && data.robots.length > 0) {
                            loadRobots(data.robots);
                        } else {
                            document.getElementById('robots-container').innerHTML = 
                                '<div class="robot-card"><p>No robots available.</p></div>';
                        }
                    })
                    .catch(err => {
                        console.error('Error fetching robots:', err);
                        document.getElementById('robots-container').innerHTML = 
                            '<div class="robot-card"><p>Error loading robots: ' + err.message + '</p></div>';
                    });
            };
            
            ws.onmessage = function(event) {
                const data = JSON.parse(event.data);
                handleWebSocketMessage(data);
            };
            
            ws.onclose = function() {
                document.getElementById('status-indicator').style.color = 'red';
                document.getElementById('status-text').textContent = 'Disconnected';
                
                // Attempt to reconnect
                if (reconnectAttempts < maxReconnectAttempts) {
                    reconnectAttempts++;
                    setTimeout(connect, 2000 * reconnectAttempts);
                } else {
                    document.getElementById('status-text').textContent = 'Connection failed';
                }
            };
            
            ws.onerror = function(error) {
                console.error('WebSocket error:', error);
            };
        }
        
        function loadRobots(robotNames) {
            const container = document.getElementById('robots-container');
            container.innerHTML = '';
            
            robotNames.forEach(robotName => {
                // Create robot card
                const robotCard = document.createElement('div');
                robotCard.className = 'robot-card';
                robotCard.id = `robot-${robotName}`;
                
                robotCard.innerHTML = `
                    <div class="robot-header">
                        <h2>${robotName}</h2>
                        <span class="robot-status status-inactive">Inactive</span>
                    </div>
                    <div class="visualization">
                        <canvas id="canvas-${robotName}"></canvas>
                    </div>
                    <div class="controls">
                        <button onclick="sendCommand('${robotName}', 'move_forward')">Forward</button>
                        <button onclick="sendCommand('${robotName}', 'move_backward')">Backward</button>
                        <button onclick="sendCommand('${robotName}', 'turn_left')">Left</button>
                        <button onclick="sendCommand('${robotName}', 'turn_right')">Right</button>
                        <button onclick="sendCommand('${robotName}', 'stop')">Stop</button>
                    </div>
                    <div class="sensor-data" id="sensor-data-${robotName}">No sensor data available</div>
                `;
                
                container.appendChild(robotCard);
                
                // Subscribe to robot updates
                if (ws && ws.readyState === WebSocket.OPEN) {
                    ws.send(JSON.stringify({
                        type: "subscribe",
                        robot: robotName
                    }));
                }
                
                // Initialize canvas for visualization
                initCanvas(robotName);
            });
        }
        
        function initCanvas(robotName) {
            const canvas = document.getElementById(`canvas-${robotName}`);
            const ctx = canvas.getContext('2d');
            
            // Set canvas dimensions
            canvas.width = canvas.offsetWidth;
            canvas.height = canvas.offsetHeight;
            
            // Draw placeholder
            ctx.fillStyle = '#ddd';
            ctx.fillRect(0, 0, canvas.width, canvas.height);
            ctx.fillStyle = '#666';
            ctx.font = '16px Arial';
            ctx.textAlign = 'center';
            ctx.fillText('Visualization will appear here', canvas.width/2, canvas.height/2);
        }
        
        function handleWebSocketMessage(data) {
            if (data.type === 'robot_update' && data.robot) {
                updateRobotCard(data.robot, data.data);
            } else if (data.type === 'error') {
                console.error('Error from server:', data.error);
            }
        }
        
        function updateRobotCard(robotName, robotData) {
            const statusElem = document.querySelector(`#robot-${robotName} .robot-status`);
            const sensorDataElem = document.getElementById(`sensor-data-${robotName}`);
            const canvas = document.getElementById(`canvas-${robotName}`);
            
            if (statusElem) {
                statusElem.textContent = 'Active';
                statusElem.className = 'robot-status status-active';
            }
            
            if (sensorDataElem && robotData.sensor_data) {
                sensorDataElem.textContent = JSON.stringify(robotData.sensor_data, null, 2);
            }
            
            if (canvas && robotData.visualization) {
                // Update canvas with visualization data
                updateCanvas(canvas, robotData.visualization);
            }
        }
        
        function updateCanvas(canvas, visualizationData) {
            const ctx = canvas.getContext('2d');
            
            // Clear canvas
            ctx.clearRect(0, 0, canvas.width, canvas.height);
            
            if (visualizationData.image) {
                // Draw image data if available
                const img = new Image();
                img.onload = function() {
                    ctx.drawImage(img, 0, 0, canvas.width, canvas.height);
                };
                img.src = visualizationData.image;
            } else if (visualizationData.position) {
                // Draw simple position visualization
                const pos = visualizationData.position;
                const scale = Math.min(canvas.width, canvas.height) / 100;
                
                // Draw environment
                ctx.fillStyle = '#f0f0f0';
                ctx.fillRect(0, 0, canvas.width, canvas.height);
                
                // Draw grid
                ctx.strokeStyle = '#ddd';
                ctx.lineWidth = 1;
                for (let i = 0; i <= 100; i += 10) {
                    // Vertical lines
                    ctx.beginPath();
                    ctx.moveTo(i * scale, 0);
                    ctx.lineTo(i * scale, canvas.height);
                    ctx.stroke();
                    
                    // Horizontal lines
                    ctx.beginPath();
                    ctx.moveTo(0, i * scale);
                    ctx.lineTo(canvas.width, i * scale);
                    ctx.stroke();
                }
                
                // Draw robot
                const x = pos.x * scale;
                const y = canvas.height - pos.y * scale;
                
                ctx.fillStyle = '#4CAF50';
                ctx.beginPath();
                ctx.arc(x, y, 5, 0, 2 * Math.PI);
                ctx.fill();
                
                // Draw direction
                const angle = pos.theta || 0;
                const lineLength = 15;
                ctx.strokeStyle = '#333';
                ctx.lineWidth = 2;
                ctx.beginPath();
                ctx.moveTo(x, y);
                ctx.lineTo(
                    x + lineLength * Math.cos(angle),
                    y - lineLength * Math.sin(angle)
                );
                ctx.stroke();
            }
        }
        
        function sendCommand(robotName, command, parameters = {}) {
            if (ws && ws.readyState === WebSocket.OPEN) {
                ws.send(JSON.stringify({
                    type: "command",
                    robot: robotName,
                    command: command,
                    parameters: parameters
                }));
            } else {
                console.error('WebSocket not connected');
            }
        }
        
        // Connect when page loads
        window.onload = connect;
    </script>
</body>
</html>
"""
        
        # Write index.html to static_dir
        with open(self.static_dir / "index.html", "w") as f:
            f.write(index_html)
    
    def add_robot(self, robot):
        """
        Add robot to dashboard.
        
        Args:
            robot: Robot instance to monitor
        """
        self.robots[robot.name] = robot
    
    def add_simulation(self, simulation):
        """
        Add simulation to dashboard.
        
        Args:
            simulation: Simulation instance to monitor
        """
        sim_name = getattr(simulation, "name", f"simulation_{len(self.simulations)}")
        self.simulations[sim_name] = simulation
    
    def start(self, background: bool = True):
        """
        Start the dashboard server.
        
        Args:
            background: Run in background thread if True
        """
        if not HAS_FASTAPI:
            logger.error("FastAPI is required for dashboard functionality")
            return
        
        if self.running:
            logger.warning("Dashboard is already running")
            return
        
        self.running = True
        
        # Start update thread
        self.update_thread = threading.Thread(
            target=self._update_loop,
            daemon=True
        )
        self.update_thread.start()
        
        if background:
            # Start server in background thread
            server_thread = threading.Thread(
                target=self._run_server,
                daemon=True
            )
            server_thread.start()
            logger.info(f"Dashboard started on http://{self.host}:{self.port}")
            return server_thread
        else:
            # Run server in current thread (blocking)
            self._run_server()
    
    def _run_server(self):
        """Run the FastAPI server."""
        uvicorn.run(
            self.app,
            host=self.host,
            port=self.port,
            log_level="info"
        )
    
    def _update_loop(self):
        """Update loop for sending real-time updates to clients."""
        while self.running:
            try:
                # Update robot data
                for name, robot in self.robots.items():
                    if self.connection_manager.has_subscribers(f"robot:{name}"):
                        # Get robot data
                        data = self._get_robot_visualization_data(robot)
                        
                        # Send update to subscribers
                        asyncio_task = self.connection_manager.broadcast(
                            {"type": "robot_update", "robot": name, "data": data},
                            f"robot:{name}"
                        )
                        
                        # Force event loop to run the task if we're not in an async context
                        import asyncio
                        try:
                            loop = asyncio.get_event_loop()
                            if loop.is_running():
                                # Loop is already running, just create task
                                loop.create_task(asyncio_task)
                            else:
                                # Run the task
                                loop.run_until_complete(asyncio_task)
                        except RuntimeError:
                            # No event loop in this thread, create new one
                            loop = asyncio.new_event_loop()
                            asyncio.set_event_loop(loop)
                            loop.run_until_complete(asyncio_task)
                
                # Update simulation data
                for name, sim in self.simulations.items():
                    if self.connection_manager.has_subscribers(f"simulation:{name}"):
                        # Get simulation data
                        data = self._get_simulation_visualization_data(sim)
                        
                        # Send update to subscribers
                        asyncio_task = self.connection_manager.broadcast(
                            {"type": "simulation_update", "simulation": name, "data": data},
                            f"simulation:{name}"
                        )
                        
                        # Force event loop to run the task
                        import asyncio
                        try:
                            loop = asyncio.get_event_loop()
                            if loop.is_running():
                                loop.create_task(asyncio_task)
                            else:
                                loop.run_until_complete(asyncio_task)
                        except RuntimeError:
                            loop = asyncio.new_event_loop()
                            asyncio.set_event_loop(loop)
                            loop.run_until_complete(asyncio_task)
            
            except Exception as e:
                logger.error(f"Error in dashboard update loop: {e}")
            
            # Sleep before next update
            time.sleep(self.update_interval)
    
    def _get_robot_visualization_data(self, robot):
        """Get visualization data for a robot."""
        data = {}
        
        # Position data
        try:
            data["position"] = robot.get_position()
        except:
            data["position"] = {"x": 0, "y": 0, "theta": 0}
        
        # Sensor data
        try:
            data["sensor_data"] = robot.get_sensor_data()
        except:
            data["sensor_data"] = {}
        
        # Camera image if available
        try:
            camera_data = robot.get_sensor_data("camera")
            if camera_data and "image" in camera_data:
                # Convert image to base64 for sending via JSON
                image = camera_data["image"]
                if isinstance(image, np.ndarray):
                    pil_img = Image.fromarray(image)
                    buffered = io.BytesIO()
                    pil_img.save(buffered, format="JPEG")
                    img_str = base64.b64encode(buffered.getvalue()).decode()
                    data["visualization"] = {"image": f"data:image/jpeg;base64,{img_str}"}
        except:
            pass
        
        # Add simple visualization if no camera
        if "visualization" not in data:
            data["visualization"] = {"position": data["position"]}
        
        return data
    
    def _get_simulation_visualization_data(self, simulation):
        """Get visualization data for a simulation."""
        data = {}
        
        # Robot position in simulation
        try:
            data["position"] = simulation.robot.get_position()
        except:
            data["position"] = {"x": 0, "y": 0, "theta": 0}
        
        # Simulation state
        try:
            data["time"] = simulation.time
            data["running"] = simulation.running
            data["obstacles"] = simulation.obstacles
        except:
            data["time"] = 0
            data["running"] = False
            data["obstacles"] = []
        
        # Add visualization data
        data["visualization"] = {
            "position": data["position"],
            "obstacles": data["obstacles"],
            "environment_size": getattr(simulation, "environment_size", (100, 100))
        }
        
        return data
    
    def stop(self):
        """Stop the dashboard server."""
        self.running = False
        if self.update_thread:
            self.update_thread.join(timeout=1.0)
        logger.info("Dashboard stopped")


class ConnectionManager:
    """WebSocket connection manager."""
    
    def __init__(self):
        """Initialize connection manager."""
        self.active_connections = []
        self.subscriptions = {}
    
    async def connect(self, websocket: WebSocket):
        """Connect a new client."""
        await websocket.accept()
        self.active_connections.append(websocket)
    
    def disconnect(self, websocket: WebSocket):
        """Disconnect a client."""
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)
        
        # Remove any subscriptions
        for topic, subscribers in list(self.subscriptions.items()):
            if websocket in subscribers:
                subscribers.remove(websocket)
                if not subscribers:
                    del self.subscriptions[topic]
    
    def subscribe(self, websocket: WebSocket, topic: str):
        """Subscribe a client to a topic."""
        if topic not in self.subscriptions:
            self.subscriptions[topic] = []
        
        if websocket not in self.subscriptions[topic]:
            self.subscriptions[topic].append(websocket)
    
    def has_subscribers(self, topic: str) -> bool:
        """Check if a topic has subscribers."""
        return topic in self.subscriptions and len(self.subscriptions[topic]) > 0
    
    async def broadcast(self, message: Dict, topic: Optional[str] = None):
        """
        Broadcast a message to clients.
        
        Args:
            message: Message to broadcast
            topic: Optional topic to filter recipients
        """
        if topic:
            # Send to subscribers of the topic
            if topic in self.subscriptions:
                disconnected = []
                for websocket in self.subscriptions[topic]:
                    try:
                        await websocket.send_json(message)
                    except WebSocketDisconnect:
                        disconnected.append(websocket)
                
                # Clean up any disconnected websockets
                for websocket in disconnected:
                    self.disconnect(websocket)
        else:
            # Broadcast to all connected clients
            disconnected = []
            for websocket in self.active_connections:
                try:
                    await websocket.send_json(message)
                except WebSocketDisconnect:
                    disconnected.append(websocket)
            
            # Clean up any disconnected websockets
            for websocket in disconnected:
                self.disconnect(websocket) 
'''
Database utilities for robotics applications.
'''

import os
import sqlite3
import json
import time
import logging
from typing import Dict, List, Optional, Any, Union
from pathlib import Path
import threading

try:
    from sqlite_utils import Database
    HAS_SQLITE_UTILS = True
except ImportError:
    HAS_SQLITE_UTILS = False
    print("sqlite-utils not found. Using standard sqlite3 instead.")

logger = logging.getLogger(__name__)


class RoboticsDB:
    '''
    SQLite database for robotics applications.
    
    This class provides a simple interface for storing and retrieving
    robotics data in a SQLite database.
    '''
    
    def __init__(self, db_path: Union[str, Path] = "robotics.db"):
        '''
        Initialize a robotics database.
        
        Args:
            db_path: Path to SQLite database file
        '''
        self.db_path = Path(db_path)
        self.lock = threading.RLock()
        
        # Ensure directory exists
        os.makedirs(self.db_path.parent, exist_ok=True)
        
        # Initialize database
        self._initialize_db()
    
    def _initialize_db(self):
        '''Initialize database tables.'''
        logger.info(f"Initializing database at {self.db_path}")
        
        with self.lock:
            conn = self._get_connection()
            try:
                cursor = conn.cursor()
                
                # Create tables if they don't exist
                
                # Observations table
                cursor.execute('''
                CREATE TABLE IF NOT EXISTS observations (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    robot_id TEXT,
                    timestamp REAL,
                    type TEXT,
                    data TEXT,
                    metadata TEXT
                )
                ''')
                
                # Commands table
                cursor.execute('''
                CREATE TABLE IF NOT EXISTS commands (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    robot_id TEXT,
                    timestamp REAL,
                    command TEXT,
                    parameters TEXT,
                    result TEXT,
                    status TEXT
                )
                ''')
                
                # LLM prompts table
                cursor.execute('''
                CREATE TABLE IF NOT EXISTS llm_prompts (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp REAL,
                    prompt TEXT,
                    response TEXT,
                    model TEXT,
                    metadata TEXT
                )
                ''')
                
                # Create indices
                cursor.execute('CREATE INDEX IF NOT EXISTS idx_observations_robot_time ON observations (robot_id, timestamp)')
                cursor.execute('CREATE INDEX IF NOT EXISTS idx_observations_type ON observations (type)')
                cursor.execute('CREATE INDEX IF NOT EXISTS idx_commands_robot_time ON commands (robot_id, timestamp)')
                cursor.execute('CREATE INDEX IF NOT EXISTS idx_llm_prompts_time ON llm_prompts (timestamp)')
                
                conn.commit()
            finally:
                conn.close()
    
    def _get_connection(self) -> sqlite3.Connection:
        '''
        Get a SQLite connection.
        
        Returns:
            SQLite connection
        '''
        return sqlite3.connect(self.db_path)
    
    def log_observation(self, observation: Dict[str, Any]) -> int:
        '''
        Log an observation to the database.
        
        Args:
            observation: Observation data
            
        Returns:
            Observation ID
        '''
        with self.lock:
            conn = self._get_connection()
            try:
                cursor = conn.cursor()
                
                # Set timestamp if not provided
                if "timestamp" not in observation:
                    observation["timestamp"] = time.time()
                
                # Convert data and metadata to JSON if they're dicts
                if "data" in observation and isinstance(observation["data"], dict):
                    observation["data"] = json.dumps(observation["data"])
                
                if "metadata" in observation and isinstance(observation["metadata"], dict):
                    observation["metadata"] = json.dumps(observation["metadata"])
                
                # Insert into database
                cursor.execute(
                    '''
                    INSERT INTO observations (robot_id, timestamp, type, data, metadata)
                    VALUES (?, ?, ?, ?, ?)
                    ''',
                    (
                        observation.get("robot_id", ""),
                        observation["timestamp"],
                        observation.get("type", ""),
                        observation.get("data", ""),
                        observation.get("metadata", ""),
                    )
                )
                
                conn.commit()
                
                # Get the last row ID
                observation_id = cursor.lastrowid
                
                return observation_id
            finally:
                conn.close()
    
    def get_observation(self, observation_id: int) -> Optional[Dict[str, Any]]:
        '''
        Get an observation from the database.
        
        Args:
            observation_id: Observation ID
            
        Returns:
            Observation data or None if not found
        '''
        with self.lock:
            conn = self._get_connection()
            try:
                cursor = conn.cursor()
                
                # Query database
                cursor.execute(
                    'SELECT id, robot_id, timestamp, type, data, metadata FROM observations WHERE id = ?',
                    (observation_id,)
                )
                
                row = cursor.fetchone()
                
                if row is None:
                    return None
                
                # Parse JSON fields
                observation = {
                    "id": row[0],
                    "robot_id": row[1],
                    "timestamp": row[2],
                    "type": row[3],
                    "data": row[4],
                    "metadata": row[5],
                }
                
                for field in ["data", "metadata"]:
                    if observation[field]:
                        try:
                            observation[field] = json.loads(observation[field])
                        except json.JSONDecodeError:
                            pass
                
                return observation
            finally:
                conn.close()
    
    def get_latest_observations(
        self,
        limit: int = 10,
        robot_id: Optional[str] = None,
        observation_type: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        '''
        Get the latest observations from the database.
        
        Args:
            limit: Maximum number of observations to return
            robot_id: Optional robot ID to filter by
            observation_type: Optional observation type to filter by
            
        Returns:
            List of observation data
        '''
        with self.lock:
            conn = self._get_connection()
            try:
                cursor = conn.cursor()
                
                # Build query
                query = 'SELECT id, robot_id, timestamp, type, data, metadata FROM observations'
                params = []
                
                where_clauses = []
                if robot_id:
                    where_clauses.append("robot_id = ?")
                    params.append(robot_id)
                
                if observation_type:
                    where_clauses.append("type = ?")
                    params.append(observation_type)
                
                if where_clauses:
                    query += ' WHERE ' + ' AND '.join(where_clauses)
                
                query += ' ORDER BY timestamp DESC LIMIT ?'
                params.append(limit)
                
                # Execute query
                cursor.execute(query, params)
                
                # Process results
                observations = []
                for row in cursor.fetchall():
                    observation = {
                        "id": row[0],
                        "robot_id": row[1],
                        "timestamp": row[2],
                        "type": row[3],
                        "data": row[4],
                        "metadata": row[5],
                    }
                    
                    # Parse JSON fields
                    for field in ["data", "metadata"]:
                        if observation[field]:
                            try:
                                observation[field] = json.loads(observation[field])
                            except json.JSONDecodeError:
                                pass
                    
                    observations.append(observation)
                
                return observations
            finally:
                conn.close()
    
    def log_command(self, command_data: Dict[str, Any]) -> int:
        '''
        Log a command to the database.
        
        Args:
            command_data: Command data
            
        Returns:
            Command ID
        '''
        with self.lock:
            conn = self._get_connection()
            try:
                cursor = conn.cursor()
                
                # Set timestamp if not provided
                if "timestamp" not in command_data:
                    command_data["timestamp"] = time.time()
                
                # Convert parameters and result to JSON if they're dicts
                if "parameters" in command_data and isinstance(command_data["parameters"], dict):
                    command_data["parameters"] = json.dumps(command_data["parameters"])
                
                if "result" in command_data and isinstance(command_data["result"], dict):
                    command_data["result"] = json.dumps(command_data["result"])
                
                # Insert into database
                cursor.execute(
                    '''
                    INSERT INTO commands (robot_id, timestamp, command, parameters, result, status)
                    VALUES (?, ?, ?, ?, ?, ?)
                    ''',
                    (
                        command_data.get("robot_id", ""),
                        command_data["timestamp"],
                        command_data.get("command", ""),
                        command_data.get("parameters", ""),
                        command_data.get("result", ""),
                        command_data.get("status", ""),
                    )
                )
                
                conn.commit()
                
                # Get the last row ID
                command_id = cursor.lastrowid
                
                return command_id
            finally:
                conn.close()
    
    def log_llm_prompt(self, prompt_data: Dict[str, Any]) -> int:
        '''
        Log an LLM prompt to the database.
        
        Args:
            prompt_data: Prompt data
            
        Returns:
            Prompt ID
        '''
        with self.lock:
            conn = self._get_connection()
            try:
                cursor = conn.cursor()
                
                # Set timestamp if not provided
                if "timestamp" not in prompt_data:
                    prompt_data["timestamp"] = time.time()
                
                # Convert metadata to JSON if it's a dict
                if "metadata" in prompt_data and isinstance(prompt_data["metadata"], dict):
                    prompt_data["metadata"] = json.dumps(prompt_data["metadata"])
                
                # Insert into database
                cursor.execute(
                    '''
                    INSERT INTO llm_prompts (timestamp, prompt, response, model, metadata)
                    VALUES (?, ?, ?, ?, ?)
                    ''',
                    (
                        prompt_data["timestamp"],
                        prompt_data.get("prompt", ""),
                        prompt_data.get("response", ""),
                        prompt_data.get("model", ""),
                        prompt_data.get("metadata", ""),
                    )
                )
                
                conn.commit()
                
                # Get the last row ID
                prompt_id = cursor.lastrowid
                
                return prompt_id
            finally:
                conn.close()
    
    def query(self, sql: str, params: Optional[List[Any]] = None) -> List[Dict[str, Any]]:
        '''
        Execute a SQL query on the database.
        
        Args:
            sql: SQL query
            params: Optional query parameters
            
        Returns:
            Query results
        '''
        with self.lock:
            conn = self._get_connection()
            try:
                # Set row factory to return dictionaries
                conn.row_factory = sqlite3.Row
                
                cursor = conn.cursor()
                
                # Execute query
                cursor.execute(sql, params or [])
                
                # Fetch results
                rows = cursor.fetchall()
                
                # Convert to list of dictionaries
                results = [dict(row) for row in rows]
                
                return results
            finally:
                conn.close() 
"""
LLM agents for robotics control and decision making.
"""

import logging
import os
import json
from typing import Dict, List, Optional, Any, Union, Callable

try:
    import openai
    from langchain.agents import AgentExecutor, create_openai_tools_agent
    from langchain.agents.format_scratchpad import format_to_openai_tool_messages
    from langchain.chat_models import ChatOpenAI
    from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
    from langchain.schema import AIMessage, HumanMessage
    from langchain.schema.runnable import RunnablePassthrough
    from langchain.tools import Tool, BaseTool
    from langchain.memory import ConversationBufferMemory
    HAS_LANGCHAIN = True
except ImportError:
    HAS_LANGCHAIN = False
    print("LangChain not found. LLM agent functionality will be limited.")

try:
    import dspy
    HAS_DSPY = True
except ImportError:
    HAS_DSPY = False
    print("DSPy not found. Some advanced LLM capabilities will be disabled.")

from openrobotics.config import config
from openrobotics.robot_control.robot import Robot

logger = logging.getLogger(__name__)


class LLMAgent:
    """
    LLM agent for robotics control and decision making.
    
    This class provides a high-level interface for using LLMs to control
    robots and make decisions in robotics applications.
    """
    
    def __init__(
        self,
        robot: Robot,
        model: str = "gpt-4",
        api_key: Optional[str] = None,
        temperature: float = 0.2,
        database = None,
        tools: Optional[List] = None,
        memory_key: str = "chat_history",
        verbose: bool = False,
    ):
        """
        Initialize an LLM agent.
        
        Args:
            robot: Robot instance to control
            model: LLM model name
            api_key: API key for the LLM provider
            temperature: Temperature for LLM generation
            database: Optional database instance for data access
            tools: Optional list of additional tools
            memory_key: Key for conversation memory
            verbose: Whether to enable verbose logging
        """
        self.robot = robot
        self.model_name = model
        self.temperature = temperature
        self.database = database
        self.verbose = verbose
        self.memory_key = memory_key
        
        # Get API key from config if not provided
        if api_key is None:
            if "gpt" in model.lower() or "openai" in model.lower():
                api_key = config.get("llm", "openai_api_key")
                if not api_key:
                    api_key = os.environ.get("OPENAI_API_KEY")
            elif "claude" in model.lower() or "anthropic" in model.lower():
                api_key = config.get("llm", "anthropic_api_key")
                if not api_key:
                    api_key = os.environ.get("ANTHROPIC_API_KEY")
        
        if HAS_LANGCHAIN:
            # Create LLM
            self.llm = ChatOpenAI(
                model=model,
                temperature=temperature,
                api_key=api_key,
                verbose=verbose
            )
            
            # Set up memory
            self.memory = ConversationBufferMemory(
                memory_key=memory_key,
                return_messages=True
            )
            
            # Create tools
            self.tools = self._create_default_tools()
            if tools:
                self.tools.extend(tools)
            
            # Create agent
            self.agent_executor = self._create_agent()
        else:
            self.llm = SimpleOpenAI(model=model, api_key=api_key, temperature=temperature)
            self.memory = None
            self.tools = []
            self.agent_executor = None
    
    def _create_default_tools(self) -> List:
        """Create default tools for the agent."""
        if not HAS_LANGCHAIN:
            return []
            
        robot = self.robot
        database = self.database
        
        tools = [
            Tool(
                name="robot_execute_motion",
                description="Execute a motion command on the robot. Commands include 'move_forward', 'move_backward', 'turn_left', 'turn_right', 'stop'.",
                func=lambda cmd: robot.execute_motion(cmd)
            ),
            Tool(
                name="robot_get_position",
                description="Get the current position of the robot.",
                func=lambda _: json.dumps(robot.get_position())
            ),
            Tool(
                name="robot_get_sensor_data",
                description="Get the latest sensor data from the robot. Can specify sensor name or get all sensors.",
                func=lambda sensor_name=None: json.dumps(robot.get_sensor_data(sensor_name))
            ),
        ]
        
        # Add database tools if database is provided
        if database:
            tools.extend([
                Tool(
                    name="database_query",
                    description="Query the robotics database using SQL.",
                    func=lambda query: json.dumps(database.query(query), default=str)
                ),
                Tool(
                    name="database_get_latest_observations",
                    description="Get the latest observations from the database.",
                    func=lambda limit=10: json.dumps(database.get_latest_observations(limit), default=str)
                ),
            ])
        
        return tools
    
    def _create_agent(self):
        """Create LangChain agent."""
        if not HAS_LANGCHAIN:
            return None
            
        # Create prompt
        prompt = ChatPromptTemplate.from_messages([
            ("system", "You are an advanced AI assistant that helps control a robot. "
                     "You have access to the robot's sensors and can execute motion commands. "
                     "Always interpret commands carefully and prioritize safety. "
                     "If something is unclear, ask for clarification rather than making assumptions. "
                     "When executing motion commands, start with small, safe movements. "
                     "You can use the available tools to interact with the robot and database."),
            MessagesPlaceholder(variable_name=self.memory_key),
            ("human", "{input}"),
            MessagesPlaceholder(variable_name="agent_scratchpad"),
        ])
        
        # Create agent
        agent = create_openai_tools_agent(self.llm, self.tools, prompt)
        
        # Create agent executor
        agent_executor = AgentExecutor(
            agent=agent,
            tools=self.tools,
            verbose=self.verbose,
            memory=self.memory,
            handle_parsing_errors=True
        )
        
        return agent_executor
    
    def query(self, query: str) -> str:
        """
        Send a query to the LLM.
        
        Args:
            query: Query text
            
        Returns:
            LLM response
        """
        if HAS_LANGCHAIN:
            response = self.llm.invoke(query)
            return response.content
        else:
            response = self.llm.generate(query)
            return response
    
    def execute(self, instruction: str) -> Dict[str, Any]:
        """
        Execute an instruction using the agent.
        
        Args:
            instruction: Instruction text
            
        Returns:
            Agent response
        """
        if HAS_LANGCHAIN and self.agent_executor:
            return self.agent_executor.invoke({"input": instruction})
        else:
            # Fallback to simple query when LangChain is not available
            response = self.query(instruction)
            return {"output": response}


class SimpleOpenAI:
    """Simple OpenAI client for when LangChain is not available."""
    
    def __init__(self, model: str, api_key: Optional[str] = None, temperature: float = 0.7):
        """
        Initialize a simple OpenAI client.
        
        Args:
            model: Model name
            api_key: API key
            temperature: Temperature for generation
        """
        self.model = model
        self.temperature = temperature
        
        # Set API key
        if api_key:
            openai.api_key = api_key
        elif os.environ.get("OPENAI_API_KEY"):
            openai.api_key = os.environ.get("OPENAI_API_KEY")
        else:
            raise ValueError("OpenAI API key must be provided")
    
    def generate(self, prompt: str) -> str:
        """
        Generate text from prompt.
        
        Args:
            prompt: Prompt text
            
        Returns:
            Generated text
        """
        try:
            response = openai.ChatCompletion.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=self.temperature,
            )
            return response.choices[0].message.content
        except Exception as e:
            logger.error(f"Error generating text: {e}")
            return f"Error: {str(e)}"


class MultiAgentOrchestrator:
    """
    Orchestrator for multiple LLM agents.
    
    This class coordinates multiple specialized LLM agents for
    complex robotics tasks.
    """
    
    def __init__(
        self,
        robot: Robot,
        database = None,
        model: str = "gpt-4",
        api_key: Optional[str] = None,
        temperature: float = 0.2,
        verbose: bool = False,
    ):
        """
        Initialize a multi-agent orchestrator.
        
        Args:
            robot: Robot instance
            database: Optional database instance
            model: LLM model name
            api_key: API key for LLM provider
            temperature: Temperature for LLM generation
            verbose: Whether to enable verbose logging
        """
        self.robot = robot
        self.database = database
        self.model_name = model
        self.api_key = api_key
        self.temperature = temperature
        self.verbose = verbose
        
        # Create specialized agents
        self.agents = {
            "perception": self._create_perception_agent(),
            "planning": self._create_planning_agent(),
            "execution": self._create_execution_agent(),
            "safety": self._create_safety_agent(),
        }
        
        # Create orchestration LLM
        if HAS_LANGCHAIN:
            self.orchestration_llm = ChatOpenAI(
                model=model,
                temperature=temperature,
                api_key=api_key,
                verbose=verbose
            )
        else:
            self.orchestration_llm = SimpleOpenAI(
                model=model,
                api_key=api_key,
                temperature=temperature
            )
    
    def _create_perception_agent(self) -> LLMAgent:
        """Create a specialized agent for perception tasks."""
        return LLMAgent(
            robot=self.robot,
            model=self.model_name,
            api_key=self.api_key,
            temperature=self.temperature,
            database=self.database,
            verbose=self.verbose
        )
    
    def _create_planning_agent(self) -> LLMAgent:
        """Create a specialized agent for planning tasks."""
        return LLMAgent(
            robot=self.robot,
            model=self.model_name,
            api_key=self.api_key,
            temperature=self.temperature,
            database=self.database,
            verbose=self.verbose
        )
    
    def _create_execution_agent(self) -> LLMAgent:
        """Create a specialized agent for execution tasks."""
        return LLMAgent(
            robot=self.robot,
            model=self.model_name,
            api_key=self.api_key,
            temperature=self.temperature,
            database=self.database,
            verbose=self.verbose
        )
    
    def _create_safety_agent(self) -> LLMAgent:
        """Create a specialized agent for safety monitoring."""
        return LLMAgent(
            robot=self.robot,
            model=self.model_name,
            api_key=self.api_key,
            temperature=0.1,  # Lower temperature for more deterministic safety checks
            database=self.database,
            verbose=self.verbose
        )
    
    def execute_task(self, task_description: str) -> Dict[str, Any]:
        """
        Execute a complex task using multiple agents.
        
        Args:
            task_description: Description of the task to execute
            
        Returns:
            Task execution results
        """
        if HAS_LANGCHAIN:
            # Implementation with LangChain
            from langchain.prompts import ChatPromptTemplate
            from langchain.chains import LLMChain
            
            # Create prompt for orchestration
            orchestration_prompt = ChatPromptTemplate.from_messages([
                ("system", "You are an orchestrator AI that coordinates multiple specialized agents for robotics tasks. "
                         "Your job is to break down complex tasks into subtasks and delegate them to the appropriate agents. "
                         "Available agents: perception, planning, execution, safety. "
                         "Always check with the safety agent before executing any physical actions."),
                ("human", "{input}"),
            ])
            
            # Create orchestration chain
            orchestration_chain = LLMChain(
                llm=self.orchestration_llm,
                prompt=orchestration_prompt,
                verbose=self.verbose
            )
            
            # Get orchestration plan
            orchestration_result = orchestration_chain.run(input=task_description)
            
            if self.verbose:
                logger.info(f"Orchestration plan: {orchestration_result}")
            
            # Create subtasks JSON
            subtasks_prompt = ChatPromptTemplate.from_messages([
                ("system", "Parse the following orchestration plan into structured subtasks for each agent. "
                         "Return a JSON object with keys 'perception', 'planning', 'execution', and 'safety', "
                         "each containing a list of subtasks for that agent."),
                ("human", orchestration_result),
            ])
            
            subtasks_chain = LLMChain(
                llm=self.orchestration_llm,
                prompt=subtasks_prompt,
                verbose=self.verbose
            )
            
            subtasks_json = subtasks_chain.run(input=orchestration_result)
            
            try:
                # Parse JSON response
                subtasks = json.loads(subtasks_json)
            except json.JSONDecodeError:
                logger.error(f"Failed to parse subtasks JSON: {subtasks_json}")
                # Fallback to a simple approach
                subtasks = {
                    "perception": ["Analyze the environment"],
                    "planning": ["Plan a path to the target"],
                    "execution": ["Execute the planned path"],
                    "safety": ["Monitor for obstacles"],
                }
        else:
            # Simple fallback when LangChain is not available
            subtasks = {
                "perception": ["Analyze the environment"],
                "planning": ["Plan a path to the target"],
                "execution": ["Execute the planned path"],
                "safety": ["Monitor for obstacles"],
            }
        
        # Execute subtasks with each agent
        results = {}
        
        # First, perception
        perception_results = []
        for subtask in subtasks.get("perception", []):
            result = self.agents["perception"].execute(subtask)
            perception_results.append(result)
        results["perception"] = perception_results
        
        # Then, planning
        planning_results = []
        for subtask in subtasks.get("planning", []):
            # Enhance planning with perception results
            enhanced_subtask = f"{subtask} based on these perceptions: {json.dumps(perception_results)}"
            result = self.agents["planning"].execute(enhanced_subtask)
            planning_results.append(result)
        results["planning"] = planning_results
        
        # Safety check before execution
        safety_results = []
        for subtask in subtasks.get("safety", []):
            # Check safety with planned actions
            safety_check = f"{subtask} for these planned actions: {json.dumps(planning_results)}"
            result = self.agents["safety"].execute(safety_check)
            safety_results.append(result)
            
            # If safety check fails, abort execution
            if isinstance(result, dict) and "output" in result and ("unsafe" in result["output"].lower() or "danger" in result["output"].lower()):
                logger.warning(f"Safety check failed: {result}")
                results["safety"] = safety_results
                results["execution"] = ["Aborted due to safety concerns"]
                return results
        
        results["safety"] = safety_results
        
        # Finally, execution
        execution_results = []
        for subtask in subtasks.get("execution", []):
            # Execute with planning results
            enhanced_subtask = f"{subtask} following this plan: {json.dumps(planning_results)}"
            result = self.agents["execution"].execute(enhanced_subtask)
            execution_results.append(result)
        results["execution"] = execution_results
        
        return results
    
    def get_agent(self, agent_type: str) -> Optional[LLMAgent]:
        """
        Get a specific agent by type.
        
        Args:
            agent_type: Type of agent ('perception', 'planning', 'execution', 'safety')
            
        Returns:
            LLMAgent instance or None if not found
        """
        return self.agents.get(agent_type)
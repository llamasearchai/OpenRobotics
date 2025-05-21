"""
LangChain chains for robotics applications.
"""

from typing import Dict, List, Optional, Any, Union, Callable

try:
    from langchain.chains import LLMChain, SequentialChain
    from langchain.chat_models import ChatOpenAI
    from langchain.prompts import ChatPromptTemplate
    from langchain.memory import ConversationBufferMemory
    HAS_LANGCHAIN = True
except ImportError:
    HAS_LANGCHAIN = False

from openrobotics.config import config


class RoboticsChain:
    """
    Collection of LangChain chains for robotics applications.
    
    This class provides pre-configured chains for common robotics tasks.
    """
    
    @staticmethod
    def create_instruction_parser(
        model: str = "gpt-4",
        temperature: float = 0.2,
        api_key: Optional[str] = None,
        memory: Optional[Any] = None,
        verbose: bool = False,
    ):
        """
        Create a chain for parsing natural language instructions.
        
        Args:
            model: LLM model name
            temperature: Temperature for LLM generation
            api_key: API key for LLM provider
            memory: Optional conversation memory
            verbose: Whether to enable verbose logging
            
        Returns:
            LLMChain for parsing instructions
        """
        if not HAS_LANGCHAIN:
            return None
            
        # Create LLM
        llm = ChatOpenAI(
            model=model,
            temperature=temperature,
            api_key=api_key,
            verbose=verbose
        )
        
        # Create prompt
        prompt = ChatPromptTemplate.from_messages([
            ("system", "You are an AI that converts natural language instructions into structured commands for a robot. "
                     "Your output should be a JSON object with the following structure:\n"
                     "{\n"
                     "  \"action\": \"move_forward\",  # The primary action to perform\n"
                     "  \"parameters\": {  # Parameters for the action\n"
                     "    \"distance\": 0.5,  # Example parameter\n"
                     "    \"speed\": 0.2  # Example parameter\n"
                     "  },\n"
                     "  \"constraints\": [  # Optional constraints\n"
                     "    \"avoid_obstacles\",\n"
                     "    \"max_acceleration: 0.1\"\n"
                     "  ]\n"
                     "}\n"
                     "Only include parameters that are specified or can be reasonably inferred."),
            ("human", "{instruction}"),
        ])
        
        # Create chain
        chain = LLMChain(
            llm=llm,
            prompt=prompt,
            memory=memory,
            verbose=verbose,
            output_key="parsed_instruction"
        )
        
        return chain
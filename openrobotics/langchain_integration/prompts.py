"""
Prompt templates for robotics applications.
"""

from typing import Dict, List, Optional, Any

try:
    from langchain.prompts import ChatPromptTemplate, PromptTemplate, MessagesPlaceholder
    from langchain.prompts.chat import SystemMessagePromptTemplate, HumanMessagePromptTemplate
    HAS_LANGCHAIN = True
except ImportError:
    HAS_LANGCHAIN = False


def create_prompt_template(
    template_type: str, 
    custom_instructions: Optional[str] = None
):
    """
    Create a prompt template for robotics applications.
    
    Args:
        template_type: Type of template to create
        custom_instructions: Optional custom instructions to include
        
    Returns:
        ChatPromptTemplate instance
    """
    if not HAS_LANGCHAIN:
        return None
        
    if template_type == "robot_control":
        system_message = (
            "You are an AI assistant that helps control a robot. "
            "You can interpret sensor data and issue commands to the robot. "
            "Always prioritize safety and avoid actions that could cause damage. "
            "When in doubt, choose the more conservative action. "
            "{custom_instructions}"
        )
        
        return ChatPromptTemplate.from_messages([
            SystemMessagePromptTemplate.from_template(system_message.format(
                custom_instructions=custom_instructions or ""
            )),
            MessagesPlaceholder(variable_name="chat_history"),
            HumanMessagePromptTemplate.from_template("{input}")
        ])
    
    elif template_type == "sensor_interpretation":
        system_message = (
            "You are an AI assistant that interprets sensor data from a robot. "
            "Your task is to analyze the sensor readings and extract useful information "
            "that can help with navigation, manipulation, or other robotic tasks. "
            "Focus on identifying patterns, anomalies, and potential hazards. "
            "{custom_instructions}"
        )
        
        return ChatPromptTemplate.from_messages([
            SystemMessagePromptTemplate.from_template(system_message.format(
                custom_instructions=custom_instructions or ""
            )),
            HumanMessagePromptTemplate.from_template(
                "Here is the sensor data:\n{sensor_data}\n\nPlease interpret this data."
            )
        ])
    
    else:
        # Default general-purpose template
        system_message = (
            "You are an AI assistant that helps with robotics tasks. "
            "Provide clear, concise responses that can be used in a robotics context. "
            "{custom_instructions}"
        )
        
        return ChatPromptTemplate.from_messages([
            SystemMessagePromptTemplate.from_template(system_message.format(
                custom_instructions=custom_instructions or ""
            )),
            MessagesPlaceholder(variable_name="chat_history"),
            HumanMessagePromptTemplate.from_template("{input}")
        ])
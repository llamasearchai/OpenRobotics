"""
Integration with local LLM models for robotics applications.
"""

import os
import logging
import json
import time
from typing import Dict, List, Any, Optional, Union, Callable
import base64
import tempfile
from pathlib import Path

import numpy as np
import requests

logger = logging.getLogger(__name__)

class LocalLLMProvider:
    """Base class for local LLM providers."""
    
    def __init__(self, model_name: str):
        self.model_name = model_name
    
    def generate(self, prompt: str, **kwargs) -> str:
        """Generate text from prompt."""
        raise NotImplementedError("Subclasses must implement generate")
    
    def embed(self, text: str) -> List[float]:
        """Generate embeddings from text."""
        raise NotImplementedError("Subclasses must implement embed")


class OllamaProvider(LocalLLMProvider):
    """Ollama local LLM provider."""
    
    def __init__(
        self, 
        model_name: str = "llama2", 
        host: str = "localhost", 
        port: int = 11434,
        timeout: float = 60.0
    ):
        """
        Initialize an Ollama provider.
        
        Args:
            model_name: Ollama model name
            host: Ollama host
            port: Ollama port
            timeout: Request timeout in seconds
        """
        super().__init__(model_name)
        self.base_url = f"http://{host}:{port}"
        self.timeout = timeout
    
    def generate(
        self, 
        prompt: str, 
        system_prompt: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 2048,
        stream: bool = False,
        **kwargs
    ) -> Union[str, Callable]:
        """
        Generate text from prompt using Ollama.
        
        Args:
            prompt: User prompt
            system_prompt: Optional system prompt
            temperature: Sampling temperature (0.0-1.0)
            max_tokens: Maximum tokens to generate
            stream: Whether to stream the response
            
        Returns:
            Generated text or generator if streaming
        """
        url = f"{self.base_url}/api/generate"
        
        payload = {
            "model": self.model_name,
            "prompt": prompt,
            "temperature": temperature,
            "num_predict": max_tokens,
            "stream": stream
        }
        
        if system_prompt:
            payload["system"] = system_prompt
        
        # Add any additional kwargs to payload
        payload.update({k: v for k, v in kwargs.items() if k not in payload})
        
        if stream:
            def stream_generator():
                try:
                    with requests.post(url, json=payload, stream=True, timeout=self.timeout) as response:
                        response.raise_for_status()
                        for line in response.iter_lines():
                            if not line:
                                continue
                            
                            try:
                                chunk = json.loads(line)
                                if "response" in chunk:
                                    yield chunk["response"]
                                if chunk.get("done", False):
                                    break
                            except json.JSONDecodeError:
                                logger.warning(f"Failed to decode JSON: {line}")
                except Exception as e:
                    logger.error(f"Error streaming from Ollama: {str(e)}")
                    raise
            
            return stream_generator()
        else:
            try:
                response = requests.post(url, json=payload, timeout=self.timeout)
                response.raise_for_status()
                return response.json()["response"]
            except Exception as e:
                logger.error(f"Error calling Ollama: {str(e)}")
                raise
    
    def embed(self, text: str) -> List[float]:
        """
        Generate embeddings from text using Ollama.
        
        Args:
            text: Text to embed
            
        Returns:
            List of embedding values
        """
        url = f"{self.base_url}/api/embeddings"
        
        payload = {
            "model": self.model_name,
            "prompt": text
        }
        
        try:
            response = requests.post(url, json=payload, timeout=self.timeout)
            response.raise_for_status()
            return response.json()["embedding"]
        except Exception as e:
            logger.error(f"Error getting embeddings from Ollama: {str(e)}")
            raise
    
    def chat(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.7,
        max_tokens: int = 2048,
        stream: bool = False,
        **kwargs
    ) -> Union[str, Callable]:
        """
        Chat with Ollama model.
        
        Args:
            messages: List of message dictionaries (role, content)
            temperature: Sampling temperature (0.0-1.0)
            max_tokens: Maximum tokens to generate
            stream: Whether to stream the response
            
        Returns:
            Generated text or generator if streaming
        """
        url = f"{self.base_url}/api/chat"
        
        payload = {
            "model": self.model_name,
            "messages": messages,
            "temperature": temperature,
            "num_predict": max_tokens,
            "stream": stream
        }
        
        # Add any additional kwargs to payload
        payload.update({k: v for k, v in kwargs.items() if k not in payload})
        
        if stream:
            def stream_generator():
                try:
                    with requests.post(url, json=payload, stream=True, timeout=self.timeout) as response:
                        response.raise_for_status()
                        for line in response.iter_lines():
                            if not line:
                                continue
                            
                            try:
                                chunk = json.loads(line)
                                if "message" in chunk and "content" in chunk["message"]:
                                    yield chunk["message"]["content"]
                                if chunk.get("done", False):
                                    break
                            except json.JSONDecodeError:
                                logger.warning(f"Failed to decode JSON: {line}")
                except Exception as e:
                    logger.error(f"Error streaming from Ollama: {str(e)}")
                    raise
            
            return stream_generator()
        else:
            try:
                response = requests.post(url, json=payload, timeout=self.timeout)
                response.raise_for_status()
                return response.json()["message"]["content"]
            except Exception as e:
                logger.error(f"Error calling Ollama chat: {str(e)}")
                raise


class LlamaProvider(LocalLLMProvider):
    """Llama.cpp server LLM provider."""
    
    def __init__(
        self, 
        model_name: str = "llama2", 
        host: str = "localhost", 
        port: int = 8080,
        timeout: float = 60.0
    ):
        """
        Initialize a Llama.cpp server provider.
        
        Args:
            model_name: Model name (for reference only)
            host: Llama.cpp server host
            port: Llama.cpp server port
            timeout: Request timeout in seconds
        """
        super().__init__(model_name)
        self.base_url = f"http://{host}:{port}"
        self.timeout = timeout
    
    def generate(
        self, 
        prompt: str, 
        system_prompt: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 2048,
        stream: bool = False,
        **kwargs
    ) -> Union[str, Callable]:
        """
        Generate text from prompt using Llama.cpp server.
        
        Args:
            prompt: User prompt
            system_prompt: Optional system prompt
            temperature: Sampling temperature (0.0-1.0)
            max_tokens: Maximum tokens to generate
            stream: Whether to stream the response
            
        Returns:
            Generated text or generator if streaming
        """
        url = f"{self.base_url}/completion"
        
        payload = {
            "prompt": prompt,
            "temperature": temperature,
            "n_predict": max_tokens,
            "stream": stream
        }
        
        if system_prompt:
            payload["system_prompt"] = system_prompt
        
        # Add any additional kwargs to payload
        payload.update({k: v for k, v in kwargs.items() if k not in payload})
        
        if stream:
            def stream_generator():
                try:
                    with requests.post(url, json=payload, stream=True, timeout=self.timeout) as response:
                        response.raise_for_status()
                        for line in response.iter_lines():
                            if not line:
                                continue
                            
                            try:
                                chunk = json.loads(line)
                                if "content" in chunk:
                                    yield chunk["content"]
                                if chunk.get("stop", False):
                                    break
                            except json.JSONDecodeError:
                                logger.warning(f"Failed to decode JSON: {line}")
                except Exception as e:
                    logger.error(f"Error streaming from Llama.cpp server: {str(e)}")
                    raise
            
            return stream_generator()
        else:
            try:
                response = requests.post(url, json=payload, timeout=self.timeout)
                response.raise_for_status()
                return response.json()["content"]
            except Exception as e:
                logger.error(f"Error calling Llama.cpp server: {str(e)}")
                raise
    
    def embed(self, text: str) -> List[float]:
        """
        Generate embeddings from text using Llama.cpp server.
        
        Args:
            text: Text to embed
            
        Returns:
            List of embedding values
        """
        url = f"{self.base_url}/embedding"
        
        payload = {
            "content": text
        }
        
        try:
            response = requests.post(url, json=payload, timeout=self.timeout)
            response.raise_for_status()
            return response.json()["embedding"]
        except Exception as e:
            logger.error(f"Error getting embeddings from Llama.cpp server: {str(e)}")
            raise


class MLXProvider(LocalLLMProvider):
    """MLX model provider using local models."""
    
    def __init__(
        self, 
        model_name: str = "mlx-community/Mistral-7B-v0.1-mlx",
        model_path: Optional[str] = None
    ):
        """
        Initialize an MLX provider.
        
        Args:
            model_name: MLX model name/identifier
            model_path: Optional path to model files
        """
        super().__init__(model_name)
        self.model_path = model_path
        self._model = None
        self._tokenizer = None
    
    def _load_model(self):
        """Load MLX model and tokenizer."""
        try:
            from mlx_lm import load, generate
            from mlx_lm.utils import generate_step
            import mlx.core as mx
            
            model_path = self.model_path or self.model_name
            self._model, self._tokenizer = load(model_path)
            logger.info(f"Loaded MLX model {self.model_name}")
        except ImportError:
            logger.error("Failed to import mlx_lm. Make sure it's installed")
            raise
        except Exception as e:
            logger.error(f"Error loading MLX model: {str(e)}")
            raise
    
    def generate(
        self, 
        prompt: str, 
        system_prompt: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 2048,
        **kwargs
    ) -> str:
        """
        Generate text from prompt using MLX model.
        
        Args:
            prompt: User prompt
            system_prompt: Optional system prompt
            temperature: Sampling temperature (0.0-1.0)
            max_tokens: Maximum tokens to generate
            
        Returns:
            Generated text
        """
        if self._model is None or self._tokenizer is None:
            self._load_model()
        
        # Format prompt with system if provided
        formatted_prompt = prompt
        if system_prompt:
            formatted_prompt = f"{system_prompt}\n\n{prompt}"
        
        # Generate text
        try:
            from mlx_lm import generate
            
            generation_args = {
                "temp": temperature,
                "max_tokens": max_tokens,
            }
            # Add any additional kwargs to generation_args
            generation_args.update({k: v for k, v in kwargs.items() if k not in generation_args})
            
            generated_text = generate(self._model, self._tokenizer, formatted_prompt, **generation_args)
            return generated_text
        except Exception as e:
            logger.error(f"Error generating text with MLX model: {str(e)}")
            raise
    
    def embed(self, text: str) -> List[float]:
        """
        Generate embeddings from text using MLX model.
        
        Args:
            text: Text to embed
            
        Returns:
            List of embedding values
        """
        if self._model is None or self._tokenizer is None:
            self._load_model()
        
        try:
            import mlx.core as mx
            
            # Tokenize text
            tokens = self._tokenizer.encode(text)
            
            # Get embeddings from model
            embeddings = self._model.embed(mx.array([tokens]))
            
            # Return embeddings as list
            return embeddings.squeeze().tolist()
        except Exception as e:
            logger.error(f"Error generating embeddings with MLX model: {str(e)}")
            raise 
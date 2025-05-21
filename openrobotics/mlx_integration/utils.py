"""
Utilities for MLX integration including error handling and logging.
"""

import logging
from typing import TypeVar, Callable, Optional, Any
import functools
import mlx.core as mx
import time

# Create logger
logger = logging.getLogger(__name__)
T = TypeVar('T')

class MLXError(Exception):
    """Base class for MLX integration errors."""
    pass

class ModelInitializationError(MLXError):
    """Error during model initialization."""
    pass

class InferenceError(MLXError):
    """Error during model inference."""
    pass

class TrainingError(MLXError):
    """Error during model training."""
    pass

def handle_mlx_errors(func: Callable[..., T]) -> Callable[..., T]:
    """
    Decorator to handle common MLX errors and log them appropriately.
    
    Args:
        func: Function to wrap with error handling
        
    Returns:
        Wrapped function with error handling
    """
    @functools.wraps(func)
    def wrapper(*args, **kwargs) -> T:
        try:
            return func(*args, **kwargs)
        except mx.MLXError as e:
            logger.error(f"MLX runtime error in {func.__name__}: {str(e)}")
            raise InferenceError(f"MLX runtime error: {str(e)}") from e
        except ValueError as e:
            logger.error(f"Invalid value in {func.__name__}: {str(e)}")
            raise ModelInitializationError(f"Invalid value: {str(e)}") from e
        except RuntimeError as e:
            logger.error(f"Runtime error in {func.__name__}: {str(e)}")
            raise TrainingError(f"Runtime error: {str(e)}") from e
        except Exception as e:
            logger.error(f"Unexpected error in {func.__name__}: {str(e)}")
            raise MLXError(f"Unexpected error: {str(e)}") from e
    return wrapper

def validate_tensor_shape(tensor: mx.array, expected_shape: tuple, name: str = "tensor"):
    """
    Validate tensor shape with helpful error message.
    
    Args:
        tensor: Tensor to validate
        expected_shape: Expected shape tuple
        name: Name of tensor for error message
        
    Raises:
        ValueError: If shape doesn't match
    """
    if tensor.shape != expected_shape:
        raise ValueError(
            f"Invalid {name} shape. Got {tensor.shape}, expected {expected_shape}"
        )

def log_execution_time(func: Callable[..., T]) -> Callable[..., T]:
    """
    Decorator to log function execution time.
    
    Args:
        func: Function to wrap with timing
        
    Returns:
        Wrapped function with timing
    """
    @functools.wraps(func)
    def wrapper(*args, **kwargs) -> T:
        start_time = time.perf_counter()
        result = func(*args, **kwargs)
        elapsed = time.perf_counter() - start_time
        logger.debug(f"{func.__name__} executed in {elapsed:.4f} seconds")
        return result
    return wrapper

def check_mlx_available() -> bool:
    """
    Check if MLX is available and properly configured.
    
    Returns:
        bool: True if MLX is available, False otherwise
    """
    try:
        mx.array([1, 2, 3])
        return True
    except ImportError:
        logger.warning("MLX not available - running in compatibility mode")
        return False
    except Exception as e:
        logger.warning(f"MLX available but not working properly: {str(e)}")
        return False 
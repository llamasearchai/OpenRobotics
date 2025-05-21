"""
Model optimization tools for MLX models.
"""

import logging
import time
from typing import Dict, Any, Optional, List, Tuple, Union
import numpy as np
import mlx.core as mx
import mlx.nn as nn

from openrobotics.mlx_integration.models import MLXModel
from openrobotics.mlx_integration.utils import handle_mlx_errors, log_execution_time

logger = logging.getLogger(__name__)

class ModelOptimizer:
    """Optimization utilities for MLX models."""
    
    @staticmethod
    @handle_mlx_errors
    def quantize_weights(model: MLXModel, bit_precision: int = 8) -> MLXModel:
        """
        Quantize model weights to reduce memory footprint.
        
        Args:
            model: The model to quantize
            bit_precision: Bit precision for quantization (8 or 16)
            
        Returns:
            Quantized model
        """
        if bit_precision not in [8, 16]:
            raise ValueError(f"Unsupported bit precision: {bit_precision}. Use 8 or 16.")
        
        params = model.model.parameters()
        quantized_params = {}
        
        # Create quantized model
        quantized_model = type(model)(
            name=f"{model.name}_quantized",
            model_dir=model.model_dir
        )
        quantized_model.config = model.config.copy()
        quantized_model.config["quantized"] = True
        quantized_model.config["bit_precision"] = bit_precision
        
        # Initialize model architecture
        quantized_model._initialize_model()
        
        # Quantize parameters
        for name, param in params.items():
            if bit_precision == 16:
                # Half precision
                quantized_params[name] = param.astype(mx.float16)
            elif bit_precision == 8:
                # 8-bit quantization
                param_min = mx.min(param)
                param_max = mx.max(param)
                scale = (param_max - param_min) / 255.0
                zero_point = -mx.round(param_min / scale).astype(mx.int8)
                
                # Store as int8 with scale and zero_point
                quantized = mx.clip(
                    mx.round(param / scale) + zero_point,
                    0, 255
                ).astype(mx.uint8)
                
                quantized_params[name] = {
                    "data": quantized,
                    "scale": scale,
                    "zero_point": zero_point
                }
        
        # Store quantization metadata
        quantized_model._quantized_params = quantized_params
        
        return quantized_model
    
    @staticmethod
    def dequantize_param(param_dict):
        """Dequantize a quantized parameter."""
        if isinstance(param_dict, dict) and "data" in param_dict:
            # Dequantize
            return param_dict["scale"] * (param_dict["data"].astype(mx.float32) - param_dict["zero_point"])
        return param_dict
    
    @staticmethod
    def dequantize_for_inference(model: MLXModel) -> MLXModel:
        """
        Dequantize a quantized model for inference.
        
        Args:
            model: Quantized model
            
        Returns:
            Model with dequantized weights for inference
        """
        if not hasattr(model, "_quantized_params"):
            return model
        
        params = {}
        for name, param in model._quantized_params.items():
            params[name] = ModelOptimizer.dequantize_param(param)
        
        # Update model with dequantized parameters
        model.model.update(params)
        
        return model
    
    @staticmethod
    @log_execution_time
    def prune_model(model: MLXModel, sparsity: float = 0.5) -> MLXModel:
        """
        Prune model weights to increase sparsity.
        
        Args:
            model: The model to prune
            sparsity: Target sparsity level (0.0-1.0)
            
        Returns:
            Pruned model
        """
        if not 0.0 <= sparsity < 1.0:
            raise ValueError(f"Sparsity must be between 0.0 and 1.0, got {sparsity}")
        
        params = model.model.parameters()
        pruned_params = {}
        
        # Create pruned model
        pruned_model = type(model)(
            name=f"{model.name}_pruned",
            model_dir=model.model_dir
        )
        pruned_model.config = model.config.copy()
        pruned_model.config["pruned"] = True
        pruned_model.config["sparsity"] = sparsity
        
        # Initialize model architecture
        pruned_model._initialize_model()
        
        # Prune parameters
        for name, param in params.items():
            # Skip biases and non-weight parameters
            if len(param.shape) <= 1 or "bias" in name:
                pruned_params[name] = param
                continue
                
            # Calculate magnitude-based threshold
            abs_param = mx.abs(param)
            threshold = mx.quantile(abs_param, sparsity)
            
            # Create mask
            mask = abs_param > threshold
            
            # Apply mask (multiply by 0/1 mask)
            pruned_params[name] = param * mask
        
        # Update model with pruned parameters
        pruned_model.model.update(pruned_params)
        
        return pruned_model 
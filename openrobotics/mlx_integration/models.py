"""
MLX model implementations for robotics applications.
"""

import os
import json
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional, Union, Literal

import mlx.core as mx
import mlx.nn as nn
import numpy as np
import requests

from openrobotics.config import config
from openrobotics.mlx_integration.utils import (
    handle_mlx_errors,
    validate_tensor_shape,
    log_execution_time,
    MLXError,
    ModelInitializationError,
    InferenceError
)


class MLXModel:
    """
    Base class for MLX models in OpenRobotics.
    
    This class provides common functionality for loading, saving, and
    running inference with MLX models.
    """
    
    def __init__(
        self,
        name: str,
        model_dir: Optional[str] = None,
        dtype: str = "float16",
    ):
        """
        Initialize an MLX model.
        
        Args:
            name: Model name
            model_dir: Directory containing model weights (if None, uses default from config)
            dtype: Data type for model weights ("float16" or "float32")
        """
        self.name = name
        self.model_dir = model_dir or config.get("mlx", "models_dir")
        self.dtype = getattr(mx, dtype)
        self.model = None
        self.config = {}
        
        # Create model directory if it doesn't exist
        os.makedirs(self.model_dir, exist_ok=True)
    
    def _get_model_path(self) -> Tuple[Path, Path]:
        """Get paths for model weights and config."""
        model_dir = Path(self.model_dir) / self.name
        weights_path = model_dir / "weights.npz"
        config_path = model_dir / "config.json"
        return weights_path, config_path
    
    def save(self, model_dir: Optional[str] = None):
        """
        Save model weights and configuration.
        
        Args:
            model_dir: Directory to save model (if None, uses instance model_dir)
        """
        if self.model is None:
            raise ValueError("Model must be initialized before saving")
        
        save_dir = Path(model_dir or self.model_dir) / self.name
        os.makedirs(save_dir, exist_ok=True)
        
        weights_path = save_dir / "weights.npz"
        config_path = save_dir / "config.json"
        
        # Save weights
        mx.save(weights_path, self.model.parameters())
        
        # Save config
        with open(config_path, 'w') as f:
            json.dump(self.config, f)
    
    def load(self, model_dir: Optional[str] = None):
        """
        Load model weights and configuration.
        
        Args:
            model_dir: Directory containing model (if None, uses instance model_dir)
        """
        load_dir = Path(model_dir or self.model_dir) / self.name
        weights_path = load_dir / "weights.npz"
        config_path = load_dir / "config.json"
        
        if not weights_path.exists():
            raise FileNotFoundError(f"Model weights not found at {weights_path}")
        
        if not config_path.exists():
            raise FileNotFoundError(f"Model config not found at {config_path}")
        
        # Load config
        with open(config_path, 'r') as f:
            self.config = json.load(f)
        
        # Initialize model architecture based on config
        self._initialize_model()
        
        # Load weights
        weights = mx.load(str(weights_path))
        self.model.update(weights)
    
    def _initialize_model(self):
        """
        Initialize model architecture based on config.
        
        This method should be implemented by subclasses.
        """
        raise NotImplementedError("Subclasses must implement _initialize_model")
    
    @handle_mlx_errors
    @log_execution_time
    def predict(self, inputs: Union[np.ndarray, mx.array]) -> mx.array:
        """
        Run model inference with error handling and timing.
        """
        if self.model is None:
            raise ModelInitializationError("Model must be initialized before prediction")
        
        if isinstance(inputs, np.ndarray):
            inputs = mx.array(inputs, dtype=self.dtype)
        
        return self.model(inputs)


class VisionModel(MLXModel):
    """
    Vision model implementation using MLX.
    
    This class implements common vision model architectures for
    robotics applications such as object detection and segmentation.
    """
    
    def __init__(
        self,
        name: str,
        model_type: str = "detection",
        input_shape: Tuple[int, int, int] = (224, 224, 3),
        num_classes: int = 80,
        backbone: str = "resnet50",
        model_dir: Optional[str] = None,
        dtype: str = "float16",
    ):
        """
        Initialize a vision model.
        
        Args:
            name: Model name
            model_type: Type of vision model ("detection", "segmentation", "classification")
            input_shape: Input image shape (height, width, channels)
            num_classes: Number of output classes
            backbone: Backbone architecture
            model_dir: Directory for model weights
            dtype: Data type for model weights
        """
        super().__init__(name, model_dir, dtype)
        
        self.config = {
            "model_type": model_type,
            "input_shape": input_shape,
            "num_classes": num_classes,
            "backbone": backbone,
        }
        
        self._initialize_model()
    
    def _initialize_model(self):
        """Initialize model architecture based on config."""
        model_type = self.config["model_type"]
        input_shape = self.config["input_shape"]
        num_classes = self.config["num_classes"]
        backbone = self.config["backbone"]
        
        if model_type == "classification":
            self.model = self._create_classification_model(input_shape, num_classes, backbone)
        elif model_type == "detection":
            self.model = self._create_detection_model(input_shape, num_classes, backbone)
        elif model_type == "segmentation":
            self.model = self._create_segmentation_model(input_shape, num_classes, backbone)
        else:
            raise ValueError(f"Unsupported model type: {model_type}")
    
    def _create_classification_model(self, input_shape, num_classes, backbone):
        """Create classification model architecture."""
        if backbone == "resnet18":
            return nn.Sequential(
                nn.Conv2d(input_shape[2], 64, kernel_size=7, stride=2, padding=3),
                nn.BatchNorm2d(64),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
                self._resnet_block(64, 64, stride=1),
                self._resnet_block(64, 64, stride=1),
                self._resnet_block(64, 128, stride=2),
                self._resnet_block(128, 128, stride=1),
                self._resnet_block(128, 256, stride=2),
                self._resnet_block(256, 256, stride=1),
                self._resnet_block(256, 512, stride=2),
                self._resnet_block(512, 512, stride=1),
                nn.AdaptiveAvgPool2d((1, 1)),
                nn.Flatten(),
                nn.Linear(512, num_classes),
            )
        elif backbone == "resnet50":
            # Simplified ResNet50 for demonstration
            return nn.Sequential(
                nn.Conv2d(input_shape[2], 64, kernel_size=7, stride=2, padding=3),
                nn.BatchNorm2d(64),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
                self._bottleneck_block(64, 64, 256, stride=1),
                self._bottleneck_block(256, 64, 256, stride=1),
                self._bottleneck_block(256, 64, 256, stride=1),
                self._bottleneck_block(256, 128, 512, stride=2),
                self._bottleneck_block(512, 128, 512, stride=1),
                self._bottleneck_block(512, 128, 512, stride=1),
                self._bottleneck_block(512, 128, 512, stride=1),
                self._bottleneck_block(512, 256, 1024, stride=2),
                self._bottleneck_block(1024, 256, 1024, stride=1),
                self._bottleneck_block(1024, 256, 1024, stride=1),
                self._bottleneck_block(1024, 256, 1024, stride=1),
                self._bottleneck_block(1024, 256, 1024, stride=1),
                self._bottleneck_block(1024, 256, 1024, stride=1),
                self._bottleneck_block(1024, 512, 2048, stride=2),
                self._bottleneck_block(2048, 512, 2048, stride=1),
                self._bottleneck_block(2048, 512, 2048, stride=1),
                nn.AdaptiveAvgPool2d((1, 1)),
                nn.Flatten(),
                nn.Linear(2048, num_classes),
            )
        else:
            raise ValueError(f"Unsupported backbone: {backbone}")
    
    def _create_detection_model(self, input_shape, num_classes, backbone):
        """Create object detection model architecture."""
        # Simplified SSD-like detection model
        if backbone == "resnet50":
            feature_extractor = nn.Sequential(
                nn.Conv2d(input_shape[2], 64, kernel_size=7, stride=2, padding=3),
                nn.BatchNorm2d(64),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
                self._bottleneck_block(64, 64, 256, stride=1),
                self._bottleneck_block(256, 64, 256, stride=1),
                self._bottleneck_block(256, 64, 256, stride=1),
                self._bottleneck_block(256, 128, 512, stride=2),
                self._bottleneck_block(512, 128, 512, stride=1),
                self._bottleneck_block(512, 128, 512, stride=1),
                self._bottleneck_block(512, 128, 512, stride=1),
            )
            
            # Detection heads
            loc_head = nn.Sequential(
                nn.Conv2d(512, 512, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.Conv2d(512, 4 * 6, kernel_size=1)  # 6 anchor boxes per location, 4 coordinates
            )
            
            conf_head = nn.Sequential(
                nn.Conv2d(512, 512, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.Conv2d(512, num_classes * 6, kernel_size=1)  # 6 anchor boxes per location
            )
            
            # Custom detection forward function
            class DetectionModel(nn.Module):
                def __init__(self, feature_extractor, loc_head, conf_head):
                    super().__init__()
                    self.feature_extractor = feature_extractor
                    self.loc_head = loc_head
                    self.conf_head = conf_head
                
                def __call__(self, x):
                    features = self.feature_extractor(x)
                    loc_preds = self.loc_head(features)
                    conf_preds = self.conf_head(features)
                    return (loc_preds, conf_preds)
            
            return DetectionModel(feature_extractor, loc_head, conf_head)
        else:
            raise ValueError(f"Unsupported backbone for detection: {backbone}")
    
    def _create_segmentation_model(self, input_shape, num_classes, backbone):
        """Create segmentation model architecture."""
        # Simplified U-Net-like segmentation model
        if backbone == "unet":
            class UNet(nn.Module):
                def __init__(self, in_channels, num_classes):
                    super().__init__()
                    
                    # Encoder
                    self.enc1 = self._double_conv(in_channels, 64)
                    self.enc2 = self._double_conv(64, 128)
                    self.enc3 = self._double_conv(128, 256)
                    self.enc4 = self._double_conv(256, 512)
                    
                    self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
                    
                    # Bottleneck
                    self.bottleneck = self._double_conv(512, 1024)
                    
                    # Decoder
                    self.upconv4 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
                    self.dec4 = self._double_conv(1024, 512)
                    
                    self.upconv3 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
                    self.dec3 = self._double_conv(512, 256)
                    
                    self.upconv2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
                    self.dec2 = self._double_conv(256, 128)
                    
                    self.upconv1 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
                    self.dec1 = self._double_conv(128, 64)
                    
                    self.final_conv = nn.Conv2d(64, num_classes, kernel_size=1)
                
                def _double_conv(self, in_channels, out_channels):
                    return nn.Sequential(
                        nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
                        nn.BatchNorm2d(out_channels),
                        nn.ReLU(),
                        nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
                        nn.BatchNorm2d(out_channels),
                        nn.ReLU()
                    )
                
                def __call__(self, x):
                    # Encoder
                    e1 = self.enc1(x)
                    e2 = self.enc2(self.pool(e1))
                    e3 = self.enc3(self.pool(e2))
                    e4 = self.enc4(self.pool(e3))
                    
                    # Bottleneck
                    b = self.bottleneck(self.pool(e4))
                    
                    # Decoder
                    d4 = self.upconv4(b)
                    d4 = mx.concatenate([d4, e4], axis=1)
                    d4 = self.dec4(d4)
                    
                    d3 = self.upconv3(d4)
                    d3 = mx.concatenate([d3, e3], axis=1)
                    d3 = self.dec3(d3)
                    
                    d2 = self.upconv2(d3)
                    d2 = mx.concatenate([d2, e2], axis=1)
                    d2 = self.dec2(d2)
                    
                    d1 = self.upconv1(d2)
                    d1 = mx.concatenate([d1, e1], axis=1)
                    d1 = self.dec1(d1)
                    
                    out = self.final_conv(d1)
                    return out
            
            return UNet(input_shape[2], num_classes)
        else:
            raise ValueError(f"Unsupported backbone for segmentation: {backbone}")
    
    def _resnet_block(self, in_channels, out_channels, stride=1):
        """Create a ResNet basic block."""
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
        )
    
    def _bottleneck_block(self, in_channels, mid_channels, out_channels, stride=1):
        """Create a ResNet bottleneck block."""
        return nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=1, stride=1),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(),
            nn.Conv2d(mid_channels, mid_channels, kernel_size=3, stride=stride, padding=1),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(),
            nn.Conv2d(mid_channels, out_channels, kernel_size=1, stride=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
        )
    
    @handle_mlx_errors
    def preprocess_image(self, image: np.ndarray) -> mx.array:
        """
        Preprocess image with error handling.
        """
        try:
            input_shape = self.config["input_shape"]
            validate_tensor_shape(
                mx.array(image), 
                (image.shape[0], image.shape[1], image.shape[2]),
                "input image"
            )
            # Resize image if needed
            if image.shape[0] != input_shape[0] or image.shape[1] != input_shape[1]:
                from PIL import Image
                import numpy as np
                
                pil_image = Image.fromarray(image)
                pil_image = pil_image.resize((input_shape[1], input_shape[0]))
                image = np.array(pil_image)
            
            # Normalize pixel values
            image = image.astype(np.float32) / 255.0
            
            # Convert to MLX array
            return mx.array(image, dtype=self.dtype)
        except Exception as e:
            logger.error(f"Image preprocessing failed: {str(e)}")
            raise InferenceError("Image preprocessing failed") from e
    
    def predict_with_preprocessing(self, image: np.ndarray) -> mx.array:
        """
        Run prediction with automatic preprocessing.
        
        Args:
            image: Input image as numpy array (H, W, C) with values in [0, 255]
            
        Returns:
            Model predictions
        """
        processed_image = self.preprocess_image(image)
        # Add batch dimension
        processed_image = processed_image.reshape(1, *processed_image.shape)
        return self.predict(processed_image)


class ControlModel(MLXModel):
    """
    Control model implementation using MLX.
    
    This class implements neural network controllers for robotics applications
    such as inverse dynamics, trajectory optimization, and reinforcement learning.
    """
    
    def __init__(
        self,
        name: str,
        model_type: str = "policy",
        input_dim: int = 10,
        output_dim: int = 6,
        hidden_dims: List[int] = [64, 64],
        activation: str = "relu",
        model_dir: Optional[str] = None,
        dtype: str = "float16",
    ):
        """
        Initialize a control model.
        
        Args:
            name: Model name
            model_type: Type of control model ("policy", "dynamics", "value")
            input_dim: Input dimension (state dimension)
            output_dim: Output dimension (action dimension)
            hidden_dims: List of hidden layer dimensions
            activation: Activation function ("relu", "tanh", "sigmoid")
            model_dir: Directory for model weights
            dtype: Data type for model weights
        """
        super().__init__(name, model_dir, dtype)
        
        self.config = {
            "model_type": model_type,
            "input_dim": input_dim,
            "output_dim": output_dim,
            "hidden_dims": hidden_dims,
            "activation": activation,
        }
        
        self._initialize_model()
    
    def _initialize_model(self):
        """Initialize model architecture based on config."""
        model_type = self.config["model_type"]
        input_dim = self.config["input_dim"]
        output_dim = self.config["output_dim"]
        hidden_dims = self.config["hidden_dims"]
        activation = self.config["activation"]
        
        if activation == "relu":
            act_fn = nn.ReLU()
        elif activation == "tanh":
            act_fn = nn.Tanh()
        elif activation == "sigmoid":
            act_fn = nn.Sigmoid()
        else:
            raise ValueError(f"Unsupported activation: {activation}")
        
        if model_type == "policy":
            self.model = self._create_mlp(input_dim, output_dim, hidden_dims, act_fn)
        elif model_type == "dynamics":
            # Dynamics model predicts next state given current state and action
            self.model = self._create_dynamics_model(input_dim, output_dim, hidden_dims, act_fn)
        elif model_type == "value":
            # Value function for reinforcement learning
            self.model = self._create_mlp(input_dim, 1, hidden_dims, act_fn)
        else:
            raise ValueError(f"Unsupported model type: {model_type}")
    
    def _create_mlp(self, input_dim, output_dim, hidden_dims, activation):
        """Create a multi-layer perceptron."""
        layers = []
        prev_dim = input_dim
        
        for h_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, h_dim))
            layers.append(activation)
            prev_dim = h_dim
        
        layers.append(nn.Linear(prev_dim, output_dim))
        
        return nn.Sequential(*layers)
    
    def _create_dynamics_model(self, state_dim, action_dim, hidden_dims, activation):
        """Create a dynamics model predicting next state."""
        class DynamicsModel(nn.Module):
            def __init__(self, state_dim, action_dim, hidden_dims, activation):
                super().__init__()
                
                self.state_dim = state_dim
                self.action_dim = action_dim
                
                # MLP for predicting state change
                input_dim = state_dim + action_dim
                layers = []
                prev_dim = input_dim
                
                for h_dim in hidden_dims:
                    layers.append(nn.Linear(prev_dim, h_dim))
                    layers.append(activation)
                    prev_dim = h_dim
                
                layers.append(nn.Linear(prev_dim, state_dim))
                
                self.mlp = nn.Sequential(*layers)
            
            def __call__(self, x):
                # x is concatenated [state, action]
                delta = self.mlp(x)
                
                # Extract state from input
                state = x[:, :self.state_dim]
                
                # Return next state (current state + delta)
                next_state = state + delta
                return next_state
        
        return DynamicsModel(state_dim, action_dim, hidden_dims, activation)
    
    def normalize_input(self, x: np.ndarray, mean: np.ndarray, std: np.ndarray) -> mx.array:
        """
        Normalize input data.
        
        Args:
            x: Input data
            mean: Mean for normalization
            std: Standard deviation for normalization
            
        Returns:
            Normalized data as MLX array
        """
        if isinstance(x, np.ndarray):
            x = (x - mean) / (std + 1e-8)
            return mx.array(x, dtype=self.dtype)
        else:
            x = (x - mx.array(mean)) / (mx.array(std) + 1e-8)
            return x
    
    def denormalize_output(self, y: mx.array, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
        """
        Denormalize output data.
        
        Args:
            y: Output data from model
            mean: Mean for denormalization
            std: Standard deviation for denormalization
            
        Returns:
            Denormalized data as numpy array
        """
        y_np = y.array if hasattr(y, 'array') else y
        return y_np * std + mean


class OllamaModel(MLXModel):
    """Wrapper for Ollama local models with MLX compatibility."""
    
    def __init__(
        self,
        name: str,
        model_type: Literal["text", "vision", "multimodal"] = "text",
        model_dir: Optional[str] = None,
        dtype: str = "float16",
        ollama_host: str = "http://localhost:11434"
    ):
        super().__init__(name, model_dir, dtype)
        self.ollama_host = ollama_host
        self.model_type = model_type
        self.config = {
            "model_type": model_type,
            "ollama_host": ollama_host
        }
    
    def predict(self, inputs: Union[str, np.ndarray, mx.array]) -> Union[str, mx.array]:
        """Run inference with Ollama model."""
        if self.model_type == "text":
            return self._predict_text(inputs)
        elif self.model_type == "vision":
            return self._predict_vision(inputs)
        else:
            raise ValueError(f"Unsupported model type: {self.model_type}")
    
    def _predict_text(self, prompt: str) -> str:
        """Generate text completion."""
        response = requests.post(
            f"{self.ollama_host}/api/generate",
            json={
                "model": self.name,
                "prompt": prompt,
                "stream": False
            }
        )
        response.raise_for_status()
        return response.json()["response"]
    
    def _predict_vision(self, image: Union[np.ndarray, mx.array]) -> mx.array:
        """Run vision model inference."""
        if isinstance(image, mx.array):
            image = np.array(image)
        
        # Convert image to base64
        from PIL import Image
        import base64
        import io
        
        pil_img = Image.fromarray(image)
        buffered = io.BytesIO()
        pil_img.save(buffered, format="JPEG")
        img_str = base64.b64encode(buffered.getvalue()).decode('utf-8')
        
        response = requests.post(
            f"{self.ollama_host}/api/generate",
            json={
                "model": self.name,
                "images": [img_str],
                "stream": False
            }
        )
        response.raise_for_status()
        return mx.array(response.json()["response"])
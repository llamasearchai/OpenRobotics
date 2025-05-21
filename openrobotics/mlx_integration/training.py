"""
Training utilities for MLX models in OpenRobotics.
"""

import time
import logging
from typing import (
    Any, Callable, Dict, List, Optional, Tuple, TypeVar, Union,
    Literal, Protocol, runtime_checkable
)

import numpy as np

try:
    import mlx.core as mx
    import mlx.nn as nn
    import mlx.optimizers as optim
    HAS_MLX = True
except ImportError:
    # Create mock classes for systems without MLX
    class MockMLX:
        def __getattr__(self, name):
            return self
        def __call__(self, *args, **kwargs):
            return self
    mx = MockMLX()
    nn = MockMLX()
    optim = MockMLX()
    HAS_MLX = False

from openrobotics.mlx_integration.models import MLXModel

logger = logging.getLogger(__name__)

T = TypeVar('T')
ArrayLike = Union[np.ndarray, mx.array]

class LossFunction(Protocol):
    """Protocol for loss functions."""
    def __call__(self, y_pred: ArrayLike, y_true: ArrayLike) -> mx.array:
        ...

@runtime_checkable
class DatasetProtocol(Protocol):
    """Protocol for dataset objects."""
    def batch(self, batch_size: int) -> Tuple[ArrayLike, ArrayLike]:
        ...
    
    def __len__(self) -> int:
        ...

class Trainer:
    """Training loop and utilities for MLX models."""
    
    def __init__(
        self,
        model: nn.Module,
        loss_fn: LossFunction,
        learning_rate: float = 1e-3,
        max_grad_norm: Optional[float] = None
    ):
        self.model = model
        self.loss_fn = loss_fn
        self.optimizer = optim.Adam(learning_rate=learning_rate)
        self.max_grad_norm = max_grad_norm
    
    def train_step(
        self,
        x: ArrayLike,
        y: ArrayLike,
    ) -> float:
        """Single training step with type hints."""
        def loss_fn(model):
            y_pred = model(x)
            return self.loss_fn(y_pred, y)
        
        grad_fn = mx.value_and_grad(self.model.model, loss_fn)
        loss, grads = grad_fn(self.model.model)
        self.optimizer.update(self.model.model, grads)
        return loss.item()
    
    def train(
        self,
        train_data: Union[Tuple[ArrayLike, ArrayLike], DatasetProtocol],
        val_data: Optional[Union[Tuple[ArrayLike, ArrayLike], DatasetProtocol]] = None,
        batch_size: int = 32,
        epochs: int = 10,
        callbacks: Optional[List[Callable[[int, Dict[str, Any]], None]]] = None,
    ) -> Dict[str, List[float]]:
        """Full training loop with type hints."""
        if not HAS_MLX:
            raise ImportError("MLX is not installed. Cannot train model.")
            
        x_train, y_train = train_data
        
        # Convert to numpy for easier batch processing
        x_train_np = x_train.array if hasattr(x_train, 'array') else x_train
        y_train_np = y_train.array if hasattr(y_train, 'array') else y_train
        
        # Training metrics
        history = {
            'train_loss': [],
            'val_loss': [] if val_data is not None else None,
            'learning_rate': [],
        }
        
        # Training loop
        for epoch in range(epochs):
            epoch_start_time = time.time()
            
            # Batch training
            batch_losses = []
            num_batches = int(np.ceil(len(x_train_np) / batch_size))
            
            for batch_idx in range(num_batches):
                # Get batch data
                start_idx = batch_idx * batch_size
                end_idx = min(start_idx + batch_size, len(x_train_np))
                x_batch = x_train_np[start_idx:end_idx]
                y_batch = y_train_np[start_idx:end_idx]
                
                # Train on batch
                batch_loss = self.train_step(x_batch, y_batch)
                batch_losses.append(batch_loss)
            
            # Compute average epoch loss
            epoch_loss = np.mean(batch_losses)
            history['train_loss'].append(epoch_loss)
            
            # Validation
            val_loss = None
            if val_data is not None:
                val_loss = self.evaluate(*val_data)
                history['val_loss'].append(val_loss)
            
            # Store current learning rate
            if hasattr(self.optimizer, 'learning_rate'):
                history['learning_rate'].append(self.optimizer.learning_rate)
            
            # Print progress
            epoch_time = time.time() - epoch_start_time
            val_str = f", val_loss: {val_loss:.4f}" if val_loss is not None else ""
            logger.info(f"Epoch {epoch + 1}/{epochs} - {epoch_time:.2f}s - loss: {epoch_loss:.4f}{val_str}")
            
            # Call callbacks
            if callbacks:
                for callback in callbacks:
                    callback(epoch, history)
        
        return history
    
    def evaluate(
        self,
        x: Union[np.ndarray, 'mx.array'],
        y: Union[np.ndarray, 'mx.array'],
    ) -> float:
        """
        Evaluate model on validation data.
        
        Args:
            x: Input data
            y: Target data
            
        Returns:
            Validation loss
        """
        if not HAS_MLX:
            raise ImportError("MLX is not installed. Cannot evaluate model.")
            
        # Convert numpy arrays to MLX arrays if needed
        if isinstance(x, np.ndarray):
            x = mx.array(x)
        if isinstance(y, np.ndarray):
            y = mx.array(y)
        
        # Forward pass
        y_pred = self.model(x)
        
        # Compute loss
        loss = self.loss_fn(y_pred, y)
        
        return loss.item()
    
    def save_model(self, path: str):
        """
        Save the model.
        
        Args:
            path: Path to save model
        """
        if not HAS_MLX:
            raise ImportError("MLX is not installed. Cannot save model.")
            
        if self.model is not None:
            mx.save(path, self.model.parameters())
    
    def load_model(self, path: str):
        """
        Load the model.
        
        Args:
            path: Path to load model from
        """
        if not HAS_MLX:
            raise ImportError("MLX is not installed. Cannot load model.")
            
        params = mx.load(path)
        self.model.update(params)

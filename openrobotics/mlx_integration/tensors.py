"""
Tensor operations optimized for MLX and robotics applications.
"""

import mlx.core as mx

class TensorOps:
    """Optimized tensor operations for robotics."""
    
    @staticmethod
    def normalize(x: mx.array) -> mx.array:
        """Normalize tensor to [0, 1] range."""
        return (x - x.min()) / (x.max() - x.min())
    
    @staticmethod
    def preprocess_image(image: mx.array) -> mx.array:
        """Preprocess image tensor for vision models."""
        # Convert to float32 and normalize
        image = image.astype(mx.float32)
        return TensorOps.normalize(image)
    
    @staticmethod
    def batch_operations(arrays: list[mx.array]) -> mx.array:
        """Stack and process batch of tensors."""
        return mx.stack(arrays) 
"""
Tests for MLX integration utilities.
"""

import unittest
import mlx.core as mx
from openrobotics.mlx_integration.utils import (
    handle_mlx_errors,
    validate_tensor_shape,
    MLXError,
    ModelInitializationError
)

class TestMLXUtils(unittest.TestCase):
    def test_handle_mlx_errors(self):
        @handle_mlx_errors
        def good_func(x):
            return x * 2
        
        @handle_mlx_errors
        def bad_func(x):
            raise mx.MLXError("Test error")
        
        # Test good function
        result = good_func(mx.array([1, 2, 3]))
        self.assertTrue(mx.all(result == mx.array([2, 4, 6])))
        
        # Test error handling
        with self.assertRaises(InferenceError):
            bad_func(mx.array([1, 2, 3]))
    
    def test_validate_tensor_shape(self):
        x = mx.ones((3, 4))
        validate_tensor_shape(x, (3, 4))
        
        with self.assertRaises(ValueError):
            validate_tensor_shape(x, (4, 3))

if __name__ == "__main__":
    unittest.main() 
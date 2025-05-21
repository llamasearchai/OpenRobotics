import unittest
import mlx.core as mx
from openrobotics.mlx_integration import MLXModel, VisionModel, ControlModel, TensorOps, Trainer

class TestMLXIntegration(unittest.TestCase):
    def setUp(self):
        self.vision_config = {"output_dim": 10}
        self.control_config = {"input_dim": 20, "output_dim": 5}
    
    def test_vision_model(self):
        model = VisionModel(self.vision_config)
        x = mx.random.uniform(shape=(1, 3, 224, 224))
        y = model.forward(x)
        self.assertEqual(y.shape[-1], self.vision_config["output_dim"])
    
    def test_tensor_ops(self):
        x = mx.random.uniform(shape=(10, 10))
        normalized = TensorOps.normalize(x)
        self.assertTrue(mx.all(normalized >= 0))
        self.assertTrue(mx.all(normalized <= 1))
    
    def test_trainer(self):
        model = ControlModel(self.control_config)
        def loss_fn(y_pred, y):
            return mx.mean((y_pred - y)**2)
        trainer = Trainer(model, loss_fn)
        # Test would need actual training data in a real scenario

if __name__ == "__main__":
    unittest.main() 
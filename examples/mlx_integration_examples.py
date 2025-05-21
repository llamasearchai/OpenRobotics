"""
MLX Integration Examples for OpenRobotics.
"""

import mlx.core as mx
from openrobotics.mlx_integration import (
    VisionModel,
    ControlModel,
    OllamaModel,
    TensorOps,
    Trainer,
    MLXBenchmark
)

def vision_model_example():
    """Example of using VisionModel for image classification."""
    print("\n=== Vision Model Example ===")
    
    # Initialize model
    model = VisionModel(
        name="resnet50_classifier",
        model_type="classification",
        input_shape=(224, 224, 3),
        num_classes=10,
        backbone="resnet50"
    )
    
    # Generate random image
    image = mx.random.uniform(shape=(1, 3, 224, 224))
    
    # Run inference
    predictions = model.predict(image)
    print(f"Predictions shape: {predictions.shape}")
    print(f"Sample predictions: {predictions[0, :5]}")

def ollama_model_example():
    """Example of using Ollama local models."""
    print("\n=== Ollama Model Example ===")
    
    # Text generation
    text_model = OllamaModel(name="llama2", model_type="text")
    response = text_model.predict("Explain robotics in simple terms")
    print(f"LLM Response: {response[:200]}...")  # Print first 200 chars
    
    # Vision model (if you have a vision model like llava)
    try:
        vision_model = OllamaModel(name="llava", model_type="vision")
        image = mx.random.uniform(shape=(224, 224, 3)) * 255
        description = vision_model.predict(image)
        print(f"Image description: {description}")
    except Exception as e:
        print(f"Vision model not available: {e}")

def benchmark_example():
    """Example of benchmarking model performance."""
    print("\n=== Benchmark Example ===")
    
    # Initialize model
    model = ControlModel(
        name="simple_policy",
        model_type="policy",
        input_dim=10,
        output_dim=6
    )
    
    # Benchmark inference
    benchmark = MLXBenchmark.benchmark_model(
        model=model,
        input_shape=(1, 10)
    )
    print("Inference Benchmark Results:")
    for k, v in benchmark.items():
        print(f"{k:>20}: {v}")

if __name__ == "__main__":
    vision_model_example()
    ollama_model_example()
    benchmark_example() 
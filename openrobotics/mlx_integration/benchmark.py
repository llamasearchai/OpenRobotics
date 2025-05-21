"""
Performance benchmarking for MLX models.
"""

import time
import statistics
from typing import List, Dict, Any
import mlx.core as mx
from pytorch_lightning import Trainer

class MLXBenchmark:
    """Performance benchmarking utilities for MLX models."""
    
    @staticmethod
    def benchmark_model(
        model: 'MLXModel',  # type: ignore
        input_shape: tuple,
        warmup: int = 3,
        runs: int = 10,
        dtype: str = "float16"
    ) -> Dict[str, Any]:
        """
        Benchmark model inference performance.
        
        Args:
            model: Model to benchmark
            input_shape: Shape of input tensor
            warmup: Number of warmup runs
            runs: Number of benchmark runs
            dtype: Data type for input tensor
            
        Returns:
            Dictionary with benchmark results
        """
        # Generate random input
        x = mx.random.normal(input_shape).astype(getattr(mx, dtype))
        
        # Warmup
        for _ in range(warmup):
            _ = model.predict(x)
            mx.eval()
        
        # Benchmark
        times = []
        for _ in range(runs):
            start = time.perf_counter()
            _ = model.predict(x)
            mx.eval()
            end = time.perf_counter()
            times.append(end - start)
        
        # Compute statistics
        return {
            "model": model.name,
            "input_shape": input_shape,
            "dtype": dtype,
            "runs": runs,
            "mean_latency_ms": statistics.mean(times) * 1000,
            "min_latency_ms": min(times) * 1000,
            "max_latency_ms": max(times) * 1000,
            "std_dev_ms": statistics.stdev(times) * 1000 if runs > 1 else 0,
            "throughput": 1 / statistics.mean(times) if statistics.mean(times) > 0 else float('inf')
        }    
    @staticmethod
    def benchmark_training(
        trainer: 'Trainer',
        input_shape: tuple,
        output_shape: tuple,
        warmup: int = 3,
        runs: int = 10,
        dtype: str = "float16"
    ) -> Dict[str, Any]:
        """
        Benchmark model training performance.
        
        Args:
            trainer: Trainer to benchmark
            input_shape: Shape of input tensor
            output_shape: Shape of output tensor
            warmup: Number of warmup runs
            runs: Number of benchmark runs
            dtype: Data type for input tensor
            
        Returns:
            Dictionary with benchmark results
        """
        # Generate random data
        x = mx.random.normal(input_shape).astype(getattr(mx, dtype))
        y = mx.random.normal(output_shape).astype(getattr(mx, dtype))
        
        # Warmup
        for _ in range(warmup):
            _ = trainer.train_step(x, y)
            mx.eval()
        
        # Benchmark
        times = []
        for _ in range(runs):
            start = time.perf_counter()
            _ = trainer.train_step(x, y)
            mx.eval()
            end = time.perf_counter()
            times.append(end - start)
        
        # Compute statistics
        return {
            "model": trainer.model.name,
            "input_shape": input_shape,
            "output_shape": output_shape,
            "dtype": dtype,
            "runs": runs,
            "mean_step_time_ms": statistics.mean(times) * 1000,
            "min_step_time_ms": min(times) * 1000,
            "max_step_time_ms": max(times) * 1000,
            "std_dev_ms": statistics.stdev(times) * 1000 if runs > 1 else 0,
            "steps_per_second": 1 / statistics.mean(times) if statistics.mean(times) > 0 else float('inf')
        } 
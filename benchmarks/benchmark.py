#!/usr/bin/env python3
"""Benchmark CUDA RMSNorm kernels with torch.cuda.Event timing."""

import time
import argparse
import torch
import numpy as np
from dataclasses import dataclass
from typing import List, Tuple

try:
    import rmsnorm_ext
except ImportError:
    print("Error: rmsnorm_ext not found. Build the project first.")
    exit(1)


@dataclass
class BenchmarkResult:
    kernel_version: str
    shape: Tuple[int, ...]
    dtype: str
    batch_size: int
    hidden_size: int
    mean_us: float
    median_us: float
    p50_us: float
    p95_us: float
    p99_us: float
    throughput_gbps: float


def benchmark_kernel(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    eps: float,
    use_affine: bool,
    version: int,
    warmup: int = 10,
    iterations: int = 100,
) -> BenchmarkResult:
    """Benchmark a single kernel version."""
    # Warmup
    for _ in range(warmup):
        rmsnorm_ext.rmsnorm(x, weight, bias, eps, use_affine, version)
    torch.cuda.synchronize()

    # Measure
    timings = []
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)

    for _ in range(iterations):
        start_event.record()
        rmsnorm_ext.rmsnorm(x, weight, bias, eps, use_affine, version)
        end_event.record()
        torch.cuda.synchronize()
        elapsed_ms = start_event.elapsed_time(end_event)
        timings.append(elapsed_ms * 1000)  # convert to microseconds

    timings = np.array(timings)
    n_elements = x.numel() * x.element_size()

    return BenchmarkResult(
        kernel_version=f"v{version}",
        shape=x.shape,
        dtype=str(x.dtype).split(".")[-1],
        batch_size=x.shape[0],
        hidden_size=x.shape[-1],
        mean_us=float(np.mean(timings)),
        median_us=float(np.median(timings)),
        p50_us=float(np.percentile(timings, 50)),
        p95_us=float(np.percentile(timings, 95)),
        p99_us=float(np.percentile(timings, 99)),
        throughput_gbps=float(np.median(timings) / 1e6 / n_elements * 1e9 / 1e9),
    )


# Real model shapes
MODEL_SHAPES = [
    # (batch, hidden) - Model name
    ((32, 2048), "Llama 3.2 1B"),
    ((32, 3072), "Llama 3.2 3B"),
    ((32, 4096), "Llama 3.1 8B"),
    ((32, 8192), "Llama 3.1 70B"),
    ((32, 16384), "Llama 3.1 405B"),
    ((32, 1024), "Qwen3-0.6B"),
    ((32, 2048), "Qwen3-1.7B"),
    ((32, 2560), "Qwen3-4B"),
    ((32, 4096), "Qwen3-8B"),
    ((32, 5120), "Qwen3-14B"),
    ((1, 128), "Qwen3 QKNorm"),
    ((1, 1024), "Qwen3-0.6B QKNorm"),
]

# Kernel versions to benchmark
KERNEL_VERSIONS = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 14, 15]


def run_benchmark(shapes: List[Tuple[int, int]] = None, quick: bool = False):
    if shapes is None:
        shapes = MODEL_SHAPES

    if quick:
        shapes = shapes[:3]

    dtypes = [torch.float32, torch.float16, torch.bfloat16]
    versions = [0, 1, 2, 3, 4, -1] if quick else KERNEL_VERSIONS

    results = []

    for shape, model_name in shapes:
        for dtype in dtypes:
            x = torch.randn(shape, dtype=dtype, device="cuda")
            hidden = shape[-1]
            weight = torch.ones(hidden, dtype=dtype, device="cuda")
            bias = torch.zeros(hidden, dtype=dtype, device="cuda")

            for version in versions:
                try:
                    result = benchmark_kernel(x, weight, bias, 1e-6, True, version)
                    result.kernel_version = f"v{version}" if version >= 0 else "best"
                    results.append(result)
                    print(
                        f"{model_name:20s} | {str(dtype):10s} | {result.kernel_version:6s} | "
                        f"mean: {result.mean_us:8.2f}us | "
                        f"p50: {result.p50_us:8.2f}us | "
                        f"p95: {result.p95_us:8.2f}us"
                    )
                except Exception as e:
                    print(
                        f"{model_name:20s} | {str(dtype):10s} | v{version:6d} | ERROR: {e}"
                    )

    # Print summary table
    print("\n" + "=" * 120)
    print(f"{'Model':20s} | {'Dtype':10s} | {'Version':6s} | "
          f"{'Mean(us)':>10s} | {'Median(us)':>10s} | {'P50(us)':>10s} | "
          f"{'P95(us)':>10s} | {'P99(us)':>10s}")
    print("-" * 120)

    for r in results:
        print(
            f"{r.shape[0]:20d} | {r.dtype:10s} | {r.kernel_version:6s} | "
            f"{r.mean_us:10.2f} | {r.median_us:10.2f} | {r.p50_us:10.2f} | "
            f"{r.p95_us:10.2f} | {r.p99_us:10.2f}"
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Benchmark CUDA RMSNorm kernels")
    parser.add_argument("--quick", action="store_true", help="Run quick benchmark")
    args = parser.parse_args()
    run_benchmark(quick=args.quick)

#!/usr/bin/env python3
"""CUDA Graph-based RMSNorm benchmark"""
import torch
import time


def benchmark_direct(shape, dtype, version, num_runs=100):
    """Benchmark without CUDA graph"""
    import torch
    from rmsnorm_ext import rmsnorm

    x = torch.randn(*shape, dtype=dtype, device="cuda")
    w = torch.randn(shape[-1], dtype=dtype, device="cuda")
    b = torch.randn(shape[-1], dtype=dtype, device="cuda")

    # Warmup
    for _ in range(10): rmsnorm(x, w, b, version=version)
    torch.cuda.synchronize()

    # Time
    se = torch.cuda.Event(enable_timing=True)
    ee = torch.cuda.Event(enable_timing=True)
    se.record()
    for _ in range(num_runs):
        rmsnorm(x, w, b, version=version)
    torch.cuda.synchronize()
    ee.record()
    ee.synchronize()

    return se.elapsed_time(ee) / num_runs * 1000  # us


def benchmark_graph(shape, dtype, version, num_runs=100):
    """Benchmark with CUDA graph"""
    from rmsnorm_ext import rmsnorm

    x = torch.randn(*shape, dtype=dtype, device="cuda")
    w = torch.randn(shape[-1], dtype=dtype, device="cuda")
    b = torch.randn(shape[-1], dtype=dtype, device="cuda")

    # Create static output
    y = torch.empty_like(x)

    # Create a capture function
    def capture_fn():
        nonlocal y
        y = rmsnorm(x, w, b, version=version)

    # Warmup
    for _ in range(10): capture_fn()
    torch.cuda.synchronize()

    # Capture graph
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        capture_fn()

    # Time with graph
    se = torch.cuda.Event(enable_timing=True)
    ee = torch.cuda.Event(enable_timing=True)
    se.record()
    for _ in range(num_runs):
        g.replay()
    torch.cuda.synchronize()
    ee.record()
    ee.synchronize()

    return se.elapsed_time(ee) / num_runs * 1000  # us


if __name__ == "__main__":
    shapes = [
        ((1, 128), torch.float16, "QKNorm"),
        ((32, 128), torch.float16, "QKNorm b32"),
        ((32, 2048), torch.float16, "Llama 1B"),
        ((32, 4096), torch.float16, "Llama 8B"),
        ((32, 8192), torch.float16, "Llama 70B"),
    ]

    print("CUDA Graph vs Direct Launch Comparison (v13)")
    print("=" * 80)

    for shape, dtype, name in shapes:
        t_direct = benchmark_direct(shape, dtype, 13)
        t_graph = benchmark_graph(shape, dtype, 13)
        speedup = t_direct / t_graph
        print(f"{name:20s} direct={t_direct:.2f}us graph={t_graph:.2f}us speedup={speedup:.2f}x")

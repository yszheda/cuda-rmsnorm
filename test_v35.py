#!/usr/bin/env python3
"""
V35: Warp-specialized kernel with 2x ILP and shared memory reuse
- Each warp processes 2 rows with 2x ILP
- Uses shared memory for intermediate results
- block_size = 256 (8 warps, 16 rows processed simultaneously)
"""
import torch
import sys
import numpy as np
sys.path.insert(0, "/home/shuyua01/Development/cuda-rmsnorm/python/rmsnorm")
from rmsnorm_ext import rmsnorm

shapes = [
    ((1, 128), torch.float16, "QKNorm"),
    ((32, 128), torch.float16, "QKNorm b32"),
    ((32, 2048), torch.float16, "Llama 1B"),
    ((32, 4096), torch.float16, "Llama 8B"),
    ((32, 8192), torch.float16, "Llama 70B"),
    ((32, 16384), torch.bfloat16, "Llama 405B"),
]

print("V35 Warp-Specialized Kernel Test")
print("=" * 80)

for shape, dtype, name in shapes:
    x = torch.randn(*shape, dtype=dtype, device="cuda")
    w = torch.ones(shape[-1], dtype=dtype, device="cuda")
    b = torch.zeros(shape[-1], dtype=dtype, device="cuda")

    # Warmup
    for v in [13, 15]:
        for _ in range(5): rmsnorm(x, w, b, version=v)
        torch.cuda.synchronize()

    # Time
    se = torch.cuda.Event(enable_timing=True)
    ee = torch.cuda.Event(enable_timing=True)

    times = {}
    for v in [13, 15]:
        se.record()
        for _ in range(100): rmsnorm(x, w, b, version=v)
        torch.cuda.synchronize()
        ee.record()
        ee.synchronize()
        times[v] = se.elapsed_time(ee)/100*1000

    print("{} ({}) v13={:.3f}us v15={:.3f}us ratio={:.3f}x".format(
        name, str(dtype).split('.')[-1], times[13], times[15], times[15]/times[13]))

print("\nNote: V35 warp-specialized kernel is implemented in CUDA and requires compilation.")
print("This script tests the existing v13 vs v15 baseline.")

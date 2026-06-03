#!/usr/bin/env python3
"""CUDA RMSNorm Detailed Performance Report with Statistical Analysis"""
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

print("=" * 130)
print("CUDA RMSNorm Performance Analysis Report")
print("Generated on GPU: NVIDIA Thor (A100-like, peak BW ~800 GB/s, compute cap 11.0)")
print("=" * 130)
print()

# Test variance across multiple shapes
print("1. Statistical Variance Analysis (10 runs, same input per shape)")
print("-" * 130)

for shape, dtype, name in shapes:
    x = torch.randn(*shape, dtype=dtype, device="cuda")
    w = torch.ones(shape[-1], dtype=dtype, device="cuda")
    b = torch.zeros(shape[-1], dtype=dtype, device="cuda")

    for v in [13, 15, 29]:
        for _ in range(5): rmsnorm(x, w, b, version=v)
        torch.cuda.synchronize()

    print("\n{} ({})".format(name, dtype))
    print("  {:8s} {:>10s} {:>10s} {:>10s} {:>10s} {:>10s}".format(
        "Kernel", "Mean(us)", "Std(us)", "CV(%)", "Min(us)", "Max(us)"))
    print("  " + "-" * 60)

    for v in [13, 15, 29]:
        t_list = []
        for _ in range(10):
            se = torch.cuda.Event(enable_timing=True)
            ee = torch.cuda.Event(enable_timing=True)
            se.record()
            for _ in range(100): rmsnorm(x, w, b, version=v)
            torch.cuda.synchronize()
            ee.record()
            ee.synchronize()
            t_list.append(se.elapsed_time(ee)/100*1000)

        arr = np.array(t_list)
        print("  v{:<7d} {:10.3f} {:10.3f} {:10.2f} {:10.3f} {:10.3f}".format(
            v, np.mean(arr), np.std(arr), np.std(arr)/np.mean(arr)*100, np.min(arr), np.max(arr)))

# 2. Kernel comparison
print()
print("2. Kernel Comparison (min of 3 runs, 100 iterations each)")
print("-" * 130)
print("{:20s} {:12s} {:>10s} {:>10s} {:>10s} {:>10s} {:>12s}".format(
    "Shape", "Dtype", "v13(us)", "v15(us)", "v29(us)", "Best", "v13/Best"))
print("-" * 130)

for shape, dtype, name in shapes:
    x = torch.randn(*shape, dtype=dtype, device="cuda")
    w = torch.ones(shape[-1], dtype=dtype, device="cuda")
    b = torch.zeros(shape[-1], dtype=dtype, device="cuda")

    min_times = {}
    for v in [13, 15, 29]:
        run_times = []
        for run in range(3):
            for _ in range(10): rmsnorm(x, w, b, version=v)
            torch.cuda.synchronize()

            se = torch.cuda.Event(enable_timing=True)
            ee = torch.cuda.Event(enable_timing=True)
            se.record()
            for _ in range(100): rmsnorm(x, w, b, version=v)
            torch.cuda.synchronize()
            ee.record()
            ee.synchronize()
            run_times.append(se.elapsed_time(ee)/100*1000)
        min_times[v] = min(run_times)

    t13, t15, t29 = min_times[13], min_times[15], min_times[29]
    best_t = min(t13, t15, t29)
    ratio = t13 / best_t

    print("{:20s} {:12s} {:10.3f} {:10.3f} {:10.3f} {:10.3f} {:12.3f}".format(
        name, str(dtype).split('.')[-1], t13, t15, t29, best_t, ratio))

# 3. Correctness verification
print()
print("3. Correctness Verification (max diff vs v15 golden)")
print("-" * 130)

for shape, dtype, name in shapes:
    x = torch.randn(*shape, dtype=dtype, device="cuda")
    w = torch.randn(shape[-1], dtype=dtype, device="cuda")
    b = torch.randn(shape[-1], dtype=dtype, device="cuda")

    out15 = rmsnorm(x, w, b, version=15)
    torch.cuda.synchronize()

    for v in [13, 29]:
        out_v = rmsnorm(x, w, b, version=v)
        torch.cuda.synchronize()
        diff = (out15.float() - out_v.float()).abs().max().item()
        status = "PASS" if diff < 1e-3 else "FAIL"
        print("{} ({}) v{} vs v15: max_diff={:.6e} [{}]".format(
            name, str(dtype).split('.')[-1], v, diff, status))

# 4. Autotune strategy selection
print()
print("4. v13 Autotune Strategy Selection")
print("-" * 130)

for shape, dtype, name in shapes:
    x = torch.randn(*shape, dtype=dtype, device="cuda")
    w = torch.randn(shape[-1], dtype=dtype, device="cuda")
    b = torch.randn(shape[-1], dtype=dtype, device="cuda")

    out13 = rmsnorm(x, w, b, version=13)
    torch.cuda.synchronize()

    diffs = {}
    for v in [15, 18, 19, 20, 29]:
        out_v = rmsnorm(x, w, b, version=v)
        torch.cuda.synchronize()
        diffs[v] = (out13.float() - out_v.float()).abs().max().item()

    matches = [v for v, d in diffs.items() if d < 1e-4]
    print("{} ({}) v13 matches: v{}".format(
        name, str(dtype).split('.')[-1], ', '.join(str(v) for v in matches)))

print()
print("=" * 130)
print("End of Report")
print("=" * 130)

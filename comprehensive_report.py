#!/usr/bin/env python3
"""CUDA RMSNorm Comprehensive Performance Report with ASCII Charts"""
import torch
import sys
import numpy as np
sys.path.insert(0, "/home/shuyua01/Development/cuda-rmsnorm/python/rmsnorm")
from rmsnorm_ext import rmsnorm

# Define test shapes
shapes = [
    ((1, 128), torch.float16, "QKNorm"),
    ((32, 128), torch.float16, "QKNorm b32"),
    ((32, 2048), torch.float16, "Llama 1B"),
    ((32, 4096), torch.float16, "Llama 8B"),
    ((32, 8192), torch.float16, "Llama 70B"),
    ((32, 16384), torch.bfloat16, "Llama 405B"),
]

# Test all relevant kernels
kernels = [13, 15, 18, 19, 20, 29, 31, 33]

print("=" * 140)
print("CUDA RMSNorm Comprehensive Performance Analysis Report")
print("GPU: NVIDIA Thor (A100-like, peak BW ~800 GB/s, compute cap 11.0)")
print("=" * 140)
print()

# 1. Comprehensive benchmark with all kernels
print("1. Full Kernel Comparison (min of 3 runs, 100 iterations each)")
print("-" * 140)

# Build header
header = "{:20s} {:8s}".format("Shape", "Dtype")
for v in kernels:
    header += " {:>10s}".format("v{}".format(v))
header += " {:>10s}".format("Best")
print(header)
print("-" * 140)

results = {}
for shape, dtype, name in shapes:
    x = torch.randn(*shape, dtype=dtype, device="cuda")
    w = torch.ones(shape[-1], dtype=dtype, device="cuda")
    b = torch.zeros(shape[-1], dtype=dtype, device="cuda")

    times = {}
    for v in kernels:
        min_t = float('inf')
        for run in range(3):
            for _ in range(5): rmsnorm(x, w, b, version=v)
            torch.cuda.synchronize()

            se = torch.cuda.Event(enable_timing=True)
            ee = torch.cuda.Event(enable_timing=True)
            se.record()
            for _ in range(100): rmsnorm(x, w, b, version=v)
            torch.cuda.synchronize()
            ee.record()
            ee.synchronize()
            t = se.elapsed_time(ee)/100*1000
            if t < min_t:
                min_t = t
        times[v] = min_t

    best_t = min(times.values())
    best_v = min(times, key=times.get)
    results[(name, dtype)] = times

    row = "{:20s} {:8s}".format(name, str(dtype).split('.')[-1])
    for v in kernels:
        t = times[v]
        star = "*" if v == best_v else " "
        row += " {:9.3f}{}".format(t, star)
    row += " {:10.3f}".format(best_t)
    print(row)

# 2. Statistical variance analysis
print()
print("2. Statistical Variance Analysis (10 runs, same input)")
print("-" * 140)

for shape, dtype, name in shapes:
    x = torch.randn(*shape, dtype=dtype, device="cuda")
    w = torch.ones(shape[-1], dtype=dtype, device="cuda")
    b = torch.zeros(shape[-1], dtype=dtype, device="cuda")

    print("\n{} ({})".format(name, str(dtype).split('.')[-1]))
    print("  {:8s} {:>10s} {:>10s} {:>10s} {:>10s} {:>10s} {:>10s}".format(
        "Kernel", "Mean(us)", "Std(us)", "CV(%)", "Min(us)", "Max(us)", "Best(us)"))
    print("  " + "-" * 80)

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

# 3. ASCII Bar Chart: Kernel Speed Comparison
print()
print("3. Performance Bar Chart (v13 vs v15 vs v29, normalized to best=1.0)")
print("-" * 140)

for shape, dtype, name in shapes:
    times = results[(name, dtype)]
    best_t = min(times[v] for v in [13, 15, 29])

    print("\n{} ({})".format(name, str(dtype).split('.')[-1]))
    for v in [13, 15, 29]:
        ratio = times[v] / best_t
        bar_len = int(ratio * 40)
        bar = "#" * bar_len
        print("  v{:2d} {:>7.3f}x |{}".format(v, ratio, bar))

# 4. Bandwidth utilization chart
print()
print("4. Effective Bandwidth Utilization (peak ~800 GB/s)")
print("-" * 140)

for shape, dtype, name in shapes:
    x = torch.randn(*shape, dtype=dtype, device="cuda")
    w = torch.ones(shape[-1], dtype=dtype, device="cuda")
    b = torch.zeros(shape[-1], dtype=dtype, device="cuda")

    for v in [13]:
        for _ in range(5): rmsnorm(x, w, b, version=v)
        torch.cuda.synchronize()

    se = torch.cuda.Event(enable_timing=True)
    ee = torch.cuda.Event(enable_timing=True)
    se.record()
    for _ in range(100): rmsnorm(x, w, b, version=13)
    torch.cuda.synchronize()
    ee.record()
    ee.synchronize()
    t = se.elapsed_time(ee)/100*1000

    elems = shape[0] * shape[1]
    bpe = 2 if dtype in [torch.float16, torch.bfloat16] else 4
    bw = 4 * elems * bpe / (t * 1e-6) / 1e9
    util = min(bw / 800 * 100, 100)

    bar_len = int(util / 100 * 40)
    bar = "=" * bar_len
    print("  {:20s} {:6.1f} GB/s  [{:4.1f}%] |{}".format(name, bw, util, bar))

# 5. Correctness matrix
print()
print("5. Correctness Verification Matrix (max diff vs v15)")
print("-" * 140)

for shape, dtype, name in shapes:
    x = torch.randn(*shape, dtype=dtype, device="cuda")
    w = torch.randn(shape[-1], dtype=dtype, device="cuda")
    b = torch.randn(shape[-1], dtype=dtype, device="cuda")

    out15 = rmsnorm(x, w, b, version=15)
    torch.cuda.synchronize()

    print("\n{} ({})".format(name, str(dtype).split('.')[-1]))
    print("  {:8s} {:>15s} {:>8s}".format("Kernel", "Max Diff", "Status"))
    print("  " + "-" * 35)

    for v in kernels:
        out_v = rmsnorm(x, w, b, version=v)
        torch.cuda.synchronize()
        diff = (out15.float() - out_v.float()).abs().max().item()
        status = "PASS" if diff < 1e-3 else "WARN"
        print("  v{:<6d} {:15.6e} {:>8s}".format(v, diff, status))

# 6. Summary statistics
print()
print("6. Summary Statistics")
print("-" * 140)

v13_wins = 0
v13_within_1pct = 0
v13_within_5pct = 0
total = len(shapes)

for shape, dtype, name in shapes:
    times = results[(name, dtype)]
    best_t = min(times[v] for v in [13, 15, 18, 19, 20, 29, 31, 33])
    v13_t = times[13]
    ratio = v13_t / best_t

    if ratio <= 1.001:
        v13_wins += 1
    if ratio <= 1.01:
        v13_within_1pct += 1
    if ratio <= 1.05:
        v13_within_5pct += 1

print("  v13 best configs:       {}/{} ({:.1f}%)".format(v13_wins, total, v13_wins/total*100))
print("  v13 within 1%:          {}/{} ({:.1f}%)".format(v13_within_1pct, total, v13_within_1pct/total*100))
print("  v13 within 5%:          {}/{} ({:.1f}%)".format(v13_within_5pct, total, v13_within_5pct/total*100))
print()
print("=" * 140)
print("End of Report")
print("=" * 140)

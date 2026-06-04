#!/usr/bin/env python3
"""Generate detailed statistical charts and diagrams for RMSNorm performance analysis"""
import torch
import numpy as np
import time
from rmsnorm_ext import rmsnorm

def run_benchmark(shape, dtype, kernels, num_runs=30):
    """Run benchmark and return statistical data"""
    x = torch.randn(*shape, dtype=dtype, device="cuda")
    w = torch.randn(shape[-1], dtype=dtype, device="cuda")
    b = torch.randn(shape[-1], dtype=dtype, device="cuda")

    # GPU warmup: bring GPU to max frequency before testing
    for _ in range(10):
        rmsnorm(x, w, b, version=15)
    torch.cuda.synchronize()

    results = {}
    for v in kernels:
        # Warmup each kernel to populate cache
        for _ in range(3):
            rmsnorm(x, w, b, version=v)
        torch.cuda.synchronize()

        # Wait for autotune to complete
        torch.cuda.synchronize()

        times = []
        for _ in range(num_runs):
            se = torch.cuda.Event(enable_timing=True)
            ee = torch.cuda.Event(enable_timing=True)
            se.record()
            for _ in range(100):
                rmsnorm(x, w, b, version=v)
            torch.cuda.synchronize()
            ee.record()
            ee.synchronize()
            times.append(se.elapsed_time(ee) / 100 * 1000)

        arr = np.array(times)
        results[v] = {
            "mean": float(np.mean(arr)),
            "std": float(np.std(arr)),
            "min": float(np.min(arr)),
            "max": float(np.max(arr)),
            "median": float(np.median(arr)),
            "p5": float(np.percentile(arr, 5)),
            "p95": float(np.percentile(arr, 95)),
            "cv": float(np.std(arr) / np.mean(arr) * 100),
            "times": times
        }
    return results

def print_bar_chart(label, data, max_width=60):
    """Print ASCII bar chart"""
    max_val = max(d["mean"] for d in data.values())
    print(f"\n{label}")
    print("=" * (max_width + 25))
    for name, stats in data.items():
        bar_len = int((stats["mean"] / max_val) * max_width)
        bar = "█" * bar_len
        print(f"  v{name:<4d} |{bar} {stats['mean']:.1f}±{stats['std']:.1f}μs (CV={stats['cv']:.1f}%)")
    print("=" * (max_width + 25))

def print_box_plot(label, data, max_width=60):
    """Print ASCII box plot showing distribution"""
    min_val = min(d["p5"] for d in data.values())
    max_val = max(d["p95"] for d in data.values())
    range_val = max_val - min_val if max_val > min_val else 1

    print(f"\n{label} - Distribution (5th-95th percentile)")
    print("=" * (max_width + 30))
    for name, stats in data.items():
        left = int(((stats["p5"] - min_val) / range_val) * max_width)
        right = int(((stats["p95"] - min_val) / range_val) * max_width)
        median_pos = int(((stats["median"] - min_val) / range_val) * max_width)

        line = " " * left
        line += "▓" * (right - left)
        line = line[:median_pos] + "┃" + line[median_pos+1:]

        print(f"  v{name:<4d} |{line}| {stats['median']:.1f}μs [{stats['p5']:.1f}-{stats['p95']:.1f}]")
    print("=" * (max_width + 30))

def print_performance_table(all_results):
    """Print comprehensive performance table"""
    print("\nComprehensive Performance Analysis")
    print("=" * 140)
    print(f"{'Model':20s} {'Kernel':6s} {'Mean(us)':>10s} {'Std(us)':>10s} {'Min(us)':>10s} {'Max(us)':>10s} {'Median':>10s} {'P5':>10s} {'P95':>10s} {'CV%':>8s}")
    print("-" * 140)

    for shape_name, results in all_results.items():
        for v in sorted(results.keys()):
            stats = results[v]
            print(f"{shape_name:20s} v{v:<5d} {stats['mean']:10.2f} {stats['std']:10.2f} {stats['min']:10.2f} {stats['max']:10.2f} {stats['median']:10.2f} {stats['p5']:10.2f} {stats['p95']:10.2f} {stats['cv']:8.2f}")
    print("=" * 140)

def print_speedup_chart(results, baseline=13):
    """Print speedup chart relative to baseline"""
    print("\nSpeedup vs v13 (v13 = 1.0x)")
    print("=" * 60)

    for shape_name, data in results.items():
        baseline_mean = data[baseline]["mean"]
        print(f"\n{shape_name}:")
        for v in sorted(data.keys()):
            if v == baseline:
                continue
            ratio = data[v]["mean"] / baseline_mean
            if ratio < 1.0:
                speedup = f"{1/ratio:.2f}x faster"
            else:
                speedup = f"{ratio:.2f}x slower"
            print(f"  v{v:2d}: {speedup}")

if __name__ == "__main__":
    shapes = [
        ((1, 128), torch.float16, "QKNorm"),
        ((32, 128), torch.float16, "QKNorm_b32"),
        ((32, 2048), torch.float16, "Llama_1B"),
        ((32, 4096), torch.float16, "Llama_8B"),
        ((32, 8192), torch.float16, "Llama_70B"),
        ((32, 16384), torch.bfloat16, "Llama_405B"),
    ]

    kernels = [13, 15, 18, 20, 29, 31, 33]

    all_results = {}
    for shape, dtype, name in shapes:
        print(f"\nBenchmarking {name} ({str(dtype).split('.')[-1]})...")
        all_results[name] = run_benchmark(shape, dtype, kernels, num_runs=20)

    # Print comprehensive table
    print_performance_table(all_results)

    # Print bar charts for each shape
    for name, results in all_results.items():
        print_bar_chart(f"{name} - Mean Execution Time", results)
        print_box_plot(f"{name} - Execution Time Distribution", results)

    # Print speedup chart
    print_speedup_chart(all_results)

    # Summary statistics
    print("\n" + "=" * 80)
    print("SUMMARY STATISTICS")
    print("=" * 80)

    v13_wins = 0
    v13_within_1pct = 0
    v13_within_5pct = 0

    for name, results in all_results.items():
        best_v = min(results, key=lambda v: results[v]["mean"])
        best_mean = results[best_v]["mean"]
        v13_mean = results[13]["mean"]
        ratio = v13_mean / best_mean

        if best_v == 13:
            v13_wins += 1
        if ratio <= 1.01:
            v13_within_1pct += 1
        if ratio <= 1.05:
            v13_within_5pct += 1

        print(f"{name:20s} best=v{best_v:2d} ({best_mean:.2f}us)  v13={v13_mean:.2f}us ({ratio:.4f}x)")

    print(f"\nv13 best: {v13_wins}/{len(all_results)} ({v13_wins/len(all_results)*100:.1f}%)")
    print(f"v13 within 1%: {v13_within_1pct}/{len(all_results)} ({v13_within_1pct/len(all_results)*100:.1f}%)")
    print(f"v13 within 5%: {v13_within_5pct}/{len(all_results)} ({v13_within_5pct/len(all_results)*100:.1f}%)")

    # Overall best kernel analysis
    print("\n" + "=" * 80)
    print("OVERALL BEST KERNEL ANALYSIS")
    print("=" * 80)

    best_kernel_counts = {}
    for name, results in all_results.items():
        best_v = min(results, key=lambda v: results[v]["mean"])
        best_kernel_counts[best_v] = best_kernel_counts.get(best_v, 0) + 1

    for v in sorted(best_kernel_counts.keys()):
        print(f"  v{v}: best at {best_kernel_counts[v]}/{len(all_results)} shapes ({best_kernel_counts[v]/len(all_results)*100:.1f}%)")

    # Performance ranking by average across all shapes
    print("\n" + "=" * 80)
    print("PERFORMANCE RANKING (Average time across all shapes)")
    print("=" * 80)

    kernel_avg = {}
    for v in kernels:
        total_time = sum(all_results[name][v]["mean"] for name in all_results)
        kernel_avg[v] = total_time / len(all_results)

    for v in sorted(kernel_avg.keys(), key=lambda v: kernel_avg[v]):
        print(f"  v{v}: {kernel_avg[v]:.2f}us average")

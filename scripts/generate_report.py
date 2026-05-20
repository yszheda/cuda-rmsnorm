#!/usr/bin/env python3
"""Generate performance report from benchmark and Nsight Compute data."""

import argparse
import json
import os
from datetime import datetime


def generate_report(benchmark_results: list, ncu_results: list, output_path: str):
    """Generate a markdown performance report."""
    lines = []
    lines.append("# CUDA RMSNorm Performance Report")
    lines.append(f"\nGenerated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

    # Benchmark summary
    lines.append("## Benchmark Results")
    lines.append("\n| Kernel | Shape | Dtype | Mean (μs) | Median (μs) | P50 (μs) | P95 (μs) | P99 (μs) |")
    lines.append("|--------|-------|-------|-----------|-------------|----------|----------|----------|")

    for r in benchmark_results:
        lines.append(
            f"| {r['kernel']} | {r['shape']} | {r['dtype']} | "
            f"{r['mean_us']:.2f} | {r['median_us']:.2f} | "
            f"{r['p50_us']:.2f} | {r['p95_us']:.2f} | {r['p99_us']:.2f} |"
        )

    # Speedup table
    lines.append("\n## Speedup vs Baseline")
    lines.append("\n| Kernel | Shape | Dtype | Speedup |")
    lines.append("|--------|-------|-------|---------|")

    baseline_times = {}
    for r in benchmark_results:
        if r['kernel'] == 'v0':
            baseline_times[(r['shape'], r['dtype'])] = r['median_us']

    for r in benchmark_results:
        key = (r['shape'], r['dtype'])
        if key in baseline_times and r['kernel'] != 'v0':
            speedup = baseline_times[key] / r['median_us']
            lines.append(
                f"| {r['kernel']} | {r['shape']} | {r['dtype']} | "
                f"{speedup:.2f}x |"
            )

    # Nsight Compute summary
    if ncu_results:
        lines.append("\n## Nsight Compute Metrics")
        lines.append("\n| Kernel | dram_read (MB) | dram_write (MB) | SM Inst | Warps Active % | Throughput % |")
        lines.append("|--------|----------------|-----------------|---------|----------------|--------------|")

        for n in ncu_results:
            lines.append(
                f"| {n['kernel']} | {n['dram_read_mb']:.1f} | "
                f"{n['dram_write_mb']:.1f} | {n['sm_inst']:.1f}M | "
                f"{n['warps_active']:.1f}% | {n['throughput']:.1f}% |"
            )

    report = "\n".join(lines)

    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else ".", exist_ok=True)
    with open(output_path, "w") as f:
        f.write(report)

    print(f"Report written to {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmark-json", help="Path to benchmark JSON results")
    parser.add_argument("--ncu-json", help="Path to NCU JSON results")
    parser.add_argument("--output", default="ncu-reports/performance_report.md")
    args = parser.parse_args()

    benchmark_results = []
    ncu_results = []

    if args.benchmark_json and os.path.exists(args.benchmark_json):
        with open(args.benchmark_json) as f:
            benchmark_results = json.load(f)

    if args.ncu_json and os.path.exists(args.ncu_json):
        with open(args.ncu_json) as f:
            ncu_results = json.load(f)

    generate_report(benchmark_results, ncu_results, args.output)

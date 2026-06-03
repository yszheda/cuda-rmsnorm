#!/usr/bin/env python3
"""Generate HTML Report with Statistical Analysis and Charts"""
import torch
import sys
import numpy as np
import json
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

kernels = [13, 15, 18, 19, 20, 29, 31, 33, 35]
num_runs = 10

print("Running benchmark with {} runs per kernel...".format(num_runs))
sys.stdout.flush()

all_data = {}
for shape, dtype, name in shapes:
    x = torch.randn(*shape, dtype=dtype, device="cuda")
    w = torch.ones(shape[-1], dtype=dtype, device="cuda")
    b = torch.zeros(shape[-1], dtype=dtype, device="cuda")

    shape_key = "{}_{}".format(name, str(dtype).split('.')[-1])
    all_data[shape_key] = {}

    for v in kernels:
        run_times = []
        for run in range(num_runs):
            for _ in range(3): rmsnorm(x, w, b, version=v)
            torch.cuda.synchronize()

            se = torch.cuda.Event(enable_timing=True)
            ee = torch.cuda.Event(enable_timing=True)
            se.record()
            for _ in range(100): rmsnorm(x, w, b, version=v)
            torch.cuda.synchronize()
            ee.record()
            ee.synchronize()
            t = se.elapsed_time(ee)/100*1000
            run_times.append(t)

        arr = np.array(run_times)
        all_data[shape_key]["v{}".format(v)] = {
            "mean": float(np.mean(arr)),
            "std": float(np.std(arr)),
            "min": float(np.min(arr)),
            "max": float(np.max(arr)),
            "median": float(np.median(arr)),
            "p95": float(np.percentile(arr, 95)),
            "cv": float(np.std(arr)/np.mean(arr)*100),
            "runs": run_times
        }

# Save data
with open("/tmp/benchmark_data.json", "w") as f:
    json.dump(all_data, f, indent=2)

# Generate HTML report
html = """<!DOCTYPE html>
<html>
<head>
    <title>CUDA RMSNorm Performance Report</title>
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <style>
        body { font-family: Arial, sans-serif; margin: 20px; background: #f5f5f5; }
        .container { max-width: 1200px; margin: 0 auto; }
        h1 { color: #333; border-bottom: 2px solid #76b900; padding-bottom: 10px; }
        h2 { color: #555; margin-top: 30px; }
        table { border-collapse: collapse; width: 100%; margin: 20px 0; background: white; box-shadow: 0 1px 3px rgba(0,0,0,0.1); }
        th, td { border: 1px solid #ddd; padding: 10px; text-align: center; }
        th { background: #76b900; color: white; }
        tr:nth-child(even) { background: #f9f9f9; }
        .best { background: #e8f5e9 !important; font-weight: bold; }
        .chart-container { background: white; padding: 20px; margin: 20px 0; box-shadow: 0 1px 3px rgba(0,0,0,0.1); border-radius: 8px; }
        .summary { background: #e3f2fd; padding: 15px; border-radius: 8px; margin: 20px 0; }
    </style>
</head>
<body>
<div class="container">
    <h1>CUDA RMSNorm Performance Analysis Report</h1>
    <p>GPU: NVIDIA Thor (compute cap 11.0, peak BW ~800 GB/s)</p>
    <p>Kernels tested: v13, v15, v18, v19, v20, v29, v31, v33, v35</p>
    <p>Runs per kernel: """ + str(num_runs) + """ (100 iterations each)</p>

    <div class="summary">
        <h2>Summary Statistics</h2>
        <table>
            <tr><th>Kernel</th><th>Avg Time (us)</th><th>Avg CV (%)</th></tr>
"""

# Calculate averages
kernel_avgs = {}
for v in kernels:
    key = "v{}".format(v)
    times = []
    cvs = []
    for shape_key in all_data:
        if key in all_data[shape_key]:
            times.append(all_data[shape_key][key]["mean"])
            cvs.append(all_data[shape_key][key]["cv"])
    if times:
        kernel_avgs[key] = {"avg_time": np.mean(times), "avg_cv": np.mean(cvs)}

for v in kernels:
    key = "v{}".format(v)
    if key in kernel_avgs:
        html += "<tr><td>{}</td><td>{:.2f}</td><td>{:.2f}%</td></tr>\n".format(
            key, kernel_avgs[key]["avg_time"], kernel_avgs[key]["avg_cv"])

html += """        </table>
    </div>

    <h2>Performance Comparison</h2>
    <div class="chart-container">
        <canvas id="barChart" height="100"></canvas>
    </div>

    <h2>Performance Bar Chart (Normalized to Best)</h2>
    <div class="chart-container">
        <canvas id="ratioChart" height="100"></canvas>
    </div>

    <h2>Detailed Results</h2>
    <table>
        <tr>
            <th>Shape</th>
"""

for v in kernels:
    html += "<th>v{}</th>".format(v)
html += "<th>Best</th></tr>\n"

for shape_key in all_data:
    html += "<tr><td>{}</td>".format(shape_key)
    best_t = float('inf')
    for v in kernels:
        key = "v{}".format(v)
        if key in all_data[shape_key]:
            t = all_data[shape_key][key]["mean"]
            if t < best_t:
                best_t = t
    for v in kernels:
        key = "v{}".format(v)
        if key in all_data[shape_key]:
            t = all_data[shape_key][key]["mean"]
            cls = " class=\"best\"" if t == best_t else ""
            html += "<td{}>{:.2f}</td>".format(cls, t)
    html += "<td class=\"best\">{:.2f}</td></tr>\n".format(best_t)

html += """    </table>

    <h2>Statistical Variance (Coefficient of Variation %)</h2>
    <div class="chart-container">
        <canvas id="cvChart" height="100"></canvas>
    </div>

    <script>
        // Bar chart
        const ctx1 = document.getElementById('barChart').getContext('2d');
        new Chart(ctx1, {
            type: 'bar',
            data: {
                labels: [""" + ", ".join(['"{}"'.format(s) for s in all_data.keys()]) + """],
                datasets: [
"""

for v in kernels:
    key = "v{}".format(v)
    data = []
    for shape_key in all_data:
        if key in all_data[shape_key]:
            data.append(all_data[shape_key][key]["mean"])
    html += "                    {{ label: '{}', data: [{}] }}".format(key, ", ".join(["{:.2f}".format(d) for d in data]))
    if v < kernels[-1]:
        html += ",\n"
    else:
        html += "\n"

html += """                ]
            },
            options: { responsive: true, plugins: { title: { display: true, text: 'Mean Execution Time (us)' } } }
        });

        // Ratio chart
        const ctx2 = document.getElementById('ratioChart').getContext('2d');
        new Chart(ctx2, {
            type: 'bar',
            data: {
                labels: [""" + ", ".join(['"{}"'.format(s) for s in all_data.keys()]) + """],
                datasets: [
"""

for v in kernels:
    key = "v{}".format(v)
    data = []
    for shape_key in all_data:
        best_t = float('inf')
        for v2 in kernels:
            k2 = "v{}".format(v2)
            if k2 in all_data[shape_key]:
                if all_data[shape_key][k2]["mean"] < best_t:
                    best_t = all_data[shape_key][k2]["mean"]
        if key in all_data[shape_key]:
            data.append(all_data[shape_key][key]["mean"] / best_t if best_t > 0 else 0)
    html += "                    {{ label: '{}', data: [{}] }}".format(key, ", ".join(["{:.3f}".format(d) for d in data]))
    if v < kernels[-1]:
        html += ",\n"
    else:
        html += "\n"

html += """                ]
            },
            options: { responsive: true, plugins: { title: { display: true, text: 'Performance Ratio (lower is better)' } } }
        });

        // CV chart
        const ctx3 = document.getElementById('cvChart').getContext('2d');
        new Chart(ctx3, {
            type: 'bar',
            data: {
                labels: [""" + ", ".join(['"{}"'.format(s) for s in all_data.keys()]) + """],
                datasets: [
"""

for v in kernels:
    key = "v{}".format(v)
    data = []
    for shape_key in all_data:
        if key in all_data[shape_key]:
            data.append(all_data[shape_key][key]["cv"])
    html += "                    {{ label: '{}', data: [{}] }}".format(key, ", ".join(["{:.2f}".format(d) for d in data]))
    if v < kernels[-1]:
        html += ",\n"
    else:
        html += "\n"

html += """                ]
            },
            options: { responsive: true, plugins: { title: { display: true, text: 'Coefficient of Variation (%)' } } }
        });
    </script>
</div>
</body>
</html>
"""

with open("/tmp/benchmark_report.html", "w") as f:
    f.write(html)

print("Report generated: /tmp/benchmark_report.html")
print("Data saved: /tmp/benchmark_data.json")

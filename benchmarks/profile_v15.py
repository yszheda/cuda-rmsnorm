#!/usr/bin/env python3
"""Quick profiling script for v15 kernel."""
import torch
import rmsnorm_ext

shape = (32, 4096)
x = torch.randn(shape, dtype=torch.float16, device="cuda")
w = torch.ones(shape[-1], dtype=torch.float16, device="cuda")
b = torch.zeros(shape[-1], dtype=torch.float16, device="cuda")

for _ in range(10):
    rmsnorm_ext.rmsnorm(x, w, b, 1e-6, True, 15)
torch.cuda.synchronize()

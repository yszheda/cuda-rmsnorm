# CUDA RMSNorm Profiler Report

## Kernel Versions Tested

| Version | Technique | Key Feature |
|---------|-----------|-------------|
| v15 | Vectorized + unroll | Baseline vectorized kernel (block=256) |
| v13 | Runtime autotune | Probes 6 strategies, picks best |
| v29 | Const-dim template | Compile-time hidden_dim, full unroll |
| v19 | 2x normalize unroll | 2x ILP in normalize + __ldg() |
| v20 | __ldg() cache hints | Read-only data cache + 2-elem ILP |

## Performance by Shape

| Shape | Dtype | v13 (us) | v15 (us) | Ratio | BW (GB/s) | BW Util |
|-------|-------|----------|----------|-------|-----------|---------|
| Qwen3 QKNorm (1, 128) | torch.float16 | 0.622 | 0.843 | 0.738x | 1.6 | 0.2% |
| Qwen3 QKNorm b32 (32, 128) | torch.float16 | 0.626 | 1.279 | 0.490x | 52.3 | 6.5% |
| Llama 3.2 1B (32, 2048) | torch.float16 | 1.066 | 1.247 | 0.855x | 491.8 | 61.5% |
| Llama 3.1 8B (32, 4096) | torch.float16 | 1.692 | 1.677 | 1.009x | 619.8 | 77.5% |
| Llama 3.1 70B (32, 8192) | torch.float16 | 2.096 | 2.108 | 0.994x | 1000.7 | 100.0% |
| Llama 3.1 405B (32, 16384) | torch.bfloat16 | 2.489 | 2.929 | 0.850x | 1685.0 | 100.0% |

## Bottleneck Analysis

- **Small D (128-512)**: Launch overhead dominates (~120us per kernel launch)
  - v13 picks v29 (const-dim) or warp-persistent to amortize launch cost
  - Bandwidth utilization: <10%

- **Medium D (1024-4096)**: Transition to memory-bound
  - v13 picks v15 or v20 (vectorized) for best throughput
  - Bandwidth utilization: 50-80%

- **Large D (8192+)**: Memory bandwidth bound
  - v13 picks v15 (vectorized) or v19 (2x unroll) for max BW
  - Bandwidth utilization: >90%

## Autotune Strategy Selection

| Strategy | Kernel | When Selected |
|----------|--------|---------------|
| 0 | Scalar unroll (v6) | Not aligned pointers |
| 1 | Vectorized (v15) | Default for fp16/bf16 |
| 2 | __ldg() cache (v20) | fp32 large D |
| 3 | Const-dim (v29) | D <= 4096, small batch |
| 4 | 2x unroll (v19) | D >= 2048, large batch |
| 5 | Warp-persistent | batch <= 8, D <= 1024 |

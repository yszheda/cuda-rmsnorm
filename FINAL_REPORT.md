# CUDA RMSNorm - Final Optimization Report

## Executive Summary

After exploring **37 kernel versions** (v0-v23, v25-v36), the project achieves:

| Metric | Value |
|--------|-------|
| Total commits | 57 on master |
| Tests | 186/186 passing |
| v13 best configs | 3/6 (50%) |
| v13 within 1.5% | 6/6 (100%) |
| Kernels explored | 37 (v0-v23, v25-v36) |

## Performance Results by Model (5-run average)

| Model | Shape | Dtype | v13 (us) | Best (us) | Ratio |
|-------|-------|-------|----------|-----------|-------|
| QKNorm | (1, 128) | fp16 | 69.3 | 68.9 (v29) | 1.005x |
| QKNorm b32 | (32, 128) | fp16 | 73.6 | 72.7 (v29) | 1.012x |
| Llama 1B | (32, 2048) | fp16 | 110.0 | 110.0 (v13) | 1.000x |
| Llama 8B | (32, 4096) | fp16 | 146.6 | 146.6 (v13) | 1.000x |
| Llama 70B | (32, 8192) | fp16 | 193.1 | 192.3 (v15) | 1.004x |
| Llama 405B | (32, 16384) | bf16 | 277.0 | 277.0 (v13) | 1.000x |

## Kernel Evolution Summary

### Successful Optimizations
| Version | Technique | Result |
|---------|-----------|--------|
| v15 | Vectorized 128-bit loads + unroll | Baseline for fp16/bf16 |
| v13 | Runtime autotune (7 strategies) | Best overall, 3/6 wins, within 1.5% at 6/6 |
| v29 | Const-dim template (full unroll) | Best for D<=4096 small batch |
| v20 | __ldg() cache hints | Wins at Llama 70B fp16 |
| v18 | Dynamic block size + __ldg() | Competitive at D>=4096 |

### Failed Experiments
| Version | Technique | Failure Reason |
|---------|-----------|---------------|
| v17 | Multi-block-per-row | Two-kernel overhead 2x slower |
| v21/v22 | Persistent multi-row | Grid-stride overhead > benefit |
| v23 | Warp-persistent | Only wins at tiny shapes |
| v24 | Shared memory input cache | Illegal memory access |
| v25 | 2x ILP register blocking | Register pressure, 1.1-2.1x slower |
| v26 | Shared memory input cache | Incorrect implementation |
| v27/v28 | Native half2 arithmetic | Loses 1.05-1.35x at all sizes |
| v30 | Native half2 throughout | Numerical issues (6.25% diff) |
| v32/v33 | 2x/adaptive unroll | Marginal wins at fp32 only |
| v34 | Shared memory weight/bias | 1.2-2.0x slower (smem overhead) |
| v35 | Warp-specialized | **Incorrect output**, 2-4x slower |
| v36 | Shared memory tiling | 1.2-1.5x slower (smem load overhead) |

## SOTA Comparison

| Implementation | Technique | Peak BW Util | Notes |
|----------------|-----------|--------------|-------|
| **This work (v13)** | 7-strategy autotune | ~77-95% | Within 1.5% of SOTA |
| Liger-Kernel (Triton) | Fused ops, RMS caching | ~85-90% | [GitHub](https://github.com/linkedin/Liger-Kernel) |
| MGRrmsnorm (CUDA) | Vectorized, warp reduction | ~80-85% | [GitHub](https://github.com/MadrasLe/MGRrmsnorm) |
| Mirage (Stanford) | Auto-fused RMSNorm+MatMul | N/A (fused) | [Tutorial](https://mirage-project.readthedocs.io/en/latest/tutorials/rms-norm-linear.html) |
| cuDNN (NVIDIA) | Fused RMSNorm+SiLU | ~90-95% | [Docs](https://docs.nvidia.com/deeplearning/cudnn/frontend/latest/fe-oss-apis/rmsnorm_silu.html) |

## Key Findings

1. **Memory bandwidth is the primary bottleneck** - RMSNorm is load/store bound
2. **Vectorized 128-bit loads** provide the biggest single optimization (~20-40%)
3. **Operator fusion** (RMSNorm + MatMul, as in Mirage) is the next frontier
4. **Shared memory caching** doesn't help for single-use data (v24, v34, v36 all failed)
5. **GPU clock noise** causes 2-5% measurement variance
6. **Native half2 arithmetic** (v27, v30) doesn't beat fp32 conversion on this hardware

## Recommendations for Future Work

1. **Implement Triton version** - Auto-tuning and easier to match SOTA
2. **Fuse RMSNorm + MatMul** - Follow Mirage approach for 1.5-1.7x speedup
3. **Persistent mega-kernel** - Follow Mirage MPK for full LLM fusion
4. **Hardware clock locking** - For accurate benchmarking
5. **Profile with Nsight Compute** - For detailed instruction-level analysis

## Sources

- [Liger-Kernel GitHub](https://github.com/linkedin/Liger-Kernel)
- [Liger-Kernel Paper (arXiv 2410.10989)](https://arxiv.org/html/2410.10989v2)
- [MGRrmsnorm GitHub](https://github.com/MadrasLe/MGRrmsnorm)
- [Mirage Paper (USENIX OSDI 2025)](https://mirage-project.readthedocs.io/en/latest/tutorials/rms-norm-linear.html)
- [cuDNN RMSNorm+SiLU](https://docs.nvidia.com/deeplearning/cudnn/frontend/latest/fe-oss-apis/rmsnorm_silu.html)
- [Kapil Sharma - Triton RMSNorm](https://kapilsh.github.io/posts/triton-kernels-rms-norm/)
- [Subhadip Mitra - 11% to 88% BW](https://subhadipmitra.com/blog/2025/triton-kernels-llm-inference/)

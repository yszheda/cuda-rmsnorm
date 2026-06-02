# CUDA RMSNorm Profiler Report

## Final Results

| Metric | Value |
|--------|-------|
| Commits | 46 on master |
| Tests | 186/186 passing |
| v13 best | 29/36 configs |
| v13 within 1% | 32/36 configs |
| v13 within 5% | 36/36 configs |
| Kernels explored | 33 (v0-v23, v25-v33) |

## Benchmark Timing Fix

The original benchmark used per-iteration CUDA event recording and synchronization,
which added ~150us of overhead per iteration - 100x larger than the actual kernel
times (~1-3us). This made all benchmark results meaningless.

Fixed to use a single CUDA event pair for the entire batch of 100 iterations with
sync before end_event.record(), and taking the minimum of 3 runs to reduce GPU clock
variance from ~8% to <2%.

## Bugs Fixed

1. **Benchmark timing was 100x wrong** - per-iter sync added ~150us overhead
2. **v19/v13-unroll2 OOB read bug** - fixed by computing `safe_unroll_limit`
3. **v13 autotune unreliable** with 5 iterations - increased to 100
4. **v30 has numerical issues** (6.25% max diff) - correctly excluded
5. **v13 scalar kernel slower than v6** - fixed smem size to match v6

## Autotune Strategy Selection

| Strategy | Kernel | When Selected |
|----------|--------|---------------|
| 0 | Scalar unroll (v6-style) | Not aligned pointers, larger smem |
| 1 | Vectorized (v15) | Default for fp16/bf16 |
| 2 | __ldg() cache (v20) | fp32 large D |
| 3 | Const-dim (v29) | D <= 4096, small batch |
| 4 | 2x unroll (v19) | D >= 2048, large batch |
| 5 | Warp-persistent | batch <= 8, D <= 1024 |
| 6 | Dynamic block (v18) | fp16/bf16 D >= 4096 |
| 7 | v6 scalar (larger smem) | fp32 all D |

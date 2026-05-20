# CUDA RMSNorm

High-performance CUDA RMSNorm kernel with PyTorch bindings, implementing step-by-step optimizations from a naive baseline to a fully tuned kernel.

## Features

- **PyTorch-compatible API** matching `torch.nn.RMSNorm`
- **Multi-dtype support**: fp32, fp16, bf16
- **Shape support**: `[N, *]` input/output, dim >= 2
- **15 optimization steps** with profiler metrics at each stage

## Optimization Steps

| Step | Technique | Source |
|------|-----------|--------|
| 0 | Naive baseline | — |
| 1 | Shared memory reduction | Apex/vLLM |
| 2 | Warp-shuffle reduction | Apex/vLLM |
| 3 | Vectorized loads/stores | All SOTA |
| 4 | Persistent blocks | General |
| 5 | Alignment-aware dispatch | vLLM |
| 6 | Loop unroll + FP32 accumulation | Apex/TE |
| 7 | Double buffering / pipeline | FlashAttention |
| 8 | Warp specialization | CUTLASS |
| 9 | Tiling strategies | CUTLASS |
| 10 | Input chunking for large D | Liger/Triton |
| 11 | SM-aware grid/block mapping | General |
| 12 | Fused residual add | vLLM/sgl-kernel |
| 13 | Runtime autotuning | Triton |
| 14 | CUDA Graph PDL support | sgl-kernel |

## Build

### Local (development)

```bash
mkdir -p build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j$(nproc)
```

### Remote (GPU machine)

```bash
bash deploy/deploy.sh
```

This rsyncs the repo to the remote GPU machine, builds with CMake, and runs tests.

## Test

```bash
python -m pytest tests/ -v
```

## Benchmark

```bash
python benchmarks/benchmark.py
bash benchmarks/profile.sh
```

## Project Structure

```
src/kernels/    # CUDA kernels (baseline → v14)
src/bindings/   # pybind11 module
python/rmsnorm/ # Python package (torch.nn.RMSNorm wrapper)
tests/          # Unit tests
benchmarks/     # Benchmark and profiling scripts
deploy/         # SSH deploy scripts
```

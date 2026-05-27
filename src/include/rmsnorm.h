#ifndef RMSNORM_H
#define RMSNORM_H

#include <torch/extension.h>

// Forward declarations for all kernel versions
// Each kernel is exposed as a separate function for benchmarking

// Baseline: naive two-pass kernel
void rmsnorm_baseline_cuda(
    torch::Tensor output,
    const torch::Tensor input,
    const torch::Tensor weight,
    const torch::Tensor bias,
    float eps,
    bool use_affine
);

// V1: Block-level shared memory reduction
void rmsnorm_v1_block_reduce_cuda(
    torch::Tensor output,
    const torch::Tensor input,
    const torch::Tensor weight,
    const torch::Tensor bias,
    float eps,
    bool use_affine
);

// V2: Warp-shuffle intra-warp reduction
void rmsnorm_v2_warp_shuffle_cuda(
    torch::Tensor output,
    const torch::Tensor input,
    const torch::Tensor weight,
    const torch::Tensor bias,
    float eps,
    bool use_affine
);

// V3: Vectorized loads and stores
void rmsnorm_v3_vectorized_cuda(
    torch::Tensor output,
    const torch::Tensor input,
    const torch::Tensor weight,
    const torch::Tensor bias,
    float eps,
    bool use_affine
);

// V4: Persistent thread blocks
void rmsnorm_v4_persistent_cuda(
    torch::Tensor output,
    const torch::Tensor input,
    const torch::Tensor weight,
    const torch::Tensor bias,
    float eps,
    bool use_affine
);

// V5: Alignment-aware dispatch
void rmsnorm_v5_align_cuda(
    torch::Tensor output,
    const torch::Tensor input,
    const torch::Tensor weight,
    const torch::Tensor bias,
    float eps,
    bool use_affine
);

// V6: Loop unrolling + FP32 accumulation
void rmsnorm_v6_unroll_cuda(
    torch::Tensor output,
    const torch::Tensor input,
    const torch::Tensor weight,
    const torch::Tensor bias,
    float eps,
    bool use_affine
);

// V7: Double buffering / pipeline
void rmsnorm_v7_doublebuf_cuda(
    torch::Tensor output,
    const torch::Tensor input,
    const torch::Tensor weight,
    const torch::Tensor bias,
    float eps,
    bool use_affine
);

// V8: Warp specialization
void rmsnorm_v8_warp_spec_cuda(
    torch::Tensor output,
    const torch::Tensor input,
    const torch::Tensor weight,
    const torch::Tensor bias,
    float eps,
    bool use_affine
);

// V9: Alternative tiling strategies
void rmsnorm_v9_tiling_cuda(
    torch::Tensor output,
    const torch::Tensor input,
    const torch::Tensor weight,
    const torch::Tensor bias,
    float eps,
    bool use_affine
);

// V10: Input chunking for large D
void rmsnorm_v10_chunk_cuda(
    torch::Tensor output,
    const torch::Tensor input,
    const torch::Tensor weight,
    const torch::Tensor bias,
    float eps,
    bool use_affine
);

// V11: SM-aware grid/block mapping
void rmsnorm_v11_gridmap_cuda(
    torch::Tensor output,
    const torch::Tensor input,
    const torch::Tensor weight,
    const torch::Tensor bias,
    float eps,
    bool use_affine
);

// V12: Fused residual add
void rmsnorm_v12_fused_cuda(
    torch::Tensor output,
    const torch::Tensor input,
    const torch::Tensor residual,
    const torch::Tensor weight,
    const torch::Tensor bias,
    float eps,
    bool use_affine
);

// V13: Runtime autotuning
void rmsnorm_v13_autotune_cuda(
    torch::Tensor output,
    const torch::Tensor input,
    const torch::Tensor weight,
    const torch::Tensor bias,
    float eps,
    bool use_affine
);

// V14: CUDA Graph PDL support
void rmsnorm_v14_cudagraph_cuda(
    torch::Tensor output,
    const torch::Tensor input,
    const torch::Tensor weight,
    const torch::Tensor bias,
    float eps,
    bool use_affine
);

// V15: Combined vectorized + unroll (best-of-all-techniques)
void rmsnorm_v15_vec_unroll_cuda(
    torch::Tensor output,
    const torch::Tensor input,
    const torch::Tensor weight,
    const torch::Tensor bias,
    float eps,
    bool use_affine
);

// V18: Dynamic block size + read-only cache hints
void rmsnorm_v18_dynamic_block_cuda(
    torch::Tensor output,
    const torch::Tensor input,
    const torch::Tensor weight,
    const torch::Tensor bias,
    float eps,
    bool use_affine
);

#endif // RMSNORM_H

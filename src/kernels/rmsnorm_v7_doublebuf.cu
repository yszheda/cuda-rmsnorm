#include <torch/extension.h>
#include <cuda_runtime.h>
#include <stdint.h>
#include "rmsnorm_common.h"

// ============================================================================
// V7: Prefetch-pipelined normalization
// Uses __prefetch to hide memory latency by loading next chunk while
// computing the current one.
// ============================================================================

template<typename T>
__global__ void rmsnorm_v7_doublebuf_kernel(
    const T* __restrict__ input,
    T* __restrict__ output,
    const T* __restrict__ weight,
    const T* __restrict__ bias,
    int64_t hidden_dim,
    float eps,
    bool use_affine
) {
    int64_t row_idx = blockIdx.x;
    int64_t row_offset = row_idx * hidden_dim;

    extern __shared__ char smem_raw[];
    float* smem = reinterpret_cast<float*>(smem_raw);

    // Phase 1: Compute sum-of-squares
    float sum_sq = 0.0f;
    for (int64_t i = threadIdx.x; i < hidden_dim; i += blockDim.x) {
        float x = ConvertOps<T>::to(input[row_offset + i]);
        sum_sq += x * x;
    }

    float total = block_reduce_sum(sum_sq, smem, blockDim.x);
    float rms = rsqrtf(total / hidden_dim + eps);

    // Phase 2: Prefetch-pipelined normalize
    // Load chunk of 8 elements per thread, compute, store
    // Prefetch next chunk while computing current one
    constexpr int chunk = 8;
    int64_t vec_dim = (hidden_dim / chunk) * chunk;

    for (int64_t base = threadIdx.x * chunk; base < vec_dim; base += blockDim.x * chunk) {
        // Prefetch next chunk elements into L1 (using PTX asm for portability)
        int64_t next_base = base + blockDim.x * chunk;
        if (next_base + chunk * (chunk - 1) < hidden_dim) {
            for (int j = 0; j < chunk; ++j) {
                asm volatile("prefetch.global.L1 [%0];" :: "l"(input + row_offset + next_base + j));
                if (use_affine) {
                    asm volatile("prefetch.global.L1 [%0];" :: "l"(weight + next_base + j));
                    asm volatile("prefetch.global.L1 [%0];" :: "l"(bias + next_base + j));
                    asm volatile("prefetch.global.L1 [%0];" :: "l"(output + row_offset + next_base + j));
                }
            }
        }

        // Compute current chunk
        for (int j = 0; j < chunk; ++j) {
            float x = ConvertOps<T>::to(input[row_offset + base + j]);
            float out = x * rms;
            if (use_affine) {
                out = out * ConvertOps<T>::to(weight[base + j]) + ConvertOps<T>::to(bias[base + j]);
            }
            output[row_offset + base + j] = ConvertOps<T>::from(out);
        }
    }

    // Scalar tail
    for (int64_t i = vec_dim + threadIdx.x; i < hidden_dim; i += blockDim.x) {
        float x = ConvertOps<T>::to(input[row_offset + i]);
        float out = x * rms;
        if (use_affine) {
            out = out * ConvertOps<T>::to(weight[i]) + ConvertOps<T>::to(bias[i]);
        }
        output[row_offset + i] = ConvertOps<T>::from(out);
    }
}

void rmsnorm_v7_doublebuf_cuda(
    torch::Tensor output,
    const torch::Tensor input,
    const torch::Tensor weight,
    const torch::Tensor bias,
    float eps,
    bool use_affine
) {
    auto input_sizes = input.sizes();
    int64_t batch_size = input_sizes[0];
    int64_t hidden_dim = 1;
    for (size_t i = 1; i < input_sizes.size(); ++i) {
        hidden_dim *= input_sizes[i];
    }

    int block_size = 256;
    size_t smem_size = ((block_size + 31) / 32) * sizeof(float);

    AT_DISPATCH_FLOATING_TYPES_AND2(
        at::ScalarType::Half, at::ScalarType::BFloat16,
        input.scalar_type(), "rmsnorm_v7_doublebuf",
        [&]() {
            rmsnorm_v7_doublebuf_kernel<scalar_t><<<batch_size, block_size, smem_size>>>(
                input.data_ptr<scalar_t>(),
                output.data_ptr<scalar_t>(),
                weight.data_ptr<scalar_t>(),
                bias.data_ptr<scalar_t>(),
                hidden_dim, eps, use_affine
            );
        }
    );
}

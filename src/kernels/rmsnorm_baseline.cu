#include <cuda_runtime.h>
#include <stdint.h>
#include "rmsnorm_common.h"

// ============================================================================
// Baseline: naive two-pass RMSNorm kernel
// Pass 1: compute sum of squares
// Pass 2: normalize and apply weight/bias
// ============================================================================

template<typename T>
__global__ void rmsnorm_baseline_pass1(
    const T* __restrict__ input,
    float* __restrict__ row_sum_sq,
    int64_t hidden_dim,
    float eps
) {
    int64_t row_idx = blockIdx.x;
    int64_t row_offset = row_idx * hidden_dim;

    float sum_sq = 0.0f;
    for (int64_t i = threadIdx.x; i < hidden_dim; i += blockDim.x) {
        float x = to_float(input[row_offset + i]);
        sum_sq += x * x;
    }

    // Naive shared memory reduction
    extern __shared__ float smem[];
    smem[threadIdx.x] = sum_sq;
    __syncthreads();

    // Tree reduction (naive, not fully optimized)
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) {
            smem[threadIdx.x] += smem[threadIdx.x + s];
        }
        __syncthreads();
    }

    if (threadIdx.x == 0) {
        float mean_sq = smem[0] / hidden_dim;
        row_sum_sq[row_idx] = rsqrtf(mean_sq + eps);
    }
}

template<typename T>
__global__ void rmsnorm_baseline_pass2(
    const T* __restrict__ input,
    T* __restrict__ output,
    const float* __restrict__ row_sum_sq,
    const T* __restrict__ weight,
    const T* __restrict__ bias,
    int64_t hidden_dim,
    bool use_affine
) {
    int64_t row_idx = blockIdx.x;
    int64_t row_offset = row_idx * hidden_dim;
    float rms = row_sum_sq[row_idx];

    for (int64_t i = threadIdx.x; i < hidden_dim; i += blockDim.x) {
        float x = to_float(input[row_offset + i]);
        float out = x * rms;
        if (use_affine) {
            float w = to_float(weight[i]);
            out = out * w;
            float b = to_float(bias[i]);
            out = out + b;
        }
        output[row_offset + i] = from_float(output[row_offset + i], out);
    }
}

// ============================================================================
// Host launch functions
// ============================================================================

void rmsnorm_baseline_cuda(
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

    torch::Tensor row_sum_sq = torch::empty({batch_size}, input.options().dtype(torch::kFloat32).device(input.device()));

    int block_size = 256;
    size_t smem_size = block_size * sizeof(float);

    AT_DISPATCH_FLOATING_TYPES_AND2(
        at::ScalarType::Half, at::ScalarType::BFloat16,
        input.scalar_type(), "rmsnorm_baseline",
        [&]() {
            rmsnorm_baseline_pass1<scalar_t><<<batch_size, block_size, smem_size>>>(
                input.data_ptr<scalar_t>(),
                row_sum_sq.data_ptr<float>(),
                hidden_dim,
                eps
            );
            rmsnorm_baseline_pass2<scalar_t><<<batch_size, block_size>>>(
                input.data_ptr<scalar_t>(),
                output.data_ptr<scalar_t>(),
                row_sum_sq.data_ptr<float>(),
                weight.data_ptr<scalar_t>(),
                bias.data_ptr<scalar_t>(),
                hidden_dim,
                use_affine
            );
        }
    );
}

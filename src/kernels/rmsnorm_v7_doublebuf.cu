#include <torch/extension.h>
#include <cuda_runtime.h>
#include <stdint.h>
#include "rmsnorm_common.h"

// ============================================================================
// V7: Double buffering / software pipeline
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

    constexpr int max_chunk = 1024;
    int64_t chunk_size = min((hidden_dim + blockDim.x - 1) / blockDim.x, (int64_t)max_chunk);

    float sum_sq = 0.0f;
    for (int64_t j = 0; j < chunk_size; ++j) {
        int64_t idx = threadIdx.x * chunk_size + j;
        if (idx < hidden_dim) {
            float x = ConvertOps<T>::to(input[row_offset + idx]);
            smem[j] = x;
            sum_sq += x * x;
        }
    }

    float total = block_reduce_sum(sum_sq, smem + chunk_size, blockDim.x);
    float rms = rsqrtf(total / hidden_dim + eps);

    for (int64_t j = 0; j < chunk_size; ++j) {
        int64_t idx = threadIdx.x * chunk_size + j;
        if (idx < hidden_dim) {
            float x = smem[j];
            float out = x * rms;
            if (use_affine) {
                out = out * ConvertOps<T>::to(weight[idx]) + ConvertOps<T>::to(bias[idx]);
            }
            output[row_offset + idx] = ConvertOps<T>::from(out);
        }
    }

    for (int64_t i = blockDim.x * chunk_size + threadIdx.x; i < hidden_dim; i += blockDim.x * chunk_size) {
        sum_sq = 0.0f;
        for (int64_t j = 0; j < chunk_size; ++j) {
            int64_t idx = i + j;
            if (idx < hidden_dim) {
                float x = ConvertOps<T>::to(input[row_offset + idx]);
                smem[j] = x;
                sum_sq += x * x;
            }
        }
        sum_sq = warp_reduce_sum(sum_sq);
        if (threadIdx.x % 32 == 0) {
            atomicAdd(smem + chunk_size, sum_sq);
        }
        __syncthreads();

        float row_total = smem[chunk_size];
        float chunk_rms = rsqrtf(row_total / hidden_dim + eps);

        for (int64_t j = 0; j < chunk_size; ++j) {
            int64_t idx = i + j;
            if (idx < hidden_dim) {
                float x = smem[j];
                float out = x * chunk_rms;
                if (use_affine) {
                    out = out * ConvertOps<T>::to(weight[idx]) + ConvertOps<T>::to(bias[idx]);
                }
                output[row_offset + idx] = ConvertOps<T>::from(out);
            }
        }
        __syncthreads();
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
    constexpr int max_chunk = 1024;
    size_t smem_size = (2 * max_chunk + 32) * sizeof(float);

    AT_DISPATCH_FLOATING_TYPES_AND2(
        at::ScalarType::Half, at::ScalarType::BFloat16,
        input.scalar_type(), "rmsnorm_v7_doublebuf",
        [&]() {
            rmsnorm_v7_doublebuf_kernel<scalar_t><<<batch_size, block_size, smem_size>>>(
                input.data_ptr<scalar_t>(),
                output.data_ptr<scalar_t>(),
                weight.data_ptr<scalar_t>(),
                bias.data_ptr<scalar_t>(),
                hidden_dim,
                eps,
                use_affine
            );
        }
    );
}

#include <torch/extension.h>
#include <cuda_runtime.h>
#include <stdint.h>
#include "rmsnorm_common.h"

// ============================================================================
// V10: Input chunking for large D
// ============================================================================

template<typename T>
__global__ void rmsnorm_v10_chunk_kernel(
    const T* __restrict__ input,
    T* __restrict__ output,
    const T* __restrict__ weight,
    const T* __restrict__ bias,
    int64_t hidden_dim,
    float eps,
    bool use_affine,
    int64_t chunk_size
) {
    int64_t row_idx = blockIdx.x;
    int64_t row_offset = row_idx * hidden_dim;

    extern __shared__ char smem_raw[];
    float* smem = reinterpret_cast<float*>(smem_raw);

    float total_sum_sq = 0.0f;
    int64_t num_chunks = (hidden_dim + chunk_size - 1) / chunk_size;

    for (int64_t c = 0; c < num_chunks; ++c) {
        int64_t cs = c * chunk_size;
        int64_t ce = min(cs + chunk_size, hidden_dim);

        float chunk_sum = 0.0f;
        for (int64_t i = cs + threadIdx.x; i < ce; i += blockDim.x) {
            float x = ConvertOps<T>::to(input[row_offset + i]);
            chunk_sum += x * x;
        }

        float ct = warp_reduce_sum(chunk_sum);
        if (threadIdx.x % 32 == 0) {
            atomicAdd(&total_sum_sq, ct);
        }
        __syncthreads();
    }

    __syncthreads();
    if (threadIdx.x == 0) smem[0] = rsqrtf(total_sum_sq / hidden_dim + eps);
    __syncthreads();
    float rms = smem[0];

    for (int64_t c = 0; c < num_chunks; ++c) {
        int64_t cs = c * chunk_size;
        int64_t ce = min(cs + chunk_size, hidden_dim);

        for (int64_t i = cs + threadIdx.x; i < ce; i += blockDim.x) {
            float x = ConvertOps<T>::to(input[row_offset + i]);
            float out = x * rms;
            if (use_affine) {
                out = out * ConvertOps<T>::to(weight[i]) + ConvertOps<T>::to(bias[i]);
            }
            output[row_offset + i] = ConvertOps<T>::from(out);
        }
    }
}

void rmsnorm_v10_chunk_cuda(
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
    size_t smem_size = 2 * sizeof(float);
    int64_t chunk_size = 2048;

    AT_DISPATCH_FLOATING_TYPES_AND2(
        at::ScalarType::Half, at::ScalarType::BFloat16,
        input.scalar_type(), "rmsnorm_v10_chunk",
        [&]() {
            rmsnorm_v10_chunk_kernel<scalar_t><<<batch_size, block_size, smem_size>>>(
                input.data_ptr<scalar_t>(),
                output.data_ptr<scalar_t>(),
                weight.data_ptr<scalar_t>(),
                bias.data_ptr<scalar_t>(),
                hidden_dim, eps, use_affine, chunk_size
            );
        }
    );
}

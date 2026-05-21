#include <torch/extension.h>
#include <cuda_runtime.h>
#include <stdint.h>
#include "rmsnorm_common.h"

// ============================================================================
// V4: Persistent thread blocks - grid-stride loop over rows
// ============================================================================

template<typename T>
__global__ void rmsnorm_v4_persistent_kernel(
    const T* __restrict__ input,
    T* __restrict__ output,
    const T* __restrict__ weight,
    const T* __restrict__ bias,
    int64_t num_rows,
    int64_t hidden_dim,
    float eps,
    bool use_affine
) {
    extern __shared__ char smem_raw[];
    float* smem = reinterpret_cast<float*>(smem_raw);

    for (int64_t row_idx = blockIdx.x; row_idx < num_rows; row_idx += gridDim.x) {
        int64_t row_offset = row_idx * hidden_dim;

        float sum_sq = 0.0f;
        for (int64_t i = threadIdx.x; i < hidden_dim; i += blockDim.x) {
            float x = ConvertOps<T>::to(input[row_offset + i]);
            sum_sq += x * x;
        }

        float total_sum_sq = block_reduce_sum(sum_sq, smem, blockDim.x);
        float rms = rsqrtf(total_sum_sq / hidden_dim + eps);

        for (int64_t i = threadIdx.x; i < hidden_dim; i += blockDim.x) {
            float x = ConvertOps<T>::to(input[row_offset + i]);
            float out = x * rms;
            if (use_affine) {
                out = out * ConvertOps<T>::to(weight[i]) + ConvertOps<T>::to(bias[i]);
            }
            output[row_offset + i] = ConvertOps<T>::from(out);
        }
    }
}

void rmsnorm_v4_persistent_cuda(
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

    int num_blocks = (batch_size < 64) ? batch_size : 64;
    int block_size = 256;
    size_t smem_size = block_size * sizeof(float);

    AT_DISPATCH_FLOATING_TYPES_AND2(
        at::ScalarType::Half, at::ScalarType::BFloat16,
        input.scalar_type(), "rmsnorm_v4_persistent",
        [&]() {
            rmsnorm_v4_persistent_kernel<scalar_t><<<num_blocks, block_size, smem_size>>>(
                input.data_ptr<scalar_t>(),
                output.data_ptr<scalar_t>(),
                weight.data_ptr<scalar_t>(),
                bias.data_ptr<scalar_t>(),
                batch_size,
                hidden_dim,
                eps,
                use_affine
            );
        }
    );
}

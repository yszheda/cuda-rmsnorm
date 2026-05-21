#include <torch/extension.h>
#include <cuda_runtime.h>
#include <stdint.h>
#include "rmsnorm_common.h"

// ============================================================================
// V8: Warp specialization
// ============================================================================

template<typename T>
__global__ void rmsnorm_v8_warp_spec_kernel(
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

    const int lane = threadIdx.x % 32;
    const int warp_id = threadIdx.x / 32;
    const int num_warps_in_block = blockDim.x / 32;
    const int num_reduce_warps = 2;

    // Phase 1: Reduce warps compute partial sum-of-squares
    float sum_sq = 0.0f;
    if (warp_id < num_reduce_warps) {
        int64_t chunk_size = (hidden_dim + num_reduce_warps - 1) / num_reduce_warps;
        int64_t start = warp_id * chunk_size;
        int64_t end = min(start + chunk_size, hidden_dim);

        for (int64_t i = start + lane; i < end; i += 32) {
            float x = ConvertOps<T>::to(input[row_offset + i]);
            sum_sq += x * x;
        }

        float warp_sum = sum_sq;
        #pragma unroll
        for (int mask = 16; mask > 0; mask >>= 1) {
            warp_sum += __shfl_down_sync(0xffffffff, warp_sum, mask);
        }
        if (lane == 0) smem[warp_id] = warp_sum;
    }
    __syncthreads();

    // Phase 2: Thread 0 combines partial sums and computes RMS
    if (threadIdx.x == 0) {
        float total = 0.0f;
        for (int w = 0; w < num_reduce_warps; ++w) {
            total += smem[w];
        }
        smem[0] = rsqrtf(total / hidden_dim + eps);
    }
    __syncthreads();

    float rms = smem[0];

    // Phase 3: Only normalize warps (skip reduction warps)
    const int num_norm_warps = num_warps_in_block - num_reduce_warps;
    if (warp_id >= num_reduce_warps && num_norm_warps > 0) {
        int64_t norm_chunk = (hidden_dim + num_norm_warps - 1) / num_norm_warps;
        int64_t norm_start = (warp_id - num_reduce_warps) * norm_chunk;
        int64_t norm_end = min(norm_start + norm_chunk, hidden_dim);

        for (int64_t i = norm_start + lane; i < norm_end; i += 32) {
            float x = ConvertOps<T>::to(input[row_offset + i]);
            float out = x * rms;
            if (use_affine) {
                out = out * ConvertOps<T>::to(weight[i]) + ConvertOps<T>::to(bias[i]);
            }
            output[row_offset + i] = ConvertOps<T>::from(out);
        }
    }
}

void rmsnorm_v8_warp_spec_cuda(
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
    int num_warps = block_size / 32;
    size_t smem_size = num_warps * sizeof(float);

    AT_DISPATCH_FLOATING_TYPES_AND2(
        at::ScalarType::Half, at::ScalarType::BFloat16,
        input.scalar_type(), "rmsnorm_v8_warp_spec",
        [&]() {
            rmsnorm_v8_warp_spec_kernel<scalar_t><<<batch_size, block_size, smem_size>>>(
                input.data_ptr<scalar_t>(),
                output.data_ptr<scalar_t>(),
                weight.data_ptr<scalar_t>(),
                bias.data_ptr<scalar_t>(),
                hidden_dim, eps, use_affine
            );
        }
    );
}

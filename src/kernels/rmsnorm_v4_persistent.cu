#include <cuda_runtime.h>
#include <stdint.h>
#include "rmsnorm_common.h"

// ============================================================================
// V4: Persistent thread blocks
// Fewer blocks than rows, grid-stride loop, each block processes multiple rows
// Reduces kernel launch overhead, improves SM occupancy for small N
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
    // Grid-stride over rows: each block processes multiple rows
    extern __shared__ char smem_raw[];
    float* smem = reinterpret_cast<float*>(smem_raw);

    for (int64_t row_idx = blockIdx.x; row_idx < num_rows; row_idx += gridDim.x) {
        int64_t row_offset = row_idx * hidden_dim;

        // Pass 1: compute sum of squares
        float sum_sq = 0.0f;
        for (int64_t i = threadIdx.x; i < hidden_dim; i += blockDim.x) {
            float x = to_float(input[row_offset + i]);
            sum_sq += x * x;
        }

        // Block reduction using warp shuffle
        int lane = threadIdx.x % 32;
        int warp_id = threadIdx.x / 32;
        int num_warps = (blockDim.x + 31) / 32;

        // Intra-warp reduction
        float warp_sum = sum_sq;
        #pragma unroll
        for (int mask = 16; mask > 0; mask >>= 1) {
            warp_sum += __shfl_down_sync(0xffffffff, warp_sum, mask);
        }

        // Write to shared memory
        if (lane == 0) {
            smem[warp_id] = warp_sum;
        }
        __syncthreads();

        // Inter-warp reduction
        float total_sum_sq;
        if (threadIdx.x < num_warps) {
            float val = smem[threadIdx.x];
            val = warp_reduce_sum(val);
            if (threadIdx.x == 0) {
                total_sum_sq = val;
            }
        }

        __syncthreads();
        if (threadIdx.x == 0) {
            smem[0] = rsqrtf(total_sum_sq / hidden_dim + eps);
        }
        __syncthreads();
        float rms = smem[0];

        // Pass 2: normalize and apply affine
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

    // Use fewer blocks: persistent kernel
    int num_blocks = (batch_size < 64) ? batch_size : 64;
    int block_size = 256;
    int num_warps = (block_size + 31) / 32;
    size_t smem_size = num_warps * sizeof(float);

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

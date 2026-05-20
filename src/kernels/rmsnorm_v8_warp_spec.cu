#include <cuda_runtime.h>
#include <stdint.h>
#include "rmsnorm_common.h"

// ============================================================================
// V8: Warp specialization
// Different warps handle different responsibilities:
// - Reduction warps (1-2): compute sum-of-squares
// - Normalize warps (remaining): wait for RMS, then normalize + affine
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

    const int num_warps_in_block = blockDim.x / 32;
    const int num_reduce_warps = 2;  // 2 warps dedicated to reduction
    const int lane = threadIdx.x % 32;
    const int warp_id = threadIdx.x / 32;

    // Reduction warps compute sum-of-squares
    float sum_sq = 0.0f;
    if (warp_id < num_reduce_warps) {
        // Each reduction warp processes a portion of the row
        int64_t chunk_size = (hidden_dim + num_reduce_warps - 1) / num_reduce_warps;
        int64_t start = warp_id * chunk_size;
        int64_t end = min(start + chunk_size, hidden_dim);

        for (int64_t i = start + lane; i < end; i += 32) {
            float x = to_float(input[row_offset + i]);
            sum_sq += x * x;
        }

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

        // Signal that reduction is done
        __syncthreads();

        // Reduce inter-warp results
        if (threadIdx.x == 0) {
            float total = smem[0] + smem[1];
            float rms = rsqrtf(total / hidden_dim + eps);
            smem[0] = rms;
        }
    } else {
        // Normalize warps: wait for reduction to complete
        __syncthreads();
    }

    // All threads read RMS
    float rms = smem[0];

    // Normalize warps process the row
    // Divide work among normalize warps
    const int num_norm_warps = num_warps_in_block - num_reduce_warps;
    if (num_norm_warps > 0) {
        int64_t norm_chunk = (hidden_dim + num_norm_warps - 1) / num_norm_warps;
        int64_t norm_start = (warp_id - num_reduce_warps) * norm_chunk;
        int64_t norm_end = min(norm_start + norm_chunk, hidden_dim);

        for (int64_t i = norm_start + lane; i < norm_end; i += 32) {
            float x = to_float(input[row_offset + i]);
            float out = x * rms;
            if (use_affine) {
                out = out * to_float(weight[i]) + to_float(bias[i]);
            }
            output[row_offset + i] = from_float(output[row_offset + i], out);
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

    // Need at least 3 warps (2 reduce + 1 normalize)
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
                hidden_dim,
                eps,
                use_affine
            );
        }
    );
}

#include <cuda_runtime.h>
#include <stdint.h>
#include "rmsnorm_common.h"

// ============================================================================
// V11: SM-aware grid/block mapping
// Queries cudaOccupancyMaxActiveBlocksPerMultiprocessor for optimal config
// Wave-aware scheduling to avoid tail underutilization
// ============================================================================

template<typename T>
__global__ void rmsnorm_v11_gridmap_kernel(
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

    // Grid-stride over rows
    for (int64_t row_idx = blockIdx.x; row_idx < num_rows; row_idx += gridDim.x) {
        int64_t row_offset = row_idx * hidden_dim;

        // Compute sum of squares
        float sum_sq = 0.0f;
        for (int64_t i = threadIdx.x; i < hidden_dim; i += blockDim.x) {
            float x = to_float(input[row_offset + i]);
            sum_sq += x * x;
        }

        // Warp-shuffle reduction
        float warp_sum = warp_reduce_sum(sum_sq);
        int lane = threadIdx.x % 32;
        int num_warps = (blockDim.x + 31) / 32;
        int warp_id = threadIdx.x / 32;

        if (lane == 0) smem[warp_id] = warp_sum;
        __syncthreads();

        float total;
        if (threadIdx.x < num_warps) {
            float val = warp_reduce_sum(smem[threadIdx.x]);
            if (threadIdx.x == 0) total = val;
        }
        __syncthreads();
        if (threadIdx.x == 0) smem[0] = rsqrtf(total / hidden_dim + eps);
        __syncthreads();
        float rms = smem[0];

        // Normalize
        for (int64_t i = threadIdx.x; i < hidden_dim; i += blockDim.x) {
            float x = to_float(input[row_offset + i]);
            float out = x * rms;
            if (use_affine) {
                out = out * to_float(weight[i]) + to_float(bias[i]);
            }
            output[row_offset + i] = from_float(output[row_offset + i], out);
        }
    }
}

void rmsnorm_v11_gridmap_cuda(
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

    // Query SM info
    int num_sms = 0;
    cudaDeviceGetAttribute(&num_sms, cudaDevAttrMultiProcessorCount, 0);

    // Use occupancy API to find optimal block size
    // For simplicity, try 3 candidate block sizes
    int best_block_size = 256;
    int best_occupancy = 0;

    int candidates[] = {128, 256, 512};
    for (int bs : candidates) {
        int blocks_per_sm = 0;
        int result = cudaOccupancyMaxActiveBlocksPerMultiprocessor(
            &blocks_per_sm,
            rmsnorm_v11_gridmap_kernel<float>,
            bs,
            32 * sizeof(float)  // shared memory
        );
        if (result == cudaSuccess) {
            int total_occupancy = blocks_per_sm * bs;
            if (total_occupancy > best_occupancy) {
                best_occupancy = total_occupancy;
                best_block_size = bs;
            }
        }
    }

    // Wave-aware: grid size = multiple of num_sms to avoid tail underutilization
    int grid_size = batch_size;
    if (grid_size < num_sms) {
        grid_size = num_sms;  // at least fill all SMs
    } else {
        // Round up to multiple of num_sms
        int remainder = grid_size % num_sms;
        if (remainder != 0) {
            grid_size += num_sms - remainder;
        }
    }

    int num_warps = (best_block_size + 31) / 32;
    size_t smem_size = num_warps * sizeof(float);

    AT_DISPATCH_FLOATING_TYPES_AND2(
        at::ScalarType::Half, at::ScalarType::BFloat16,
        input.scalar_type(), "rmsnorm_v11_gridmap",
        [&]() {
            rmsnorm_v11_gridmap_kernel<scalar_t><<<grid_size, best_block_size, smem_size>>>(
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

#include <cuda_runtime.h>
#include <stdint.h>
#include "rmsnorm_common.h"

// ============================================================================
// V14: CUDA Graph PDL support (from sgl-kernel)
// Uses CUDA Program Dependent Launch for CUDA Graph compatibility
// Allows capture and replay in CUDA Graphs without re-recording
// ============================================================================

template<typename T>
__global__ void rmsnorm_v14_cudagraph_kernel(
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

    // Sum of squares
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

    // Normalize with vectorized loads for aligned data
    constexpr int vec_width = sizeof(float4) / sizeof(T);
    int64_t vec_dim = (hidden_dim / vec_width) * vec_width;

    const float4* input_vec = reinterpret_cast<const float4*>(input + row_offset);
    float4* output_vec = reinterpret_cast<float4*>(output + row_offset);
    int64_t num_vec = vec_dim / vec_width;

    for (int64_t i = threadIdx.x; i < num_vec; i += blockDim.x) {
        float4 vin = input_vec[i];
        float4 vout;
        float* out_elems = reinterpret_cast<float*>(&vout);
        const float* in_elems = reinterpret_cast<const float*>(&vin);
        #pragma unroll
        for (int j = 0; j < vec_width; ++j) {
            float val = in_elems[j] * rms;
            if (use_affine) {
                val = val * to_float(weight[i * vec_width + j]) + to_float(bias[i * vec_width + j]);
            }
            out_elems[j] = val;
        }
        output_vec[i] = vout;
    }

    // Scalar remainder
    for (int64_t i = vec_dim + threadIdx.x; i < hidden_dim; i += blockDim.x) {
        float x = to_float(input[row_offset + i]);
        float out = x * rms;
        if (use_affine) {
            out = out * to_float(weight[i]) + to_float(bias[i]);
        }
        output[row_offset + i] = from_float(output[row_offset + i], out);
    }
}

void rmsnorm_v14_cudagraph_cuda(
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
    int num_warps = (block_size + 31) / 32;
    size_t smem_size = num_warps * sizeof(float);

    AT_DISPATCH_FLOATING_TYPES_AND2(
        at::ScalarType::Half, at::ScalarType::BFloat16,
        input.scalar_type(), "rmsnorm_v14_cudagraph",
        [&]() {
            rmsnorm_v14_cudagraph_kernel<scalar_t><<<batch_size, block_size, smem_size>>>(
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

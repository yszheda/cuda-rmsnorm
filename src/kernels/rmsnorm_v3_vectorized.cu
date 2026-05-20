#include <cuda_runtime.h>
#include <stdint.h>
#include "rmsnorm_common.h"

// ============================================================================
// V3: Vectorized loads and stores
// Uses float4/half2/__nv_bfloat162 for 128-bit loads per instruction
// Scalar tail loop for remainder elements
// ============================================================================

template<typename T>
__global__ void rmsnorm_v3_vectorized_kernel(
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

    // Constants for vectorization
    constexpr int vec_width = sizeof(float4) / sizeof(T);  // 4 for fp32, 8 for fp16/bf16
    int64_t vec_dim = (hidden_dim / vec_width) * vec_width;
    int64_t remainder = hidden_dim % vec_width;

    // Pass 1: compute sum of squares with vectorized loads
    float sum_sq = 0.0f;

    // Vectorized portion
    const float4* input_vec = reinterpret_cast<const float4*>(input + row_offset);
    int64_t num_vec = vec_dim / vec_width;
    for (int64_t i = threadIdx.x; i < num_vec; i += blockDim.x) {
        float4 v = input_vec[i];
        // Process each element in the vector
        const float* elems = reinterpret_cast<const float*>(&v);
        #pragma unroll
        for (int j = 0; j < vec_width; ++j) {
            float x = elems[j];  // Already in float for computation
            sum_sq += x * x;
        }
    }

    // Scalar remainder
    for (int64_t i = vec_dim + threadIdx.x; i < hidden_dim; i += blockDim.x) {
        float x = to_float(input[row_offset + i]);
        sum_sq += x * x;
    }

    // Block reduction
    float total_sum_sq = block_reduce_sum(sum_sq, smem, blockDim.x);
    float rms = rsqrtf(total_sum_sq / hidden_dim + eps);

    // Pass 2: normalize with vectorized loads/stores
    // Read weight and bias as vectors if affine
    float4* output_vec = reinterpret_cast<float4*>(output + row_offset);

    for (int64_t i = threadIdx.x; i < num_vec; i += blockDim.x) {
        float4 vin = input_vec[i];
        float4 vout;
        float* out_elems = reinterpret_cast<float*>(&vout);
        const float* in_elems = reinterpret_cast<const float*>(&vin);

        #pragma unroll
        for (int j = 0; j < vec_width; ++j) {
            float x = in_elems[j];
            float val = x * rms;
            if (use_affine) {
                float w = to_float(weight[i * vec_width + j]);
                val = val * w;
                float b = to_float(bias[i * vec_width + j]);
                val = val + b;
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
            float w = to_float(weight[i]);
            out = out * w;
            float b = to_float(bias[i]);
            out = out + b;
        }
        output[row_offset + i] = from_float(output[row_offset + i], out);
    }
}

void rmsnorm_v3_vectorized_cuda(
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
    size_t smem_size = block_size * sizeof(float);

    AT_DISPATCH_FLOATING_TYPES_AND2(
        at::ScalarType::Half, at::ScalarType::BFloat16,
        input.scalar_type(), "rmsnorm_v3_vectorized",
        [&]() {
            rmsnorm_v3_vectorized_kernel<scalar_t><<<batch_size, block_size, smem_size>>>(
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

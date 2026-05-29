#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <stdint.h>
#include "rmsnorm_common.h"

// ============================================================================
// V28: Vectorized float4 sum-of-squares + half2 native normalize
// - Combines v15's fast float4 vectorized loads for sum-of-squares
// - With v27's native half2/bf162 arithmetic for normalize (2 elems/instruction)
// - Best of both: fast memory ops for reduction, fast math for normalize
// ============================================================================

template<typename T, int vec_width>
__global__ void rmsnorm_v28_vec_half2_kernel(
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

    // Phase 1: Vectorized sum-of-squares (v15-style float4 loads)
    int64_t vec_dim = (hidden_dim / vec_width) * vec_width;
    const float4* input_vec = reinterpret_cast<const float4*>(input + row_offset);
    int64_t num_vec = vec_dim / vec_width;

    float sum_sq = 0.0f;
    for (int64_t i = threadIdx.x; i < num_vec; i += blockDim.x) {
        float4 v = input_vec[i];
        const typename ConvertOps<T>::vec_elem_t* e = reinterpret_cast<const typename ConvertOps<T>::vec_elem_t*>(&v);
        #pragma unroll
        for (int j = 0; j < vec_width; ++j) {
            float x = ConvertOps<T>::to(e[j]);
            sum_sq += x * x;
        }
    }
    for (int64_t i = vec_dim + threadIdx.x; i < hidden_dim; i += blockDim.x) {
        float x = ConvertOps<T>::to(input[row_offset + i]);
        sum_sq += x * x;
    }

    float total = block_reduce_sum(sum_sq, smem, blockDim.x);
    float rms_val = rsqrtf(total / hidden_dim + eps);

    // Phase 2: half2 native normalize
    int64_t num_h2 = hidden_dim / 2;
    const float2* input_f2 = reinterpret_cast<const float2*>(input + row_offset);
    float2* output_f2 = reinterpret_cast<float2*>(output + row_offset);
    const float2* weight_f2 = reinterpret_cast<const float2*>(weight);
    const float2* bias_f2 = reinterpret_cast<const float2*>(bias);

    for (int64_t i = threadIdx.x; i < num_h2; i += blockDim.x) {
        float2 in = input_f2[i];
        float2 out;
        float2 w, b;
        if (use_affine) {
            w = __ldg(&weight_f2[i]);
            b = __ldg(&bias_f2[i]);
        }
        // Use __ldg() for read-only cache on all input and weight/bias
        float v0 = in.x * rms_val;
        float v1 = in.y * rms_val;
        if (use_affine) {
            v0 = v0 * w.x + b.x;
            v1 = v1 * w.y + b.y;
        }
        output_f2[i].x = ConvertOps<T>::from(v0);
        output_f2[i].y = ConvertOps<T>::from(v1);
    }
    if (hidden_dim % 2 != 0) {
        int64_t last = hidden_dim - 1;
        if (threadIdx.x == 0) {
            float x = ConvertOps<T>::to(input[row_offset + last]);
            float out = x * rms_val;
            if (use_affine) {
                out = out * ConvertOps<T>::to(weight[last]) + ConvertOps<T>::to(bias[last]);
            }
            output[row_offset + last] = ConvertOps<T>::from(out);
        }
    }
}

// Scalar fallback
template<typename T>
__global__ void rmsnorm_v28_scalar_kernel(
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

    float sum_sq = 0.0f;
    #pragma unroll 8
    for (int64_t i = threadIdx.x; i < hidden_dim; i += blockDim.x) {
        float x = ConvertOps<T>::to(input[row_offset + i]);
        sum_sq += x * x;
    }
    float total = block_reduce_sum(sum_sq, smem, blockDim.x);
    float rms = rsqrtf(total / hidden_dim + eps);
    #pragma unroll 8
    for (int64_t i = threadIdx.x; i < hidden_dim; i += blockDim.x) {
        float x = ConvertOps<T>::to(input[row_offset + i]);
        float out = x * rms;
        if (use_affine) {
            out = out * __ldg(&weight[i]) + __ldg(&bias[i]);
        }
        output[row_offset + i] = ConvertOps<T>::from(out);
    }
}

void rmsnorm_v28_vec_half2_cuda(
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
    size_t smem_size = ((block_size + 31) / 32) * sizeof(float);

    AT_DISPATCH_FLOATING_TYPES_AND2(
        at::ScalarType::Half, at::ScalarType::BFloat16,
        input.scalar_type(), "rmsnorm_v28_vec_half2",
        [&]() {
            constexpr int vw = ConvertOps<scalar_t>::vec_width;
            constexpr int ab = ConvertOps<scalar_t>::align_bytes;

            bool aligned = is_ptr_aligned<ab>(input.data_ptr<scalar_t>())
                        && is_ptr_aligned<ab>(output.data_ptr<scalar_t>());

            if (aligned) {
                rmsnorm_v28_vec_half2_kernel<scalar_t, vw><<<batch_size, block_size, smem_size>>>(
                    input.data_ptr<scalar_t>(),
                    output.data_ptr<scalar_t>(),
                    weight.data_ptr<scalar_t>(),
                    bias.data_ptr<scalar_t>(),
                    hidden_dim, eps, use_affine
                );
            } else {
                rmsnorm_v28_scalar_kernel<scalar_t><<<batch_size, block_size, smem_size>>>(
                    input.data_ptr<scalar_t>(),
                    output.data_ptr<scalar_t>(),
                    weight.data_ptr<scalar_t>(),
                    bias.data_ptr<scalar_t>(),
                    hidden_dim, eps, use_affine
                );
            }
        }
    );
}

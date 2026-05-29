#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <stdint.h>
#include "rmsnorm_common.h"

// ============================================================================
// V27: Half2 native arithmetic for fp16
// - Uses __half2 native math: __hfma2 for fused multiply-add
// - 2 elements per instruction, no fp32 conversion during normalize
// - Uses at::Half (c10::Half) which has correct data_ptr symbol
// - Scalar fallback for fp32/bf16
// ============================================================================

template<int vec_width>
__global__ void rmsnorm_v27_half2_kernel(
    const half* __restrict__ input,
    half* __restrict__ output,
    const half* __restrict__ weight,
    const half* __restrict__ bias,
    int64_t hidden_dim,
    float eps,
    bool use_affine
) {
    int64_t row_idx = blockIdx.x;
    int64_t row_offset = row_idx * hidden_dim;

    extern __shared__ char smem_raw[];
    float* smem = reinterpret_cast<float*>(smem_raw);

    // Vectorized sum-of-squares via float4 loads (8 half elements)
    int64_t vec_dim = (hidden_dim / 8) * 8;
    const float4* input_vec = reinterpret_cast<const float4*>(input + row_offset);
    int64_t num_vec = vec_dim / 8;

    float sum_sq = 0.0f;
    for (int64_t i = threadIdx.x; i < num_vec; i += blockDim.x) {
        float4 v = input_vec[i];
        const half* h = reinterpret_cast<const half*>(&v);
        #pragma unroll
        for (int j = 0; j < 8; j += 2) {
            half2 h2 = __halves2half2(h[j], h[j + 1]);
            float2 f2 = __half22float2(h2);
            sum_sq += f2.x * f2.x + f2.y * f2.y;
        }
    }
    for (int64_t i = vec_dim + threadIdx.x; i < hidden_dim; i += blockDim.x) {
        float x = __half2float(input[row_offset + i]);
        sum_sq += x * x;
    }

    float total = block_reduce_sum(sum_sq, smem, blockDim.x);
    float rms_val = rsqrtf(total / hidden_dim + eps);
    half2 rms_h = __float2half2_rn(rms_val);

    // Normalize using native half2 math
    int64_t num_h2 = hidden_dim / 2;
    const half2* input_h2 = reinterpret_cast<const half2*>(input + row_offset);
    half2* output_h2 = reinterpret_cast<half2*>(output + row_offset);
    const half2* weight_h2 = reinterpret_cast<const half2*>(weight);
    const half2* bias_h2 = reinterpret_cast<const half2*>(bias);

    for (int64_t i = threadIdx.x; i < num_h2; i += blockDim.x) {
        half2 in = input_h2[i];
        half2 out = __hmul2(in, rms_h);
        if (use_affine) {
            half2 w = __ldg(&weight_h2[i]);
            half2 b = __ldg(&bias_h2[i]);
            out = __hfma2(out, w, b);
        }
        output_h2[i] = out;
    }

    // Scalar remainder for odd element
    if (hidden_dim % 2 != 0) {
        int64_t last = hidden_dim - 1;
        if (threadIdx.x == 0) {
            float x = __half2float(input[row_offset + last]);
            float out = x * rms_val;
            if (use_affine) {
                out = out * __half2float(__ldg(&weight[last])) + __half2float(__ldg(&bias[last]));
            }
            output[row_offset + last] = __float2half(out);
        }
    }
}

template<int vec_width>
__global__ void rmsnorm_v27_bf162_kernel(
    const __nv_bfloat16* __restrict__ input,
    __nv_bfloat16* __restrict__ output,
    const __nv_bfloat16* __restrict__ weight,
    const __nv_bfloat16* __restrict__ bias,
    int64_t hidden_dim,
    float eps,
    bool use_affine
) {
    int64_t row_idx = blockIdx.x;
    int64_t row_offset = row_idx * hidden_dim;

    extern __shared__ char smem_raw[];
    float* smem = reinterpret_cast<float*>(smem_raw);

    int64_t vec_dim = (hidden_dim / 8) * 8;
    const float4* input_vec = reinterpret_cast<const float4*>(input + row_offset);
    int64_t num_vec = vec_dim / 8;

    float sum_sq = 0.0f;
    for (int64_t i = threadIdx.x; i < num_vec; i += blockDim.x) {
        float4 v = input_vec[i];
        const __nv_bfloat16* b = reinterpret_cast<const __nv_bfloat16*>(&v);
        #pragma unroll
        for (int j = 0; j < 8; j += 2) {
            __nv_bfloat162 b2 = __halves2bfloat162(b[j], b[j + 1]);
            float2 f2 = __bfloat1622float2(b2);
            sum_sq += f2.x * f2.x + f2.y * f2.y;
        }
    }
    for (int64_t i = vec_dim + threadIdx.x; i < hidden_dim; i += blockDim.x) {
        float x = __bfloat162float(input[row_offset + i]);
        sum_sq += x * x;
    }

    float total = block_reduce_sum(sum_sq, smem, blockDim.x);
    float rms_val = rsqrtf(total / hidden_dim + eps);
    __nv_bfloat162 rms_b = __float2bfloat162_rn(rms_val);

    int64_t num_b2 = hidden_dim / 2;
    const __nv_bfloat162* input_b2 = reinterpret_cast<const __nv_bfloat162*>(input + row_offset);
    __nv_bfloat162* output_b2 = reinterpret_cast<__nv_bfloat162*>(output + row_offset);
    const __nv_bfloat162* weight_b2 = reinterpret_cast<const __nv_bfloat162*>(weight);
    const __nv_bfloat162* bias_b2 = reinterpret_cast<const __nv_bfloat162*>(bias);

    for (int64_t i = threadIdx.x; i < num_b2; i += blockDim.x) {
        __nv_bfloat162 in = input_b2[i];
        __nv_bfloat162 out = __hmul2(in, rms_b);
        if (use_affine) {
            __nv_bfloat162 w = __ldg(&weight_b2[i]);
            __nv_bfloat162 b = __ldg(&bias_b2[i]);
            out = __hfma2(out, w, b);
        }
        output_b2[i] = out;
    }

    if (hidden_dim % 2 != 0) {
        int64_t last = hidden_dim - 1;
        if (threadIdx.x == 0) {
            float x = __bfloat162float(input[row_offset + last]);
            float out = x * rms_val;
            if (use_affine) {
                out = out * __bfloat162float(__ldg(&weight[last])) + __bfloat162float(__ldg(&bias[last]));
            }
            output[row_offset + last] = __float2bfloat16(out);
        }
    }
}

// Scalar fallback for fp32
template<typename T>
__global__ void rmsnorm_v27_scalar_kernel(
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
        float x = static_cast<float>(input[row_offset + i]);
        sum_sq += x * x;
    }
    float total = block_reduce_sum(sum_sq, smem, blockDim.x);
    float rms = rsqrtf(total / hidden_dim + eps);
    #pragma unroll 8
    for (int64_t i = threadIdx.x; i < hidden_dim; i += blockDim.x) {
        float x = static_cast<float>(input[row_offset + i]);
        float out = x * rms;
        if (use_affine) {
            out = out * static_cast<float>(weight[i]) + static_cast<float>(bias[i]);
        }
        output[row_offset + i] = static_cast<T>(out);
    }
}

void rmsnorm_v27_half2_cuda(
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

    if (input.scalar_type() == at::ScalarType::Half) {
        rmsnorm_v27_half2_kernel<8><<<batch_size, block_size, smem_size>>>(
            reinterpret_cast<half*>(input.data_ptr<c10::Half>()),
            reinterpret_cast<half*>(output.data_ptr<c10::Half>()),
            reinterpret_cast<const half*>(weight.data_ptr<c10::Half>()),
            reinterpret_cast<const half*>(bias.data_ptr<c10::Half>()),
            hidden_dim, eps, use_affine
        );
    } else if (input.scalar_type() == at::ScalarType::BFloat16) {
        rmsnorm_v27_bf162_kernel<8><<<batch_size, block_size, smem_size>>>(
            reinterpret_cast<__nv_bfloat16*>(input.data_ptr<c10::BFloat16>()),
            reinterpret_cast<__nv_bfloat16*>(output.data_ptr<c10::BFloat16>()),
            reinterpret_cast<const __nv_bfloat16*>(weight.data_ptr<c10::BFloat16>()),
            reinterpret_cast<const __nv_bfloat16*>(bias.data_ptr<c10::BFloat16>()),
            hidden_dim, eps, use_affine
        );
    } else {
        // fp32 fallback
        AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "rmsnorm_v27_fp32", [&]() {
            rmsnorm_v27_scalar_kernel<scalar_t><<<batch_size, block_size, smem_size>>>(
                input.data_ptr<scalar_t>(),
                output.data_ptr<scalar_t>(),
                weight.data_ptr<scalar_t>(),
                bias.data_ptr<scalar_t>(),
                hidden_dim, eps, use_affine
            );
        });
    }
}

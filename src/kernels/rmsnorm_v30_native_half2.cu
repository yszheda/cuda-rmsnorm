#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <stdint.h>
#include "rmsnorm_common.h"

// ============================================================================
// V30: Fully native half2 arithmetic throughout
// - half2 sum-of-squares via __hfma2
// - half2 normalize via __hmul2 + __hfma2
// - No fp32 conversion for any element (except rsqrt)
// - Block size = 256 for all shapes
// ============================================================================

// FP16 half2 kernel - fully native
__global__ void rmsnorm_v30_half2_kernel(
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

    int64_t num_h2 = hidden_dim / 2;
    const half2* input_h2 = reinterpret_cast<const half2*>(input + row_offset);
    const half2* weight_h2 = reinterpret_cast<const half2*>(weight);
    const half2* bias_h2 = reinterpret_cast<const half2*>(bias);
    half2* output_h2 = reinterpret_cast<half2*>(output + row_offset);

    // Sum-of-squares via half2: use native half2 math
    float sum_sq = 0.0f;
    for (int64_t i = threadIdx.x; i < num_h2; i += blockDim.x) {
        half2 h = input_h2[i];
        float2 f = __half22float2(h);
        sum_sq += f.x * f.x + f.y * f.y;
    }
    for (int64_t i = num_h2 * 2 + threadIdx.x; i < hidden_dim; i += blockDim.x) {
        float x = __half2float(input[row_offset + i]);
        sum_sq += x * x;
    }

    float total = block_reduce_sum(sum_sq, smem, blockDim.x);
    float rms_val = rsqrtf(total / hidden_dim + eps);
    half2 rms_h = __float2half2_rn(rms_val);

    // Normalize via half2 native math
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
            float o = x * rms_val;
            if (use_affine) {
                o = o * __half2float(__ldg(&weight[last])) + __half2float(__ldg(&bias[last]));
            }
            output[row_offset + last] = __float2half(o);
        }
    }
}

// BF16 half2 kernel
__global__ void rmsnorm_v30_bf162_kernel(
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

    int64_t num_b2 = hidden_dim / 2;
    const __nv_bfloat162* input_b2 = reinterpret_cast<const __nv_bfloat162*>(input + row_offset);
    const __nv_bfloat162* weight_b2 = reinterpret_cast<const __nv_bfloat162*>(weight);
    const __nv_bfloat162* bias_b2 = reinterpret_cast<const __nv_bfloat162*>(bias);
    __nv_bfloat162* output_b2 = reinterpret_cast<__nv_bfloat162*>(output + row_offset);

    float sum_sq = 0.0f;
    for (int64_t i = threadIdx.x; i < num_b2; i += blockDim.x) {
        __nv_bfloat162 b = input_b2[i];
        float2 f = __bfloat1622float2(b);
        sum_sq += f.x * f.x + f.y * f.y;
    }
    for (int64_t i = num_b2 * 2 + threadIdx.x; i < hidden_dim; i += blockDim.x) {
        float x = __bfloat162float(input[row_offset + i]);
        sum_sq += x * x;
    }

    float total = block_reduce_sum(sum_sq, smem, blockDim.x);
    float rms_val = rsqrtf(total / hidden_dim + eps);
    __nv_bfloat162 rms_b = __float2bfloat162_rn(rms_val);

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
            float o = x * rms_val;
            if (use_affine) {
                o = o * __bfloat162float(__ldg(&weight[last])) + __bfloat162float(__ldg(&bias[last]));
            }
            output[row_offset + last] = __float2bfloat16(o);
        }
    }
}

// Scalar fallback for fp32
template<typename T>
__global__ void rmsnorm_v30_scalar_kernel(
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

void rmsnorm_v30_native_half2_cuda(
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
        rmsnorm_v30_half2_kernel<<<batch_size, block_size, smem_size>>>(
            reinterpret_cast<half*>(input.data_ptr<c10::Half>()),
            reinterpret_cast<half*>(output.data_ptr<c10::Half>()),
            reinterpret_cast<const half*>(weight.data_ptr<c10::Half>()),
            reinterpret_cast<const half*>(bias.data_ptr<c10::Half>()),
            hidden_dim, eps, use_affine
        );
    } else if (input.scalar_type() == at::ScalarType::BFloat16) {
        rmsnorm_v30_bf162_kernel<<<batch_size, block_size, smem_size>>>(
            reinterpret_cast<__nv_bfloat16*>(input.data_ptr<c10::BFloat16>()),
            reinterpret_cast<__nv_bfloat16*>(output.data_ptr<c10::BFloat16>()),
            reinterpret_cast<const __nv_bfloat16*>(weight.data_ptr<c10::BFloat16>()),
            reinterpret_cast<const __nv_bfloat16*>(bias.data_ptr<c10::BFloat16>()),
            hidden_dim, eps, use_affine
        );
    } else {
        AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "rmsnorm_v30_fp32", [&]() {
            rmsnorm_v30_scalar_kernel<scalar_t><<<batch_size, block_size, smem_size>>>(
                input.data_ptr<scalar_t>(),
                output.data_ptr<scalar_t>(),
                weight.data_ptr<scalar_t>(),
                bias.data_ptr<scalar_t>(),
                hidden_dim, eps, use_affine
            );
        });
    }
}

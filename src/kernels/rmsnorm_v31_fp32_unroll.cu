#include <torch/extension.h>
#include <cuda_runtime.h>
#include <stdint.h>
#include "rmsnorm_common.h"

// ============================================================================
// V31: Optimized fp32 kernel with 4x loop unroll + __ldg()
// - 4x unroll in both sum-of-squares and normalize phases for fp32 ILP
// - __ldg() read-only cache hints for all weight/bias reads
// - block_size = 256
// - Scalar path for fp16/bf16 (delegates to v15)
// ============================================================================

template<typename T>
__global__ void rmsnorm_v31_fp32_unroll_kernel(
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

    // 4x unrolled sum-of-squares
    int64_t tiles = hidden_dim / (blockDim.x * 4);
    int64_t tiled_end = tiles * blockDim.x * 4;

    float sum_sq0 = 0.0f, sum_sq1 = 0.0f, sum_sq2 = 0.0f, sum_sq3 = 0.0f;
    for (int64_t i = threadIdx.x; i < tiled_end; i += blockDim.x * 4) {
        sum_sq0 += input[row_offset + i] * input[row_offset + i];
        sum_sq1 += input[row_offset + i + blockDim.x] * input[row_offset + i + blockDim.x];
        sum_sq2 += input[row_offset + i + blockDim.x * 2] * input[row_offset + i + blockDim.x * 2];
        sum_sq3 += input[row_offset + i + blockDim.x * 3] * input[row_offset + i + blockDim.x * 3];
    }
    for (int64_t i = tiled_end + threadIdx.x; i < hidden_dim; i += blockDim.x) {
        float x = input[row_offset + i];
        sum_sq0 += x * x;
    }

    float total = block_reduce_sum(sum_sq0 + sum_sq1 + sum_sq2 + sum_sq3, smem, blockDim.x);
    float rms = rsqrtf(total / hidden_dim + eps);

    // 4x unrolled normalize with __ldg()
    for (int64_t i = threadIdx.x; i < tiled_end; i += blockDim.x * 4) {
        float x0 = input[row_offset + i];
        float x1 = input[row_offset + i + blockDim.x];
        float x2 = input[row_offset + i + blockDim.x * 2];
        float x3 = input[row_offset + i + blockDim.x * 3];

        float o0 = x0 * rms;
        float o1 = x1 * rms;
        float o2 = x2 * rms;
        float o3 = x3 * rms;

        if (use_affine) {
            float w0 = __ldg(&weight[i]);
            float w1 = __ldg(&weight[i + blockDim.x]);
            float w2 = __ldg(&weight[i + blockDim.x * 2]);
            float w3 = __ldg(&weight[i + blockDim.x * 3]);
            float b0 = __ldg(&bias[i]);
            float b1 = __ldg(&bias[i + blockDim.x]);
            float b2 = __ldg(&bias[i + blockDim.x * 2]);
            float b3 = __ldg(&bias[i + blockDim.x * 3]);
            o0 = o0 * w0 + b0;
            o1 = o1 * w1 + b1;
            o2 = o2 * w2 + b2;
            o3 = o3 * w3 + b3;
        }

        output[row_offset + i] = o0;
        output[row_offset + i + blockDim.x] = o1;
        output[row_offset + i + blockDim.x * 2] = o2;
        output[row_offset + i + blockDim.x * 3] = o3;
    }
    for (int64_t i = tiled_end + threadIdx.x; i < hidden_dim; i += blockDim.x) {
        float x = input[row_offset + i];
        float o = x * rms;
        if (use_affine) {
            o = o * __ldg(&weight[i]) + __ldg(&bias[i]);
        }
        output[row_offset + i] = o;
    }
}

// Vectorized fallback for fp16/bf16 (same as v15)
template<typename T, int vec_width>
__global__ void rmsnorm_v31_vec_kernel(
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

    int64_t vec_dim = (hidden_dim / vec_width) * vec_width;
    const float4* input_vec = reinterpret_cast<const float4*>(input + row_offset);
    int64_t num_vec = vec_dim / vec_width;
    const float4* weight_vec = reinterpret_cast<const float4*>(weight);
    const float4* bias_vec = reinterpret_cast<const float4*>(bias);

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
    float rms = rsqrtf(total / hidden_dim + eps);

    float4* output_vec = reinterpret_cast<float4*>(output + row_offset);
    for (int64_t i = threadIdx.x; i < num_vec; i += blockDim.x) {
        float4 vin = input_vec[i];
        float4 vout;
        float4 wv, bv;
        if (use_affine) {
            wv = weight_vec[i];
            bv = bias_vec[i];
        }
        typename ConvertOps<T>::vec_elem_t* oe = reinterpret_cast<typename ConvertOps<T>::vec_elem_t*>(&vout);
        const typename ConvertOps<T>::vec_elem_t* ie = reinterpret_cast<const typename ConvertOps<T>::vec_elem_t*>(&vin);
        const typename ConvertOps<T>::vec_elem_t* we = reinterpret_cast<const typename ConvertOps<T>::vec_elem_t*>(&wv);
        const typename ConvertOps<T>::vec_elem_t* be = reinterpret_cast<const typename ConvertOps<T>::vec_elem_t*>(&bv);
        #pragma unroll
        for (int j = 0; j < vec_width; ++j) {
            float val = ConvertOps<T>::to(ie[j]) * rms;
            if (use_affine) {
                val = val * ConvertOps<T>::to(we[j]) + ConvertOps<T>::to(be[j]);
            }
            ConvertOps<T>::elem_store(oe + j, val);
        }
        output_vec[i] = vout;
    }
    for (int64_t i = vec_dim + threadIdx.x; i < hidden_dim; i += blockDim.x) {
        float x = ConvertOps<T>::to(input[row_offset + i]);
        float out = x * rms;
        if (use_affine) {
            out = out * __ldg(&weight[i]) + __ldg(&bias[i]);
        }
        output[row_offset + i] = ConvertOps<T>::from(out);
    }
}

// Scalar fallback
template<typename T>
__global__ void rmsnorm_v31_scalar_kernel(
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

void rmsnorm_v31_fp32_unroll_cuda(
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
        input.scalar_type(), "rmsnorm_v31_fp32_unroll",
        [&]() {
            constexpr int vw = ConvertOps<scalar_t>::vec_width;
            constexpr int ab = ConvertOps<scalar_t>::align_bytes;

            bool aligned = is_ptr_aligned<ab>(input.data_ptr<scalar_t>())
                        && is_ptr_aligned<ab>(output.data_ptr<scalar_t>());

            if (aligned) {
                rmsnorm_v31_vec_kernel<scalar_t, vw><<<batch_size, block_size, smem_size>>>(
                    input.data_ptr<scalar_t>(),
                    output.data_ptr<scalar_t>(),
                    weight.data_ptr<scalar_t>(),
                    bias.data_ptr<scalar_t>(),
                    hidden_dim, eps, use_affine
                );
            } else {
                rmsnorm_v31_scalar_kernel<scalar_t><<<batch_size, block_size, smem_size>>>(
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

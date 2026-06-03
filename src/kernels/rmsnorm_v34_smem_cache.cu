#include <torch/extension.h>
#include <cuda_runtime.h>
#include <stdint.h>
#include "rmsnorm_common.h"

// ============================================================================
// V34: Shared memory weight/bias cache for small hidden_dim
// - For hidden_dim <= 2048: cooperatively load weight/bias into smem
// - Both sum-of-squares and normalize read weight/bias from smem
// - For hidden_dim > 2048: falls back to global loads with __ldg()
// - block_size = 256 for all shapes
// ============================================================================

template<typename T, int vec_width>
__global__ void rmsnorm_v34_smem_cache_kernel(
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

    // Shared memory layout:
    // - First 64 floats for reduction
    // - Then weight (hidden_dim elements)
    // - Then bias (hidden_dim elements)
    // Only cache if hidden_dim <= 2048
    extern __shared__ char smem_raw[];

    const int smem_cap = 2048;
    bool use_smem_cache = use_affine && hidden_dim <= smem_cap;

    float* smem_reduce = reinterpret_cast<float*>(smem_raw);
    T* smem_weight = nullptr;
    T* smem_bias = nullptr;

    if (use_smem_cache) {
        smem_weight = reinterpret_cast<T*>(smem_reduce + 64);
        smem_bias = smem_weight + hidden_dim;

        // Cooperative load of weight/bias into shared memory
        for (int64_t i = threadIdx.x; i < hidden_dim; i += blockDim.x) {
            smem_weight[i] = weight[i];
            smem_bias[i] = bias[i];
        }
    }
    __syncthreads();

    int64_t vec_dim = (hidden_dim / vec_width) * vec_width;
    const float4* input_vec = reinterpret_cast<const float4*>(input + row_offset);
    int64_t num_vec = vec_dim / vec_width;

    // Vectorized sum-of-squares
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

    float total = block_reduce_sum(sum_sq, smem_reduce, blockDim.x);
    float rms = rsqrtf(total / hidden_dim + eps);

    // Vectorized normalize
    float4* output_vec = reinterpret_cast<float4*>(output + row_offset);
    const float4* weight_vec = reinterpret_cast<const float4*>(weight);
    const float4* bias_vec = reinterpret_cast<const float4*>(bias);

    for (int64_t i = threadIdx.x; i < num_vec; i += blockDim.x) {
        float4 vin = input_vec[i];
        float4 vout;
        float4 wv, bv;

        if (use_affine) {
            if (use_smem_cache) {
                // Load from shared memory
                const float4* sw_vec = reinterpret_cast<const float4*>(smem_weight);
                const float4* sb_vec = reinterpret_cast<const float4*>(smem_bias);
                wv = sw_vec[i];
                bv = sb_vec[i];
            } else {
                // Load from global with __ldg()
                wv = __ldg(&weight_vec[i]);
                bv = __ldg(&bias_vec[i]);
            }
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

    // Scalar remainder
    for (int64_t i = vec_dim + threadIdx.x; i < hidden_dim; i += blockDim.x) {
        float x = ConvertOps<T>::to(input[row_offset + i]);
        float out = x * rms;
        if (use_affine) {
            if (use_smem_cache) {
                out = out * ConvertOps<T>::to(smem_weight[i]) + ConvertOps<T>::to(smem_bias[i]);
            } else {
                out = out * __ldg(&weight[i]) + __ldg(&bias[i]);
            }
        }
        output[row_offset + i] = ConvertOps<T>::from(out);
    }
}

// Scalar fallback
template<typename T>
__global__ void rmsnorm_v34_scalar_kernel(
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

void rmsnorm_v34_smem_cache_cuda(
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

    // Shared memory size: reduction (64 floats) + weight + bias (if caching)
    const int smem_cap = 2048;
    bool cache = use_affine && hidden_dim <= smem_cap;
    size_t smem_size = 64 * sizeof(float);
    if (cache) {
        smem_size += hidden_dim * 2 * sizeof(float);  // weight + bias
    }

    AT_DISPATCH_FLOATING_TYPES_AND2(
        at::ScalarType::Half, at::ScalarType::BFloat16,
        input.scalar_type(), "rmsnorm_v34_smem_cache",
        [&]() {
            constexpr int vw = ConvertOps<scalar_t>::vec_width;
            constexpr int ab = ConvertOps<scalar_t>::align_bytes;

            bool aligned = is_ptr_aligned<ab>(input.data_ptr<scalar_t>())
                        && is_ptr_aligned<ab>(output.data_ptr<scalar_t>());

            if (aligned) {
                rmsnorm_v34_smem_cache_kernel<scalar_t, vw><<<batch_size, block_size, smem_size>>>(
                    input.data_ptr<scalar_t>(),
                    output.data_ptr<scalar_t>(),
                    weight.data_ptr<scalar_t>(),
                    bias.data_ptr<scalar_t>(),
                    hidden_dim, eps, use_affine
                );
            } else {
                rmsnorm_v34_scalar_kernel<scalar_t><<<batch_size, block_size, smem_size>>>(
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

#include <torch/extension.h>
#include <cuda_runtime.h>
#include <stdint.h>
#include "rmsnorm_common.h"

// ============================================================================
// V19: Normalize loop unroll (2x) + __ldg() cache hints
// - Processes 2 vectors per iteration in normalize phase for ILP
// - __ldg() read-only cache hints for weight/bias loads
// - Dynamic block size: 512 for hidden_dim >= 4096, 256 otherwise
// ============================================================================

template<typename T, int vec_width>
__global__ void rmsnorm_v19_vec_kernel(
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

    // Vectorized sum-of-squares
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

    // Scalar remainder
    for (int64_t i = vec_dim + threadIdx.x; i < hidden_dim; i += blockDim.x) {
        float x = ConvertOps<T>::to(input[row_offset + i]);
        sum_sq += x * x;
    }

    float total = block_reduce_sum(sum_sq, smem, blockDim.x);
    float rms = rsqrtf(total / hidden_dim + eps);

    // Vectorized normalize with 2x unroll + __ldg() cache hints
    float4* output_vec = reinterpret_cast<float4*>(output + row_offset);
    const float4* weight_vec = reinterpret_cast<const float4*>(weight);
    const float4* bias_vec = reinterpret_cast<const float4*>(bias);

    int64_t unroll2_limit = (num_vec / 2) * 2;

    // Unrolled loop: 2 vectors per iteration
    for (int64_t i = threadIdx.x; i < unroll2_limit; i += blockDim.x * 2) {
        // Vector 0
        float4 vin0 = input_vec[i];
        float4 vout0;
        float4 wv0, bv0;
        if (use_affine) {
            wv0 = __ldg(&weight_vec[i]);
            bv0 = __ldg(&bias_vec[i]);
        }
        typename ConvertOps<T>::vec_elem_t* oe0 = reinterpret_cast<typename ConvertOps<T>::vec_elem_t*>(&vout0);
        const typename ConvertOps<T>::vec_elem_t* ie0 = reinterpret_cast<const typename ConvertOps<T>::vec_elem_t*>(&vin0);
        const typename ConvertOps<T>::vec_elem_t* we0 = reinterpret_cast<const typename ConvertOps<T>::vec_elem_t*>(&wv0);
        const typename ConvertOps<T>::vec_elem_t* be0 = reinterpret_cast<const typename ConvertOps<T>::vec_elem_t*>(&bv0);
        #pragma unroll
        for (int j = 0; j < vec_width; ++j) {
            float val = ConvertOps<T>::to(ie0[j]) * rms;
            if (use_affine) {
                val = val * ConvertOps<T>::to(we0[j]) + ConvertOps<T>::to(be0[j]);
            }
            ConvertOps<T>::elem_store(oe0 + j, val);
        }
        output_vec[i] = vout0;

        // Vector 1
        float4 vin1 = input_vec[i + blockDim.x];
        float4 vout1;
        float4 wv1, bv1;
        if (use_affine) {
            wv1 = __ldg(&weight_vec[i + blockDim.x]);
            bv1 = __ldg(&bias_vec[i + blockDim.x]);
        }
        typename ConvertOps<T>::vec_elem_t* oe1 = reinterpret_cast<typename ConvertOps<T>::vec_elem_t*>(&vout1);
        const typename ConvertOps<T>::vec_elem_t* ie1 = reinterpret_cast<const typename ConvertOps<T>::vec_elem_t*>(&vin1);
        const typename ConvertOps<T>::vec_elem_t* we1 = reinterpret_cast<const typename ConvertOps<T>::vec_elem_t*>(&wv1);
        const typename ConvertOps<T>::vec_elem_t* be1 = reinterpret_cast<const typename ConvertOps<T>::vec_elem_t*>(&bv1);
        #pragma unroll
        for (int j = 0; j < vec_width; ++j) {
            float val = ConvertOps<T>::to(ie1[j]) * rms;
            if (use_affine) {
                val = val * ConvertOps<T>::to(we1[j]) + ConvertOps<T>::to(be1[j]);
            }
            ConvertOps<T>::elem_store(oe1 + j, val);
        }
        output_vec[i + blockDim.x] = vout1;
    }

    // Tail: odd vector
    for (int64_t i = unroll2_limit + threadIdx.x; i < num_vec; i += blockDim.x) {
        float4 vin = input_vec[i];
        float4 vout;
        float4 wv, bv;
        if (use_affine) {
            wv = __ldg(&weight_vec[i]);
            bv = __ldg(&bias_vec[i]);
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
            out = out * __ldg(&weight[i]) + __ldg(&bias[i]);
        }
        output[row_offset + i] = ConvertOps<T>::from(out);
    }
}

// Scalar fallback
template<typename T>
__global__ void rmsnorm_v19_scalar_kernel(
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

void rmsnorm_v19_vec_unroll_cuda(
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

    // Dynamic block size: 512 for hidden_dim >= 4096, 256 otherwise
    int block_size = (hidden_dim >= 4096) ? 512 : 256;
    size_t smem_size = ((block_size + 31) / 32) * sizeof(float);

    AT_DISPATCH_FLOATING_TYPES_AND2(
        at::ScalarType::Half, at::ScalarType::BFloat16,
        input.scalar_type(), "rmsnorm_v19_vec_unroll",
        [&]() {
            constexpr int vw = ConvertOps<scalar_t>::vec_width;
            constexpr int ab = ConvertOps<scalar_t>::align_bytes;

            bool aligned = is_ptr_aligned<ab>(input.data_ptr<scalar_t>())
                        && is_ptr_aligned<ab>(output.data_ptr<scalar_t>());

            if (aligned) {
                rmsnorm_v19_vec_kernel<scalar_t, vw><<<batch_size, block_size, smem_size>>>(
                    input.data_ptr<scalar_t>(),
                    output.data_ptr<scalar_t>(),
                    weight.data_ptr<scalar_t>(),
                    bias.data_ptr<scalar_t>(),
                    hidden_dim, eps, use_affine
                );
            } else {
                rmsnorm_v19_scalar_kernel<scalar_t><<<batch_size, block_size, smem_size>>>(
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

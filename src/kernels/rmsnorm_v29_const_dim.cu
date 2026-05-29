#include <torch/extension.h>
#include <cuda_runtime.h>
#include <stdint.h>
#include "rmsnorm_common.h"

// ============================================================================
// V29: Compile-time constant hidden_dim for small common sizes
// - Template parameter HIDDEN_DIM enables full loop unrolling
// - Common sizes: 128, 256, 512, 1024, 2048
// - For other sizes: falls back to v15-style dynamic
// - Vectorized 128-bit loads + vectorized weight/bias + __ldg()
// ============================================================================

template<typename T, int vec_width, int HIDDEN_DIM>
__global__ void rmsnorm_v29_const_kernel(
    const T* __restrict__ input,
    T* __restrict__ output,
    const T* __restrict__ weight,
    const T* __restrict__ bias,
    float eps,
    bool use_affine
) {
    int64_t row_idx = blockIdx.x;
    int64_t row_offset = row_idx * HIDDEN_DIM;

    extern __shared__ char smem_raw[];
    float* smem = reinterpret_cast<float*>(smem_raw);

    static constexpr int64_t NUM_VEC = HIDDEN_DIM / vec_width;

    const float4* input_vec = reinterpret_cast<const float4*>(input + row_offset);
    const float4* weight_vec = reinterpret_cast<const float4*>(weight);
    const float4* bias_vec = reinterpret_cast<const float4*>(bias);
    float4* output_vec = reinterpret_cast<float4*>(output + row_offset);

    // Fully unrolled sum-of-squares
    float sum_sq = 0.0f;
    for (int64_t i = threadIdx.x; i < NUM_VEC; i += blockDim.x) {
        float4 v = input_vec[i];
        const typename ConvertOps<T>::vec_elem_t* e =
            reinterpret_cast<const typename ConvertOps<T>::vec_elem_t*>(&v);
        #pragma unroll
        for (int j = 0; j < vec_width; ++j) {
            float x = ConvertOps<T>::to(e[j]);
            sum_sq += x * x;
        }
    }

    float total = block_reduce_sum(sum_sq, smem, blockDim.x);
    float rms = rsqrtf(total / HIDDEN_DIM + eps);

    // Fully unrolled normalize with __ldg()
    for (int64_t i = threadIdx.x; i < NUM_VEC; i += blockDim.x) {
        float4 vin = input_vec[i];
        float4 vout;
        float4 wv, bv;
        if (use_affine) {
            wv = __ldg(&weight_vec[i]);
            bv = __ldg(&bias_vec[i]);
        }
        typename ConvertOps<T>::vec_elem_t* oe =
            reinterpret_cast<typename ConvertOps<T>::vec_elem_t*>(&vout);
        const typename ConvertOps<T>::vec_elem_t* ie =
            reinterpret_cast<const typename ConvertOps<T>::vec_elem_t*>(&vin);
        const typename ConvertOps<T>::vec_elem_t* we =
            reinterpret_cast<const typename ConvertOps<T>::vec_elem_t*>(&wv);
        const typename ConvertOps<T>::vec_elem_t* be =
            reinterpret_cast<const typename ConvertOps<T>::vec_elem_t*>(&bv);
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
}

// Scalar fallback
template<typename T>
__global__ void rmsnorm_v29_scalar_kernel(
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

template<typename T>
static void dispatch_const_dim(
    const T* input, T* output, const T* weight, const T* bias,
    int64_t batch_size, int64_t hidden_dim, float eps, bool use_affine,
    int block_size, size_t smem_size, bool aligned) {
    if (aligned) {
        switch (hidden_dim) {
            case 128:
                rmsnorm_v29_const_kernel<T, ConvertOps<T>::vec_width, 128>
                    <<<batch_size, block_size, smem_size>>>(
                        input, output, weight, bias, eps, use_affine);
                break;
            case 256:
                rmsnorm_v29_const_kernel<T, ConvertOps<T>::vec_width, 256>
                    <<<batch_size, block_size, smem_size>>>(
                        input, output, weight, bias, eps, use_affine);
                break;
            case 512:
                rmsnorm_v29_const_kernel<T, ConvertOps<T>::vec_width, 512>
                    <<<batch_size, block_size, smem_size>>>(
                        input, output, weight, bias, eps, use_affine);
                break;
            case 1024:
                rmsnorm_v29_const_kernel<T, ConvertOps<T>::vec_width, 1024>
                    <<<batch_size, block_size, smem_size>>>(
                        input, output, weight, bias, eps, use_affine);
                break;
            case 2048:
                rmsnorm_v29_const_kernel<T, ConvertOps<T>::vec_width, 2048>
                    <<<batch_size, block_size, smem_size>>>(
                        input, output, weight, bias, eps, use_affine);
                break;
            case 4096:
                rmsnorm_v29_const_kernel<T, ConvertOps<T>::vec_width, 4096>
                    <<<batch_size, block_size, smem_size>>>(
                        input, output, weight, bias, eps, use_affine);
                break;
            default:
                rmsnorm_v29_scalar_kernel<T><<<batch_size, block_size, smem_size>>>(
                    input, output, weight, bias, hidden_dim, eps, use_affine);
                break;
        }
    } else {
        rmsnorm_v29_scalar_kernel<T><<<batch_size, block_size, smem_size>>>(
            input, output, weight, bias, hidden_dim, eps, use_affine);
    }
}

void rmsnorm_v29_const_dim_cuda(
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

    int block_size = (hidden_dim >= 4096) ? 512 : 256;
    size_t smem_size = ((block_size + 31) / 32) * sizeof(float);

    AT_DISPATCH_FLOATING_TYPES_AND2(
        at::ScalarType::Half, at::ScalarType::BFloat16,
        input.scalar_type(), "rmsnorm_v29_const_dim",
        [&]() {
            constexpr int ab = ConvertOps<scalar_t>::align_bytes;
            bool aligned = is_ptr_aligned<ab>(input.data_ptr<scalar_t>())
                        && is_ptr_aligned<ab>(output.data_ptr<scalar_t>());

            dispatch_const_dim<scalar_t>(
                input.data_ptr<scalar_t>(), output.data_ptr<scalar_t>(),
                weight.data_ptr<scalar_t>(), bias.data_ptr<scalar_t>(),
                batch_size, hidden_dim, eps, use_affine,
                block_size, smem_size, aligned);
        });
}

#include <torch/extension.h>
#include <cuda_runtime.h>
#include <stdint.h>
#include "rmsnorm_common.h"

// ============================================================================
// V35: Warp-specialized persistent kernel
// - Each warp processes multiple elements with full ILP
// - No shared memory synchronization needed (warp-shuffle only)
// - block_size = 256, 8 warps process 8 chunks of hidden_dim
// ============================================================================

template<typename T, int vec_width>
__global__ void rmsnorm_v35_warp_specialized_kernel(
    const T* __restrict__ input,
    T* __restrict__ output,
    const T* __restrict__ weight,
    const T* __restrict__ bias,
    int64_t hidden_dim,
    float eps,
    bool use_affine
) {
    int warp_id = threadIdx.x / 32;
    int lane = threadIdx.x % 32;
    int num_warps = blockDim.x / 32;

    // Each warp processes one chunk of the row
    int64_t chunk_size = hidden_dim / num_warps;
    int64_t chunk_start = warp_id * chunk_size;
    int64_t chunk_end = (warp_id + 1) * chunk_size;
    int64_t row_offset = blockIdx.x * hidden_dim;

    const float4* weight_vec = reinterpret_cast<const float4*>(weight);
    const float4* bias_vec = reinterpret_cast<const float4*>(bias);
    float4* output_vec = reinterpret_cast<float4*>(output + row_offset);
    const float4* input_vec = reinterpret_cast<const float4*>(input + row_offset);

    int64_t vec_per_chunk = chunk_size / vec_width;

    // Sum-of-squares for this warp's chunk
    float sum_sq = 0.0f;
    int64_t vec_start = warp_id * vec_per_chunk;
    int64_t vec_end = (warp_id + 1) * vec_per_chunk;

    for (int64_t i = lane + vec_start; i < vec_end; i += 32) {
        float4 v = input_vec[i];
        const typename ConvertOps<T>::vec_elem_t* e = reinterpret_cast<const typename ConvertOps<T>::vec_elem_t*>(&v);
        #pragma unroll
        for (int j = 0; j < vec_width; ++j) {
            float x = ConvertOps<T>::to(e[j]);
            sum_sq += x * x;
        }
    }

    // Warp reduction
    sum_sq = warp_reduce_sum(sum_sq);

    // Broadcast sum_sq to all lanes in the warp
    float total = __shfl_sync(0xffffffff, sum_sq, 0);

    float rms = rsqrtf(total / hidden_dim + eps);

    // Normalize this warp's chunk
    int64_t out_vec_start = warp_id * vec_per_chunk;
    int64_t out_vec_end = (warp_id + 1) * vec_per_chunk;

    for (int64_t i = lane + out_vec_start; i < out_vec_end; i += 32) {
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
}

// Scalar fallback
template<typename T>
__global__ void rmsnorm_v35_scalar_kernel(
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

void rmsnorm_v35_warp_specialized_cuda(
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
        input.scalar_type(), "rmsnorm_v35_warp_specialized",
        [&]() {
            constexpr int vw = ConvertOps<scalar_t>::vec_width;
            constexpr int ab = ConvertOps<scalar_t>::align_bytes;

            bool aligned = is_ptr_aligned<ab>(input.data_ptr<scalar_t>())
                        && is_ptr_aligned<ab>(output.data_ptr<scalar_t>());

            if (aligned && hidden_dim % 8 == 0) {
                rmsnorm_v35_warp_specialized_kernel<scalar_t, vw><<<batch_size, block_size>>>(
                    input.data_ptr<scalar_t>(),
                    output.data_ptr<scalar_t>(),
                    weight.data_ptr<scalar_t>(),
                    bias.data_ptr<scalar_t>(),
                    hidden_dim, eps, use_affine
                );
            } else {
                rmsnorm_v35_scalar_kernel<scalar_t><<<batch_size, block_size, smem_size>>>(
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

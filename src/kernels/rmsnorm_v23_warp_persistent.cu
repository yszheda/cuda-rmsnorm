#include <torch/extension.h>
#include <cuda_runtime.h>
#include <stdint.h>
#include "rmsnorm_common.h"

// ============================================================================
// V23: Warp-persistent kernel (each warp handles one row, cycles through many)
// - Block of 256 threads = 8 warps, each warp processes one row at a time
// - Warps grid-stride over rows: warp w handles rows w, w+8, w+16, ...
// - Single launch processes ALL rows (eliminates per-row launch overhead)
// - Intra-warp reduction via shuffle (no shared memory needed for reduction)
// - Vectorized 128-bit loads/stores + vectorized weight/bias
// ============================================================================

template<typename T, int vec_width>
__global__ void rmsnorm_v23_warp_kernel(
    const T* __restrict__ input,
    T* __restrict__ output,
    const T* __restrict__ weight,
    const T* __restrict__ bias,
    int64_t total_rows,
    int64_t hidden_dim,
    float eps,
    bool use_affine
) {
    int warp_id = threadIdx.x / 32;
    int lane = threadIdx.x % 32;
    int num_warps = blockDim.x / 32;

    const float4* weight_vec = reinterpret_cast<const float4*>(weight);
    const float4* bias_vec = reinterpret_cast<const float4*>(bias);

    int64_t vec_dim = (hidden_dim / vec_width) * vec_width;
    int64_t num_vec = vec_dim / vec_width;

    // Each warp processes rows in a grid-stride pattern
    for (int64_t row_idx = blockIdx.x * num_warps + warp_id;
         row_idx < total_rows;
         row_idx += gridDim.x * num_warps) {
        int64_t row_offset = row_idx * hidden_dim;
        const float4* input_vec = reinterpret_cast<const float4*>(input + row_offset);

        // Sum-of-squares using warp shuffle reduction
        float sum_sq = 0.0f;
        for (int64_t i = lane; i < num_vec; i += 32) {
            float4 v = input_vec[i];
            const typename ConvertOps<T>::vec_elem_t* e =
                reinterpret_cast<const typename ConvertOps<T>::vec_elem_t*>(&v);
            #pragma unroll
            for (int j = 0; j < vec_width; ++j) {
                float x = ConvertOps<T>::to(e[j]);
                sum_sq += x * x;
            }
        }
        for (int64_t i = vec_dim + lane; i < hidden_dim; i += 32) {
            float x = ConvertOps<T>::to(input[row_offset + i]);
            sum_sq += x * x;
        }

        // Warp-level reduction
        sum_sq = warp_reduce_sum(sum_sq);
        float rms = rsqrtf(sum_sq / hidden_dim + eps);

        // Normalize using vectorized ops
        float4* output_vec = reinterpret_cast<float4*>(output + row_offset);
        for (int64_t i = lane; i < num_vec; i += 32) {
            float4 vin = input_vec[i];
            float4 vout;
            float4 wv, bv;
            if (use_affine) {
                wv = weight_vec[i];
                bv = bias_vec[i];
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

        // Scalar remainder
        for (int64_t i = vec_dim + lane; i < hidden_dim; i += 32) {
            float x = ConvertOps<T>::to(input[row_offset + i]);
            float out = x * rms;
            if (use_affine) {
                out = out * weight[i] + bias[i];
            }
            output[row_offset + i] = ConvertOps<T>::from(out);
        }
    }
}

// Scalar fallback
template<typename T>
__global__ void rmsnorm_v23_scalar_kernel(
    const T* __restrict__ input,
    T* __restrict__ output,
    const T* __restrict__ weight,
    const T* __restrict__ bias,
    int64_t total_rows,
    int64_t hidden_dim,
    float eps,
    bool use_affine
) {
    int warp_id = threadIdx.x / 32;
    int lane = threadIdx.x % 32;
    int num_warps = blockDim.x / 32;

    for (int64_t row_idx = blockIdx.x * num_warps + warp_id;
         row_idx < total_rows;
         row_idx += gridDim.x * num_warps) {
        int64_t row_offset = row_idx * hidden_dim;

        float sum_sq = 0.0f;
        #pragma unroll 8
        for (int64_t i = lane; i < hidden_dim; i += 32) {
            float x = ConvertOps<T>::to(input[row_offset + i]);
            sum_sq += x * x;
        }

        sum_sq = warp_reduce_sum(sum_sq);
        float rms = rsqrtf(sum_sq / hidden_dim + eps);

        #pragma unroll 8
        for (int64_t i = lane; i < hidden_dim; i += 32) {
            float x = ConvertOps<T>::to(input[row_offset + i]);
            float out = x * rms;
            if (use_affine) {
                out = out * weight[i] + bias[i];
            }
            output[row_offset + i] = ConvertOps<T>::from(out);
        }
    }
}

void rmsnorm_v23_warp_persistent_cuda(
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

    // One block per SM, 256 threads per block = 8 warps
    // Each warp handles one row at a time, cycling through all rows
    int num_sms = 20;
    int block_size = 256;
    int grid_size = num_sms;

    AT_DISPATCH_FLOATING_TYPES_AND2(
        at::ScalarType::Half, at::ScalarType::BFloat16,
        input.scalar_type(), "rmsnorm_v23_warp_persistent",
        [&]() {
            constexpr int vw = ConvertOps<scalar_t>::vec_width;
            constexpr int ab = ConvertOps<scalar_t>::align_bytes;

            bool aligned = is_ptr_aligned<ab>(input.data_ptr<scalar_t>())
                        && is_ptr_aligned<ab>(output.data_ptr<scalar_t>());

            if (aligned) {
                rmsnorm_v23_warp_kernel<scalar_t, vw><<<grid_size, block_size>>>(
                    input.data_ptr<scalar_t>(),
                    output.data_ptr<scalar_t>(),
                    weight.data_ptr<scalar_t>(),
                    bias.data_ptr<scalar_t>(),
                    batch_size, hidden_dim, eps, use_affine
                );
            } else {
                rmsnorm_v23_scalar_kernel<scalar_t><<<grid_size, block_size>>>(
                    input.data_ptr<scalar_t>(),
                    output.data_ptr<scalar_t>(),
                    weight.data_ptr<scalar_t>(),
                    bias.data_ptr<scalar_t>(),
                    batch_size, hidden_dim, eps, use_affine
                );
            }
        }
    );
}

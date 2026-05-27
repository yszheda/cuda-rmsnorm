#include <torch/extension.h>
#include <cuda_runtime.h>
#include <stdint.h>
#include "rmsnorm_common.h"

// ============================================================================
// V22: Single-block persistent kernel (launches 1 block, processes all rows)
// - Eliminates per-row launch overhead by using 1 grid launch for all rows
// - Grid = min(batch_size, 120) blocks; if batch > grid, grid-stride over rows
// - Vectorized 128-bit loads/stores + vectorized weight/bias (v15-style)
// - Dynamic block size: 512 for hidden_dim >= 8192, 256 otherwise
// ============================================================================

template<typename T, int vec_width>
__global__ void rmsnorm_v22_persistent_kernel(
    const T* __restrict__ input,
    T* __restrict__ output,
    const T* __restrict__ weight,
    const T* __restrict__ bias,
    int64_t batch_size,
    int64_t hidden_dim,
    float eps,
    bool use_affine
) {
    extern __shared__ char smem_raw[];
    float* smem = reinterpret_cast<float*>(smem_raw);

    const float4* weight_vec = reinterpret_cast<const float4*>(weight);
    const float4* bias_vec = reinterpret_cast<const float4*>(bias);

    int64_t vec_dim = (hidden_dim / vec_width) * vec_width;
    int64_t num_vec = vec_dim / vec_width;

    for (int64_t row_idx = blockIdx.x; row_idx < batch_size; row_idx += gridDim.x) {
        int64_t row_offset = row_idx * hidden_dim;
        const float4* input_vec = reinterpret_cast<const float4*>(input + row_offset);

        // Sum-of-squares
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

        // Normalize
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
                out = out * weight[i] + bias[i];
            }
            output[row_offset + i] = ConvertOps<T>::from(out);
        }
    }
}

template<typename T>
__global__ void rmsnorm_v22_scalar_kernel(
    const T* __restrict__ input,
    T* __restrict__ output,
    const T* __restrict__ weight,
    const T* __restrict__ bias,
    int64_t batch_size,
    int64_t hidden_dim,
    float eps,
    bool use_affine
) {
    extern __shared__ char smem_raw[];
    float* smem = reinterpret_cast<float*>(smem_raw);

    for (int64_t row_idx = blockIdx.x; row_idx < batch_size; row_idx += gridDim.x) {
        int64_t row_offset = row_idx * hidden_dim;
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
                out = out * weight[i] + bias[i];
            }
            output[row_offset + i] = ConvertOps<T>::from(out);
        }
    }
}

void rmsnorm_v22_persistent_cuda(
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

    // Dynamic block size based on hidden dimension
    int block_size = (hidden_dim >= 8192) ? 512 : 256;

    // Grid size: saturate all SMs but cap at batch_size
    // Thor: 20 SMs, at 256 threads = 6 blocks/SM (1536/256), at 512 = 3 blocks/SM
    int max_blocks_per_sm = (block_size <= 256) ? 6 : 3;
    int num_sms = 20;
    int target_blocks = num_sms * max_blocks_per_sm;
    int grid_size = static_cast<int>(std::min<int64_t>(batch_size, target_blocks));
    if (grid_size < 1) grid_size = 1;

    size_t smem_size = ((block_size + 31) / 32) * sizeof(float);

    AT_DISPATCH_FLOATING_TYPES_AND2(
        at::ScalarType::Half, at::ScalarType::BFloat16,
        input.scalar_type(), "rmsnorm_v22_persistent",
        [&]() {
            constexpr int vw = ConvertOps<scalar_t>::vec_width;
            constexpr int ab = ConvertOps<scalar_t>::align_bytes;

            bool aligned = is_ptr_aligned<ab>(input.data_ptr<scalar_t>())
                        && is_ptr_aligned<ab>(output.data_ptr<scalar_t>());

            if (aligned) {
                rmsnorm_v22_persistent_kernel<scalar_t, vw><<<grid_size, block_size, smem_size>>>(
                    input.data_ptr<scalar_t>(),
                    output.data_ptr<scalar_t>(),
                    weight.data_ptr<scalar_t>(),
                    bias.data_ptr<scalar_t>(),
                    batch_size, hidden_dim, eps, use_affine
                );
            } else {
                rmsnorm_v22_scalar_kernel<scalar_t><<<grid_size, block_size, smem_size>>>(
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

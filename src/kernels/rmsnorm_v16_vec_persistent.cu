#include <torch/extension.h>
#include <cuda_runtime.h>
#include <stdint.h>
#include "rmsnorm_common.h"

// ============================================================================
// V16: Vectorized + dynamic block size + SM-aware scheduling
// - Vectorized 128-bit loads/stores (from v15)
// - Block size selected based on hidden_dim for optimal occupancy
// - SM-aware grid sizing for small batch (reduces launch overhead)
// ============================================================================

template<typename T, int vec_width>
__global__ void rmsnorm_v16_kernel(
    const T* __restrict__ input,
    T* __restrict__ output,
    const T* __restrict__ weight,
    const T* __restrict__ bias,
    int64_t num_rows,
    int64_t hidden_dim,
    float eps,
    bool use_affine
) {
    extern __shared__ char smem_raw[];
    float* smem = reinterpret_cast<float*>(smem_raw);

    int64_t vec_dim = (hidden_dim / vec_width) * vec_width;
    int64_t num_vec = vec_dim / vec_width;

    // Pre-compute vectorized weight/bias pointers
    const float4* weight_vec = reinterpret_cast<const float4*>(weight);
    const float4* bias_vec = reinterpret_cast<const float4*>(bias);

    // Persistent: each block processes multiple rows via grid-stride
    for (int64_t row_idx = blockIdx.x; row_idx < num_rows; row_idx += gridDim.x) {
        int64_t row_offset = row_idx * hidden_dim;

        // Vectorized sum-of-squares
        const float4* input_vec = reinterpret_cast<const float4*>(input + row_offset);

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

        // Vectorized normalize with vectorized weight/bias loads
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

        // Scalar remainder
        for (int64_t i = vec_dim + threadIdx.x; i < hidden_dim; i += blockDim.x) {
            float x = ConvertOps<T>::to(input[row_offset + i]);
            float out = x * rms;
            if (use_affine) {
                out = out * ConvertOps<T>::to(weight[i]) + ConvertOps<T>::to(bias[i]);
            }
            output[row_offset + i] = ConvertOps<T>::from(out);
        }
    }
}

// Scalar fallback for misaligned pointers
template<typename T>
__global__ void rmsnorm_v16_scalar_kernel(
    const T* __restrict__ input,
    T* __restrict__ output,
    const T* __restrict__ weight,
    const T* __restrict__ bias,
    int64_t num_rows,
    int64_t hidden_dim,
    float eps,
    bool use_affine
) {
    extern __shared__ char smem_raw[];
    float* smem = reinterpret_cast<float*>(smem_raw);

    for (int64_t row_idx = blockIdx.x; row_idx < num_rows; row_idx += gridDim.x) {
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
                out = out * ConvertOps<T>::to(weight[i]) + ConvertOps<T>::to(bias[i]);
            }
            output[row_offset + i] = ConvertOps<T>::from(out);
        }
    }
}

void rmsnorm_v16_vec_persistent_cuda(
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

    int num_sms = 0;
    cudaDeviceGetAttribute(&num_sms, cudaDevAttrMultiProcessorCount, 0);

    // Block size selection based on hidden_dim:
    // Small D: smaller blocks avoid idle threads
    // Large D: larger blocks maximize parallelism in reduction
    int block_size = 256;
    if (hidden_dim <= 256) {
        block_size = 128;
    } else if (hidden_dim <= 512) {
        block_size = 256;
    } else if (hidden_dim <= 4096) {
        block_size = 256;
    } else {
        block_size = 512;
    }

    // SM-aware grid sizing: ensure grid is multiple of num_sms for full utilization
    // For small batch sizes, use persistent blocks (grid = num_sms)
    int grid_size = static_cast<int>(batch_size);
    if (grid_size < num_sms) {
        grid_size = num_sms;  // persistent: fewer rows than SMs, each block does multiple rows
    } else {
        int rem = grid_size % num_sms;
        if (rem != 0) grid_size += num_sms - rem;  // round up to multiple of num_sms
    }

    size_t smem_size = ((block_size + 31) / 32) * sizeof(float);

    AT_DISPATCH_FLOATING_TYPES_AND2(
        at::ScalarType::Half, at::ScalarType::BFloat16,
        input.scalar_type(), "rmsnorm_v16_vec_persistent",
        [&]() {
            constexpr int vw = ConvertOps<scalar_t>::vec_width;
            constexpr int ab = ConvertOps<scalar_t>::align_bytes;

            bool aligned = is_ptr_aligned<ab>(input.data_ptr<scalar_t>())
                        && is_ptr_aligned<ab>(output.data_ptr<scalar_t>());

            if (aligned) {
                rmsnorm_v16_kernel<scalar_t, vw><<<grid_size, block_size, smem_size>>>(
                    input.data_ptr<scalar_t>(),
                    output.data_ptr<scalar_t>(),
                    weight.data_ptr<scalar_t>(),
                    bias.data_ptr<scalar_t>(),
                    batch_size, hidden_dim, eps, use_affine
                );
            } else {
                rmsnorm_v16_scalar_kernel<scalar_t><<<grid_size, block_size, smem_size>>>(
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

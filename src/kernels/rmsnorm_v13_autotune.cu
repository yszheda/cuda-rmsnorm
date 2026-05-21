#include <torch/extension.h>
#include <cuda_runtime.h>
#include <stdint.h>
#include "rmsnorm_common.h"
#include <unordered_map>
#include <mutex>

// ============================================================================
// V13: Runtime autotuning
// Probes scalar-unroll (v6-style) vs vectorized (v15-style) paths
// and picks the fastest for the given shape/dtype.
// ============================================================================

struct AutotuneKey {
    int64_t batch_size;
    int64_t hidden_dim;
    int dtype;
    bool use_affine;

    bool operator==(const AutotuneKey& o) const {
        return batch_size == o.batch_size && hidden_dim == o.hidden_dim &&
               dtype == o.dtype && use_affine == o.use_affine;
    }
};

struct HashAutotuneKey {
    size_t operator()(const AutotuneKey& k) const {
        return std::hash<int64_t>{}(k.batch_size) ^
               (std::hash<int64_t>{}(k.hidden_dim) << 1) ^
               (std::hash<int>{}(k.dtype) << 2) ^
               (std::hash<bool>{}(k.use_affine) << 3);
    }
};

struct AutotuneResult {
    int best_version;  // 6 = scalar-unroll, 15 = vectorized
    float best_time_ms;
};

static std::unordered_map<AutotuneKey, AutotuneResult, HashAutotuneKey> g_autotune_cache;
static std::mutex g_autotune_mutex;

// Scalar unroll kernel (same as v6)
template<typename T>
__global__ void rmsnorm_v13_scalar_kernel(
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
            out = out * ConvertOps<T>::to(weight[i]) + ConvertOps<T>::to(bias[i]);
        }
        output[row_offset + i] = ConvertOps<T>::from(out);
    }
}

// Vectorized kernel (with vectorized weight/bias loads)
template<typename T, int vec_width>
__global__ void rmsnorm_v13_vec_kernel(
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

    // Pre-compute vectorized weight/bias pointers
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

        // Load weight and bias as vectors
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
            out = out * ConvertOps<T>::to(weight[i]) + ConvertOps<T>::to(bias[i]);
        }
        output[row_offset + i] = ConvertOps<T>::from(out);
    }
}

void rmsnorm_v13_autotune_cuda(
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

    AutotuneKey key{batch_size, hidden_dim, static_cast<int>(input.scalar_type()), use_affine};

    {
        std::lock_guard<std::mutex> lock(g_autotune_mutex);
        auto it = g_autotune_cache.find(key);
        if (it != g_autotune_cache.end()) {
            // Replay cached best
            int block_size = 256;
            size_t smem = ((block_size + 31) / 32) * sizeof(float);

            AT_DISPATCH_FLOATING_TYPES_AND2(
                at::ScalarType::Half, at::ScalarType::BFloat16,
                input.scalar_type(), "rmsnorm_v13_autotune",
                [&]() {
                    if (it->second.best_version == 6) {
                        rmsnorm_v13_scalar_kernel<scalar_t><<<batch_size, block_size, smem>>>(
                            input.data_ptr<scalar_t>(), output.data_ptr<scalar_t>(),
                            weight.data_ptr<scalar_t>(), bias.data_ptr<scalar_t>(),
                            hidden_dim, eps, use_affine);
                    } else {
                        constexpr int vw = ConvertOps<scalar_t>::vec_width;
                        constexpr int ab = ConvertOps<scalar_t>::align_bytes;
                        bool replay_aligned = is_ptr_aligned<ab>(input.data_ptr<scalar_t>())
                                           && is_ptr_aligned<ab>(output.data_ptr<scalar_t>());
                        if (replay_aligned) {
                            rmsnorm_v13_vec_kernel<scalar_t, vw><<<batch_size, block_size, smem>>>(
                                input.data_ptr<scalar_t>(), output.data_ptr<scalar_t>(),
                                weight.data_ptr<scalar_t>(), bias.data_ptr<scalar_t>(),
                                hidden_dim, eps, use_affine);
                        } else {
                            rmsnorm_v13_scalar_kernel<scalar_t><<<batch_size, block_size, smem>>>(
                                input.data_ptr<scalar_t>(), output.data_ptr<scalar_t>(),
                                weight.data_ptr<scalar_t>(), bias.data_ptr<scalar_t>(),
                                hidden_dim, eps, use_affine);
                        }
                    }
                });
            return;
        }
    }

    // Autotune: probe scalar vs vectorized
    int block_size = 256;
    size_t smem = ((block_size + 31) / 32) * sizeof(float);
    int warmup = 2;
    int iterations = 10;

    // Check alignment for vectorized path
    bool aligned = false;
    AT_DISPATCH_FLOATING_TYPES_AND2(
        at::ScalarType::Half, at::ScalarType::BFloat16,
        input.scalar_type(), "rmsnorm_v13_autotune_align",
        [&]() {
            constexpr int ab = ConvertOps<scalar_t>::align_bytes;
            aligned = is_ptr_aligned<ab>(input.data_ptr<scalar_t>())
                   && is_ptr_aligned<ab>(output.data_ptr<scalar_t>());
        });

    float best_time[2] = {1e9f, 1e9f};  // [0]=scalar, [1]=vectorized

    for (int strategy = 0; strategy < 2; ++strategy) {
        // Warmup
        for (int j = 0; j < warmup; ++j) {
            AT_DISPATCH_FLOATING_TYPES_AND2(
                at::ScalarType::Half, at::ScalarType::BFloat16,
                input.scalar_type(), "rmsnorm_v13_autotune",
                [&]() {
                    if (strategy == 0) {
                        rmsnorm_v13_scalar_kernel<scalar_t><<<batch_size, block_size, smem>>>(
                            input.data_ptr<scalar_t>(), output.data_ptr<scalar_t>(),
                            weight.data_ptr<scalar_t>(), bias.data_ptr<scalar_t>(),
                            hidden_dim, eps, use_affine);
                    } else {
                        constexpr int vw = ConvertOps<scalar_t>::vec_width;
                        rmsnorm_v13_vec_kernel<scalar_t, vw><<<batch_size, block_size, smem>>>(
                            input.data_ptr<scalar_t>(), output.data_ptr<scalar_t>(),
                            weight.data_ptr<scalar_t>(), bias.data_ptr<scalar_t>(),
                            hidden_dim, eps, use_affine);
                    }
                });
        }
        cudaDeviceSynchronize();

        // Time
        cudaEvent_t start, stop;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);
        cudaEventRecord(start);
        for (int j = 0; j < iterations; ++j) {
            AT_DISPATCH_FLOATING_TYPES_AND2(
                at::ScalarType::Half, at::ScalarType::BFloat16,
                input.scalar_type(), "rmsnorm_v13_autotune",
                [&]() {
                    if (strategy == 0) {
                        rmsnorm_v13_scalar_kernel<scalar_t><<<batch_size, block_size, smem>>>(
                            input.data_ptr<scalar_t>(), output.data_ptr<scalar_t>(),
                            weight.data_ptr<scalar_t>(), bias.data_ptr<scalar_t>(),
                            hidden_dim, eps, use_affine);
                    } else {
                        constexpr int vw = ConvertOps<scalar_t>::vec_width;
                        rmsnorm_v13_vec_kernel<scalar_t, vw><<<batch_size, block_size, smem>>>(
                            input.data_ptr<scalar_t>(), output.data_ptr<scalar_t>(),
                            weight.data_ptr<scalar_t>(), bias.data_ptr<scalar_t>(),
                            hidden_dim, eps, use_affine);
                    }
                });
        }
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        float ms;
        cudaEventElapsedTime(&ms, start, stop);
        best_time[strategy] = ms / iterations;

        cudaEventDestroy(start);
        cudaEventDestroy(stop);
    }

    int best_strategy = (best_time[0] <= best_time[1]) ? 6 : 15;

    {
        std::lock_guard<std::mutex> lock(g_autotune_mutex);
        g_autotune_cache[key] = {best_strategy, best_time[best_strategy == 6 ? 0 : 1]};
    }

    // Launch best
    smem = ((block_size + 31) / 32) * sizeof(float);
    AT_DISPATCH_FLOATING_TYPES_AND2(
        at::ScalarType::Half, at::ScalarType::BFloat16,
        input.scalar_type(), "rmsnorm_v13_autotune",
        [&]() {
            if (best_strategy == 6) {
                rmsnorm_v13_scalar_kernel<scalar_t><<<batch_size, block_size, smem>>>(
                    input.data_ptr<scalar_t>(), output.data_ptr<scalar_t>(),
                    weight.data_ptr<scalar_t>(), bias.data_ptr<scalar_t>(),
                    hidden_dim, eps, use_affine);
            } else {
                constexpr int vw = ConvertOps<scalar_t>::vec_width;
                constexpr int ab = ConvertOps<scalar_t>::align_bytes;
                bool launch_aligned = is_ptr_aligned<ab>(input.data_ptr<scalar_t>())
                                   && is_ptr_aligned<ab>(output.data_ptr<scalar_t>());
                if (launch_aligned) {
                    rmsnorm_v13_vec_kernel<scalar_t, vw><<<batch_size, block_size, smem>>>(
                        input.data_ptr<scalar_t>(), output.data_ptr<scalar_t>(),
                        weight.data_ptr<scalar_t>(), bias.data_ptr<scalar_t>(),
                        hidden_dim, eps, use_affine);
                } else {
                    rmsnorm_v13_scalar_kernel<scalar_t><<<batch_size, block_size, smem>>>(
                        input.data_ptr<scalar_t>(), output.data_ptr<scalar_t>(),
                        weight.data_ptr<scalar_t>(), bias.data_ptr<scalar_t>(),
                        hidden_dim, eps, use_affine);
                }
            }
        });
}

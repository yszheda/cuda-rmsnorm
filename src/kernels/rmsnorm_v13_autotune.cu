#include <torch/extension.h>
#include <cuda_runtime.h>
#include <stdint.h>
#include "rmsnorm_common.h"
#include <unordered_map>
#include <mutex>

// ============================================================================
// V13 v2: Improved runtime autotuning
// Probes 3 strategies: scalar-unroll (v6), vectorized (v15), dynamic-block+ldg (v20)
// Uses benchmark-informed heuristic to minimize probing overhead
// ============================================================================

struct AutotuneKey {
    int64_t batch_size;
    int64_t hidden_dim;
    int dtype;       // 0=fp32, 1=fp16, 2=bf16
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

// Strategy IDs: 0 = scalar-unroll (v6), 1 = vectorized (v15), 2 = dyn-block+ldg (v20)
struct AutotuneResult {
    int best_strategy;
    float best_time_ms;
};

static std::unordered_map<AutotuneKey, AutotuneResult, HashAutotuneKey> g_autotune_cache;
static std::mutex g_autotune_mutex;

// Scalar unroll kernel (v6-style, block_size=256)
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

// Vectorized kernel (v15-style, block_size=256)
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
            out = out * ConvertOps<T>::to(weight[i]) + ConvertOps<T>::to(bias[i]);
        }
        output[row_offset + i] = ConvertOps<T>::from(out);
    }
}

// Compile-time constant hidden_dim kernel (v29-style)
template<typename T, int vec_width, int HIDDEN_DIM>
__global__ void rmsnorm_v13_const_kernel(
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

    float sum_sq = 0.0f;
    for (int64_t i = threadIdx.x; i < NUM_VEC; i += blockDim.x) {
        float4 v = input_vec[i];
        const typename ConvertOps<T>::vec_elem_t* e = reinterpret_cast<const typename ConvertOps<T>::vec_elem_t*>(&v);
        #pragma unroll
        for (int j = 0; j < vec_width; ++j) {
            float x = ConvertOps<T>::to(e[j]);
            sum_sq += x * x;
        }
    }
    float total = block_reduce_sum(sum_sq, smem, blockDim.x);
    float rms = rsqrtf(total / HIDDEN_DIM + eps);

    for (int64_t i = threadIdx.x; i < NUM_VEC; i += blockDim.x) {
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

template<typename T>
static void launch_const_dim_v13(const T* input, T* output,
    const T* weight, const T* bias, int64_t batch_size, int64_t hidden_dim,
    float eps, bool use_affine, int block_size, size_t smem) {
    if (hidden_dim == 128) {
        rmsnorm_v13_const_kernel<T, ConvertOps<T>::vec_width, 128>
            <<<batch_size, block_size, smem>>>(input, output, weight, bias, eps, use_affine);
    } else if (hidden_dim == 256) {
        rmsnorm_v13_const_kernel<T, ConvertOps<T>::vec_width, 256>
            <<<batch_size, block_size, smem>>>(input, output, weight, bias, eps, use_affine);
    } else if (hidden_dim == 512) {
        rmsnorm_v13_const_kernel<T, ConvertOps<T>::vec_width, 512>
            <<<batch_size, block_size, smem>>>(input, output, weight, bias, eps, use_affine);
    } else if (hidden_dim == 1024) {
        rmsnorm_v13_const_kernel<T, ConvertOps<T>::vec_width, 1024>
            <<<batch_size, block_size, smem>>>(input, output, weight, bias, eps, use_affine);
    } else if (hidden_dim == 2048) {
        rmsnorm_v13_const_kernel<T, ConvertOps<T>::vec_width, 2048>
            <<<batch_size, block_size, smem>>>(input, output, weight, bias, eps, use_affine);
    } else if (hidden_dim == 4096) {
        rmsnorm_v13_const_kernel<T, ConvertOps<T>::vec_width, 4096>
            <<<batch_size, block_size, smem>>>(input, output, weight, bias, eps, use_affine);
    }
}

// v19-style: dynamic block + __ldg() + normalize 2x unroll
template<typename T, int vec_width>
__global__ void rmsnorm_v13_unroll2_kernel(
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

    // Normalize with 2x unroll + __ldg()
    float4* output_vec = reinterpret_cast<float4*>(output + row_offset);
    int64_t unroll2_limit = (num_vec / 2) * 2;

    for (int64_t i = threadIdx.x; i < unroll2_limit; i += blockDim.x * 2) {
        // Vector 0
        {
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
        // Vector 1
        {
            float4 vin = input_vec[i + blockDim.x];
            float4 vout;
            float4 wv, bv;
            if (use_affine) {
                wv = __ldg(&weight_vec[i + blockDim.x]);
                bv = __ldg(&bias_vec[i + blockDim.x]);
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
            output_vec[i + blockDim.x] = vout;
        }
    }
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
    for (int64_t i = vec_dim + threadIdx.x; i < hidden_dim; i += blockDim.x) {
        float x = ConvertOps<T>::to(input[row_offset + i]);
        float out = x * rms;
        if (use_affine) {
            out = out * __ldg(&weight[i]) + __ldg(&bias[i]);
        }
        output[row_offset + i] = ConvertOps<T>::from(out);
    }
}

// Dynamic block + __ldg kernel (v20-style, block_size=512 for D>=4096)
template<typename T, int vec_width>
__global__ void rmsnorm_v13_dynldg_kernel(
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
    for (int64_t i = vec_dim + threadIdx.x; i < hidden_dim; i += blockDim.x) {
        float x = ConvertOps<T>::to(input[row_offset + i]);
        float out = x * rms;
        if (use_affine) {
            out = out * __ldg(&weight[i]) + __ldg(&bias[i]);
        }
        output[row_offset + i] = ConvertOps<T>::from(out);
    }
}

// Launch a specific strategy (now 4 strategies: 0=scalar, 1=v15, 2=v20, 3=v29-const)
template<typename T>
static void launch_strategy_v13(int strategy, const T* input, T* output,
                                 const T* weight, const T* bias,
                                 int64_t batch_size, int64_t hidden_dim,
                                 float eps, bool use_affine,
                                 int block_size, size_t smem, bool aligned) {
    constexpr int vw = ConvertOps<T>::vec_width;

    switch (strategy) {
        case 0:  // scalar-unroll
            rmsnorm_v13_scalar_kernel<T><<<batch_size, block_size, smem>>>(
                input, output, weight, bias, hidden_dim, eps, use_affine);
            break;
        case 1:  // vectorized (v15-style, block=256)
            if (aligned) {
                rmsnorm_v13_vec_kernel<T, vw><<<batch_size, block_size, smem>>>(
                    input, output, weight, bias, hidden_dim, eps, use_affine);
            } else {
                rmsnorm_v13_scalar_kernel<T><<<batch_size, block_size, smem>>>(
                    input, output, weight, bias, hidden_dim, eps, use_affine);
            }
            break;
        case 2:  // dynamic block + __ldg (v20-style)
            if (aligned) {
                rmsnorm_v13_dynldg_kernel<T, vw><<<batch_size, block_size, smem>>>(
                    input, output, weight, bias, hidden_dim, eps, use_affine);
            } else {
                rmsnorm_v13_scalar_kernel<T><<<batch_size, block_size, smem>>>(
                    input, output, weight, bias, hidden_dim, eps, use_affine);
            }
            break;
        case 3:  // compile-time constant hidden_dim (v29-style)
            if (aligned) {
                launch_const_dim_v13<T>(input, output, weight, bias,
                    batch_size, hidden_dim, eps, use_affine, block_size, smem);
            } else {
                rmsnorm_v13_scalar_kernel<T><<<batch_size, block_size, smem>>>(
                    input, output, weight, bias, hidden_dim, eps, use_affine);
            }
            break;
        case 4:  // v19-style: 2x unroll + __ldg (fp32 large)
            if (aligned) {
                rmsnorm_v13_unroll2_kernel<T, vw><<<batch_size, block_size, smem>>>(
                    input, output, weight, bias, hidden_dim, eps, use_affine);
            } else {
                rmsnorm_v13_scalar_kernel<T><<<batch_size, block_size, smem>>>(
                    input, output, weight, bias, hidden_dim, eps, use_affine);
            }
            break;
    }
}

// Time a specific strategy
template<typename T>
static float time_strategy_v13(int strategy, const T* input, T* output,
                                const T* weight, const T* bias,
                                int64_t batch_size, int64_t hidden_dim,
                                float eps, bool use_affine,
                                int block_size, size_t smem, bool aligned,
                                int iterations) {
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // Warmup
    launch_strategy_v13<T>(strategy, input, output, weight, bias,
                           batch_size, hidden_dim, eps, use_affine,
                           block_size, smem, aligned);
    cudaDeviceSynchronize();

    cudaEventRecord(start);
    for (int j = 0; j < iterations; ++j) {
        launch_strategy_v13<T>(strategy, input, output, weight, bias,
                               batch_size, hidden_dim, eps, use_affine,
                               block_size, smem, aligned);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float ms;
    cudaEventElapsedTime(&ms, start, stop);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    return ms / iterations;
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

    // Check cache
    {
        std::lock_guard<std::mutex> lock(g_autotune_mutex);
        auto it = g_autotune_cache.find(key);
        if (it != g_autotune_cache.end()) {
            int block_size = 256;
            size_t smem = ((block_size + 31) / 32) * sizeof(float);
            AT_DISPATCH_FLOATING_TYPES_AND2(
                at::ScalarType::Half, at::ScalarType::BFloat16,
                input.scalar_type(), "rmsnorm_v13_autotune",
                [&]() {
                    constexpr int ab = ConvertOps<scalar_t>::align_bytes;
                    bool aligned = is_ptr_aligned<ab>(input.data_ptr<scalar_t>())
                                && is_ptr_aligned<ab>(output.data_ptr<scalar_t>());
                    launch_strategy_v13<scalar_t>(it->second.best_strategy,
                        input.data_ptr<scalar_t>(), output.data_ptr<scalar_t>(),
                        weight.data_ptr<scalar_t>(), bias.data_ptr<scalar_t>(),
                        batch_size, hidden_dim, eps, use_affine,
                        block_size, smem, aligned);
                });
            return;
        }
    }

    // Autotune: probe 3 strategies with lightweight timing
    int base_block = 256;
    size_t base_smem = ((base_block + 31) / 32) * sizeof(float);

    bool aligned = false;
    int dtype_code = 0;  // 0=fp32, 1=fp16, 2=bf16
    AT_DISPATCH_FLOATING_TYPES_AND2(
        at::ScalarType::Half, at::ScalarType::BFloat16,
        input.scalar_type(), "rmsnorm_v13_autotune_align",
        [&]() {
            constexpr int ab = ConvertOps<scalar_t>::align_bytes;
            aligned = is_ptr_aligned<ab>(input.data_ptr<scalar_t>())
                   && is_ptr_aligned<ab>(output.data_ptr<scalar_t>());
            if constexpr (std::is_same_v<scalar_t, c10::Half>) dtype_code = 1;
            else if constexpr (std::is_same_v<scalar_t, c10::BFloat16>) dtype_code = 2;
            else dtype_code = 0;
        });

    // Benchmark-informed heuristic:
    // - fp16/bf16 + aligned: v15 or v20 compete (probe both)
    // - fp32: v20 usually wins over v15, v6 for small shapes
    // - Not aligned: scalar only
    int warmup = 1;
    int iterations = 5;

    int best_strategy = 1;  // default: vectorized
    float best_time = 1e9f;

    if (!aligned) {
        // Scalar is only option
        best_strategy = 0;
    } else {
        // Determine which strategies to probe
        int strategies[4];
        int num_strategies = 0;

        if (dtype_code > 0) {
            // fp16/bf16: probe v15 (1), v20 (2), v29-const (3 if small D)
            strategies[num_strategies++] = 1;
            strategies[num_strategies++] = 2;
            if (hidden_dim <= 4096) {
                strategies[num_strategies++] = 3;
            }
        } else {
            // fp32: probe scalar (0), v15 (1), v20 (2), v29-const (3 if D<=4096), v19-unroll (4 if D>=4096)
            strategies[num_strategies++] = 0;
            strategies[num_strategies++] = 1;
            strategies[num_strategies++] = 2;
            if (hidden_dim <= 4096) {
                strategies[num_strategies++] = 3;
            }
            if (hidden_dim >= 4096) {
                strategies[num_strategies++] = 4;
            }
        }

        for (int s = 0; s < num_strategies; ++s) {
            int strat = strategies[s];
            int block = (strat == 2 && hidden_dim >= 4096) ? 512 : 256;
            size_t smem = ((block + 31) / 32) * sizeof(float);

            float t;
            AT_DISPATCH_FLOATING_TYPES_AND2(
                at::ScalarType::Half, at::ScalarType::BFloat16,
                input.scalar_type(), "rmsnorm_v13_autotune",
                [&]() {
                    if (strat == 3) {
                        launch_const_dim_v13<scalar_t>(
                            input.data_ptr<scalar_t>(), output.data_ptr<scalar_t>(),
                            weight.data_ptr<scalar_t>(), bias.data_ptr<scalar_t>(),
                            batch_size, hidden_dim, eps, use_affine, block, smem);
                        // Time const-dim
                        cudaEvent_t start, stop;
                        cudaEventCreate(&start);
                        cudaEventCreate(&stop);
                        rmsnorm_v13_scalar_kernel<scalar_t><<<batch_size, block, smem>>>(
                            input.data_ptr<scalar_t>(), output.data_ptr<scalar_t>(),
                            weight.data_ptr<scalar_t>(), bias.data_ptr<scalar_t>(),
                            hidden_dim, eps, use_affine);
                        cudaDeviceSynchronize();
                        cudaEventRecord(start);
                        for (int j = 0; j < iterations; ++j) {
                            launch_const_dim_v13<scalar_t>(
                                input.data_ptr<scalar_t>(), output.data_ptr<scalar_t>(),
                                weight.data_ptr<scalar_t>(), bias.data_ptr<scalar_t>(),
                                batch_size, hidden_dim, eps, use_affine, block, smem);
                        }
                        cudaEventRecord(stop);
                        cudaEventSynchronize(stop);
                        cudaEventElapsedTime(&t, start, stop);
                        cudaEventDestroy(start);
                        cudaEventDestroy(stop);
                        t /= iterations;
                    } else {
                        t = time_strategy_v13<scalar_t>(strat,
                            input.data_ptr<scalar_t>(), output.data_ptr<scalar_t>(),
                            weight.data_ptr<scalar_t>(), bias.data_ptr<scalar_t>(),
                            batch_size, hidden_dim, eps, use_affine,
                            block, smem, aligned, iterations);
                    }
                });

            if (t < best_time) {
                best_time = t;
                best_strategy = strat;
            }
        }
    }

    // Cache result
    {
        std::lock_guard<std::mutex> lock(g_autotune_mutex);
        g_autotune_cache[key] = {best_strategy, best_time};
    }

    // Launch best strategy
    int block = (best_strategy == 2 && hidden_dim >= 4096) ? 512 : 256;
    size_t smem = ((block + 31) / 32) * sizeof(float);
    AT_DISPATCH_FLOATING_TYPES_AND2(
        at::ScalarType::Half, at::ScalarType::BFloat16,
        input.scalar_type(), "rmsnorm_v13_autotune",
        [&]() {
            launch_strategy_v13<scalar_t>(best_strategy,
                input.data_ptr<scalar_t>(), output.data_ptr<scalar_t>(),
                weight.data_ptr<scalar_t>(), bias.data_ptr<scalar_t>(),
                batch_size, hidden_dim, eps, use_affine,
                block, smem, aligned);
        });
}

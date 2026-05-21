#include <torch/extension.h>
#include <cuda_runtime.h>
#include <stdint.h>
#include "rmsnorm_common.h"
#include <unordered_map>
#include <mutex>

// ============================================================================
// V13: Runtime autotuning
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
    int block_size;
    int num_blocks;
    float best_time_ms;
};

static std::unordered_map<AutotuneKey, AutotuneResult, HashAutotuneKey> g_autotune_cache;
static std::mutex g_autotune_mutex;

template<typename T>
__global__ void rmsnorm_v13_autotune_kernel(
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
        for (int64_t i = threadIdx.x; i < hidden_dim; i += blockDim.x) {
            float x = ConvertOps<T>::to(input[row_offset + i]);
            sum_sq += x * x;
        }

        float total = block_reduce_sum(sum_sq, smem, blockDim.x);
        float rms = rsqrtf(total / hidden_dim + eps);

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
            auto& r = it->second;
            int nb = r.num_blocks > 0 ? r.num_blocks : batch_size;
            size_t smem = ((r.block_size + 31) / 32) * sizeof(float);
            AT_DISPATCH_FLOATING_TYPES_AND2(
                at::ScalarType::Half, at::ScalarType::BFloat16,
                input.scalar_type(), "rmsnorm_v13_autotune",
                [&]() {
                    rmsnorm_v13_autotune_kernel<scalar_t><<<nb, r.block_size, smem>>>(
                        input.data_ptr<scalar_t>(), output.data_ptr<scalar_t>(),
                        weight.data_ptr<scalar_t>(), bias.data_ptr<scalar_t>(),
                        batch_size, hidden_dim, eps, use_affine);
                });
            return;
        }
    }

    int block_sizes[] = {128, 256, 512};
    float best_time = 1e9f;
    int best_block = 256;
    int best_nb = batch_size;

    for (int bs : block_sizes) {
        int nb = (batch_size < 64) ? 64 : batch_size;
        size_t smem = ((bs + 31) / 32) * sizeof(float);

        for (int j = 0; j < 5; ++j) {
            AT_DISPATCH_FLOATING_TYPES_AND2(
                at::ScalarType::Half, at::ScalarType::BFloat16,
                input.scalar_type(), "rmsnorm_v13_autotune",
                [&]() {
                    rmsnorm_v13_autotune_kernel<scalar_t><<<nb, bs, smem>>>(
                        input.data_ptr<scalar_t>(), output.data_ptr<scalar_t>(),
                        weight.data_ptr<scalar_t>(), bias.data_ptr<scalar_t>(),
                        batch_size, hidden_dim, eps, use_affine);
                });
        }
        cudaDeviceSynchronize();

        cudaEvent_t start, stop;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);
        cudaEventRecord(start);
        for (int j = 0; j < 10; ++j) {
            AT_DISPATCH_FLOATING_TYPES_AND2(
                at::ScalarType::Half, at::ScalarType::BFloat16,
                input.scalar_type(), "rmsnorm_v13_autotune",
                [&]() {
                    rmsnorm_v13_autotune_kernel<scalar_t><<<nb, bs, smem>>>(
                        input.data_ptr<scalar_t>(), output.data_ptr<scalar_t>(),
                        weight.data_ptr<scalar_t>(), bias.data_ptr<scalar_t>(),
                        batch_size, hidden_dim, eps, use_affine);
                });
        }
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        float ms;
        cudaEventElapsedTime(&ms, start, stop);
        float avg = ms / 10;

        cudaEventDestroy(start);
        cudaEventDestroy(stop);

        if (avg < best_time) {
            best_time = avg;
            best_block = bs;
            best_nb = nb;
        }
    }

    {
        std::lock_guard<std::mutex> lock(g_autotune_mutex);
        g_autotune_cache[key] = {best_block, best_nb, best_time};
    }

    size_t smem = ((best_block + 31) / 32) * sizeof(float);
    AT_DISPATCH_FLOATING_TYPES_AND2(
        at::ScalarType::Half, at::ScalarType::BFloat16,
        input.scalar_type(), "rmsnorm_v13_autotune",
        [&]() {
            rmsnorm_v13_autotune_kernel<scalar_t><<<best_nb, best_block, smem>>>(
                input.data_ptr<scalar_t>(), output.data_ptr<scalar_t>(),
                weight.data_ptr<scalar_t>(), bias.data_ptr<scalar_t>(),
                batch_size, hidden_dim, eps, use_affine);
        });
}

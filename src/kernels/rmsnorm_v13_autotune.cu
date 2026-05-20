#include <cuda_runtime.h>
#include <stdint.h>
#include "rmsnorm_common.h"
#include <unordered_map>
#include <mutex>

// ============================================================================
// V13: Runtime autotuning
// Probes multiple configs at first call, caches the best per (N, D, dtype)
// ============================================================================

struct AutotuneKey {
    int64_t batch_size;
    int64_t hidden_dim;
    int dtype;
    bool use_affine;

    bool operator==(const AutotuneKey& other) const {
        return batch_size == other.batch_size &&
               hidden_dim == other.hidden_dim &&
               dtype == other.dtype &&
               use_affine == other.use_affine;
    }
};

struct HashAutotuneKey {
    size_t operator()(const AutotuneKey& k) const {
        size_t h1 = std::hash<int64_t>{}(k.batch_size);
        size_t h2 = std::hash<int64_t>{}(k.hidden_dim);
        size_t h3 = std::hash<int>{}(k.dtype);
        size_t h4 = std::hash<bool>{}(k.use_affine);
        return h1 ^ (h2 << 1) ^ (h3 << 2) ^ (h4 << 3);
    }
};

struct AutotuneResult {
    int block_size;
    int num_blocks;  // -1 means use batch_size
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
            float x = to_float(input[row_offset + i]);
            sum_sq += x * x;
        }

        float total = block_reduce_sum(sum_sq, smem, blockDim.x);
        float rms = rsqrtf(total / hidden_dim + eps);

        for (int64_t i = threadIdx.x; i < hidden_dim; i += blockDim.x) {
            float x = to_float(input[row_offset + i]);
            float out = x * rms;
            if (use_affine) {
                out = out * to_float(weight[i]) + to_float(bias[i]);
            }
            output[row_offset + i] = from_float(output[row_offset + i], out);
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

    // Check cache
    {
        std::lock_guard<std::mutex> lock(g_autotune_mutex);
        auto it = g_autotune_cache.find(key);
        if (it != g_autotune_cache.end()) {
            const auto& result = it->second;
            int num_blocks = result.num_blocks > 0 ? result.num_blocks : batch_size;
            int num_warps = (result.block_size + 31) / 32;
            size_t smem = num_warps * sizeof(float);

            AT_DISPATCH_FLOATING_TYPES_AND2(
                at::ScalarType::Half, at::ScalarType::BFloat16,
                input.scalar_type(), "rmsnorm_v13_autotune",
                [&]() {
                    rmsnorm_v13_autotune_kernel<scalar_t><<<num_blocks, result.block_size, smem>>>(
                        input.data_ptr<scalar_t>(),
                        output.data_ptr<scalar_t>(),
                        weight.data_ptr<scalar_t>(),
                        bias.data_ptr<scalar_t>(),
                        batch_size,
                        hidden_dim,
                        eps,
                        use_affine
                    );
                }
            );
            return;
        }
    }

    // Autotune: test multiple configurations
    int block_sizes[] = {128, 256, 512};
    int num_block_options[] = {0, 0, 0};  // 0 = use batch_size

    float best_time = 1e9f;
    int best_block = 256;
    int best_num_blocks = 0;

    for (int i = 0; i < 3; ++i) {
        int bs = block_sizes[i];
        int nb = (batch_size < 64) ? 64 : batch_size;

        // Warmup
        int num_warps = (bs + 31) / 32;
        size_t smem = num_warps * sizeof(float);

        AT_DISPATCH_FLOATING_TYPES_AND2(
            at::ScalarType::Half, at::ScalarType::BFloat16,
            input.scalar_type(), "rmsnorm_v13_autotune",
            [&]() {
                rmsnorm_v13_autotune_kernel<scalar_t><<<nb, bs, smem>>>(
                    input.data_ptr<scalar_t>(),
                    output.data_ptr<scalar_t>(),
                    weight.data_ptr<scalar_t>(),
                    bias.data_ptr<scalar_t>(),
                    batch_size,
                    hidden_dim,
                    eps,
                    use_affine
                );
            }
        );
        cudaDeviceSynchronize();

        // Time
        cudaEvent_t start, stop;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);
        cudaEventRecord(start);

        int iterations = 10;
        for (int j = 0; j < iterations; ++j) {
            AT_DISPATCH_FLOATING_TYPES_AND2(
                at::ScalarType::Half, at::ScalarType::BFloat16,
                input.scalar_type(), "rmsnorm_v13_autotune",
                [&]() {
                    rmsnorm_v13_autotune_kernel<scalar_t><<<nb, bs, smem>>>(
                        input.data_ptr<scalar_t>(),
                        output.data_ptr<scalar_t>(),
                        weight.data_ptr<scalar_t>(),
                        bias.data_ptr<scalar_t>(),
                        batch_size,
                        hidden_dim,
                        eps,
                        use_affine
                    );
                }
            );
        }
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);

        float ms;
        cudaEventElapsedTime(&ms, start, stop);
        float avg_time = ms / iterations;

        cudaEventDestroy(start);
        cudaEventDestroy(stop);

        if (avg_time < best_time) {
            best_time = avg_time;
            best_block = bs;
            best_num_blocks = nb;
        }
    }

    // Cache result
    {
        std::lock_guard<std::mutex> lock(g_autotune_mutex);
        g_autotune_cache[key] = {best_block, best_num_blocks, best_time};
    }

    // Run with best config
    int num_warps = (best_block + 31) / 32;
    size_t smem = num_warps * sizeof(float);

    AT_DISPATCH_FLOATING_TYPES_AND2(
        at::ScalarType::Half, at::ScalarType::BFloat16,
        input.scalar_type(), "rmsnorm_v13_autotune",
        [&]() {
            rmsnorm_v13_autotune_kernel<scalar_t><<<best_num_blocks, best_block, smem>>>(
                input.data_ptr<scalar_t>(),
                output.data_ptr<scalar_t>(),
                weight.data_ptr<scalar_t>(),
                bias.data_ptr<scalar_t>(),
                batch_size,
                hidden_dim,
                eps,
                use_affine
            );
        }
    );
}

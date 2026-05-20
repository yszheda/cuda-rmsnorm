#ifndef RMSNORM_COMMON_H
#define RMSNORM_COMMON_H

#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <type_traits>
#include <cstdint>

// ============================================================================
// Type traits and scalar/vector type mappings
// ============================================================================

template<typename T>
struct VecType;

template<>
struct VecType<float> {
    using scalar_t = float;
    using vec2_t = float2;
    using vec4_t = float4;
    static constexpr int max_vec_width = 4;  // float4 = 128-bit
};

template<>
struct VecType<half> {
    using scalar_t = half;
    using vec2_t = half2;
    using vec4_t = float4;  // half2 packed into float4 for 128-bit load
    static constexpr int max_vec_width = 8;  // 8 x half = 128-bit
};

template<>
struct VecType<__nv_bfloat16> {
    using scalar_t = __nv_bfloat16;
    using vec2_t = __nv_bfloat162;
    using vec4_t = float4;  // bf16x2 packed into float4 for 128-bit load
    static constexpr int max_vec_width = 8;  // 8 x bf16 = 128-bit
};

// ============================================================================
// Pointer alignment check
// ============================================================================

template<int alignment_bytes>
__device__ __forceinline__ bool is_aligned(const void* ptr) {
    return (reinterpret_cast<uintptr_t>(ptr) % alignment_bytes) == 0;
}

// ============================================================================
// Vectorized load/store helpers
// ============================================================================

template<typename T>
__device__ __forceinline__ void vec_load_4(const T* src, T* dst, int width) {
    for (int i = 0; i < width; ++i) {
        dst[i] = src[i];
    }
}

template<>
__device__ __forceinline__ void vec_load_4<float>(const float* src, float* dst, int width) {
    const float4* src4 = reinterpret_cast<const float4*>(src);
    float4* dst4 = reinterpret_cast<float4*>(dst);
    *dst4 = *src4;
}

template<>
__device__ __forceinline__ void vec_load_4<half>(const half* src, half* dst, int width) {
    const float4* src4 = reinterpret_cast<const float4*>(src);
    float4* dst4 = reinterpret_cast<float4*>(dst);
    *dst4 = *src4;
}

template<>
__device__ __forceinline__ void vec_load_4<__nv_bfloat16>(const __nv_bfloat16* src, __nv_bfloat16* dst, int width) {
    const float4* src4 = reinterpret_cast<const float4*>(src);
    float4* dst4 = reinterpret_cast<float4*>(dst);
    *dst4 = *src4;
}

template<typename T>
__device__ __forceinline__ void vec_store_4(const T* src, T* dst, int width) {
    for (int i = 0; i < width; ++i) {
        dst[i] = src[i];
    }
}

template<>
__device__ __forceinline__ void vec_store_4<float>(const float* src, float* dst, int width) {
    const float4* src4 = reinterpret_cast<const float4*>(src);
    float4* dst4 = reinterpret_cast<float4*>(dst);
    *dst4 = *src4;
}

template<>
__device__ __forceinline__ void vec_store_4<half>(const half* src, half* dst, int width) {
    const float4* src4 = reinterpret_cast<const float4*>(src);
    float4* dst4 = reinterpret_cast<float4*>(dst);
    *dst4 = *src4;
}

template<>
__device__ __forceinline__ void vec_store_4<__nv_bfloat16>(const __nv_bfloat16* src, __nv_bfloat16* dst, int width) {
    const float4* src4 = reinterpret_cast<const float4*>(src);
    float4* dst4 = reinterpret_cast<float4*>(dst);
    *dst4 = *src4;
}

// ============================================================================
// Convert to/from float for computation
// ============================================================================

__device__ __forceinline__ float to_float(float x) { return x; }
__device__ __forceinline__ float to_float(half x) { return __half2float(x); }
__device__ __forceinline__ float to_float(__nv_bfloat16 x) { return __bfloat162float(x); }

__device__ __forceinline__ float from_float(float x) { return x; }
__device__ __forceinline__ half from_float(half, float x) { return __float2half(x); }
__device__ __forceinline__ __nv_bfloat16 from_float(__nv_bfloat16, float x) { return __float2bfloat16(x); }

// ============================================================================
// Warp-level reduction helpers
// ============================================================================

__device__ __forceinline__ float warp_reduce_sum(float val, int warp_size = 32) {
    #pragma unroll
    for (int mask = warp_size / 2; mask > 0; mask /= 2) {
        val += __shfl_down_sync(0xffffffff, val, mask);
    }
    return val;
}

// ============================================================================
// Block-level reduction via shared memory
// ============================================================================

__device__ __forceinline__ float block_reduce_sum(float val, float* smem, int block_dim) {
    int lane = threadIdx.x % 32;
    int warp_id = threadIdx.x / 32;
    int num_warps = (block_dim + 31) / 32;

    // Intra-warp reduction
    val = warp_reduce_sum(val);

    // Write warp results to shared memory
    if (lane == 0) {
        smem[warp_id] = val;
    }
    __syncthreads();

    // Inter-warp reduction
    if (threadIdx.x < num_warps) {
        val = smem[threadIdx.x];
    } else {
        val = 0.0f;
    }
    val = warp_reduce_sum(val);

    return val;
}

// ============================================================================
// Occupancy calculator
// ============================================================================

__host__ inline int compute_optimal_block_size(int num_sms, int desired_blocks_per_sm = 4) {
    // Simple heuristic: try 256, 512, 1024 and pick best
    // In practice, query cudaOccupancyMaxActiveBlocksPerMultiprocessor
    int candidates[] = {256, 512, 1024};
    int best = 256;
    int best_occupancy = 0;

    for (int bs : candidates) {
        int blocks_per_sm = 1024 / bs;  // rough estimate
        if (blocks_per_sm > desired_blocks_per_sm) {
            blocks_per_sm = desired_blocks_per_sm;
        }
        int occupancy = blocks_per_sm * bs;
        if (occupancy > best_occupancy) {
            best_occupancy = occupancy;
            best = bs;
        }
    }
    return best;
}

#endif // RMSNORM_COMMON_H

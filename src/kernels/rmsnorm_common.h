#ifndef RMSNORM_COMMON_H
#define RMSNORM_COMMON_H

#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <ATen/cuda/CUDAContext.h>
#include <type_traits>
#include <cstdint>

// ============================================================================
// Conversion helpers via type traits to avoid overload ambiguity
// ============================================================================

template<typename T>
struct ConvertOps;

template<>
struct ConvertOps<float> {
    __device__ __forceinline__ static float to(float x) { return x; }
    __device__ __forceinline__ static float from(float x) { return x; }
    __device__ __forceinline__ static void elem_store(float* dst, float val) { *dst = val; }
    using vec4_t = float4;
    static constexpr int vec_width = 4;
    static constexpr int align_bytes = 16;
    using vec_elem_t = float;
};

template<>
struct ConvertOps<half> {
    __device__ __forceinline__ static float to(half x) { return __half2float(x); }
    __device__ __forceinline__ static half from(float x) { return __float2half(x); }
    __device__ __forceinline__ static void elem_store(half* dst, float val) { *dst = __float2half(val); }
    using vec4_t = float4;
    static constexpr int vec_width = 8;
    static constexpr int align_bytes = 16;
    using vec_elem_t = half;
};

template<>
struct ConvertOps<__nv_bfloat16> {
    __device__ __forceinline__ static float to(__nv_bfloat16 x) { return __bfloat162float(x); }
    __device__ __forceinline__ static __nv_bfloat16 from(float x) { return __float2bfloat16(x); }
    __device__ __forceinline__ static void elem_store(__nv_bfloat16* dst, float val) { *dst = __float2bfloat16(val); }
    using vec4_t = float4;
    static constexpr int vec_width = 8;
    static constexpr int align_bytes = 16;
    using vec_elem_t = __nv_bfloat16;
};

// PyTorch c10 type specializations
template<>
struct ConvertOps<c10::Half> {
    __device__ __forceinline__ static float to(c10::Half x) { return __half2float(static_cast<half>(x)); }
    __device__ __forceinline__ static c10::Half from(float x) { return static_cast<c10::Half>(__float2half(x)); }
    __device__ __forceinline__ static void elem_store(half* dst, float val) { *dst = __float2half(val); }
    using vec4_t = float4;
    static constexpr int vec_width = 8;
    static constexpr int align_bytes = 16;
    using vec_elem_t = half;
};

template<>
struct ConvertOps<c10::BFloat16> {
    __device__ __forceinline__ static float to(c10::BFloat16 x) { return __bfloat162float(static_cast<__nv_bfloat16>(x)); }
    __device__ __forceinline__ static c10::BFloat16 from(float x) { return static_cast<c10::BFloat16>(__float2bfloat16(x)); }
    __device__ __forceinline__ static void elem_store(__nv_bfloat16* dst, float val) { *dst = __float2bfloat16(val); }
    using vec4_t = float4;
    static constexpr int vec_width = 8;
    static constexpr int align_bytes = 16;
    using vec_elem_t = __nv_bfloat16;
};

// Double specialization (required by AT_DISPATCH_FLOATING_TYPES_AND2)
template<>
struct ConvertOps<double> {
    __device__ __forceinline__ static float to(double x) { return static_cast<float>(x); }
    __device__ __forceinline__ static double from(float x) { return static_cast<double>(x); }
    __device__ __forceinline__ static void elem_store(double* dst, float val) { *dst = static_cast<double>(val); }
    using vec4_t = float4;
    static constexpr int vec_width = 2;
    static constexpr int align_bytes = 16;
    using vec_elem_t = double;
};

// ============================================================================
// Vectorized load/store helpers
// ============================================================================

template<typename T>
__device__ __forceinline__ void vec_load(const T* src, T* dst) {
    const float4* src4 = reinterpret_cast<const float4*>(src);
    float4* dst4 = reinterpret_cast<float4*>(dst);
    *dst4 = *src4;
}

template<typename T>
__device__ __forceinline__ void vec_store(const T* src, T* dst) {
    const float4* src4 = reinterpret_cast<const float4*>(src);
    float4* dst4 = reinterpret_cast<float4*>(dst);
    *dst4 = *src4;
}

// ============================================================================
// Host-side alignment check
// ============================================================================

template<int alignment_bytes>
static inline bool is_ptr_aligned(const void* ptr) {
    return (reinterpret_cast<uintptr_t>(ptr) % alignment_bytes) == 0;
}

// ============================================================================
// Warp-level reduction helpers
// ============================================================================

__device__ __forceinline__ float warp_reduce_sum(float val) {
    #pragma unroll
    for (int mask = 16; mask > 0; mask >>= 1) {
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

    val = warp_reduce_sum(val);

    if (lane == 0) {
        smem[warp_id] = val;
    }
    __syncthreads();

    if (threadIdx.x < num_warps) {
        val = smem[threadIdx.x];
    } else {
        val = 0.0f;
    }
    val = warp_reduce_sum(val);

    // Broadcast: thread 0 writes to smem[0], all threads read it back
    if (threadIdx.x == 0) smem[0] = val;
    __syncthreads();
    return smem[0];
}

#endif // RMSNORM_COMMON_H

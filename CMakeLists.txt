cmake_minimum_required(VERSION 3.18)
project(rmsnorm_cuda LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# CUDA architecture
if(DEFINED ENV{PYTORCH_CUDA_ARCH})
    set(CMAKE_CUDA_ARCHITECTURES $ENV{PYTORCH_CUDA_ARCH})
else()
    set(CMAKE_CUDA_ARCHITECTURES "native")
endif()

# Find CUDA toolkit
find_package(CUDAToolkit REQUIRED)

# Find PyTorch (provides torch/extension.h and ATen macros)
execute_process(
    COMMAND python3 -c "import torch; print(torch.utils.cmake_prefix_path)"
    OUTPUT_VARIABLE TORCH_CMAKE_PATH
    OUTPUT_STRIP_TRAILING_WHITESPACE
)
list(APPEND CMAKE_PREFIX_PATH "${TORCH_CMAKE_PATH}")
find_package(Torch REQUIRED)

# Fetch pybind11
include(FetchContent)
FetchContent_Declare(
    pybind11
    GIT_REPOSITORY https://github.com/pybind/pybind11.git
    GIT_TAG        v2.13.1
)
FetchContent_MakeAvailable(pybind11)

# Source files
set(KERNEL_SOURCES
    src/kernels/rmsnorm_baseline.cu
    src/kernels/rmsnorm_v1_block_reduce.cu
    src/kernels/rmsnorm_v2_warp_shuffle.cu
    src/kernels/rmsnorm_v3_vectorized.cu
    src/kernels/rmsnorm_v4_persistent.cu
    src/kernels/rmsnorm_v5_align.cu
    src/kernels/rmsnorm_v6_unroll.cu
    src/kernels/rmsnorm_v7_doublebuf.cu
    src/kernels/rmsnorm_v8_warp_spec.cu
    src/kernels/rmsnorm_v9_tiling.cu
    src/kernels/rmsnorm_v10_chunk.cu
    src/kernels/rmsnorm_v11_gridmap.cu
    src/kernels/rmsnorm_v12_fused.cu
    src/kernels/rmsnorm_v13_autotune.cu
    src/kernels/rmsnorm_v14_cudagraph.cu
)

set(BINDING_SOURCES
    src/bindings/pybind_rmsnorm.cpp
)

# Build pybind11 extension module
pybind11_add_module(rmsnorm_ext
    ${BINDING_SOURCES}
    ${KERNEL_SOURCES}
)

target_include_directories(rmsnorm_ext PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/src/include
    ${CMAKE_CURRENT_SOURCE_DIR}/src/kernels
    ${TORCH_INCLUDE_DIRS}
)

target_link_libraries(rmsnorm_ext PRIVATE
    CUDA::cudart
    ${TORCH_LIBRARIES}
)

# Link torch_python for pybind11 at::Tensor type caster
find_library(TORCH_PYTHON_LIBRARY torch_python PATHS "${TORCH_INSTALL_PREFIX}/lib")
if(TORCH_PYTHON_LIBRARY)
    target_link_libraries(rmsnorm_ext PRIVATE ${TORCH_PYTHON_LIBRARY})
endif()

# Compile flags for profiling support
target_compile_options(rmsnorm_ext PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:
        -O3
        -lineinfo
        --expt-relaxed-constexpr
        -Xcompiler=-fPIC
        -Wno-deprecated-gpu-targets
    >
)

# Suppress warnings in kernel code
set_target_properties(rmsnorm_ext PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
)

# Set output directory
set_target_properties(rmsnorm_ext PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/python/rmsnorm
)

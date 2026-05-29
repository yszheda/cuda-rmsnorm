#include <torch/extension.h>
#include "rmsnorm.h"

// Helper: validate input tensors
static void validate_input(
    const torch::Tensor& input,
    const torch::Tensor& weight,
    const torch::Tensor& bias,
    bool use_affine
) {
    TORCH_CHECK(input.dim() >= 2, "input must have at least 2 dimensions");
    TORCH_CHECK(input.is_contiguous(), "input must be contiguous");
    TORCH_CHECK(input.device().is_cuda(), "input must be on CUDA device");
    TORCH_CHECK(
        input.scalar_type() == torch::kFloat32 ||
        input.scalar_type() == torch::kFloat16 ||
        input.scalar_type() == torch::kBFloat16,
        "input dtype must be float32, float16, or bfloat16"
    );
    if (use_affine) {
        TORCH_CHECK(weight.dim() == 1, "weight must be 1D");
        TORCH_CHECK(weight.is_contiguous(), "weight must be contiguous");
        TORCH_CHECK(bias.is_contiguous(), "bias must be contiguous");
    }
}

// Standard RMSNorm (no residual)
void rmsnorm_cuda_wrapper(
    torch::Tensor output,
    const torch::Tensor input,
    const torch::Tensor weight,
    const torch::Tensor bias,
    float eps,
    bool use_affine,
    int version
) {
    validate_input(input, weight, bias, use_affine);
    TORCH_CHECK(output.sizes() == input.sizes(), "output shape must match input shape");
    TORCH_CHECK(output.is_contiguous(), "output must be contiguous");
    TORCH_CHECK(output.dtype() == input.dtype(), "output dtype must match input dtype");

    switch (version) {
        case 0:  rmsnorm_baseline_cuda(output, input, weight, bias, eps, use_affine); break;
        case 1:  rmsnorm_v1_block_reduce_cuda(output, input, weight, bias, eps, use_affine); break;
        case 2:  rmsnorm_v2_warp_shuffle_cuda(output, input, weight, bias, eps, use_affine); break;
        case 3:  rmsnorm_v3_vectorized_cuda(output, input, weight, bias, eps, use_affine); break;
        case 4:  rmsnorm_v4_persistent_cuda(output, input, weight, bias, eps, use_affine); break;
        case 5:  rmsnorm_v5_align_cuda(output, input, weight, bias, eps, use_affine); break;
        case 6:  rmsnorm_v6_unroll_cuda(output, input, weight, bias, eps, use_affine); break;
        case 7:  rmsnorm_v7_doublebuf_cuda(output, input, weight, bias, eps, use_affine); break;
        case 8:  rmsnorm_v8_warp_spec_cuda(output, input, weight, bias, eps, use_affine); break;
        case 9:  rmsnorm_v9_tiling_cuda(output, input, weight, bias, eps, use_affine); break;
        case 10: rmsnorm_v10_chunk_cuda(output, input, weight, bias, eps, use_affine); break;
        case 11: rmsnorm_v11_gridmap_cuda(output, input, weight, bias, eps, use_affine); break;
        case 13: rmsnorm_v13_autotune_cuda(output, input, weight, bias, eps, use_affine); break;
        case 14: rmsnorm_v14_cudagraph_cuda(output, input, weight, bias, eps, use_affine); break;
        case 15: rmsnorm_v15_vec_unroll_cuda(output, input, weight, bias, eps, use_affine); break;
        case 18: rmsnorm_v18_dynamic_block_cuda(output, input, weight, bias, eps, use_affine); break;
        case 19: rmsnorm_v19_vec_unroll_cuda(output, input, weight, bias, eps, use_affine); break;
        case 20: rmsnorm_v20_half2_cuda(output, input, weight, bias, eps, use_affine); break;
        case 21: rmsnorm_v21_persistent_cuda(output, input, weight, bias, eps, use_affine); break;
        case 22: rmsnorm_v22_persistent_cuda(output, input, weight, bias, eps, use_affine); break;
        case 23: rmsnorm_v23_warp_persistent_cuda(output, input, weight, bias, eps, use_affine); break;
        case 25: rmsnorm_v25_best_cuda(output, input, weight, bias, eps, use_affine); break;
        default:
            // Use the best version (v15) as default
            rmsnorm_v15_vec_unroll_cuda(output, input, weight, bias, eps, use_affine);
            break;
    }
}

// Fused add RMSNorm (with residual)
void rmsnorm_fused_cuda(
    torch::Tensor output,
    const torch::Tensor input,
    const torch::Tensor residual,
    const torch::Tensor weight,
    const torch::Tensor bias,
    float eps,
    bool use_affine
) {
    validate_input(input, weight, bias, use_affine);
    TORCH_CHECK(residual.sizes() == input.sizes(), "residual shape must match input shape");
    TORCH_CHECK(residual.is_contiguous(), "residual must be contiguous");
    TORCH_CHECK(output.sizes() == input.sizes(), "output shape must match input shape");

    rmsnorm_v12_fused_cuda(output, input, residual, weight, bias, eps, use_affine);
}

// Python binding for forward RMSNorm
torch::Tensor rmsnorm_forward(
    const torch::Tensor& input,
    const torch::Tensor& weight,
    const torch::Tensor& bias,
    float eps,
    bool use_affine,
    int version
) {
    torch::Tensor output = torch::empty_like(input);
    rmsnorm_cuda_wrapper(output, input, weight, bias, eps, use_affine, version);
    return output;
}

// Python binding for fused add RMSNorm
torch::Tensor rmsnorm_fused_add_forward(
    const torch::Tensor& input,
    const torch::Tensor& residual,
    const torch::Tensor& weight,
    const torch::Tensor& bias,
    float eps,
    bool use_affine
) {
    torch::Tensor output = torch::empty_like(input);
    rmsnorm_fused_cuda(output, input, residual, weight, bias, eps, use_affine);
    return output;
}

PYBIND11_MODULE(rmsnorm_ext, m) {
    m.doc() = "High-performance CUDA RMSNorm with PyTorch bindings";

    m.def(
        "rmsnorm",
        &rmsnorm_forward,
        "Compute RMSNorm",
        py::arg("input"),
        py::arg("weight"),
        py::arg("bias"),
        py::arg("eps") = 1e-6,
        py::arg("use_affine") = true,
        py::arg("version") = -1
    );

    m.def(
        "rmsnorm_fused_add",
        &rmsnorm_fused_add_forward,
        "Compute fused residual + RMSNorm",
        py::arg("input"),
        py::arg("residual"),
        py::arg("weight"),
        py::arg("bias"),
        py::arg("eps") = 1e-6,
        py::arg("use_affine") = true
    );
}

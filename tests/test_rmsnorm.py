import pytest
import torch
import numpy as np
from typing import List, Tuple


class TorchRMSNorm(torch.nn.Module):
    """Reference RMSNorm implementation using PyTorch as golden."""

    def __init__(self, normalized_shape, eps=1e-6, elementwise_affine=True):
        super().__init__()
        if isinstance(normalized_shape, int):
            normalized_shape = (normalized_shape,)
        self.normalized_shape = tuple(normalized_shape)
        self.eps = eps
        self.elementwise_affine = elementwise_affine
        if elementwise_affine:
            self.weight = torch.nn.Parameter(torch.ones(normalized_shape))
            self.bias = torch.nn.Parameter(torch.zeros(normalized_shape))
        else:
            self.register_parameter("weight", None)
            self.register_parameter("bias", None)

    def forward(self, x):
        original_shape = x.shape
        hidden_size = self.normalized_shape[0]
        x = x.reshape(-1, hidden_size)
        # RMSNorm: x / RMS(x) * weight + bias
        variance = x.pow(2).mean(-1, keepdim=True)
        x = x * torch.rsqrt(variance + self.eps)
        if self.elementwise_affine:
            x = x * self.weight + self.bias
        return x.view(original_shape)


def rmsnorm_reference(x, weight, bias, eps, use_affine):
    """Pure PyTorch reference implementation."""
    variance = x.pow(2).mean(-1, keepdim=True)
    out = x * torch.rsqrt(variance + eps)
    if use_affine:
        out = out * weight + bias
    return out


# Test parameters
SHAPES = [
    (2, 4),
    (8, 32),
    (64, 128),
    (256, 1024),
    (512, 2048),
    (256, 4096),
    (1, 128),
    (1, 8192),
    (1024, 5120),
    (128, 16384),
]

DTYPES = [torch.float32, torch.float16, torch.bfloat16]
EPS_VALUES = [1e-5, 1e-6, 1e-8]
AFFINE_VALUES = [True, False]

# Only test a subset of kernel versions for basic coverage
# Full version testing is done during benchmark runs
KERNEL_VERSIONS = [-1]  # -1 = best version


def get_tolerances(dtype):
    if dtype == torch.float32:
        return {"rtol": 1e-4, "atol": 1e-5}
    elif dtype in (torch.float16, torch.bfloat16):
        return {"rtol": 1e-2, "atol": 1e-2}
    return {"rtol": 1e-3, "atol": 1e-3}


class TestRMSNormCorrectness:
    @pytest.mark.parametrize("shape", SHAPES)
    @pytest.mark.parametrize("dtype", DTYPES)
    @pytest.mark.parametrize("eps", EPS_VALUES)
    @pytest.mark.parametrize("affine", AFFINE_VALUES)
    def test_correctness_vs_pytorch(self, shape, dtype, eps, affine):
        """Test CUDA RMSNorm against PyTorch golden reference."""
        try:
            import rmsnorm_ext
        except ImportError:
            pytest.skip("rmsnorm_ext not built")

        torch.manual_seed(42)
        hidden_size = shape[-1]
        x = torch.randn(shape, dtype=dtype, device="cuda")
        weight = torch.randn(hidden_size, dtype=dtype, device="cuda") if affine else torch.empty(0, dtype=dtype, device="cuda")
        bias = torch.randn(hidden_size, dtype=dtype, device="cuda") if affine else torch.empty(0, dtype=dtype, device="cuda")

        # Our implementation
        output = rmsnorm_ext.rmsnorm(x, weight, bias, eps, affine, version=-1)

        # PyTorch reference
        ref_out = rmsnorm_reference(x.float(), weight.float(), bias.float(), eps, affine)
        ref_out = ref_out.to(dtype)

        rtol, atol = get_tolerances(dtype).values()
        torch.testing.assert_close(output, ref_out, rtol=rtol, atol=atol)

    def test_numerical_stability_large_values(self):
        """Test with very large input values."""
        try:
            import rmsnorm_ext
        except ImportError:
            pytest.skip("rmsnorm_ext not built")

        x = torch.randn(64, 1024, dtype=torch.float32, device="cuda") * 1e6
        weight = torch.ones(1024, dtype=torch.float32, device="cuda")
        bias = torch.zeros(1024, dtype=torch.float32, device="cuda")

        output = rmsnorm_ext.rmsnorm(x, weight, bias, 1e-6, True)
        # Output should be normalized (no NaN/Inf)
        assert not torch.isnan(output).any()
        assert not torch.isinf(output).any()

    def test_numerical_stability_small_values(self):
        """Test with very small input values."""
        try:
            import rmsnorm_ext
        except ImportError:
            pytest.skip("rmsnorm_ext not built")

        x = torch.randn(64, 1024, dtype=torch.float32, device="cuda") * 1e-10
        weight = torch.ones(1024, dtype=torch.float32, device="cuda")
        bias = torch.zeros(1024, dtype=torch.float32, device="cuda")

        output = rmsnorm_ext.rmsnorm(x, weight, bias, 1e-6, True)
        assert not torch.isnan(output).any()
        assert not torch.isinf(output).any()

    def test_contiguous_requirement(self):
        """Test that non-contiguous input is handled properly."""
        try:
            import rmsnorm_ext
        except ImportError:
            pytest.skip("rmsnorm_ext not built")

        x = torch.randn(128, 1024, dtype=torch.float32, device="cuda")
        x_non_contig = x.T  # Non-contiguous
        weight = torch.ones(128, dtype=torch.float32, device="cuda")
        bias = torch.zeros(128, dtype=torch.float32, device="cuda")

        # Should raise an error or handle it internally
        with pytest.raises(Exception):
            rmsnorm_ext.rmsnorm(x_non_contig, weight, bias, 1e-6, True)


class TestRMSNormModule:
    """Test the torch.nn.Module wrapper."""

    @pytest.mark.parametrize("shape", SHAPES[:5])
    @pytest.mark.parametrize("dtype", DTYPES)
    def test_module_api(self, shape, dtype):
        """Test CUDARMSNorm matches torch.nn.RMSNorm interface."""
        try:
            from rmsnorm import CUDARMSNorm
        except ImportError:
            pytest.skip("rmsnorm not built")

        hidden_size = shape[-1]

        # Our module
        our_norm = CUDARMSNorm(hidden_size, eps=1e-6, elementwise_affine=True).cuda().to(dtype)
        # PyTorch module
        pt_norm = torch.nn.RMSNorm(hidden_size, eps=1e-6, elementwise_affine=True).cuda().to(dtype)

        # Copy weights
        with torch.no_grad():
            our_norm.weight.copy_(pt_norm.weight)

        x = torch.randn(shape, dtype=dtype, device="cuda")
        our_out = our_norm(x)
        pt_out = pt_norm(x)

        rtol, atol = get_tolerances(dtype).values()
        torch.testing.assert_close(our_out, pt_out, rtol=rtol, atol=atol)

    def test_module_no_affine(self):
        """Test elementwise_affine=False."""
        try:
            from rmsnorm import CUDARMSNorm
        except ImportError:
            pytest.skip("rmsnorm not built")

        norm = CUDARMSNorm(1024, elementwise_affine=False).cuda()
        assert norm.weight is None
        assert norm.bias is None

        x = torch.randn(32, 1024, dtype=torch.float32, device="cuda")
        out = norm(x)
        assert out.shape == x.shape
        assert not torch.isnan(out).any()


class TestFusedRMSNorm:
    """Test fused residual add + RMSNorm."""

    @pytest.mark.parametrize("shape", SHAPES[:3])
    @pytest.mark.parametrize("dtype", [torch.float32])
    def test_fused_correctness(self, shape, dtype):
        """Test fused add RMSNorm against reference."""
        try:
            import rmsnorm_ext
        except ImportError:
            pytest.skip("rmsnorm_ext not built")

        x = torch.randn(shape, dtype=dtype, device="cuda")
        residual = torch.randn(shape, dtype=dtype, device="cuda")
        hidden_size = shape[-1]
        weight = torch.randn(hidden_size, dtype=dtype, device="cuda")
        bias = torch.randn(hidden_size, dtype=dtype, device="cuda")

        # Fused implementation
        fused_out = rmsnorm_ext.rmsnorm_fused_add(x, residual, weight, bias, 1e-6, True)

        # Reference: add then RMSNorm
        x_fused = (x + residual).float()
        ref = rmsnorm_reference(x_fused, weight.float(), bias.float(), 1e-6, True)
        ref = ref.to(dtype)

        rtol, atol = get_tolerances(dtype).values()
        torch.testing.assert_close(fused_out, ref, rtol=rtol, atol=atol)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

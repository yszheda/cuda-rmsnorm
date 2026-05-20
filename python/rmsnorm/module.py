import torch
import torch.nn as nn
from typing import Optional

try:
    import rmsnorm_ext
except ImportError:
    raise ImportError(
        "rmsnorm_ext module not found. Please build the project with CMake first:\n"
        "  mkdir -p build && cd build\n"
        "  cmake .. -DCMAKE_BUILD_TYPE=Release\n"
        "  make -j$(nproc)"
    )


class CUDARMSNorm(nn.Module):
    """CUDA-accelerated RMSNorm matching torch.nn.RMSNorm API.

    Args:
        normalized_shape: Input shape from an expected input of size (*, H)
        eps: A value added to the denominator for numerical stability. Default: 1e-6
        elementwise_affine: A boolean value that when set to True, this module
            has learnable per-element weight and bias. Default: True
        bias: A boolean value that when set to True, this module has a learnable
            bias. Default: True
        device: Device for parameters. Default: None
        dtype: Data type for parameters. Default: None
        version: Kernel version to use. -1 = auto (best), 0-14 = specific version.
    """

    def __init__(
        self,
        normalized_shape,
        eps: float = 1e-6,
        elementwise_affine: bool = True,
        bias: bool = True,
        device=None,
        dtype=None,
        version: int = -1,
    ):
        super().__init__()
        if isinstance(normalized_shape, int):
            normalized_shape = (normalized_shape,)
        self.normalized_shape = tuple(normalized_shape)
        self.eps = eps
        self.elementwise_affine = elementwise_affine
        self.use_bias = bias
        self.version = version

        if elementwise_affine:
            self.weight = nn.Parameter(
                torch.ones(self.normalized_shape, device=device, dtype=dtype)
            )
            self.bias = nn.Parameter(
                torch.zeros(self.normalized_shape, device=device, dtype=dtype)
            ) if bias else None
        else:
            self.register_parameter("weight", None)
            self.register_parameter("bias", None)

        self.reset_parameters()

    def reset_parameters(self):
        if self.elementwise_affine:
            nn.init.ones_(self.weight)
            if self.bias is not None:
                nn.init.zeros_(self.bias)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        # Save original shape
        original_shape = input.shape
        hidden_size = self.normalized_shape[0]

        # Flatten input to 2D: [N, hidden_size]
        x = input.reshape(-1, hidden_size)

        # Ensure contiguous
        x = x.contiguous()

        # Get weight and bias
        weight = self.weight if self.weight is not None else torch.empty(0, dtype=x.dtype, device=x.device)
        bias = self.bias if self.bias is not None else torch.empty(0, dtype=x.dtype, device=x.device)

        output = rmsnorm_ext.rmsnorm(
            x, weight, bias, self.eps, self.elementwise_affine, self.version
        )

        # Reshape back to original shape
        return output.view(original_shape)

    def extra_repr(self) -> str:
        return (
            f"{self.normalized_shape}, eps={self.eps}, "
            f"elementwise_affine={self.elementwise_affine}"
        )


class CUDARMSNormFused(nn.Module):
    """Fused residual add + RMSNorm.

    Computes: output = rmsnorm(input + residual)

    Args:
        normalized_shape: Input shape from an expected input of size (*, H)
        eps: A value added to the denominator for numerical stability. Default: 1e-6
        elementwise_affine: Whether to use learnable weight and bias. Default: True
        bias: Whether to use learnable bias. Default: True
    """

    def __init__(
        self,
        normalized_shape,
        eps: float = 1e-6,
        elementwise_affine: bool = True,
        bias: bool = True,
        device=None,
        dtype=None,
    ):
        super().__init__()
        if isinstance(normalized_shape, int):
            normalized_shape = (normalized_shape,)
        self.normalized_shape = tuple(normalized_shape)
        self.eps = eps
        self.elementwise_affine = elementwise_affine
        self.use_bias = bias

        if elementwise_affine:
            self.weight = nn.Parameter(
                torch.ones(self.normalized_shape, device=device, dtype=dtype)
            )
            self.bias = nn.Parameter(
                torch.zeros(self.normalized_shape, device=device, dtype=dtype)
            ) if bias else None
        else:
            self.register_parameter("weight", None)
            self.register_parameter("bias", None)

        self.reset_parameters()

    def reset_parameters(self):
        if self.elementwise_affine:
            nn.init.ones_(self.weight)
            if self.bias is not None:
                nn.init.zeros_(self.bias)

    def forward(
        self, input: torch.Tensor, residual: torch.Tensor
    ) -> torch.Tensor:
        original_shape = input.shape
        hidden_size = self.normalized_shape[0]

        x = input.reshape(-1, hidden_size).contiguous()
        r = residual.reshape(-1, hidden_size).contiguous()

        weight = self.weight if self.weight is not None else torch.empty(0, dtype=x.dtype, device=x.device)
        bias = self.bias if self.bias is not None else torch.empty(0, dtype=x.dtype, device=x.device)

        output = rmsnorm_ext.rmsnorm_fused_add(
            x, r, weight, bias, self.eps, self.elementwise_affine
        )

        return output.view(original_shape)

    def extra_repr(self) -> str:
        return (
            f"{self.normalized_shape}, eps={self.eps}, "
            f"elementwise_affine={self.elementwise_affine}"
        )

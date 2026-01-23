"""
LSQ (Learned Step Size Quantization) Implementation

This module implements LSQ quantization following the paper:
"Learned Step Size Quantization" by Esser et al. (ICLR 2020)

Key features:
- Learnable step size parameter for each quantizer
- Gradient scaling for stable training
- Support for both weight and activation quantization
"""

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function


class LSQQuantizeFunction(Function):
    """
    Custom autograd function for LSQ quantization with STE (Straight-Through Estimator).

    Forward: Quantize input using learned step size
    Backward: Pass gradients through (STE) with proper gradient for step size
    """

    @staticmethod
    def forward(
        ctx,
        x: torch.Tensor,
        step_size: torch.Tensor,
        Q_N: int,
        Q_P: int,
        grad_scale: float
    ) -> torch.Tensor:
        """
        Forward pass: quantize input.

        Args:
            x: Input tensor
            step_size: Learned step size parameter
            Q_N: Negative quantization bound (e.g., 0 for unsigned, 128 for signed)
            Q_P: Positive quantization bound (e.g., 255 for unsigned, 127 for signed)
            grad_scale: Gradient scaling factor

        Returns:
            Quantized tensor (fake quantization - still in float)
        """
        # Quantize: round(clip(x/s, -Q_N, Q_P)) * s
        x_scaled = x / step_size
        x_clipped = torch.clamp(x_scaled, -Q_N, Q_P)
        x_rounded = torch.round(x_clipped)
        x_quant = x_rounded * step_size

        # Save for backward
        ctx.save_for_backward(x_scaled, step_size)
        ctx.Q_N = Q_N
        ctx.Q_P = Q_P
        ctx.grad_scale = grad_scale
        ctx.step_size_shape = step_size.shape

        return x_quant

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> Tuple[torch.Tensor, ...]:
        """
        Backward pass: compute gradients for input and step size.

        The gradient for step size follows Equation 3 from LSQ paper:
        ∂L/∂s = ∂L/∂v_hat * ∂v_hat/∂s

        where ∂v_hat/∂s = {
            -v/s + round(v/s)  if -Q_N < v/s < Q_P
            -Q_N               if v/s <= -Q_N
            Q_P                if v/s >= Q_P
        }
        """
        x_scaled, step_size = ctx.saved_tensors
        Q_N = ctx.Q_N
        Q_P = ctx.Q_P
        grad_scale = ctx.grad_scale
        step_size_shape = ctx.step_size_shape

        # Gradient for input (STE with clamping)
        # Pass gradient only for values within quantization range
        within_range = (x_scaled >= -Q_N) & (x_scaled <= Q_P)
        grad_x = grad_output * within_range.float()

        # Gradient for step size (Equation 3 from LSQ paper)
        below_range = x_scaled < -Q_N
        above_range = x_scaled > Q_P

        grad_step = torch.zeros_like(grad_output)
        # Within range: -v/s + round(v/s)
        grad_step = torch.where(
            within_range,
            -x_scaled + torch.round(x_scaled),
            grad_step
        )
        # Below range: -Q_N
        grad_step = torch.where(below_range, torch.full_like(grad_step, -Q_N), grad_step)
        # Above range: Q_P
        grad_step = torch.where(above_range, torch.full_like(grad_step, Q_P), grad_step)

        # Compute gradient for step size
        grad_combined = grad_output * grad_step

        # Handle per-channel quantization: sum over appropriate dimensions
        if step_size_shape != torch.Size([]):
            # Per-channel: step_size has shape like [C, 1, 1, 1] or [C, 1]
            # Sum over all dimensions except the channel dimension
            if grad_combined.dim() == 4:
                # For conv weights [out_ch, in_ch, H, W] or activations [B, C, H, W]
                if step_size_shape[0] == grad_combined.shape[0]:
                    # Weight quantization: sum over [in_ch, H, W]
                    grad_step_size = grad_combined.sum(dim=[1, 2, 3], keepdim=True) * grad_scale
                else:
                    # Activation quantization: sum over [B, H, W]
                    grad_step_size = grad_combined.sum(dim=[0, 2, 3], keepdim=True).view(step_size_shape) * grad_scale
            elif grad_combined.dim() == 2:
                # For linear weights [out, in]
                grad_step_size = grad_combined.sum(dim=1, keepdim=True) * grad_scale
            else:
                grad_step_size = grad_combined.sum() * grad_scale
                grad_step_size = grad_step_size.view(step_size_shape)
        else:
            # Scalar step_size: sum over all dimensions
            grad_step_size = grad_combined.sum() * grad_scale

        return grad_x, grad_step_size, None, None, None


class LSQQuantizer(nn.Module):
    """
    LSQ Quantizer module.

    Quantizes inputs using a learned step size parameter.

    Args:
        bits: Number of quantization bits (default: 8)
        is_activation: Whether this is for activation (unsigned) or weight (signed)
        per_channel: Whether to use per-channel quantization for weights
        num_channels: Number of channels for per-channel quantization
    """

    def __init__(
        self,
        bits: int = 8,
        is_activation: bool = True,
        per_channel: bool = False,
        num_channels: int = 1
    ):
        super().__init__()

        self.bits = bits
        self.is_activation = is_activation
        self.per_channel = per_channel
        self.num_channels = num_channels

        # Set quantization bounds
        if is_activation:
            # Unsigned quantization for activations
            self.Q_N = 0
            self.Q_P = 2 ** bits - 1  # 255 for 8-bit
        else:
            # Signed quantization for weights
            self.Q_N = 2 ** (bits - 1)  # 128 for 8-bit
            self.Q_P = 2 ** (bits - 1) - 1  # 127 for 8-bit

        # Learnable step size
        if per_channel:
            self.step_size = nn.Parameter(torch.ones(num_channels))
        else:
            self.step_size = nn.Parameter(torch.tensor(1.0))

        # Flag to indicate if step size has been initialized
        # Use register_buffer so it's saved with the model state but not a parameter
        self.register_buffer('initialized', torch.tensor(False))

    def init_step_size(self, x: torch.Tensor):
        """
        Initialize step size based on input statistics.

        For weights: s = 2 * mean(|W|) / sqrt(Q_P)
        For activations: s = 2 * mean(X) / sqrt(Q_P) (using running average)
        """
        with torch.no_grad():
            if self.is_activation:
                # For activations: use mean of tensor
                if self.per_channel:
                    # Per-channel: mean over all dims except channel
                    mean_val = x.abs().mean(dim=[0, 2, 3]) if x.dim() == 4 else x.abs().mean(dim=0)
                else:
                    mean_val = x.abs().mean()
            else:
                # For weights: use mean of absolute values
                if self.per_channel:
                    # Per-channel: mean over all dims except output channel
                    mean_val = x.abs().mean(dim=[1, 2, 3]) if x.dim() == 4 else x.abs().mean(dim=1)
                else:
                    mean_val = x.abs().mean()

            # Initialize: s = 2 * mean / sqrt(Q_P)
            init_val = 2 * mean_val / math.sqrt(self.Q_P)

            # Ensure minimum value (use torch.clamp for export compatibility)
            if self.per_channel:
                init_val = torch.clamp(init_val, min=1e-6)
            else:
                # Use torch.clamp instead of Python max() for export compatibility
                init_val = torch.clamp(init_val, min=1e-6)

            self.step_size.data.copy_(init_val if init_val.dim() > 0 else init_val.reshape([]))

        self.initialized.fill_(True)

    def compute_grad_scale(self, x: torch.Tensor) -> float:
        """
        Compute gradient scaling factor.

        From LSQ paper: g = 1 / sqrt(N * Q_P)
        where N is the number of elements being quantized.
        """
        if self.is_activation:
            # For activations: N = batch_size * height * width * channels
            n = x.numel()
        else:
            # For weights: N = number of weight elements
            n = x.numel()

        return 1.0 / math.sqrt(n * self.Q_P)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass: quantize input.

        Args:
            x: Input tensor

        Returns:
            Quantized tensor (fake quantization)
        """
        # Initialize step size on first forward pass (only during training)
        # During export/inference, step_size should already be set from training
        if self.training:
            if not self.initialized.item():
                self.init_step_size(x)

        # Get step size (expand for per-channel)
        if self.per_channel:
            if x.dim() == 4:
                # Conv weights: [out_ch, in_ch, H, W]
                if not self.is_activation:
                    step_size = self.step_size.view(-1, 1, 1, 1)
                # Activations: [B, C, H, W]
                else:
                    step_size = self.step_size.view(1, -1, 1, 1)
            else:
                step_size = self.step_size.view(-1, 1)
        else:
            step_size = self.step_size

        # Apply quantization
        if self.training:
            # During training, use custom autograd function for proper gradient computation
            grad_scale = self.compute_grad_scale(x)
            return LSQQuantizeFunction.apply(x, step_size, self.Q_N, self.Q_P, grad_scale)
        else:
            # During inference/export, use simple tensor operations (export-friendly)
            return self._quantize_inference(x, step_size)

    def _quantize_inference(self, x: torch.Tensor, step_size: torch.Tensor) -> torch.Tensor:
        """
        Quantization for inference mode (export-friendly, no custom autograd).

        This performs the same computation as the forward pass but without
        custom autograd function, making it compatible with torch.export.
        """
        x_scaled = x / step_size
        x_clipped = torch.clamp(x_scaled, -self.Q_N, self.Q_P)
        x_rounded = torch.round(x_clipped)
        x_quant = x_rounded * step_size
        return x_quant

    def extra_repr(self) -> str:
        return (
            f'bits={self.bits}, is_activation={self.is_activation}, '
            f'per_channel={self.per_channel}, Q_N={self.Q_N}, Q_P={self.Q_P}'
        )


class QuantizedConv2d(nn.Module):
    """
    Quantized 2D Convolution layer with LSQ.

    Applies LSQ quantization to both weights and activations.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        padding: int = 0,
        dilation: int = 1,
        groups: int = 1,
        bias: bool = False,
        bits: int = 8,
        per_channel_weight: bool = True
    ):
        super().__init__()

        self.conv = nn.Conv2d(
            in_channels, out_channels, kernel_size,
            stride=stride, padding=padding, dilation=dilation,
            groups=groups, bias=bias
        )

        # Weight quantizer (signed, optionally per-channel)
        self.weight_quantizer = LSQQuantizer(
            bits=bits,
            is_activation=False,
            per_channel=per_channel_weight,
            num_channels=out_channels
        )

        # Activation quantizer (unsigned)
        self.act_quantizer = LSQQuantizer(
            bits=bits,
            is_activation=True,
            per_channel=False
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Quantize input activation
        x_quant = self.act_quantizer(x)

        # Quantize weights
        w_quant = self.weight_quantizer(self.conv.weight)

        # Perform convolution with quantized values
        return F.conv2d(
            x_quant, w_quant, self.conv.bias,
            self.conv.stride, self.conv.padding,
            self.conv.dilation, self.conv.groups
        )


class QuantizedLinear(nn.Module):
    """
    Quantized Linear layer with LSQ.

    Applies LSQ quantization to both weights and activations.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        bits: int = 8,
        per_channel_weight: bool = True
    ):
        super().__init__()

        self.linear = nn.Linear(in_features, out_features, bias=bias)

        # Weight quantizer
        self.weight_quantizer = LSQQuantizer(
            bits=bits,
            is_activation=False,
            per_channel=per_channel_weight,
            num_channels=out_features
        )

        # Activation quantizer
        self.act_quantizer = LSQQuantizer(
            bits=bits,
            is_activation=True,
            per_channel=False
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Quantize input activation
        x_quant = self.act_quantizer(x)

        # Quantize weights
        w_quant = self.weight_quantizer(self.linear.weight)

        # Perform linear operation
        return F.linear(x_quant, w_quant, self.linear.bias)


def quantize_model(
    model: nn.Module,
    bits: int = 8,
    exclude_layers: Optional[list] = None
) -> nn.Module:
    """
    Convert a model to use LSQ quantization.

    Replaces Conv2d and Linear layers with their quantized versions,
    except for layers in exclude_layers.

    Args:
        model: Original model
        bits: Quantization bits
        exclude_layers: List of layer names to exclude from quantization

    Returns:
        Quantized model
    """
    if exclude_layers is None:
        exclude_layers = []

    def should_quantize(name: str) -> bool:
        for excluded in exclude_layers:
            if excluded in name:
                return False
        return True

    def replace_layers(module: nn.Module, prefix: str = ''):
        for name, child in module.named_children():
            full_name = f"{prefix}.{name}" if prefix else name

            if isinstance(child, nn.Conv2d) and should_quantize(full_name):
                # Replace with quantized conv
                new_layer = QuantizedConv2d(
                    child.in_channels, child.out_channels,
                    child.kernel_size[0],
                    stride=child.stride[0],
                    padding=child.padding[0],
                    dilation=child.dilation[0],
                    groups=child.groups,
                    bias=child.bias is not None,
                    bits=bits
                )
                # Copy weights
                new_layer.conv.weight.data.copy_(child.weight.data)
                if child.bias is not None:
                    new_layer.conv.bias.data.copy_(child.bias.data)
                setattr(module, name, new_layer)

            elif isinstance(child, nn.Linear) and should_quantize(full_name):
                # Replace with quantized linear
                new_layer = QuantizedLinear(
                    child.in_features, child.out_features,
                    bias=child.bias is not None,
                    bits=bits
                )
                # Copy weights
                new_layer.linear.weight.data.copy_(child.weight.data)
                if child.bias is not None:
                    new_layer.linear.bias.data.copy_(child.bias.data)
                setattr(module, name, new_layer)

            else:
                # Recursively replace in children
                replace_layers(child, full_name)

    replace_layers(model)
    return model


if __name__ == "__main__":
    # Test LSQ quantizer
    print("Testing LSQ Quantizer...")

    # Test weight quantizer (signed)
    weight_q = LSQQuantizer(bits=8, is_activation=False)
    weights = torch.randn(64, 64, 3, 3)
    weights_quant = weight_q(weights)
    print(f"Weight quantizer - step_size: {weight_q.step_size.item():.6f}")
    print(f"Original weights range: [{weights.min():.4f}, {weights.max():.4f}]")
    print(f"Quantized weights range: [{weights_quant.min():.4f}, {weights_quant.max():.4f}]")

    # Test activation quantizer (unsigned)
    act_q = LSQQuantizer(bits=8, is_activation=True)
    activations = torch.relu(torch.randn(2, 64, 32, 32))  # ReLU ensures non-negative
    act_quant = act_q(activations)
    print(f"\nActivation quantizer - step_size: {act_q.step_size.item():.6f}")
    print(f"Original activations range: [{activations.min():.4f}, {activations.max():.4f}]")
    print(f"Quantized activations range: [{act_quant.min():.4f}, {act_quant.max():.4f}]")

    # Test gradient flow
    print("\nTesting gradient flow...")
    x = torch.randn(2, 3, 32, 32, requires_grad=True)
    quantizer = LSQQuantizer(bits=8, is_activation=True)
    y = quantizer(x)
    loss = y.sum()
    loss.backward()
    print(f"Input gradient exists: {x.grad is not None}")
    print(f"Step size gradient: {quantizer.step_size.grad}")

    # Test quantized conv
    print("\nTesting QuantizedConv2d...")
    qconv = QuantizedConv2d(3, 64, 3, padding=1, bits=8)
    x = torch.randn(2, 3, 32, 32)
    y = qconv(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {y.shape}")

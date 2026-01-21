"""
ResNet-18 Implementation for Multi-Label Classification

This module implements ResNet-18 architecture following the original paper
"Deep Residual Learning for Image Recognition" by He et al.

The model is adapted for multi-label classification with 80 COCO categories,
using sigmoid activation instead of softmax.
"""

from typing import Optional, List, Type

import torch
import torch.nn as nn
import torchvision.models as models


def conv3x3(
    in_planes: int,
    out_planes: int,
    stride: int = 1,
    groups: int = 1,
    dilation: int = 1
) -> nn.Conv2d:
    """3x3 convolution with padding."""
    return nn.Conv2d(
        in_planes,
        out_planes,
        kernel_size=3,
        stride=stride,
        padding=dilation,
        groups=groups,
        bias=False,
        dilation=dilation
    )


def conv1x1(in_planes: int, out_planes: int, stride: int = 1) -> nn.Conv2d:
    """1x1 convolution."""
    return nn.Conv2d(
        in_planes,
        out_planes,
        kernel_size=1,
        stride=stride,
        bias=False
    )


class BasicBlock(nn.Module):
    """
    Basic residual block for ResNet-18/34.

    Architecture:
        x -> Conv3x3 -> BN -> ReLU -> Conv3x3 -> BN -> (+x) -> ReLU

    If input and output dimensions differ, a 1x1 conv is used for the shortcut.
    """

    expansion = 1

    def __init__(
        self,
        in_planes: int,
        planes: int,
        stride: int = 1,
        downsample: Optional[nn.Module] = None,
        groups: int = 1,
        base_width: int = 64,
        dilation: int = 1,
        norm_layer: Optional[Type[nn.Module]] = None
    ):
        super().__init__()

        if norm_layer is None:
            norm_layer = nn.BatchNorm2d

        if groups != 1 or base_width != 64:
            raise ValueError("BasicBlock only supports groups=1 and base_width=64")
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")

        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(in_planes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class ResNet(nn.Module):
    """
    ResNet implementation.

    Architecture for ResNet-18:
        - conv1: 7x7, 64, stride 2
        - maxpool: 3x3, stride 2
        - layer1: 2 x BasicBlock (64 channels)
        - layer2: 2 x BasicBlock (128 channels, stride 2)
        - layer3: 2 x BasicBlock (256 channels, stride 2)
        - layer4: 2 x BasicBlock (512 channels, stride 2)
        - avgpool: global average pooling
        - fc: 512 -> num_classes
    """

    def __init__(
        self,
        block: Type[BasicBlock],
        layers: List[int],
        num_classes: int = 80,
        zero_init_residual: bool = False,
        groups: int = 1,
        width_per_group: int = 64,
        norm_layer: Optional[Type[nn.Module]] = None
    ):
        super().__init__()

        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        self.in_planes = 64
        self.dilation = 1
        self.groups = groups
        self.base_width = width_per_group

        # Initial convolution layer
        self.conv1 = nn.Conv2d(
            3, self.in_planes,
            kernel_size=7,
            stride=2,
            padding=3,
            bias=False
        )
        self.bn1 = norm_layer(self.in_planes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # Residual layers
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)

        # Classifier
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        # Weight initialization
        self._initialize_weights(zero_init_residual)

    def _make_layer(
        self,
        block: Type[BasicBlock],
        planes: int,
        blocks: int,
        stride: int = 1
    ) -> nn.Sequential:
        """Create a residual layer with multiple blocks."""
        norm_layer = self._norm_layer
        downsample = None

        # Downsample if stride != 1 or input channels != output channels
        if stride != 1 or self.in_planes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.in_planes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(
            block(
                self.in_planes,
                planes,
                stride,
                downsample,
                self.groups,
                self.base_width,
                self.dilation,
                norm_layer
            )
        )
        self.in_planes = planes * block.expansion

        for _ in range(1, blocks):
            layers.append(
                block(
                    self.in_planes,
                    planes,
                    groups=self.groups,
                    base_width=self.base_width,
                    dilation=self.dilation,
                    norm_layer=norm_layer
                )
            )

        return nn.Sequential(*layers)

    def _initialize_weights(self, zero_init_residual: bool):
        """Initialize model weights."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, BasicBlock) and m.bn2.weight is not None:
                    nn.init.constant_(m.bn2.weight, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input tensor of shape (B, 3, H, W)

        Returns:
            Logits tensor of shape (B, num_classes)
        """
        # Initial layers
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        # Residual layers
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        # Classifier
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x

    def get_features(self, x: torch.Tensor) -> torch.Tensor:
        """Extract features before the classifier layer."""
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)

        return x


def resnet18(num_classes: int = 80, pretrained: bool = True) -> ResNet:
    """
    Create ResNet-18 model for multi-label classification.

    Args:
        num_classes: Number of output classes (default: 80 for COCO)
        pretrained: Whether to load ImageNet pretrained weights

    Returns:
        ResNet-18 model
    """
    model = ResNet(BasicBlock, [2, 2, 2, 2], num_classes=num_classes)

    if pretrained:
        # Load pretrained weights from torchvision
        pretrained_model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        pretrained_dict = pretrained_model.state_dict()

        # Remove fc layer weights (different number of classes)
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if 'fc' not in k}

        # Load pretrained weights
        model_dict = model.state_dict()
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)

        print("Loaded ImageNet pretrained weights for ResNet-18")

    return model


def count_parameters(model: nn.Module) -> int:
    """Count the number of trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def get_model_size_mb(model: nn.Module) -> float:
    """Get model size in megabytes."""
    param_size = 0
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()

    buffer_size = 0
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()

    size_all_mb = (param_size + buffer_size) / 1024**2
    return size_all_mb


if __name__ == "__main__":
    # Test the model
    model = resnet18(num_classes=80, pretrained=False)

    print(f"Model architecture:\n{model}")
    print(f"\nNumber of parameters: {count_parameters(model):,}")
    print(f"Model size: {get_model_size_mb(model):.2f} MB")

    # Test forward pass
    dummy_input = torch.randn(2, 3, 224, 224)
    output = model(dummy_input)
    print(f"\nInput shape: {dummy_input.shape}")
    print(f"Output shape: {output.shape}")

    # Test feature extraction
    features = model.get_features(dummy_input)
    print(f"Features shape: {features.shape}")

    # Test with sigmoid for multi-label
    probs = torch.sigmoid(output)
    print(f"Probabilities range: [{probs.min():.4f}, {probs.max():.4f}]")

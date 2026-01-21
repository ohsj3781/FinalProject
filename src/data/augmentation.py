"""
Data Augmentation Module

This module provides data augmentation pipelines for training and validation
using torchvision transforms.
"""

from typing import Dict, Any, Tuple

import torch
import torchvision.transforms as T
from torchvision.transforms import InterpolationMode


class TrainTransform:
    """
    Training data augmentation pipeline.

    Applies the following augmentations:
    - Resize to 256
    - Random crop to 224
    - Random horizontal flip
    - Color jitter (brightness, contrast, saturation, hue)
    - Normalize with ImageNet statistics
    """

    def __init__(
        self,
        resize: int = 256,
        crop_size: int = 224,
        horizontal_flip_prob: float = 0.5,
        color_jitter: Dict[str, float] = None,
        mean: Tuple[float, ...] = (0.485, 0.456, 0.406),
        std: Tuple[float, ...] = (0.229, 0.224, 0.225)
    ):
        if color_jitter is None:
            color_jitter = {
                'brightness': 0.4,
                'contrast': 0.4,
                'saturation': 0.4,
                'hue': 0.1
            }

        transforms = [
            T.Resize(resize, interpolation=InterpolationMode.BILINEAR),
            T.RandomCrop(crop_size),
            T.RandomHorizontalFlip(p=horizontal_flip_prob),
            T.ColorJitter(
                brightness=color_jitter.get('brightness', 0),
                contrast=color_jitter.get('contrast', 0),
                saturation=color_jitter.get('saturation', 0),
                hue=color_jitter.get('hue', 0)
            ),
            T.ToTensor(),
            T.Normalize(mean=mean, std=std)
        ]

        self.transform = T.Compose(transforms)

    def __call__(self, image):
        return self.transform(image)


class ValTransform:
    """
    Validation data transform pipeline.

    Applies the following transforms:
    - Resize to 256
    - Center crop to 224
    - Normalize with ImageNet statistics
    """

    def __init__(
        self,
        resize: int = 256,
        crop_size: int = 224,
        mean: Tuple[float, ...] = (0.485, 0.456, 0.406),
        std: Tuple[float, ...] = (0.229, 0.224, 0.225)
    ):
        self.transform = T.Compose([
            T.Resize(resize, interpolation=InterpolationMode.BILINEAR),
            T.CenterCrop(crop_size),
            T.ToTensor(),
            T.Normalize(mean=mean, std=std)
        ])

    def __call__(self, image):
        return self.transform(image)


class InferenceTransform:
    """
    Inference transform for deployment.

    Applies minimal preprocessing for inference:
    - Resize to target size
    - Normalize with ImageNet statistics
    """

    def __init__(
        self,
        size: int = 224,
        mean: Tuple[float, ...] = (0.485, 0.456, 0.406),
        std: Tuple[float, ...] = (0.229, 0.224, 0.225)
    ):
        self.transform = T.Compose([
            T.Resize((size, size), interpolation=InterpolationMode.BILINEAR),
            T.ToTensor(),
            T.Normalize(mean=mean, std=std)
        ])

    def __call__(self, image):
        return self.transform(image)


def get_transforms_from_config(config: Dict[str, Any]) -> Tuple[TrainTransform, ValTransform]:
    """
    Create train and validation transforms from config dictionary.

    Args:
        config: Configuration dictionary with augmentation settings

    Returns:
        tuple: (train_transform, val_transform)
    """
    aug_config = config.get('augmentation', {})
    train_config = aug_config.get('train', {})
    val_config = aug_config.get('val', {})

    # Get normalization parameters
    normalize = train_config.get('normalize', {})
    mean = tuple(normalize.get('mean', [0.485, 0.456, 0.406]))
    std = tuple(normalize.get('std', [0.229, 0.224, 0.225]))

    # Create train transform
    train_transform = TrainTransform(
        resize=train_config.get('resize', 256),
        crop_size=train_config.get('random_crop', 224),
        horizontal_flip_prob=train_config.get('horizontal_flip', 0.5),
        color_jitter=train_config.get('color_jitter', None),
        mean=mean,
        std=std
    )

    # Create validation transform
    val_transform = ValTransform(
        resize=val_config.get('resize', 256),
        crop_size=val_config.get('center_crop', 224),
        mean=mean,
        std=std
    )

    return train_transform, val_transform


def denormalize(
    tensor: torch.Tensor,
    mean: Tuple[float, ...] = (0.485, 0.456, 0.406),
    std: Tuple[float, ...] = (0.229, 0.224, 0.225)
) -> torch.Tensor:
    """
    Denormalize a tensor image with mean and standard deviation.

    Args:
        tensor: Normalized tensor image of shape (C, H, W) or (B, C, H, W)
        mean: Mean used for normalization
        std: Std used for normalization

    Returns:
        Denormalized tensor
    """
    mean = torch.tensor(mean).view(-1, 1, 1)
    std = torch.tensor(std).view(-1, 1, 1)

    if tensor.dim() == 4:
        mean = mean.unsqueeze(0)
        std = std.unsqueeze(0)

    return tensor * std + mean


if __name__ == "__main__":
    # Test transforms
    from PIL import Image
    import numpy as np

    # Create a dummy image
    dummy_image = Image.fromarray(np.random.randint(0, 255, (300, 400, 3), dtype=np.uint8))

    # Test train transform
    train_transform = TrainTransform()
    train_output = train_transform(dummy_image)
    print(f"Train transform output shape: {train_output.shape}")
    print(f"Train transform output range: [{train_output.min():.3f}, {train_output.max():.3f}]")

    # Test val transform
    val_transform = ValTransform()
    val_output = val_transform(dummy_image)
    print(f"Val transform output shape: {val_output.shape}")
    print(f"Val transform output range: [{val_output.min():.3f}, {val_output.max():.3f}]")

    # Test inference transform
    inference_transform = InferenceTransform()
    inference_output = inference_transform(dummy_image)
    print(f"Inference transform output shape: {inference_output.shape}")

    # Test denormalization
    denorm = denormalize(val_output)
    print(f"Denormalized range: [{denorm.min():.3f}, {denorm.max():.3f}]")

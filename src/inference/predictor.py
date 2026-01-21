"""
Inference Module for Multi-Label Classification

This module provides inference functionality for both PyTorch and ExecuTorch models.
"""

import os
import time
from typing import List, Tuple, Optional, Union

import numpy as np
import torch
import torch.nn as nn
from PIL import Image

from src.data.augmentation import InferenceTransform


# COCO category names
COCO_CATEGORIES = [
    'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck',
    'boat', 'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench',
    'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra',
    'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
    'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove',
    'skateboard', 'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup',
    'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange',
    'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
    'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse',
    'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink',
    'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier',
    'toothbrush'
]


class PyTorchPredictor:
    """
    Predictor class for PyTorch models.

    Args:
        model: PyTorch model
        device: Device to run inference on
        threshold: Classification threshold
        transform: Image transform (default: InferenceTransform)
    """

    def __init__(
        self,
        model: nn.Module,
        device: torch.device = None,
        threshold: float = 0.5,
        transform: Optional[object] = None
    ):
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = model.to(self.device)
        self.model.eval()
        self.threshold = threshold
        self.transform = transform or InferenceTransform()
        self.categories = COCO_CATEGORIES

    @torch.no_grad()
    def predict(
        self,
        image: Union[str, Image.Image, np.ndarray, torch.Tensor]
    ) -> Tuple[List[str], List[float], float]:
        """
        Predict tags for a single image.

        Args:
            image: Input image (path, PIL Image, numpy array, or tensor)

        Returns:
            Tuple of (tags, probabilities, inference_time_ms)
        """
        # Preprocess image
        if isinstance(image, str):
            image = Image.open(image).convert('RGB')
        elif isinstance(image, np.ndarray):
            image = Image.fromarray(image)

        if isinstance(image, Image.Image):
            input_tensor = self.transform(image)

        if isinstance(input_tensor, torch.Tensor):
            if input_tensor.dim() == 3:
                input_tensor = input_tensor.unsqueeze(0)

        input_tensor = input_tensor.to(self.device)

        # Run inference
        start_time = time.time()
        outputs = self.model(input_tensor)
        inference_time = (time.time() - start_time) * 1000  # ms

        # Post-process
        probs = torch.sigmoid(outputs).squeeze().cpu().numpy()

        # Get tags above threshold
        tags = []
        tag_probs = []

        for i, prob in enumerate(probs):
            if prob >= self.threshold:
                tags.append(self.categories[i])
                tag_probs.append(float(prob))

        # Sort by probability
        sorted_indices = np.argsort(tag_probs)[::-1]
        tags = [tags[i] for i in sorted_indices]
        tag_probs = [tag_probs[i] for i in sorted_indices]

        return tags, tag_probs, inference_time

    @torch.no_grad()
    def predict_batch(
        self,
        images: List[Union[str, Image.Image]]
    ) -> List[Tuple[List[str], List[float]]]:
        """
        Predict tags for a batch of images.

        Args:
            images: List of input images

        Returns:
            List of (tags, probabilities) tuples
        """
        # Preprocess all images
        tensors = []
        for image in images:
            if isinstance(image, str):
                image = Image.open(image).convert('RGB')
            tensor = self.transform(image)
            tensors.append(tensor)

        batch_tensor = torch.stack(tensors).to(self.device)

        # Run inference
        outputs = self.model(batch_tensor)
        probs = torch.sigmoid(outputs).cpu().numpy()

        # Post-process each image
        results = []
        for prob in probs:
            tags = []
            tag_probs = []

            for i, p in enumerate(prob):
                if p >= self.threshold:
                    tags.append(self.categories[i])
                    tag_probs.append(float(p))

            # Sort by probability
            sorted_indices = np.argsort(tag_probs)[::-1]
            tags = [tags[i] for i in sorted_indices]
            tag_probs = [tag_probs[i] for i in sorted_indices]

            results.append((tags, tag_probs))

        return results

    def get_top_k(
        self,
        image: Union[str, Image.Image, np.ndarray],
        k: int = 5
    ) -> Tuple[List[str], List[float]]:
        """
        Get top-k predicted tags regardless of threshold.

        Args:
            image: Input image
            k: Number of top predictions

        Returns:
            Tuple of (tags, probabilities)
        """
        # Preprocess image
        if isinstance(image, str):
            image = Image.open(image).convert('RGB')
        elif isinstance(image, np.ndarray):
            image = Image.fromarray(image)

        input_tensor = self.transform(image).unsqueeze(0).to(self.device)

        # Run inference
        with torch.no_grad():
            outputs = self.model(input_tensor)
            probs = torch.sigmoid(outputs).squeeze().cpu().numpy()

        # Get top-k
        top_k_indices = np.argsort(probs)[-k:][::-1]
        tags = [self.categories[i] for i in top_k_indices]
        tag_probs = [float(probs[i]) for i in top_k_indices]

        return tags, tag_probs


class ExecuTorchPredictor:
    """
    Predictor class for ExecuTorch models.

    Args:
        model_path: Path to .pte model file
        threshold: Classification threshold
        transform: Image transform (default: InferenceTransform)
    """

    def __init__(
        self,
        model_path: str,
        threshold: float = 0.5,
        transform: Optional[object] = None
    ):
        self.model_path = model_path
        self.threshold = threshold
        self.transform = transform or InferenceTransform()
        self.categories = COCO_CATEGORIES

        # Try to load ExecuTorch runtime
        try:
            from executorch.runtime import Runtime, Program
            self.program = Program(model_path)
            self.method = self.program.load_method("forward")
            self.use_executorch = True
            print(f"Loaded ExecuTorch model: {model_path}")
        except ImportError:
            print("Warning: ExecuTorch runtime not available")
            print("Falling back to simulation mode")
            self.use_executorch = False

    def _sigmoid(self, x: np.ndarray) -> np.ndarray:
        """Numpy sigmoid function."""
        return 1.0 / (1.0 + np.exp(-x))

    def predict(
        self,
        image: Union[str, Image.Image, np.ndarray]
    ) -> Tuple[List[str], List[float], float]:
        """
        Predict tags for a single image.

        Args:
            image: Input image

        Returns:
            Tuple of (tags, probabilities, inference_time_ms)
        """
        # Preprocess image
        if isinstance(image, str):
            image = Image.open(image).convert('RGB')
        elif isinstance(image, np.ndarray):
            image = Image.fromarray(image)

        input_tensor = self.transform(image)
        if isinstance(input_tensor, torch.Tensor):
            input_array = input_tensor.unsqueeze(0).numpy()
        else:
            input_array = np.expand_dims(input_tensor, 0)

        # Run inference
        start_time = time.time()

        if self.use_executorch:
            outputs = self.method.execute([input_array])
            output_array = outputs[0]
        else:
            # Simulation mode: return random predictions
            print("Warning: Using simulation mode")
            output_array = np.random.randn(1, 80)

        inference_time = (time.time() - start_time) * 1000  # ms

        # Post-process
        probs = self._sigmoid(output_array.squeeze())

        # Get tags above threshold
        tags = []
        tag_probs = []

        for i, prob in enumerate(probs):
            if prob >= self.threshold:
                tags.append(self.categories[i])
                tag_probs.append(float(prob))

        # Sort by probability
        sorted_indices = np.argsort(tag_probs)[::-1]
        tags = [tags[i] for i in sorted_indices]
        tag_probs = [tag_probs[i] for i in sorted_indices]

        return tags, tag_probs, inference_time


def create_predictor(
    model_or_path: Union[nn.Module, str],
    threshold: float = 0.5,
    device: torch.device = None
) -> Union[PyTorchPredictor, ExecuTorchPredictor]:
    """
    Factory function to create appropriate predictor.

    Args:
        model_or_path: PyTorch model or path to .pte file
        threshold: Classification threshold
        device: Device for PyTorch model

    Returns:
        Predictor instance
    """
    if isinstance(model_or_path, str):
        if model_or_path.endswith('.pte'):
            return ExecuTorchPredictor(model_or_path, threshold=threshold)
        else:
            raise ValueError(f"Unknown model format: {model_or_path}")
    elif isinstance(model_or_path, nn.Module):
        return PyTorchPredictor(model_or_path, device=device, threshold=threshold)
    else:
        raise ValueError(f"Invalid model type: {type(model_or_path)}")


def benchmark_inference(
    predictor: Union[PyTorchPredictor, ExecuTorchPredictor],
    image_path: str,
    num_runs: int = 100,
    warmup_runs: int = 10
) -> dict:
    """
    Benchmark inference performance.

    Args:
        predictor: Predictor instance
        image_path: Path to test image
        num_runs: Number of benchmark runs
        warmup_runs: Number of warmup runs

    Returns:
        Dictionary with benchmark results
    """
    image = Image.open(image_path).convert('RGB')

    # Warmup
    for _ in range(warmup_runs):
        predictor.predict(image)

    # Benchmark
    times = []
    for _ in range(num_runs):
        _, _, time_ms = predictor.predict(image)
        times.append(time_ms)

    times = np.array(times)

    return {
        'mean_ms': float(np.mean(times)),
        'std_ms': float(np.std(times)),
        'min_ms': float(np.min(times)),
        'max_ms': float(np.max(times)),
        'median_ms': float(np.median(times)),
        'p95_ms': float(np.percentile(times, 95)),
        'p99_ms': float(np.percentile(times, 99))
    }


if __name__ == "__main__":
    # Test predictor
    import sys
    from pathlib import Path

    project_root = Path(__file__).parent.parent.parent
    sys.path.insert(0, str(project_root))

    from src.models.resnet import resnet18

    # Create model
    model = resnet18(num_classes=80, pretrained=False)

    # Create predictor
    predictor = PyTorchPredictor(model, threshold=0.3)

    # Create dummy image
    dummy_image = Image.fromarray(np.random.randint(0, 255, (300, 400, 3), dtype=np.uint8))

    # Test prediction
    tags, probs, time_ms = predictor.predict(dummy_image)
    print(f"Predicted tags: {tags[:5]}")
    print(f"Probabilities: {[f'{p:.3f}' for p in probs[:5]]}")
    print(f"Inference time: {time_ms:.2f} ms")

    # Test top-k
    top_tags, top_probs = predictor.get_top_k(dummy_image, k=5)
    print(f"\nTop-5 tags: {top_tags}")
    print(f"Top-5 probs: {[f'{p:.3f}' for p in top_probs]}")

"""
Evaluation Metrics for Multi-Label Classification

This module provides metrics for evaluating multi-label classification models,
including mAP, precision, recall, and F1 score.
"""

from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from sklearn.metrics import (
    average_precision_score,
    precision_score,
    recall_score,
    f1_score
)


class MultiLabelMetrics:
    """
    Metrics calculator for multi-label classification.

    Computes mAP, precision, recall, and F1 score at a given threshold.
    """

    def __init__(
        self,
        num_classes: int = 80,
        threshold: float = 0.5
    ):
        self.num_classes = num_classes
        self.threshold = threshold
        self.reset()

    def reset(self):
        """Reset accumulated predictions and targets."""
        self.all_preds = []
        self.all_targets = []

    def update(
        self,
        preds: torch.Tensor,
        targets: torch.Tensor
    ):
        """
        Update metrics with new batch of predictions.

        Args:
            preds: Predicted probabilities (after sigmoid), shape (B, C)
            targets: Ground truth labels, shape (B, C)
        """
        if isinstance(preds, torch.Tensor):
            preds = preds.detach().cpu().numpy()
        if isinstance(targets, torch.Tensor):
            targets = targets.detach().cpu().numpy()

        self.all_preds.append(preds)
        self.all_targets.append(targets)

    def compute(self) -> Dict[str, float]:
        """
        Compute all metrics.

        Returns:
            Dictionary containing mAP, precision, recall, F1, and per-class AP
        """
        if not self.all_preds:
            return {}

        preds = np.concatenate(self.all_preds, axis=0)
        targets = np.concatenate(self.all_targets, axis=0)

        # Compute mAP (mean Average Precision)
        mAP = self._compute_map(preds, targets)

        # Binary predictions at threshold
        binary_preds = (preds >= self.threshold).astype(np.float32)

        # Compute precision, recall, F1 (sample-averaged)
        precision = precision_score(
            targets, binary_preds, average='samples', zero_division=0
        )
        recall = recall_score(
            targets, binary_preds, average='samples', zero_division=0
        )
        f1 = f1_score(
            targets, binary_preds, average='samples', zero_division=0
        )

        # Compute macro-averaged metrics
        precision_macro = precision_score(
            targets, binary_preds, average='macro', zero_division=0
        )
        recall_macro = recall_score(
            targets, binary_preds, average='macro', zero_division=0
        )
        f1_macro = f1_score(
            targets, binary_preds, average='macro', zero_division=0
        )

        return {
            'mAP': mAP,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'precision_macro': precision_macro,
            'recall_macro': recall_macro,
            'f1_macro': f1_macro
        }

    def _compute_map(
        self,
        preds: np.ndarray,
        targets: np.ndarray
    ) -> float:
        """Compute mean Average Precision."""
        aps = []

        for i in range(self.num_classes):
            # Skip classes with no positive samples
            if targets[:, i].sum() == 0:
                continue

            ap = average_precision_score(targets[:, i], preds[:, i])
            aps.append(ap)

        if not aps:
            return 0.0

        return np.mean(aps)

    def compute_per_class_ap(self) -> Dict[int, float]:
        """Compute AP for each class."""
        if not self.all_preds:
            return {}

        preds = np.concatenate(self.all_preds, axis=0)
        targets = np.concatenate(self.all_targets, axis=0)

        per_class_ap = {}

        for i in range(self.num_classes):
            if targets[:, i].sum() == 0:
                per_class_ap[i] = 0.0
            else:
                per_class_ap[i] = average_precision_score(
                    targets[:, i], preds[:, i]
                )

        return per_class_ap


def compute_optimal_threshold(
    preds: np.ndarray,
    targets: np.ndarray,
    metric: str = 'f1'
) -> Tuple[float, float]:
    """
    Find optimal threshold that maximizes the given metric.

    Args:
        preds: Predicted probabilities, shape (N, C)
        targets: Ground truth labels, shape (N, C)
        metric: Metric to optimize ('f1', 'precision', 'recall')

    Returns:
        Tuple of (optimal_threshold, best_metric_value)
    """
    thresholds = np.arange(0.1, 0.9, 0.05)
    best_threshold = 0.5
    best_value = 0.0

    for thresh in thresholds:
        binary_preds = (preds >= thresh).astype(np.float32)

        if metric == 'f1':
            value = f1_score(targets, binary_preds, average='samples', zero_division=0)
        elif metric == 'precision':
            value = precision_score(targets, binary_preds, average='samples', zero_division=0)
        elif metric == 'recall':
            value = recall_score(targets, binary_preds, average='samples', zero_division=0)
        else:
            raise ValueError(f"Unknown metric: {metric}")

        if value > best_value:
            best_value = value
            best_threshold = thresh

    return best_threshold, best_value


class AverageMeter:
    """Computes and stores the average and current value."""

    def __init__(self, name: str = ''):
        self.name = name
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val: float, n: int = 1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self) -> str:
        return f"{self.name}: {self.avg:.4f}"


def compute_top_k_accuracy(
    preds: torch.Tensor,
    targets: torch.Tensor,
    k: int = 5
) -> float:
    """
    Compute top-k recall for multi-label classification.

    For each sample, check if any of the top-k predicted classes
    are in the ground truth labels.

    Args:
        preds: Predicted probabilities, shape (B, C)
        targets: Ground truth labels, shape (B, C)
        k: Number of top predictions to consider

    Returns:
        Top-k recall value
    """
    if isinstance(preds, torch.Tensor):
        preds = preds.detach().cpu().numpy()
    if isinstance(targets, torch.Tensor):
        targets = targets.detach().cpu().numpy()

    batch_size = preds.shape[0]
    correct = 0

    for i in range(batch_size):
        # Get indices of top-k predictions
        top_k_indices = np.argsort(preds[i])[-k:]

        # Get ground truth positive indices
        gt_indices = np.where(targets[i] == 1)[0]

        if len(gt_indices) == 0:
            continue

        # Check overlap
        overlap = len(set(top_k_indices) & set(gt_indices))
        correct += overlap / len(gt_indices)

    return correct / batch_size


if __name__ == "__main__":
    # Test metrics
    batch_size = 32
    num_classes = 80

    # Random predictions and targets
    preds = torch.sigmoid(torch.randn(batch_size, num_classes))
    targets = torch.randint(0, 2, (batch_size, num_classes)).float()

    # Test MultiLabelMetrics
    metrics = MultiLabelMetrics(num_classes=num_classes, threshold=0.5)
    metrics.update(preds, targets)
    results = metrics.compute()

    print("Multi-Label Metrics:")
    for name, value in results.items():
        print(f"  {name}: {value:.4f}")

    # Test top-k accuracy
    top5_recall = compute_top_k_accuracy(preds, targets, k=5)
    print(f"\nTop-5 Recall: {top5_recall:.4f}")

    # Test optimal threshold
    preds_np = preds.numpy()
    targets_np = targets.numpy()
    opt_thresh, opt_f1 = compute_optimal_threshold(preds_np, targets_np, 'f1')
    print(f"\nOptimal Threshold: {opt_thresh:.2f} (F1: {opt_f1:.4f})")

    # Test AverageMeter
    meter = AverageMeter('loss')
    for i in range(10):
        meter.update(np.random.random())
    print(f"\n{meter}")

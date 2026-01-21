"""
Loss Functions for Multi-Label Classification

This module provides loss functions specifically designed for multi-label
classification tasks with the COCO dataset.
"""

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class AsymmetricLoss(nn.Module):
    """
    Asymmetric Loss for Multi-Label Classification.

    This loss function handles the imbalance between positive and negative
    samples in multi-label classification by applying different focusing
    parameters to positive and negative samples.

    Reference:
        "Asymmetric Loss For Multi-Label Classification" (ICCV 2021)

    Args:
        gamma_neg: Focusing parameter for negative samples (default: 4)
        gamma_pos: Focusing parameter for positive samples (default: 1)
        clip: Probability margin for negative samples (default: 0.05)
        eps: Small constant for numerical stability
        disable_torch_grad_focal_loss: Whether to disable gradient for focal term
    """

    def __init__(
        self,
        gamma_neg: float = 4,
        gamma_pos: float = 1,
        clip: float = 0.05,
        eps: float = 1e-8,
        disable_torch_grad_focal_loss: bool = False
    ):
        super().__init__()
        self.gamma_neg = gamma_neg
        self.gamma_pos = gamma_pos
        self.clip = clip
        self.eps = eps
        self.disable_torch_grad_focal_loss = disable_torch_grad_focal_loss

    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute asymmetric loss.

        Args:
            logits: Model outputs before sigmoid, shape (B, C)
            targets: Binary targets, shape (B, C)

        Returns:
            Scalar loss value
        """
        # Sigmoid activation
        probs = torch.sigmoid(logits)

        # Separate positive and negative
        probs_pos = probs
        probs_neg = 1 - probs

        # Asymmetric clipping for negatives
        if self.clip is not None and self.clip > 0:
            probs_neg = (probs_neg + self.clip).clamp(max=1)

        # Basic cross-entropy
        loss_pos = targets * torch.log(probs_pos.clamp(min=self.eps))
        loss_neg = (1 - targets) * torch.log(probs_neg.clamp(min=self.eps))

        # Asymmetric focusing
        if self.disable_torch_grad_focal_loss:
            torch.set_grad_enabled(False)

        pt_pos = probs_pos * targets + (1 - probs_pos) * (1 - targets)
        pt_neg = probs_neg * (1 - targets) + (1 - probs_neg) * targets

        focal_weight_pos = (1 - pt_pos) ** self.gamma_pos
        focal_weight_neg = (1 - pt_neg) ** self.gamma_neg

        if self.disable_torch_grad_focal_loss:
            torch.set_grad_enabled(True)

        loss_pos = loss_pos * focal_weight_pos
        loss_neg = loss_neg * focal_weight_neg

        loss = -(loss_pos + loss_neg)

        return loss.mean()


class MultiLabelBCELoss(nn.Module):
    """
    Binary Cross-Entropy Loss for Multi-Label Classification.

    A wrapper around BCEWithLogitsLoss with optional class weighting
    to handle class imbalance.

    Args:
        pos_weight: Positive class weights for handling class imbalance
        reduction: Reduction method ('mean', 'sum', 'none')
    """

    def __init__(
        self,
        pos_weight: Optional[torch.Tensor] = None,
        reduction: str = 'mean'
    ):
        super().__init__()
        self.pos_weight = pos_weight
        self.reduction = reduction

    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute BCE loss.

        Args:
            logits: Model outputs before sigmoid, shape (B, C)
            targets: Binary targets, shape (B, C)

        Returns:
            Loss value
        """
        return F.binary_cross_entropy_with_logits(
            logits,
            targets,
            pos_weight=self.pos_weight,
            reduction=self.reduction
        )


class FocalLoss(nn.Module):
    """
    Focal Loss for Multi-Label Classification.

    Focal loss focuses learning on hard examples by down-weighting
    the contribution of easy examples.

    Reference:
        "Focal Loss for Dense Object Detection" (ICCV 2017)

    Args:
        alpha: Weighting factor for positive class
        gamma: Focusing parameter (higher = more focus on hard examples)
        reduction: Reduction method ('mean', 'sum', 'none')
    """

    def __init__(
        self,
        alpha: float = 0.25,
        gamma: float = 2.0,
        reduction: str = 'mean'
    ):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute focal loss.

        Args:
            logits: Model outputs before sigmoid, shape (B, C)
            targets: Binary targets, shape (B, C)

        Returns:
            Loss value
        """
        probs = torch.sigmoid(logits)

        # Compute focal weight
        pt = probs * targets + (1 - probs) * (1 - targets)
        focal_weight = (1 - pt) ** self.gamma

        # Compute BCE loss
        bce_loss = F.binary_cross_entropy_with_logits(
            logits,
            targets,
            reduction='none'
        )

        # Apply alpha weighting
        alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)

        # Combine
        loss = alpha_t * focal_weight * bce_loss

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss


def get_loss_function(
    loss_name: str,
    pos_weight: Optional[torch.Tensor] = None,
    **kwargs
) -> nn.Module:
    """
    Factory function to create loss functions.

    Args:
        loss_name: Name of loss function ('bce', 'focal', 'asymmetric')
        pos_weight: Optional positive class weights
        **kwargs: Additional arguments for specific loss functions

    Returns:
        Loss function module
    """
    loss_name = loss_name.lower()

    if loss_name in ('bce', 'bce_with_logits'):
        return MultiLabelBCELoss(pos_weight=pos_weight)
    elif loss_name == 'focal':
        return FocalLoss(**kwargs)
    elif loss_name == 'asymmetric':
        return AsymmetricLoss(**kwargs)
    else:
        raise ValueError(f"Unknown loss function: {loss_name}")


if __name__ == "__main__":
    # Test loss functions
    batch_size = 4
    num_classes = 80

    # Random logits and targets
    logits = torch.randn(batch_size, num_classes)
    targets = torch.randint(0, 2, (batch_size, num_classes)).float()

    # Test BCE loss
    bce_loss = MultiLabelBCELoss()
    loss_bce = bce_loss(logits, targets)
    print(f"BCE Loss: {loss_bce.item():.4f}")

    # Test Focal loss
    focal_loss = FocalLoss()
    loss_focal = focal_loss(logits, targets)
    print(f"Focal Loss: {loss_focal.item():.4f}")

    # Test Asymmetric loss
    asym_loss = AsymmetricLoss()
    loss_asym = asym_loss(logits, targets)
    print(f"Asymmetric Loss: {loss_asym.item():.4f}")

    # Test with pos_weight
    pos_weight = torch.ones(num_classes) * 2.0
    weighted_bce = MultiLabelBCELoss(pos_weight=pos_weight)
    loss_weighted = weighted_bce(logits, targets)
    print(f"Weighted BCE Loss: {loss_weighted.item():.4f}")

    # Test factory function
    loss_fn = get_loss_function('asymmetric', gamma_neg=4, gamma_pos=1)
    loss_factory = loss_fn(logits, targets)
    print(f"Factory Loss (asymmetric): {loss_factory.item():.4f}")

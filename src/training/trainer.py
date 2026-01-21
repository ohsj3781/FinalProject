"""
Training Module for Multi-Label Classification

This module provides the Trainer class for training ResNet-18 models
on the COCO dataset for multi-label classification.
"""

import os
import time
from typing import Dict, Optional, Tuple, Any

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from src.utils.metrics import MultiLabelMetrics, AverageMeter


class Trainer:
    """
    Trainer class for multi-label classification.

    Handles training loop, validation, checkpointing, and logging.

    Args:
        model: The model to train
        train_loader: Training data loader
        val_loader: Validation data loader
        criterion: Loss function
        optimizer: Optimizer
        scheduler: Learning rate scheduler (optional)
        device: Device to train on
        config: Training configuration dictionary
    """

    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        criterion: nn.Module,
        optimizer: optim.Optimizer,
        scheduler: Optional[Any] = None,
        device: torch.device = None,
        config: Dict = None
    ):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.config = config or {}

        # Move model to device
        self.model = self.model.to(self.device)

        # Training settings
        self.epochs = self.config.get('epochs', 90)
        self.threshold = self.config.get('threshold', 0.5)
        self.save_dir = self.config.get('save_dir', 'checkpoints')
        self.save_freq = self.config.get('save_freq', 10)
        self.log_dir = self.config.get('log_dir', 'logs')
        self.log_freq = self.config.get('log_freq', 100)

        # Create directories
        os.makedirs(self.save_dir, exist_ok=True)
        os.makedirs(self.log_dir, exist_ok=True)

        # Tensorboard writer
        self.writer = SummaryWriter(self.log_dir)

        # Metrics
        self.metrics = MultiLabelMetrics(
            num_classes=self.config.get('num_classes', 80),
            threshold=self.threshold
        )

        # Best metrics tracking
        self.best_mAP = 0.0
        self.current_epoch = 0

    def train(self) -> Dict[str, float]:
        """
        Run full training loop.

        Returns:
            Dictionary with final metrics
        """
        print(f"Starting training for {self.epochs} epochs")
        print(f"Device: {self.device}")
        print(f"Training samples: {len(self.train_loader.dataset)}")
        print(f"Validation samples: {len(self.val_loader.dataset)}")
        print("-" * 50)

        for epoch in range(self.current_epoch, self.epochs):
            self.current_epoch = epoch

            # Train one epoch
            train_loss = self._train_epoch(epoch)

            # Validate
            val_metrics = self._validate(epoch)

            # Learning rate scheduling
            if self.scheduler is not None:
                self.scheduler.step()

            # Log epoch results
            self._log_epoch(epoch, train_loss, val_metrics)

            # Save checkpoint
            is_best = val_metrics['mAP'] > self.best_mAP
            if is_best:
                self.best_mAP = val_metrics['mAP']

            if (epoch + 1) % self.save_freq == 0 or is_best:
                self._save_checkpoint(epoch, is_best)

        self.writer.close()
        print(f"\nTraining complete. Best mAP: {self.best_mAP:.4f}")

        return {'best_mAP': self.best_mAP}

    def _train_epoch(self, epoch: int) -> float:
        """Train for one epoch."""
        self.model.train()

        loss_meter = AverageMeter('Loss')
        batch_time = AverageMeter('Batch')
        data_time = AverageMeter('Data')

        pbar = tqdm(
            self.train_loader,
            desc=f"Epoch {epoch + 1}/{self.epochs} [Train]"
        )

        end = time.time()

        for batch_idx, (images, targets) in enumerate(pbar):
            # Measure data loading time
            data_time.update(time.time() - end)

            # Move to device
            images = images.to(self.device)
            targets = targets.to(self.device)

            # Forward pass
            outputs = self.model(images)
            loss = self.criterion(outputs, targets)

            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            # Update metrics
            loss_meter.update(loss.item(), images.size(0))
            batch_time.update(time.time() - end)
            end = time.time()

            # Update progress bar
            pbar.set_postfix({
                'loss': f'{loss_meter.avg:.4f}',
                'lr': f'{self.optimizer.param_groups[0]["lr"]:.6f}'
            })

            # Log to tensorboard
            global_step = epoch * len(self.train_loader) + batch_idx
            if (batch_idx + 1) % self.log_freq == 0:
                self.writer.add_scalar('Train/Loss', loss.item(), global_step)
                self.writer.add_scalar(
                    'Train/LR',
                    self.optimizer.param_groups[0]['lr'],
                    global_step
                )

        return loss_meter.avg

    @torch.no_grad()
    def _validate(self, epoch: int) -> Dict[str, float]:
        """Validate the model."""
        self.model.eval()
        self.metrics.reset()

        loss_meter = AverageMeter('Loss')

        pbar = tqdm(
            self.val_loader,
            desc=f"Epoch {epoch + 1}/{self.epochs} [Val]"
        )

        for images, targets in pbar:
            images = images.to(self.device)
            targets = targets.to(self.device)

            # Forward pass
            outputs = self.model(images)
            loss = self.criterion(outputs, targets)

            # Update loss meter
            loss_meter.update(loss.item(), images.size(0))

            # Update metrics
            probs = torch.sigmoid(outputs)
            self.metrics.update(probs, targets)

            pbar.set_postfix({'loss': f'{loss_meter.avg:.4f}'})

        # Compute final metrics
        metrics = self.metrics.compute()
        metrics['val_loss'] = loss_meter.avg

        return metrics

    def _log_epoch(
        self,
        epoch: int,
        train_loss: float,
        val_metrics: Dict[str, float]
    ):
        """Log epoch results."""
        print(f"\nEpoch {epoch + 1}/{self.epochs}")
        print(f"  Train Loss: {train_loss:.4f}")
        print(f"  Val Loss: {val_metrics['val_loss']:.4f}")
        print(f"  mAP: {val_metrics['mAP']:.4f}")
        print(f"  Precision: {val_metrics['precision']:.4f}")
        print(f"  Recall: {val_metrics['recall']:.4f}")
        print(f"  F1: {val_metrics['f1']:.4f}")

        # Tensorboard logging
        self.writer.add_scalar('Train/Epoch_Loss', train_loss, epoch)
        self.writer.add_scalar('Val/Loss', val_metrics['val_loss'], epoch)
        self.writer.add_scalar('Val/mAP', val_metrics['mAP'], epoch)
        self.writer.add_scalar('Val/Precision', val_metrics['precision'], epoch)
        self.writer.add_scalar('Val/Recall', val_metrics['recall'], epoch)
        self.writer.add_scalar('Val/F1', val_metrics['f1'], epoch)

    def _save_checkpoint(self, epoch: int, is_best: bool = False):
        """Save model checkpoint."""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_mAP': self.best_mAP,
            'config': self.config
        }

        if self.scheduler is not None:
            checkpoint['scheduler_state_dict'] = self.scheduler.state_dict()

        # Save regular checkpoint
        checkpoint_path = os.path.join(
            self.save_dir, f'checkpoint_epoch_{epoch + 1}.pth'
        )
        torch.save(checkpoint, checkpoint_path)
        print(f"  Saved checkpoint: {checkpoint_path}")

        # Save best model
        if is_best:
            best_path = os.path.join(self.save_dir, 'best_model.pth')
            torch.save(checkpoint, best_path)
            print(f"  Saved best model: {best_path}")

    def load_checkpoint(self, checkpoint_path: str):
        """Load model checkpoint."""
        print(f"Loading checkpoint: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)

        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.current_epoch = checkpoint['epoch'] + 1
        self.best_mAP = checkpoint.get('best_mAP', 0.0)

        if self.scheduler is not None and 'scheduler_state_dict' in checkpoint:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

        print(f"  Resumed from epoch {self.current_epoch}")
        print(f"  Best mAP so far: {self.best_mAP:.4f}")


def create_optimizer(
    model: nn.Module,
    config: Dict
) -> optim.Optimizer:
    """
    Create optimizer from config.

    Args:
        model: The model to optimize
        config: Optimizer configuration

    Returns:
        Optimizer instance
    """
    opt_config = config.get('optimizer', {})
    name = opt_config.get('name', 'sgd').lower()

    params = model.parameters()

    if name == 'sgd':
        return optim.SGD(
            params,
            lr=opt_config.get('lr', 0.1),
            momentum=opt_config.get('momentum', 0.9),
            weight_decay=opt_config.get('weight_decay', 1e-4),
            nesterov=opt_config.get('nesterov', True)
        )
    elif name == 'adam':
        return optim.Adam(
            params,
            lr=opt_config.get('lr', 0.001),
            weight_decay=opt_config.get('weight_decay', 1e-4)
        )
    elif name == 'adamw':
        return optim.AdamW(
            params,
            lr=opt_config.get('lr', 0.001),
            weight_decay=opt_config.get('weight_decay', 1e-4)
        )
    else:
        raise ValueError(f"Unknown optimizer: {name}")


def create_scheduler(
    optimizer: optim.Optimizer,
    config: Dict
) -> Optional[Any]:
    """
    Create learning rate scheduler from config.

    Args:
        optimizer: The optimizer
        config: Scheduler configuration

    Returns:
        Scheduler instance or None
    """
    sched_config = config.get('scheduler', {})
    name = sched_config.get('name', 'cosine').lower()

    if name == 'cosine':
        return optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=sched_config.get('T_max', 90),
            eta_min=sched_config.get('eta_min', 0.0)
        )
    elif name == 'step':
        return optim.lr_scheduler.StepLR(
            optimizer,
            step_size=sched_config.get('step_size', 30),
            gamma=sched_config.get('gamma', 0.1)
        )
    elif name == 'multistep':
        return optim.lr_scheduler.MultiStepLR(
            optimizer,
            milestones=sched_config.get('milestones', [30, 60, 80]),
            gamma=sched_config.get('gamma', 0.1)
        )
    elif name == 'none':
        return None
    else:
        raise ValueError(f"Unknown scheduler: {name}")


if __name__ == "__main__":
    # Quick test of trainer components
    print("Testing trainer components...")

    # Mock model
    model = nn.Linear(10, 5)

    # Mock config
    config = {
        'optimizer': {'name': 'sgd', 'lr': 0.1, 'momentum': 0.9},
        'scheduler': {'name': 'cosine', 'T_max': 10}
    }

    # Create optimizer and scheduler
    optimizer = create_optimizer(model, config)
    scheduler = create_scheduler(optimizer, config)

    print(f"Optimizer: {optimizer}")
    print(f"Scheduler: {scheduler}")

    # Test LR scheduling
    for epoch in range(5):
        print(f"Epoch {epoch}: LR = {optimizer.param_groups[0]['lr']:.6f}")
        scheduler.step()

#!/usr/bin/env python3
"""
PT2E QAT Training Script

This script trains a ResNet model using PyTorch's native PT2E Quantization-Aware Training.
The trained model can be directly exported to ExecuTorch without any custom conversion.

Workflow:
1. Load FP32 pretrained model
2. Prepare for PT2E QAT (insert fake quantization ops)
3. Train with QAT
4. Convert to INT8
5. Export to ExecuTorch

Usage:
    python scripts/train_pt2e_qat.py --config configs/config.yaml --epochs 30

    # Resume from checkpoint
    python scripts/train_pt2e_qat.py --config configs/config.yaml --resume checkpoints/pt2e_qat/last.pth
"""

import argparse
import os
import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.tensorboard import SummaryWriter
import yaml
from tqdm import tqdm

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.models.resnet import get_resnet
from src.data.dataset import create_coco_dataloaders
from src.data.augmentation import get_train_transform, get_val_transform
from src.training.loss import get_loss_function
from src.utils.metrics import calculate_metrics, calculate_map


def parse_args():
    parser = argparse.ArgumentParser(description='PT2E QAT Training')
    parser.add_argument('--config', type=str, default='configs/config.yaml',
                        help='Path to config file')
    parser.add_argument('--pretrained', type=str, default=None,
                        help='Path to pretrained FP32 checkpoint')
    parser.add_argument('--resume', type=str, default=None,
                        help='Path to PT2E QAT checkpoint to resume')
    parser.add_argument('--epochs', type=int, default=30,
                        help='Number of QAT epochs')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='Learning rate for QAT (typically 1/10 of FP32 LR)')
    parser.add_argument('--batch-size', type=int, default=64,
                        help='Batch size')
    parser.add_argument('--output-dir', type=str, default='checkpoints/pt2e_qat',
                        help='Output directory')
    return parser.parse_args()


def load_config(config_path: str) -> dict:
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def prepare_pt2e_qat_model(model: nn.Module) -> nn.Module:
    """
    Prepare model for PT2E Quantization-Aware Training.

    This uses PyTorch's native PT2E QAT which is directly compatible with ExecuTorch.
    """
    from torch.ao.quantization.quantize_pt2e import prepare_qat_pt2e
    from torch.ao.quantization.quantizer.xnnpack_quantizer import (
        XNNPACKQuantizer,
        get_symmetric_quantization_config
    )
    from torch.export import export_for_training

    print("\nPreparing model for PT2E QAT...")

    # Export for training
    model.train()
    example_input = (torch.randn(1, 3, 224, 224),)
    exported = export_for_training(model, example_input)

    # Setup quantizer with per-channel weights (better accuracy)
    quantizer = XNNPACKQuantizer()
    quant_config = get_symmetric_quantization_config(
        is_per_channel=True,  # Per-channel for weights
        is_qat=True           # QAT mode
    )
    quantizer.set_global(quant_config)

    # Prepare for QAT
    prepared = prepare_qat_pt2e(exported, quantizer)

    print("  PT2E QAT preparation complete")
    print("  Quantizer: XNNPACK (symmetric, per-channel)")

    return prepared


def convert_pt2e_to_int8(prepared_model: nn.Module) -> nn.Module:
    """
    Convert PT2E QAT model to INT8.
    """
    from torch.ao.quantization.quantize_pt2e import convert_pt2e

    print("\nConverting PT2E QAT model to INT8...")
    prepared_model.eval()
    quantized = convert_pt2e(prepared_model)
    print("  Conversion complete")

    return quantized


def export_to_executorch(quantized_model: nn.Module, output_path: str):
    """
    Export quantized model to ExecuTorch.
    """
    from torch.export import export
    from executorch.exir import to_edge_transform_and_lower, EdgeCompileConfig
    from executorch.backends.xnnpack.partition.xnnpack_partitioner import XnnpackPartitioner

    print(f"\nExporting to ExecuTorch: {output_path}")

    example_input = (torch.randn(1, 3, 224, 224),)
    exported = export(quantized_model, example_input)

    partitioners = [XnnpackPartitioner()]
    edge_config = EdgeCompileConfig(_check_ir_validity=False)

    edge = to_edge_transform_and_lower(
        exported,
        compile_config=edge_config,
        partitioner=partitioners
    )

    et_program = edge.to_executorch()

    with open(output_path, 'wb') as f:
        f.write(et_program.buffer)

    file_size = os.path.getsize(output_path) / (1024 * 1024)
    print(f"  Saved: {output_path} ({file_size:.2f} MB)")

    return output_path


def train_one_epoch(model, train_loader, criterion, optimizer, device, epoch):
    """Train for one epoch."""
    model.train()

    total_loss = 0
    all_preds = []
    all_targets = []

    pbar = tqdm(train_loader, desc=f'Epoch {epoch}')
    for images, targets in pbar:
        images = images.to(device)
        targets = targets.to(device)

        optimizer.zero_grad()

        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, targets)

        # Backward pass
        loss.backward()

        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        optimizer.step()

        total_loss += loss.item()

        # Collect predictions
        with torch.no_grad():
            preds = torch.sigmoid(outputs)
            all_preds.append(preds.cpu())
            all_targets.append(targets.cpu())

        pbar.set_postfix({'loss': loss.item()})

    # Calculate metrics
    all_preds = torch.cat(all_preds, dim=0).numpy()
    all_targets = torch.cat(all_targets, dim=0).numpy()

    mAP = calculate_map(all_preds, all_targets)
    avg_loss = total_loss / len(train_loader)

    return avg_loss, mAP


def validate(model, val_loader, criterion, device):
    """Validate the model."""
    model.eval()

    total_loss = 0
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for images, targets in tqdm(val_loader, desc='Validation'):
            images = images.to(device)
            targets = targets.to(device)

            outputs = model(images)
            loss = criterion(outputs, targets)

            total_loss += loss.item()

            preds = torch.sigmoid(outputs)
            all_preds.append(preds.cpu())
            all_targets.append(targets.cpu())

    all_preds = torch.cat(all_preds, dim=0).numpy()
    all_targets = torch.cat(all_targets, dim=0).numpy()

    mAP = calculate_map(all_preds, all_targets)
    metrics = calculate_metrics(all_preds, all_targets)
    avg_loss = total_loss / len(val_loader)

    return avg_loss, mAP, metrics


def main():
    args = parse_args()
    config = load_config(args.config)

    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Setup tensorboard
    writer = SummaryWriter(log_dir=os.path.join(args.output_dir, 'logs'))

    # Create data loaders
    print("\nLoading data...")
    train_loader, val_loader = create_coco_dataloaders(
        data_dir=config['data']['data_dir'],
        train_transform=get_train_transform(config),
        val_transform=get_val_transform(config),
        batch_size=args.batch_size,
        num_workers=config['data'].get('num_workers', 4)
    )

    # Create or load model
    model_name = config['model'].get('name', 'resnet50')
    print(f"\nCreating model: {model_name}")

    # Load pretrained FP32 model
    base_model = get_resnet(model_name, num_classes=80, pretrained=False)

    if args.pretrained:
        print(f"Loading pretrained weights: {args.pretrained}")
        checkpoint = torch.load(args.pretrained, map_location='cpu', weights_only=False)
        if 'model_state_dict' in checkpoint:
            base_model.load_state_dict(checkpoint['model_state_dict'])
        else:
            base_model.load_state_dict(checkpoint)

    # Prepare for PT2E QAT
    model = prepare_pt2e_qat_model(base_model)

    # For PT2E models, we need to handle device placement differently
    # The model is a GraphModule, not a regular nn.Module
    # We'll keep it on CPU and move inputs/outputs as needed

    # Setup optimizer and scheduler
    # PT2E model parameters might be accessed differently
    try:
        params = list(model.parameters())
    except:
        params = [p for n, p in model.named_parameters()]

    optimizer = optim.AdamW(params, lr=args.lr, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs)

    # Loss function
    criterion = get_loss_function(config)

    # Resume if specified
    start_epoch = 0
    best_mAP = 0

    if args.resume:
        print(f"Resuming from: {args.resume}")
        checkpoint = torch.load(args.resume, map_location='cpu', weights_only=False)
        # Note: PT2E models have a different structure, state_dict loading might differ
        start_epoch = checkpoint.get('epoch', 0) + 1
        best_mAP = checkpoint.get('best_mAP', 0)
        if 'optimizer_state_dict' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    print(f"\n{'='*60}")
    print("Starting PT2E QAT Training")
    print(f"{'='*60}")
    print(f"  Model: {model_name}")
    print(f"  Epochs: {args.epochs}")
    print(f"  Learning rate: {args.lr}")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Output: {args.output_dir}")
    print(f"{'='*60}\n")

    # Training loop
    for epoch in range(start_epoch, args.epochs):
        # Train
        train_loss, train_mAP = train_one_epoch(
            model, train_loader, criterion, optimizer, device, epoch
        )

        # Validate
        val_loss, val_mAP, metrics = validate(model, val_loader, criterion, device)

        # Update scheduler
        scheduler.step()

        # Log to tensorboard
        writer.add_scalar('Loss/train', train_loss, epoch)
        writer.add_scalar('Loss/val', val_loss, epoch)
        writer.add_scalar('mAP/train', train_mAP, epoch)
        writer.add_scalar('mAP/val', val_mAP, epoch)
        writer.add_scalar('LR', optimizer.param_groups[0]['lr'], epoch)

        # Print progress
        print(f"\nEpoch {epoch}:")
        print(f"  Train Loss: {train_loss:.4f}, Train mAP: {train_mAP:.4f}")
        print(f"  Val Loss: {val_loss:.4f}, Val mAP: {val_mAP:.4f}")
        print(f"  Metrics: P={metrics['precision']:.4f}, R={metrics['recall']:.4f}, F1={metrics['f1']:.4f}")

        # Save checkpoint
        is_best = val_mAP > best_mAP
        best_mAP = max(val_mAP, best_mAP)

        # Save last checkpoint
        # Note: For PT2E models, we save the entire model
        torch.save({
            'epoch': epoch,
            'model': model,  # Save entire PT2E model
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'best_mAP': best_mAP,
            'val_mAP': val_mAP,
        }, os.path.join(args.output_dir, 'last.pth'))

        if is_best:
            torch.save({
                'epoch': epoch,
                'model': model,
                'best_mAP': best_mAP,
            }, os.path.join(args.output_dir, 'best.pth'))
            print(f"  New best model! mAP: {val_mAP:.4f}")

    writer.close()

    # Convert and export best model
    print("\n" + "="*60)
    print("Training complete! Converting to INT8...")
    print("="*60)

    # Load best model
    best_checkpoint = torch.load(os.path.join(args.output_dir, 'best.pth'),
                                  map_location='cpu', weights_only=False)
    best_model = best_checkpoint['model']

    # Convert to INT8
    quantized_model = convert_pt2e_to_int8(best_model)

    # Export to ExecuTorch
    pte_path = os.path.join(args.output_dir, f'{model_name}_pt2e_qat_int8.pte')
    export_to_executorch(quantized_model, pte_path)

    print(f"\n{'='*60}")
    print("PT2E QAT Complete!")
    print(f"  Best mAP: {best_mAP:.4f}")
    print(f"  Model: {pte_path}")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()

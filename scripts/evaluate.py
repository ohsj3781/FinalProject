#!/usr/bin/env python3
"""
Evaluation Script for ResNet-18 Multi-Label Classification

This script evaluates a trained model on the COCO validation set.

Usage:
    python scripts/evaluate.py --checkpoint checkpoints/best_model.pth
    python scripts/evaluate.py --checkpoint checkpoints/qat/best_model.pth --qat
"""

import argparse
import os
import sys
from pathlib import Path

import torch
import torch.nn as nn
import yaml
from tqdm import tqdm

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.models.resnet import resnet18, count_parameters, get_model_size_mb
from src.models.quantization import quantize_model
from src.data.dataset import create_coco_dataloaders
from src.data.augmentation import ValTransform
from src.utils.metrics import MultiLabelMetrics, compute_optimal_threshold


def parse_args():
    parser = argparse.ArgumentParser(description='Evaluate ResNet-18 model')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to model checkpoint')
    parser.add_argument('--config', type=str, default='configs/config.yaml',
                        help='Path to config file')
    parser.add_argument('--qat', action='store_true',
                        help='Model was trained with QAT')
    parser.add_argument('--data-dir', type=str, default=None,
                        help='Override data directory from config')
    parser.add_argument('--batch-size', type=int, default=128,
                        help='Batch size for evaluation')
    parser.add_argument('--threshold', type=float, default=None,
                        help='Classification threshold (default: find optimal)')
    parser.add_argument('--gpu', type=int, default=0,
                        help='GPU device ID')
    parser.add_argument('--save-predictions', action='store_true',
                        help='Save predictions to file')
    parser.add_argument('--output', type=str, default='evaluation_results.txt',
                        help='Output file for results')
    return parser.parse_args()


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def setup_device(gpu_id: int) -> torch.device:
    """Setup compute device."""
    if torch.cuda.is_available():
        device = torch.device(f'cuda:{gpu_id}')
        print(f"Using GPU: {torch.cuda.get_device_name(gpu_id)}")
    else:
        device = torch.device('cpu')
        print("Using CPU")
    return device


def load_model(config: dict, checkpoint_path: str, qat: bool, device: torch.device) -> nn.Module:
    """Load trained model."""
    model_config = config['model']

    # Create base model
    model = resnet18(
        num_classes=model_config['num_classes'],
        pretrained=False
    )

    # Apply quantization structure if QAT
    if qat:
        quant_config = config['quantization']
        exclude_layers = quant_config.get('exclude_layers', ['conv1', 'fc'])
        model = quantize_model(
            model,
            bits=quant_config['bits'],
            exclude_layers=exclude_layers
        )

    # Load checkpoint
    print(f"Loading checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
        if 'best_mAP' in checkpoint:
            print(f"Checkpoint best mAP: {checkpoint['best_mAP']:.4f}")
    else:
        model.load_state_dict(checkpoint)

    model = model.to(device)
    model.eval()

    return model


@torch.no_grad()
def evaluate(
    model: nn.Module,
    dataloader: torch.utils.data.DataLoader,
    device: torch.device,
    threshold: float = 0.5
) -> dict:
    """
    Evaluate model on dataset.

    Returns:
        Dictionary with evaluation metrics
    """
    model.eval()

    metrics = MultiLabelMetrics(
        num_classes=dataloader.dataset.NUM_CLASSES,
        threshold=threshold
    )

    all_preds = []
    all_targets = []

    pbar = tqdm(dataloader, desc="Evaluating")

    for images, targets in pbar:
        images = images.to(device)

        # Forward pass
        outputs = model(images)
        probs = torch.sigmoid(outputs)

        # Update metrics
        metrics.update(probs, targets)

        # Store for threshold optimization
        all_preds.append(probs.cpu())
        all_targets.append(targets)

    # Compute final metrics
    results = metrics.compute()

    # Concatenate all predictions for threshold optimization
    all_preds = torch.cat(all_preds, dim=0).numpy()
    all_targets = torch.cat(all_targets, dim=0).numpy()

    return results, all_preds, all_targets


def print_results(results: dict, threshold: float):
    """Print evaluation results."""
    print("\n" + "=" * 50)
    print("Evaluation Results")
    print("=" * 50)
    print(f"Threshold: {threshold:.2f}")
    print("-" * 50)
    print(f"mAP:                {results['mAP']:.4f}")
    print(f"Precision (sample): {results['precision']:.4f}")
    print(f"Recall (sample):    {results['recall']:.4f}")
    print(f"F1 (sample):        {results['f1']:.4f}")
    print("-" * 50)
    print(f"Precision (macro):  {results['precision_macro']:.4f}")
    print(f"Recall (macro):     {results['recall_macro']:.4f}")
    print(f"F1 (macro):         {results['f1_macro']:.4f}")
    print("=" * 50)


def save_results(results: dict, args, config: dict, output_path: str):
    """Save evaluation results to file."""
    with open(output_path, 'w') as f:
        f.write("=" * 50 + "\n")
        f.write("Evaluation Results\n")
        f.write("=" * 50 + "\n\n")

        f.write("Configuration:\n")
        f.write(f"  Checkpoint: {args.checkpoint}\n")
        f.write(f"  QAT: {args.qat}\n")
        f.write(f"  Batch Size: {args.batch_size}\n")
        f.write(f"  Threshold: {args.threshold}\n\n")

        f.write("Metrics:\n")
        for name, value in results.items():
            f.write(f"  {name}: {value:.4f}\n")

    print(f"\nResults saved to: {output_path}")


def main():
    args = parse_args()

    # Load config
    config = load_config(args.config)

    # Override data directory if specified
    if args.data_dir:
        config['data']['data_dir'] = args.data_dir

    # Setup device
    device = setup_device(args.gpu)

    # Create validation transform
    val_transform = ValTransform()

    # Create data loader
    print("\nCreating data loader...")
    data_dir = config['data']['data_dir']

    try:
        _, val_loader = create_coco_dataloaders(
            data_dir=data_dir,
            train_transform=val_transform,  # Not used, just placeholder
            val_transform=val_transform,
            batch_size=args.batch_size,
            num_workers=4
        )
        print(f"Validation samples: {len(val_loader.dataset)}")
    except FileNotFoundError:
        print(f"\nERROR: COCO dataset not found at {data_dir}")
        sys.exit(1)

    # Load model
    model = load_model(config, args.checkpoint, args.qat, device)

    print(f"\nModel info:")
    print(f"  Parameters: {count_parameters(model):,}")
    print(f"  Size: {get_model_size_mb(model):.2f} MB")

    # Initial evaluation with default threshold
    initial_threshold = args.threshold if args.threshold else 0.5
    print(f"\nEvaluating with threshold = {initial_threshold}...")

    results, all_preds, all_targets = evaluate(
        model, val_loader, device, threshold=initial_threshold
    )

    # Find optimal threshold if not specified
    if args.threshold is None:
        print("\nFinding optimal threshold...")
        opt_threshold, opt_f1 = compute_optimal_threshold(all_preds, all_targets, 'f1')
        print(f"Optimal threshold: {opt_threshold:.2f} (F1: {opt_f1:.4f})")

        # Re-evaluate with optimal threshold
        print(f"\nRe-evaluating with optimal threshold = {opt_threshold}...")
        metrics = MultiLabelMetrics(
            num_classes=val_loader.dataset.NUM_CLASSES,
            threshold=opt_threshold
        )
        metrics.all_preds = [all_preds]
        metrics.all_targets = [all_targets]
        results = metrics.compute()
        final_threshold = opt_threshold
    else:
        final_threshold = args.threshold

    # Print results
    print_results(results, final_threshold)

    # Save results
    save_results(results, args, config, args.output)

    # Save predictions if requested
    if args.save_predictions:
        import numpy as np
        pred_path = args.output.replace('.txt', '_predictions.npz')
        np.savez(
            pred_path,
            predictions=all_preds,
            targets=all_targets,
            threshold=final_threshold
        )
        print(f"Predictions saved to: {pred_path}")


if __name__ == '__main__':
    main()

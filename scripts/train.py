#!/usr/bin/env python3
"""
Training Script for ResNet-18 Multi-Label Classification

This script handles both full-precision training and quantization-aware training (QAT)
for the photo auto-tagging system.

Usage:
    # Full precision training
    python scripts/train.py --config configs/config.yaml

    # QAT training (requires pretrained full precision model)
    python scripts/train.py --config configs/config.yaml --qat --checkpoint checkpoints/best_model.pth

    # Resume training
    python scripts/train.py --config configs/config.yaml --resume checkpoints/checkpoint_epoch_50.pth
"""

import argparse
import os
import sys
from pathlib import Path

import torch
import torch.nn as nn
import yaml

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.models.resnet import resnet18, count_parameters, get_model_size_mb
from src.models.quantization import quantize_model
from src.data.dataset import create_coco_dataloaders
from src.data.augmentation import get_transforms_from_config
from src.training.trainer import Trainer, create_optimizer, create_scheduler
from src.training.loss import get_loss_function


def parse_args():
    parser = argparse.ArgumentParser(description='Train ResNet-18 for multi-label classification')
    parser.add_argument('--config', type=str, default='configs/config.yaml',
                        help='Path to config file')
    parser.add_argument('--qat', action='store_true',
                        help='Enable Quantization-Aware Training')
    parser.add_argument('--checkpoint', type=str, default=None,
                        help='Path to pretrained checkpoint (required for QAT)')
    parser.add_argument('--resume', type=str, default=None,
                        help='Path to checkpoint to resume training from')
    parser.add_argument('--data-dir', type=str, default=None,
                        help='Override data directory from config')
    parser.add_argument('--output-dir', type=str, default=None,
                        help='Override output directory from config')
    parser.add_argument('--gpu', type=int, default=0,
                        help='GPU device ID')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    return parser.parse_args()


def set_seed(seed: int, deterministic: bool = False):
    """Set random seed for reproducibility."""
    import random
    import numpy as np

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        if deterministic:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
        else:
            # Enable cudnn autotuner for faster training
            torch.backends.cudnn.deterministic = False
            torch.backends.cudnn.benchmark = True


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


def create_model(config: dict, qat: bool = False, checkpoint_path: str = None) -> nn.Module:
    """Create and optionally load model."""
    model_config = config['model']

    # Create base model
    model = resnet18(
        num_classes=model_config['num_classes'],
        pretrained=model_config.get('pretrained', True) and checkpoint_path is None
    )

    # Load checkpoint if provided
    if checkpoint_path:
        print(f"Loading checkpoint: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)
        print("Checkpoint loaded successfully")

    # Apply quantization for QAT
    if qat:
        quant_config = config['quantization']
        exclude_layers = quant_config.get('exclude_layers', ['conv1', 'fc'])

        print(f"Applying LSQ quantization ({quant_config['bits']}-bit)")
        print(f"Excluding layers: {exclude_layers}")

        model = quantize_model(
            model,
            bits=quant_config['bits'],
            exclude_layers=exclude_layers
        )

    return model


def main():
    args = parse_args()

    # Set seed (non-deterministic for faster training with cudnn.benchmark)
    set_seed(args.seed, deterministic=False)

    # Load config
    config = load_config(args.config)
    print(f"Loaded config from: {args.config}")

    # Override config with command line arguments
    if args.data_dir:
        config['data']['data_dir'] = args.data_dir
    if args.output_dir:
        config['training']['save_dir'] = args.output_dir
        config['training']['log_dir'] = os.path.join(args.output_dir, 'logs')

    # Setup device
    device = setup_device(args.gpu)

    # Get training config (use QAT config if enabled)
    if args.qat:
        if args.checkpoint is None:
            print("ERROR: QAT requires a pretrained checkpoint (--checkpoint)")
            sys.exit(1)
        train_config = config['quantization']['qat']
        save_prefix = 'qat'
    else:
        train_config = config['training']
        save_prefix = 'fp32'

    # Update save/log directories
    save_dir = os.path.join(config['training']['save_dir'], save_prefix)
    log_dir = os.path.join(config['training']['log_dir'], save_prefix)
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)

    # Create data transforms
    train_transform, val_transform = get_transforms_from_config(config)

    # Create data loaders
    print("\nCreating data loaders...")
    data_dir = config['data']['data_dir']

    try:
        train_loader, val_loader = create_coco_dataloaders(
            data_dir=data_dir,
            train_transform=train_transform,
            val_transform=val_transform,
            batch_size=train_config['batch_size'],
            num_workers=train_config.get('num_workers', config['training'].get('num_workers', 4)),
            pin_memory=train_config.get('pin_memory', config['training'].get('pin_memory', True))
        )
        print(f"Training samples: {len(train_loader.dataset)}")
        print(f"Validation samples: {len(val_loader.dataset)}")
    except FileNotFoundError as e:
        print(f"\nERROR: COCO dataset not found at {data_dir}")
        print("Please download COCO 2017 dataset and place it in the data directory.")
        print("\nExpected structure:")
        print(f"  {data_dir}/train2017/")
        print(f"  {data_dir}/val2017/")
        print(f"  {data_dir}/annotations/instances_train2017.json")
        print(f"  {data_dir}/annotations/instances_val2017.json")
        sys.exit(1)

    # Create model
    print("\nCreating model...")
    model = create_model(
        config,
        qat=args.qat,
        checkpoint_path=args.checkpoint
    )

    print(f"Model: {config['model']['name']}")
    print(f"Parameters: {count_parameters(model):,}")
    print(f"Model size: {get_model_size_mb(model):.2f} MB")

    # Get pos_weight for class imbalance (optional)
    pos_weight = None
    if config['training']['loss'].get('pos_weight') == 'auto':
        print("Computing class weights...")
        pos_weight = train_loader.dataset.get_pos_weights().to(device)

    # Create loss function
    loss_config = config['training']['loss']
    loss_kwargs = {k: v for k, v in loss_config.items() if k not in ('name', 'pos_weight')}
    criterion = get_loss_function(
        loss_config['name'],
        pos_weight=pos_weight,
        **loss_kwargs
    )

    # Create optimizer
    optimizer = create_optimizer(model, train_config)

    # Create scheduler
    scheduler = create_scheduler(optimizer, train_config)

    # Prepare trainer config
    trainer_config = {
        'epochs': train_config['epochs'],
        'threshold': config['training'].get('threshold', 0.5),
        'save_dir': save_dir,
        'save_freq': config['training'].get('save_freq', 10),
        'log_dir': log_dir,
        'log_freq': config['training'].get('log_freq', 100),
        'num_classes': config['model']['num_classes'],
        'use_amp': train_config.get('use_amp', config['training'].get('use_amp', True))
    }

    # Create trainer
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        config=trainer_config
    )

    # Resume from checkpoint if specified
    if args.resume:
        trainer.load_checkpoint(args.resume)

    # Start training
    print("\n" + "=" * 50)
    print("Starting training...")
    print("=" * 50)

    results = trainer.train()

    print("\n" + "=" * 50)
    print("Training completed!")
    print(f"Best mAP: {results['best_mAP']:.4f}")
    print(f"Checkpoints saved to: {save_dir}")
    print("=" * 50)


if __name__ == '__main__':
    main()

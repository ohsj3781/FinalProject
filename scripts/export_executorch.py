#!/usr/bin/env python3
"""
ExecuTorch Export Script for ResNet Multi-Label Classification

두 가지 export 모드:

  --ptq   FP32 모델을 PTQ(Post-Training Quantization)로 INT8 변환
          사용 예: python scripts/export_executorch.py \\
                      --checkpoint checkpoints/resnet50_fp32/best_model.pth \\
                      --model resnet50 --ptq

  --qat   QAT 학습된 모델을 INT8로 변환 (calibration 기반, 더 높은 정확도)
          사용 예: python scripts/export_executorch.py \\
                      --checkpoint checkpoints/resnet50_qat/best_model.pth \\
                      --model resnet50 --qat
"""

import argparse
import os
import sys
from pathlib import Path

import torch
import torch.nn as nn
import yaml

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.models.resnet import get_resnet, count_parameters, get_model_size_mb
from src.models.quantization import quantize_model
from src.models.int8_export import (
    export_with_lsq_scales_pt2e,
    get_model_size_breakdown
)


def parse_args():
    parser = argparse.ArgumentParser(description='Export model to ExecuTorch INT8 format')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to model checkpoint')
    parser.add_argument('--config', type=str, default='configs/config.yaml',
                        help='Path to config file')
    parser.add_argument('--model', type=str, default=None,
                        choices=['resnet18', 'resnet34', 'resnet50'],
                        help='Model architecture (overrides config)')

    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--ptq', action='store_true',
                      help='PTQ: FP32 모델을 calibration으로 INT8 변환')
    mode.add_argument('--qat', action='store_true',
                      help='QAT: QAT 학습된 모델을 calibration으로 INT8 변환 (더 정확)')

    parser.add_argument('--backend', type=str, default='xnnpack',
                        choices=['xnnpack', 'nnapi'],
                        help='ExecuTorch backend (default: xnnpack)')
    parser.add_argument('--output-dir', type=str, default='exported_models',
                        help='Output directory for exported model')
    parser.add_argument('--output-name', type=str, default=None,
                        help='Output filename (without extension)')
    parser.add_argument('--calibration-samples', type=int, default=500,
                        help='Number of calibration samples (default: 500)')
    return parser.parse_args()


def load_config(config_path: str) -> dict:
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def load_fp32_model(config: dict, checkpoint_path: str) -> nn.Module:
    """FP32 체크포인트 로드"""
    model_config = config['model']
    model_name = model_config.get('name', 'resnet18')

    print(f"Creating FP32 model: {model_name}")
    model = get_resnet(
        name=model_name,
        num_classes=model_config['num_classes'],
        pretrained=False
    )

    print(f"Loading checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    state_dict = checkpoint.get('model_state_dict', checkpoint)
    model.load_state_dict(state_dict, strict=False)
    model.eval()
    return model


def load_qat_model(config: dict, checkpoint_path: str) -> nn.Module:
    """QAT 체크포인트 로드 및 quantizer 초기화"""
    model_config = config['model']
    model_name = model_config.get('name', 'resnet18')
    quant_config = config['quantization']

    print(f"Creating QAT model: {model_name}")
    model = get_resnet(
        name=model_name,
        num_classes=model_config['num_classes'],
        pretrained=False
    )
    model = quantize_model(
        model,
        bits=quant_config['bits'],
        exclude_layers=quant_config.get('exclude_layers', ['conv1', 'fc'])
    )

    print(f"Loading checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    state_dict = checkpoint.get('model_state_dict', checkpoint)
    model.load_state_dict(state_dict, strict=False)

    # Quantizer 초기화 (dummy forward in train mode)
    print("Initializing quantizers...")
    model.train()
    with torch.no_grad():
        _ = model(torch.randn(1, 3, 224, 224))
    model.eval()
    print("Quantizers initialized")
    return model


def export_ptq(model: nn.Module, output_path: str, backend: str,
               config: dict, calibration_samples: int):
    """
    PTQ: FP32 모델 → PT2E calibration → INT8
    BatchNorm이 Conv에 fusion된 이후 calibration으로 올바른 scale을 찾음
    """
    import warnings
    warnings.filterwarnings('ignore')

    from torch.export import export, export_for_training
    from torch.ao.quantization.quantize_pt2e import prepare_pt2e, convert_pt2e
    from torch.ao.quantization.quantizer.xnnpack_quantizer import (
        XNNPACKQuantizer, get_symmetric_quantization_config
    )
    from executorch.exir import to_edge_transform_and_lower, EdgeCompileConfig
    from executorch.backends.xnnpack.partition.xnnpack_partitioner import XnnpackPartitioner

    print(f"\n{'='*60}")
    print("PTQ Export (FP32 → INT8)")
    print("="*60)

    model.eval()
    example_input = (torch.randn(1, 3, 224, 224),)

    print("\nStep 1: Exporting model for PT2E...")
    exported = export_for_training(model, example_input)
    gm = exported.module()

    print("\nStep 2: Setting up XNNPACK quantizer (per-channel symmetric)...")
    quantizer = XNNPACKQuantizer()
    quantizer.set_global(get_symmetric_quantization_config(is_per_channel=True))
    prepared = prepare_pt2e(gm, quantizer)

    print(f"\nStep 3: Calibrating with {calibration_samples} COCO samples...")
    _calibrate(prepared, calibration_samples, config)

    print("\nStep 4: Converting to INT8...")
    quantized = convert_pt2e(prepared)

    _export_to_pte(quantized, example_input, output_path, backend)


def _calibrate(prepared_model, num_samples: int, config: dict = None):
    """COCO validation 데이터로 calibration"""
    try:
        if config is not None:
            from src.data.dataset import create_coco_dataloaders
            from src.data.augmentation import ValTransform

            print("  Loading COCO validation data...")
            _, val_loader = create_coco_dataloaders(
                data_dir=config['data']['data_dir'],
                train_transform=ValTransform(),
                val_transform=ValTransform(),
                batch_size=1,
                num_workers=0
            )

            count = 0
            with torch.no_grad():
                for images, _ in val_loader:
                    if count >= num_samples:
                        break
                    prepared_model(images)
                    count += 1
                    if count % 100 == 0:
                        print(f"    {count}/{num_samples} samples...")

            print(f"  Calibration done: {count} samples")
            return
    except Exception as e:
        print(f"  Could not load COCO data: {e}")

    print("  Using random data for calibration (less accurate)...")
    with torch.no_grad():
        for i in range(num_samples):
            prepared_model(torch.randn(1, 3, 224, 224))
            if (i + 1) % 100 == 0:
                print(f"    {i+1}/{num_samples} samples...")
    print(f"  Calibration done: {num_samples} random samples")


def _export_to_pte(quantized_model, example_input, output_path: str, backend: str):
    """INT8 변환된 모델을 .pte로 저장"""
    from torch.export import export
    from executorch.exir import to_edge_transform_and_lower, EdgeCompileConfig
    from executorch.backends.xnnpack.partition.xnnpack_partitioner import XnnpackPartitioner

    print("\nStep 5: Exporting quantized model...")
    quantized_exported = export(quantized_model, example_input)

    print(f"\nStep 6: Applying {backend} backend...")
    partitioners = [XnnpackPartitioner()] if backend in ['xnnpack', 'nnapi'] else []

    print("\nStep 7: Converting to edge program...")
    edge_config = EdgeCompileConfig(_check_ir_validity=False)
    try:
        edge = to_edge_transform_and_lower(
            quantized_exported,
            compile_config=edge_config,
            partitioner=partitioners if partitioners else None
        )
    except Exception as e:
        print(f"  Warning: {e}")
        edge = to_edge_transform_and_lower(
            quantized_exported, compile_config=edge_config, partitioner=None
        )

    print("\nStep 8: Generating ExecuTorch program...")
    et_program = edge.to_executorch()

    os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
    with open(output_path, 'wb') as f:
        f.write(et_program.buffer)

    file_size_mb = os.path.getsize(output_path) / (1024 * 1024)
    print(f"\n{'='*60}")
    print("Export Successful!")
    print(f"  Output: {output_path}")
    print(f"  Size:   {file_size_mb:.2f} MB")
    print(f"  Backend: {backend} (native INT8 ops)")
    print("="*60)


def main():
    args = parse_args()
    config = load_config(args.config)

    if args.model:
        config['model']['name'] = args.model

    os.makedirs(args.output_dir, exist_ok=True)
    model_name = config['model'].get('name', 'resnet18')

    # Output path
    if args.output_name:
        output_name = args.output_name
    else:
        mode_str = 'ptq' if args.ptq else 'qat'
        output_name = f"{model_name}_multilabel_{mode_str}_{args.backend}"
    output_path = os.path.join(args.output_dir, f"{output_name}.pte")

    if args.ptq:
        model = load_fp32_model(config, args.checkpoint)
        print(f"\nModel: {model_name} | Parameters: {count_parameters(model):,} | Size: {get_model_size_mb(model):.2f} MB | Mode: PTQ")
        export_ptq(model, output_path, args.backend, config, args.calibration_samples)

    elif args.qat:
        model = load_qat_model(config, args.checkpoint)
        print(f"\nModel: {model_name} | Parameters: {count_parameters(model):,} | Size: {get_model_size_mb(model):.2f} MB | Mode: QAT → INT8")
        export_with_lsq_scales_pt2e(
            qat_model=model,
            output_path=output_path,
            config=config,
            backend=args.backend,
            calibration_samples=args.calibration_samples,
        )


if __name__ == '__main__':
    main()

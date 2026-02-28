#!/usr/bin/env python3
"""
ExecuTorch Export Script for ResNet Multi-Label Classification

This script exports a trained PyTorch model to ExecuTorch format (.pte)
for mobile deployment. Supports ResNet-18, ResNet-34, and ResNet-50.

Usage:
    # Export full precision model
    python scripts/export_executorch.py --checkpoint checkpoints/resnet50_fp32/best_model.pth --model resnet50

    # Export QAT model (fake quantization, FP32 weights)
    python scripts/export_executorch.py --checkpoint checkpoints/resnet50_qat/best_model.pth --model resnet50 --qat

    # Export with PT2E PTQ (FP32 → INT8, recommended for best accuracy)
    python scripts/export_executorch.py --checkpoint checkpoints/resnet50_fp32/best_model.pth --model resnet50 --pt2e-qat

    # Export QAT model with INT8 storage using learned LSQ step_size (★ RECOMMENDED for QAT)
    python scripts/export_executorch.py --checkpoint checkpoints/resnet50_qat/best_model.pth --model resnet50 --qat --int8-lsq
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

from src.models.resnet import get_resnet, count_parameters, get_model_size_mb
from src.models.quantization import quantize_model
from src.models.int8_export import (
    convert_lsq_to_int8,
    export_int8_to_executorch,
    export_with_lsq_scales_pt2e,
    get_model_size_breakdown
)


def parse_args():
    parser = argparse.ArgumentParser(description='Export model to ExecuTorch format')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to model checkpoint')
    parser.add_argument('--config', type=str, default='configs/config.yaml',
                        help='Path to config file')
    parser.add_argument('--model', type=str, default=None,
                        choices=['resnet18', 'resnet34', 'resnet50'],
                        help='Model architecture (overrides config)')
    parser.add_argument('--qat', action='store_true',
                        help='Model was trained with QAT')
    parser.add_argument('--pt2e-qat', action='store_true',
                        help='Use pure PT2E PTQ (FP32 → INT8, recommended for best accuracy)')
    parser.add_argument('--int8-lsq', action='store_true',
                        help='Export with INT8 storage using learned LSQ step_size (requires --qat)')
    parser.add_argument('--int8-lsq-pt2e', action='store_true',
                        help='Export with PT2E + LSQ scales injection (faster native INT8 ops)')
    parser.add_argument('--backend', type=str, default='xnnpack',
                        choices=['xnnpack', 'nnapi', 'vulkan', 'portable'],
                        help='ExecuTorch backend (default: xnnpack)')
    parser.add_argument('--output-dir', type=str, default='exported_models',
                        help='Output directory for exported model')
    parser.add_argument('--output-name', type=str, default=None,
                        help='Output filename (without extension)')
    parser.add_argument('--verify', action='store_true',
                        help='Verify exported model with random input')
    parser.add_argument('--calibration-samples', type=int, default=500,
                        help='Number of calibration samples for PT2E PTQ')
    return parser.parse_args()


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def load_model(config: dict, checkpoint_path: str, qat: bool) -> nn.Module:
    """Load trained model."""
    model_config = config['model']
    model_name = model_config.get('name', 'resnet18')

    # Create base model
    print(f"Creating model: {model_name}")
    model = get_resnet(
        name=model_name,
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
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)

    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'], strict=False)
    else:
        model.load_state_dict(checkpoint, strict=False)

    model.eval()
    return model


def prepare_model_for_export(model: nn.Module) -> nn.Module:
    """
    Prepare model for torch.export by ensuring all quantizers are initialized.
    """
    from src.models.quantization import LSQQuantizer

    # Run a dummy forward pass in TRAINING mode to initialize all quantizers
    print("Initializing quantizers with dummy forward pass...")
    model.train()
    with torch.no_grad():
        dummy_input = torch.randn(1, 3, 224, 224)
        _ = model(dummy_input)

    # Switch back to eval mode for export
    model.eval()

    # Verify all LSQQuantizer modules are initialized
    uninitialized = []
    for name, module in model.named_modules():
        if isinstance(module, LSQQuantizer):
            if not module.initialized.item():
                uninitialized.append(name)

    if uninitialized:
        print(f"Warning: Some quantizers are still uninitialized: {uninitialized}")
    else:
        print("All quantizers initialized successfully")

    return model


def export_to_executorch(
    model: nn.Module,
    output_path: str,
    backend: str = 'xnnpack',
    verify: bool = False
):
    """
    Export model to ExecuTorch format.

    Args:
        model: PyTorch model to export
        output_path: Path for output .pte file
        backend: ExecuTorch backend ('xnnpack', 'nnapi', 'vulkan', 'portable')
        verify: Whether to verify the exported model
    """
    try:
        from torch.export import export
        from executorch.exir import to_edge_transform_and_lower, EdgeCompileConfig
    except ImportError:
        print("\nERROR: ExecuTorch not installed.")
        print("Please install ExecuTorch:")
        print("  pip install executorch")
        print("\nAlternatively, export to ONNX format for manual conversion.")
        return export_to_onnx_fallback(model, output_path.replace('.pte', '.onnx'))

    print(f"\nExporting model to ExecuTorch format...")
    print(f"Backend: {backend}")

    # Create example input
    example_input = (torch.randn(1, 3, 224, 224),)

    # Export using torch.export
    print("Step 1: Capturing model with torch.export...")
    exported_program = export(model, example_input)

    # Get partitioner based on backend
    print(f"Step 2: Setting up {backend} backend...")
    partitioners = []

    if backend == 'xnnpack':
        try:
            from executorch.backends.xnnpack.partition.xnnpack_partitioner import XnnpackPartitioner
            partitioners.append(XnnpackPartitioner())
            print("  XNNPACK partitioner configured")
        except ImportError:
            print("  Warning: XNNPACK partitioner not available, using portable backend")

    elif backend == 'nnapi':
        try:
            from executorch.backends.xnnpack.partition.xnnpack_partitioner import XnnpackPartitioner
            partitioners.append(XnnpackPartitioner())
            print("  XNNPACK partitioner configured (NNAPI compatible)")
            print("  Note: NNAPI backend requires Android deployment")
        except ImportError:
            print("  Warning: NNAPI partitioner not available")

    elif backend == 'vulkan':
        try:
            from executorch.backends.vulkan.partition.vulkan_partitioner import VulkanPartitioner
            partitioners.append(VulkanPartitioner())
            print("  Vulkan partitioner configured")
        except ImportError:
            print("  Warning: Vulkan partitioner not available")

    # Convert to edge program and apply backend
    print("Step 3: Converting to edge program with backend optimizations...")
    edge_config = EdgeCompileConfig(_check_ir_validity=True)
    edge_program = to_edge_transform_and_lower(
        exported_program,
        compile_config=edge_config,
        partitioner=partitioners if partitioners else None
    )

    # Convert to ExecuTorch program and save
    print("Step 4: Generating ExecuTorch program...")
    exec_program = edge_program.to_executorch()

    # Save to file
    print(f"Step 5: Saving to {output_path}...")
    with open(output_path, 'wb') as f:
        f.write(exec_program.buffer)

    # Get file size
    file_size_mb = os.path.getsize(output_path) / (1024 * 1024)
    print(f"\nExport successful!")
    print(f"  Output: {output_path}")
    print(f"  Size: {file_size_mb:.2f} MB")

    # Verify if requested
    if verify:
        print("\nVerifying exported model...")
        verify_executorch_model(output_path, example_input[0])

    return output_path


def export_pt2e_qat_to_executorch(
    model: nn.Module,
    output_path: str,
    backend: str = 'xnnpack',
    config: dict = None,
    calibration_samples: int = 500,
    verify: bool = False
):
    """
    Export FP32 model to ExecuTorch using pure PT2E PTQ.

    This is the recommended approach for best accuracy:
    1. Start with FP32 pretrained model
    2. Apply PT2E quantization preparation
    3. Calibrate with real data
    4. Convert to INT8
    5. Export to ExecuTorch

    Args:
        model: FP32 pretrained model
        output_path: Output .pte file path
        backend: ExecuTorch backend
        config: Config for data loading
        calibration_samples: Samples for calibration
        verify: Whether to verify the exported model
    """
    import warnings
    warnings.filterwarnings('ignore')

    from torch.export import export, export_for_training
    from torch.ao.quantization.quantize_pt2e import prepare_pt2e, convert_pt2e
    from torch.ao.quantization.quantizer.xnnpack_quantizer import (
        XNNPACKQuantizer,
        get_symmetric_quantization_config
    )
    from executorch.exir import to_edge_transform_and_lower, EdgeCompileConfig
    from executorch.backends.xnnpack.partition.xnnpack_partitioner import XnnpackPartitioner

    print(f"\n{'='*60}")
    print("Pure PT2E PTQ Export (FP32 → INT8)")
    print("="*60)
    print(f"Backend: {backend}")
    print(f"Calibration samples: {calibration_samples}")

    model.eval()
    example_input = (torch.randn(1, 3, 224, 224),)

    # Step 1: Export for training
    print("\nStep 1: Exporting model for PT2E...")
    exported = export_for_training(model, example_input)
    gm = exported.module()

    # Step 2: Setup XNNPACK quantizer
    print("\nStep 2: Setting up XNNPACK quantizer...")
    quantizer = XNNPACKQuantizer()
    quant_config = get_symmetric_quantization_config(is_per_channel=True)
    quantizer.set_global(quant_config)
    print("  Using per-channel symmetric quantization")

    # Step 3: Prepare for quantization
    print("\nStep 3: Preparing for quantization...")
    prepared = prepare_pt2e(gm, quantizer)

    # Step 4: Calibration
    print(f"\nStep 4: Calibrating with {calibration_samples} samples...")
    calibrate_prepared_model(prepared, calibration_samples, config)

    # Step 5: Convert to INT8
    print("\nStep 5: Converting to INT8...")
    quantized = convert_pt2e(prepared)

    # Step 6: Export
    print("\nStep 6: Exporting quantized model...")
    quantized_exported = export(quantized, example_input)

    # Step 7: Setup backend
    print(f"\nStep 7: Setting up {backend} backend...")
    partitioners = []
    if backend in ['xnnpack', 'nnapi']:
        partitioners.append(XnnpackPartitioner())

    # Step 8: Convert to edge
    print("\nStep 8: Converting to edge program...")
    edge_config = EdgeCompileConfig(_check_ir_validity=False)
    try:
        edge = to_edge_transform_and_lower(
            quantized_exported,
            compile_config=edge_config,
            partitioner=partitioners if partitioners else None
        )
    except Exception as e:
        print(f"  Warning: Partitioner failed ({e})")
        edge = to_edge_transform_and_lower(
            quantized_exported,
            compile_config=edge_config,
            partitioner=None
        )

    # Step 9: Generate and save
    print("\nStep 9: Generating ExecuTorch program...")
    et_program = edge.to_executorch()

    print(f"\nStep 10: Saving to {output_path}...")
    with open(output_path, 'wb') as f:
        f.write(et_program.buffer)

    # Report
    file_size_mb = os.path.getsize(output_path) / (1024 * 1024)
    print(f"\n{'='*60}")
    print("Pure PT2E PTQ Export Successful!")
    print(f"  Output: {output_path}")
    print(f"  Size: {file_size_mb:.2f} MB")
    print(f"  Method: PT2E PTQ (calibration-based)")
    print("="*60)

    if verify:
        verify_executorch_model(output_path, example_input[0])

    return output_path


def calibrate_prepared_model(prepared_model, num_samples: int, config: dict = None):
    """Calibrate the prepared model with representative data."""
    try:
        if config is not None:
            from src.data.dataset import create_coco_dataloaders
            from src.data.augmentation import ValTransform

            print("  Loading calibration data from COCO dataset...")
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
                        print(f"    Calibrated {count}/{num_samples} samples...")

            print(f"  Calibration completed with {count} COCO images")
            return

    except Exception as e:
        print(f"  Could not load COCO data: {e}")

    # Fallback: Use random data
    print("  Using random data for calibration...")
    with torch.no_grad():
        for i in range(num_samples):
            random_input = torch.randn(1, 3, 224, 224)
            prepared_model(random_input)
            if (i + 1) % 100 == 0:
                print(f"    Calibrated {i+1}/{num_samples} samples...")

    print(f"  Calibration completed with {num_samples} random samples")


def export_to_onnx_fallback(model: nn.Module, output_path: str):
    """
    Fallback: Export model to ONNX format if ExecuTorch is not available.
    """
    print("\nFalling back to ONNX export...")

    example_input = torch.randn(1, 3, 224, 224)

    torch.onnx.export(
        model,
        example_input,
        output_path,
        export_params=True,
        opset_version=13,
        do_constant_folding=True,
        input_names=['input'],
        output_names=['output'],
        dynamic_axes={
            'input': {0: 'batch_size'},
            'output': {0: 'batch_size'}
        }
    )

    file_size_mb = os.path.getsize(output_path) / (1024 * 1024)
    print(f"\nONNX export successful!")
    print(f"  Output: {output_path}")
    print(f"  Size: {file_size_mb:.2f} MB")

    try:
        import onnx
        onnx_model = onnx.load(output_path)
        onnx.checker.check_model(onnx_model)
        print("  ONNX model verified successfully")
    except ImportError:
        print("  Note: Install 'onnx' package to verify the model")

    return output_path


def verify_executorch_model(model_path: str, example_input: torch.Tensor):
    """Verify ExecuTorch model with example input."""
    try:
        from executorch.runtime import Runtime, Program

        program = Program(model_path)
        runtime = Runtime.get()
        method = program.load_method("forward")

        inputs = [example_input.numpy()]
        outputs = method.execute(inputs)

        print(f"  Input shape: {example_input.shape}")
        print(f"  Output shape: {outputs[0].shape}")
        print("  Verification passed!")

    except ImportError:
        print("  Note: ExecuTorch runtime not available for verification")
    except Exception as e:
        print(f"  Verification failed: {e}")


def main():
    args = parse_args()

    # Load config
    config = load_config(args.config)

    # Override model name from command line if specified
    if args.model:
        config['model']['name'] = args.model

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    model_name = config['model'].get('name', 'resnet18')

    # --int8-lsq / --int8-lsq-pt2e는 QAT 모델이 필요하므로 자동으로 --qat 활성화
    if args.int8_lsq or args.int8_lsq_pt2e:
        args.qat = True

    # Determine mode string
    if args.int8_lsq:
        mode_str = 'INT8 with learned LSQ step_size'
    elif args.int8_lsq_pt2e:
        mode_str = 'PT2E + LSQ scales injection (native INT8 ops)'
    elif args.pt2e_qat:
        mode_str = 'PT2E PTQ (FP32 → INT8)'
    elif args.qat:
        mode_str = 'QAT (fake quantization, FP32)'
    else:
        mode_str = 'FP32'

    # Load model
    model = load_model(config, args.checkpoint, args.qat)

    # Prepare model for export (initialize quantizers)
    if args.qat:
        model = prepare_model_for_export(model)

    print(f"\nModel info:")
    print(f"  Parameters: {count_parameters(model):,}")
    print(f"  Size (PyTorch): {get_model_size_mb(model):.2f} MB")
    print(f"  Mode: {mode_str}")

    # Determine output filename
    if args.output_name:
        output_name = args.output_name
    else:
        if args.int8_lsq:
            model_type = 'int8_lsq'
        elif args.int8_lsq_pt2e:
            model_type = 'int8_lsq_pt2e'
        elif args.pt2e_qat:
            model_type = 'pt2e_ptq_int8'
        elif args.qat:
            model_type = 'qat_fp32'
        else:
            model_type = 'fp32'
        output_name = f"{model_name}_multilabel_{model_type}_{args.backend}"

    output_path = os.path.join(args.output_dir, f"{output_name}.pte")

    # Export model with appropriate method
    if args.int8_lsq:
        # ★ INT8 storage with learned LSQ step_size
        print("\n" + "="*60)
        print("INT8 Export with Learned LSQ step_size")
        print("="*60)

        # Convert QAT model → INT8 storage model
        print("\nConverting LSQ QAT model to INT8 storage...")
        int8_model = convert_lsq_to_int8(model)

        # Show size comparison
        size_info = get_model_size_breakdown(int8_model)
        original_size = get_model_size_mb(model)
        print(f"\nSize comparison:")
        print(f"  Original QAT (FP32): {original_size:.2f} MB")
        print(f"  INT8 model:          {size_info['total_mb']:.2f} MB")
        print(f"    - INT8 weights:    {size_info['int8_mb']:.2f} MB")
        print(f"    - FP32 (bias/BN):  {size_info['fp32_mb']:.2f} MB")
        print(f"  Compression ratio:   {original_size / size_info['total_mb']:.1f}x")

        # Save INT8 PyTorch model (.pt) alongside .pte
        pt_path = output_path.replace('.pte', '_int8.pt')
        torch.save({
            'model_state_dict': int8_model.state_dict(),
            'model_name': model_name,
            'quantization': 'int8_lsq',
            'original_checkpoint': args.checkpoint,
        }, pt_path)
        pt_size_mb = os.path.getsize(pt_path) / (1024 * 1024)
        print(f"\nINT8 PyTorch model saved: {pt_path} ({pt_size_mb:.2f} MB)")

        # Export to ExecuTorch
        export_int8_to_executorch(
            int8_model=int8_model,
            output_path=output_path,
            backend=args.backend,
            verify=args.verify
        )

    elif args.int8_lsq_pt2e:
        # PT2E export with LSQ scales injection (native INT8 ops)
        export_with_lsq_scales_pt2e(
            qat_model=model,
            output_path=output_path,
            config=config,
            backend=args.backend,
            calibration_samples=args.calibration_samples,
            verify=args.verify
        )

    elif args.pt2e_qat:
        # Pure PT2E export (uses FP32 model + calibration)
        fp32_model = get_resnet(model_name, num_classes=80, pretrained=False)
        checkpoint = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
        if 'model_state_dict' in checkpoint:
            fp32_model.load_state_dict(checkpoint['model_state_dict'], strict=False)
        else:
            fp32_model.load_state_dict(checkpoint, strict=False)
        fp32_model.eval()

        export_pt2e_qat_to_executorch(
            model=fp32_model,
            output_path=output_path,
            backend=args.backend,
            config=config,
            calibration_samples=args.calibration_samples,
            verify=args.verify
        )
    else:
        # Standard export (FP32 or fake-quantized QAT)
        export_to_executorch(
            model=model,
            output_path=output_path,
            backend=args.backend,
            verify=args.verify
        )


if __name__ == '__main__':
    main()

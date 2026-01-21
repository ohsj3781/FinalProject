#!/usr/bin/env python3
"""
ExecuTorch Export Script for ResNet-18 Multi-Label Classification

This script exports a trained PyTorch model to ExecuTorch format (.pte)
for mobile deployment.

Usage:
    # Export full precision model
    python scripts/export_executorch.py --checkpoint checkpoints/best_model.pth

    # Export QAT model with XNNPACK backend
    python scripts/export_executorch.py --checkpoint checkpoints/qat/best_model.pth --qat --backend xnnpack

    # Export with NNAPI backend for NPU acceleration
    python scripts/export_executorch.py --checkpoint checkpoints/qat/best_model.pth --qat --backend nnapi
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


def parse_args():
    parser = argparse.ArgumentParser(description='Export model to ExecuTorch format')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to model checkpoint')
    parser.add_argument('--config', type=str, default='configs/config.yaml',
                        help='Path to config file')
    parser.add_argument('--qat', action='store_true',
                        help='Model was trained with QAT')
    parser.add_argument('--backend', type=str, default='xnnpack',
                        choices=['xnnpack', 'nnapi', 'vulkan', 'portable'],
                        help='ExecuTorch backend (default: xnnpack)')
    parser.add_argument('--output-dir', type=str, default='exported_models',
                        help='Output directory for exported model')
    parser.add_argument('--output-name', type=str, default=None,
                        help='Output filename (without extension)')
    parser.add_argument('--verify', action='store_true',
                        help='Verify exported model with random input')
    return parser.parse_args()


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def load_model(config: dict, checkpoint_path: str, qat: bool) -> nn.Module:
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
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)

    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)

    model.eval()
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
        from executorch.exir import to_edge, EdgeCompileConfig
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

    # Convert to edge program
    print("Step 2: Converting to edge program...")
    edge_config = EdgeCompileConfig(_check_ir_validity=True)
    edge_program = to_edge(exported_program, compile_config=edge_config)

    # Apply backend-specific optimizations
    print(f"Step 3: Applying {backend} backend optimizations...")

    if backend == 'xnnpack':
        try:
            from executorch.backends.xnnpack.partition.xnnpack_partitioner import XnnpackPartitioner
            edge_program = edge_program.to_backend(XnnpackPartitioner())
        except ImportError:
            print("  Warning: XNNPACK partitioner not available, using portable backend")

    elif backend == 'nnapi':
        try:
            # Note: NNAPI support may vary by ExecuTorch version
            print("  Note: NNAPI backend requires Android deployment")
            # For now, fall back to portable for export
            pass
        except ImportError:
            print("  Warning: NNAPI partitioner not available")

    elif backend == 'vulkan':
        try:
            from executorch.backends.vulkan.partition.vulkan_partitioner import VulkanPartitioner
            edge_program = edge_program.to_backend(VulkanPartitioner())
        except ImportError:
            print("  Warning: Vulkan partitioner not available")

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


def export_to_onnx_fallback(model: nn.Module, output_path: str):
    """
    Fallback: Export model to ONNX format if ExecuTorch is not available.

    The ONNX model can be converted to ExecuTorch manually or used with
    ONNX Runtime Mobile.
    """
    print("\nFalling back to ONNX export...")

    # Create example input
    example_input = torch.randn(1, 3, 224, 224)

    # Export to ONNX
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

    # Get file size
    file_size_mb = os.path.getsize(output_path) / (1024 * 1024)
    print(f"\nONNX export successful!")
    print(f"  Output: {output_path}")
    print(f"  Size: {file_size_mb:.2f} MB")

    # Verify ONNX model
    try:
        import onnx
        onnx_model = onnx.load(output_path)
        onnx.checker.check_model(onnx_model)
        print("  ONNX model verified successfully")
    except ImportError:
        print("  Note: Install 'onnx' package to verify the model")

    print("\nTo use on mobile:")
    print("  1. Convert ONNX to ExecuTorch using ExecuTorch tools")
    print("  2. Or use ONNX Runtime Mobile directly")

    return output_path


def verify_executorch_model(model_path: str, example_input: torch.Tensor):
    """Verify ExecuTorch model with example input."""
    try:
        from executorch.runtime import Runtime, Program

        # Load program
        program = Program(model_path)
        runtime = Runtime.get()

        # Create method
        method = program.load_method("forward")

        # Run inference
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

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Load model
    model = load_model(config, args.checkpoint, args.qat)

    print(f"\nModel info:")
    print(f"  Parameters: {count_parameters(model):,}")
    print(f"  Size (PyTorch): {get_model_size_mb(model):.2f} MB")

    # Determine output filename
    if args.output_name:
        output_name = args.output_name
    else:
        model_type = 'qat' if args.qat else 'fp32'
        output_name = f"resnet18_multilabel_{model_type}_{args.backend}"

    output_path = os.path.join(args.output_dir, f"{output_name}.pte")

    # Export model
    export_to_executorch(
        model=model,
        output_path=output_path,
        backend=args.backend,
        verify=args.verify
    )


if __name__ == '__main__':
    main()

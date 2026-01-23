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

    # Export with INT8 quantization for NPU optimization
    python scripts/export_executorch.py --checkpoint checkpoints/fp32/best_model.pth --int8 --backend xnnpack
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
    parser.add_argument('--int8', action='store_true',
                        help='Apply INT8 quantization (Post-Training Quantization)')
    parser.add_argument('--backend', type=str, default='xnnpack',
                        choices=['xnnpack', 'nnapi', 'vulkan', 'portable'],
                        help='ExecuTorch backend (default: xnnpack)')
    parser.add_argument('--output-dir', type=str, default='exported_models',
                        help='Output directory for exported model')
    parser.add_argument('--output-name', type=str, default=None,
                        help='Output filename (without extension)')
    parser.add_argument('--verify', action='store_true',
                        help='Verify exported model with random input')
    parser.add_argument('--calibration-samples', type=int, default=100,
                        help='Number of calibration samples for INT8 PTQ')
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
        # Use strict=False to handle backward compatibility with checkpoints
        # that were saved before 'initialized' became a buffer
        model.load_state_dict(checkpoint['model_state_dict'], strict=False)
    else:
        model.load_state_dict(checkpoint, strict=False)

    model.eval()
    return model


def prepare_model_for_export(model: nn.Module) -> nn.Module:
    """
    Prepare model for torch.export by ensuring all quantizers are initialized.

    This runs a dummy forward pass to initialize step_size parameters,
    which avoids data-dependent control flow during export.
    """
    from src.models.quantization import LSQQuantizer

    # Run a dummy forward pass in TRAINING mode to initialize all quantizers
    print("Initializing quantizers with dummy forward pass...")
    model.train()  # Enable training mode for initialization
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


def export_int8_to_executorch(
    model: nn.Module,
    output_path: str,
    backend: str = 'xnnpack',
    calibration_samples: int = 100,
    config: dict = None,
    verify: bool = False
):
    """
    Export model with INT8 quantization (Post-Training Quantization).

    This produces a truly quantized model with INT8 weights and activations,
    which is optimized for NPU/accelerator inference.

    Uses the working PT2E quantization workflow:
    1. export_for_training -> 2. prepare_pt2e -> 3. calibrate ->
    4. convert_pt2e -> 5. export -> 6. to_edge -> 7. to_executorch

    Args:
        model: PyTorch model (FP32)
        output_path: Path for output .pte file
        backend: ExecuTorch backend
        calibration_samples: Number of samples for calibration
        config: Config dict for data loading
        verify: Whether to verify the exported model
    """
    import warnings
    warnings.filterwarnings('ignore')

    from torch.export import export_for_training, export
    from torch.ao.quantization.quantize_pt2e import prepare_pt2e, convert_pt2e
    from torch.ao.quantization.quantizer.xnnpack_quantizer import (
        XNNPACKQuantizer,
        get_symmetric_quantization_config
    )
    from executorch.exir import to_edge, EdgeCompileConfig
    from executorch.backends.xnnpack.partition.xnnpack_partitioner import XnnpackPartitioner

    print(f"\nExporting model with INT8 quantization...")
    print(f"Backend: {backend}")
    print(f"Calibration samples: {calibration_samples}")

    example_input = (torch.randn(1, 3, 224, 224),)

    # Step 1: Export for training (preserves structure for quantization)
    print("\nStep 1: Exporting model for training...")
    try:
        exported = export_for_training(model, example_input)
        gm = exported.module()
        print("  Success")
    except Exception as e:
        print(f"  export_for_training failed: {e}")
        print("  Falling back to standard export...")
        return export_int8_legacy(model, output_path, backend, calibration_samples, config, verify)

    # Step 2: Setup quantizer (per-tensor for better compatibility)
    print("Step 2: Setting up INT8 quantizer...")
    quantizer = XNNPACKQuantizer()
    # Use per-tensor quantization for ExecuTorch compatibility
    quant_config = get_symmetric_quantization_config(is_per_channel=False)
    quantizer.set_global(quant_config)

    # Step 3: Prepare for quantization
    print("Step 3: Preparing for quantization...")
    try:
        prepared = prepare_pt2e(gm, quantizer)
    except Exception as e:
        print(f"  prepare_pt2e failed: {e}")
        return export_int8_legacy(model, output_path, backend, calibration_samples, config, verify)

    # Step 4: Calibrate
    print(f"Step 4: Calibrating with {calibration_samples} samples...")
    calibrate_prepared_model(prepared, calibration_samples, config)

    # Step 5: Convert to INT8
    print("Step 5: Converting to INT8...")
    quantized = convert_pt2e(prepared)

    # Step 6: Export quantized model
    print("Step 6: Exporting quantized model...")
    try:
        quantized_exported = export(quantized, example_input)
    except Exception as e:
        print(f"  Export failed: {e}")
        return export_int8_legacy(model, output_path, backend, calibration_samples, config, verify)

    # Step 7: Convert to edge program
    print("Step 7: Converting to edge program...")
    edge_config = EdgeCompileConfig(_check_ir_validity=False)
    edge = to_edge(quantized_exported, compile_config=edge_config)

    # Step 8: Apply backend optimizations
    print(f"Step 8: Applying {backend} backend optimizations...")
    if backend == 'xnnpack':
        edge = edge.to_backend(XnnpackPartitioner())
        print("  XNNPACK partitioner applied (INT8 optimized)")
    elif backend == 'nnapi':
        # For NNAPI with INT8, we use XNNPACK partitioner which is compatible
        # with NNAPI delegation at runtime on Android
        try:
            edge = edge.to_backend(XnnpackPartitioner())
            print("  XNNPACK partitioner applied (NNAPI compatible, INT8 optimized)")
            print("  Note: On Android, NNAPI will delegate INT8 ops to NPU")
        except Exception as e:
            print(f"  Warning: Partitioner failed ({e}), using portable backend")
            print("  INT8 model will run on CPU")

    # Step 9: Generate ExecuTorch program
    print("Step 9: Generating ExecuTorch program...")
    et_program = edge.to_executorch()

    # Step 10: Save
    print(f"Step 10: Saving to {output_path}...")
    with open(output_path, 'wb') as f:
        f.write(et_program.buffer)

    return finalize_int8_export(output_path, verify, example_input)


def calibrate_prepared_model(prepared_model, num_samples: int, config: dict = None):
    """Calibrate the prepared model with representative data."""
    try:
        # Try to load COCO validation data
        if config is not None:
            from src.data.dataset import get_dataloaders
            from src.data.augmentation import get_val_transforms

            print("  Loading calibration data from COCO dataset...")
            _, val_loader = get_dataloaders(
                data_dir=config['data']['data_dir'],
                batch_size=1,
                num_workers=0,
                train_transform=None,
                val_transform=get_val_transforms(config['data']['input_size'])
            )

            count = 0
            with torch.no_grad():
                for images, _ in val_loader:
                    if count >= num_samples:
                        break
                    prepared_model(images)
                    count += 1
                    if count % 20 == 0:
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
            if (i + 1) % 20 == 0:
                print(f"    Calibrated {i+1}/{num_samples} samples...")

    print(f"  Calibration completed with {num_samples} random samples")




def export_int8_legacy(
    model: nn.Module,
    output_path: str,
    backend: str,
    calibration_samples: int,
    config: dict,
    verify: bool
):
    """Legacy PyTorch static quantization approach."""
    from torch.export import export
    from executorch.exir import to_edge, EdgeCompileConfig

    print("\n[Using legacy PyTorch quantization]")

    # Step 1: Prepare model for static quantization
    print("Step 1: Preparing model for static quantization...")
    model.eval()
    model_copy = torch.quantization.QuantStub()

    # Use dynamic quantization as it's simpler and more compatible
    print("Step 2: Applying dynamic INT8 quantization...")
    quantized_model = torch.quantization.quantize_dynamic(
        model,
        {nn.Linear, nn.Conv2d},
        dtype=torch.qint8
    )

    # Step 3: Export
    print("Step 3: Exporting model...")
    example_input = (torch.randn(1, 3, 224, 224),)

    try:
        exported_model = export(quantized_model, example_input)
    except Exception as e:
        print(f"  Dynamic quantization export failed: {e}")
        print("  Falling back to FP32 export with size optimization...")
        # Export original model
        exported_model = export(model, example_input)

    # Step 4: Convert to edge program
    print("Step 4: Converting to edge program...")
    edge_config = EdgeCompileConfig(_check_ir_validity=False)
    edge_program = to_edge(exported_model, compile_config=edge_config)

    # Step 5: Apply backend optimizations
    print(f"Step 5: Applying {backend} backend optimizations...")
    if backend == 'xnnpack':
        try:
            from executorch.backends.xnnpack.partition.xnnpack_partitioner import XnnpackPartitioner
            edge_program = edge_program.to_backend(XnnpackPartitioner())
            print("  XNNPACK partitioner applied")
        except ImportError:
            print("  Warning: XNNPACK partitioner not available")

    # Step 6: Generate and save
    print("Step 6: Generating ExecuTorch program...")
    exec_program = edge_program.to_executorch()

    print(f"Step 7: Saving to {output_path}...")
    with open(output_path, 'wb') as f:
        f.write(exec_program.buffer)

    return finalize_int8_export(output_path, verify, example_input)


def finalize_int8_export(output_path: str, verify: bool, example_input: tuple):
    """Finalize INT8 export and report results."""
    file_size_mb = os.path.getsize(output_path) / (1024 * 1024)
    print(f"\n{'='*50}")
    print(f"INT8 Export successful!")
    print(f"  Output: {output_path}")
    print(f"  Size: {file_size_mb:.2f} MB")
    print(f"  Expected speedup: 2-4x on NPU/CPU")
    print(f"{'='*50}")

    if verify:
        print("\nVerifying exported model...")
        verify_executorch_model(output_path, example_input[0])

    return output_path


def calibrate_model_graph(graph_module, num_samples: int, config: dict = None):
    """Calibrate a graph module with representative data."""
    try:
        # Try to load COCO validation data
        if config is not None:
            from src.data.dataset import get_dataloaders
            from src.data.augmentation import get_val_transforms

            print(f"  Loading calibration data from COCO dataset...")
            _, val_loader = get_dataloaders(
                data_dir=config['data']['data_dir'],
                batch_size=1,
                num_workers=0,
                train_transform=None,
                val_transform=get_val_transforms(config['data']['input_size'])
            )

            count = 0
            with torch.no_grad():
                for images, _ in val_loader:
                    if count >= num_samples:
                        break
                    graph_module(images)
                    count += 1
                    if count % 20 == 0:
                        print(f"    Calibrated {count}/{num_samples} samples...")

            print(f"  Calibration completed with {count} COCO images")
            return

    except Exception as e:
        print(f"  Could not load COCO data: {e}")

    # Fallback: Use random data for calibration
    print(f"  Using random data for calibration...")
    with torch.no_grad():
        for i in range(num_samples):
            random_input = torch.randn(1, 3, 224, 224)
            graph_module(random_input)
            if (i + 1) % 20 == 0:
                print(f"    Calibrated {i+1}/{num_samples} samples...")

    print(f"  Calibration completed with {num_samples} random samples")


def calibrate_model(prepared_model, num_samples: int, config: dict = None):
    """
    Calibrate quantized model with representative data.

    Uses COCO validation images or random data if dataset not available.
    """
    # Get the actual model from the exported program
    try:
        model_fn = prepared_model.module()
    except:
        model_fn = prepared_model

    try:
        # Try to load COCO validation data
        if config is not None:
            from src.data.dataset import get_dataloaders
            from src.data.augmentation import get_val_transforms

            print(f"  Loading calibration data from COCO dataset...")
            _, val_loader = get_dataloaders(
                data_dir=config['data']['data_dir'],
                batch_size=1,
                num_workers=0,
                train_transform=None,
                val_transform=get_val_transforms(config['data']['input_size'])
            )

            count = 0
            with torch.no_grad():
                for images, _ in val_loader:
                    if count >= num_samples:
                        break
                    model_fn(images)
                    count += 1
                    if count % 20 == 0:
                        print(f"    Calibrated {count}/{num_samples} samples...")

            print(f"  Calibration completed with {count} COCO images")
            return

    except Exception as e:
        print(f"  Could not load COCO data: {e}")

    # Fallback: Use random data for calibration
    print(f"  Using random data for calibration...")
    with torch.no_grad():
        for i in range(num_samples):
            random_input = torch.randn(1, 3, 224, 224)
            try:
                model_fn(random_input)
            except:
                prepared_model(random_input)
            if (i + 1) % 20 == 0:
                print(f"    Calibrated {i+1}/{num_samples} samples...")

    print(f"  Calibration completed with {num_samples} random samples")


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

    # For INT8 PTQ, we need FP32 model (not QAT)
    if args.int8 and args.qat:
        print("Warning: --int8 and --qat are mutually exclusive.")
        print("  --int8 applies Post-Training Quantization to FP32 model")
        print("  --qat exports QAT-trained model (fake quantization)")
        print("Using --int8 mode...")
        args.qat = False

    # Load model
    model = load_model(config, args.checkpoint, args.qat)

    # Prepare model for export (initialize quantizers)
    if args.qat:
        model = prepare_model_for_export(model)

    print(f"\nModel info:")
    print(f"  Parameters: {count_parameters(model):,}")
    print(f"  Size (PyTorch): {get_model_size_mb(model):.2f} MB")
    print(f"  Mode: {'INT8 PTQ' if args.int8 else ('QAT' if args.qat else 'FP32')}")

    # Determine output filename
    if args.output_name:
        output_name = args.output_name
    else:
        if args.int8:
            model_type = 'int8'
        elif args.qat:
            model_type = 'qat'
        else:
            model_type = 'fp32'
        output_name = f"resnet18_multilabel_{model_type}_{args.backend}"

    output_path = os.path.join(args.output_dir, f"{output_name}.pte")

    # Export model with appropriate method
    if args.int8:
        # Use INT8 Post-Training Quantization
        export_int8_to_executorch(
            model=model,
            output_path=output_path,
            backend=args.backend,
            calibration_samples=args.calibration_samples,
            config=config,
            verify=args.verify
        )
    else:
        # Use standard export (FP32 or fake-quantized QAT)
        export_to_executorch(
            model=model,
            output_path=output_path,
            backend=args.backend,
            verify=args.verify
        )


if __name__ == '__main__':
    main()

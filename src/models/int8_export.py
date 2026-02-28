"""
INT8 Export Module - LSQ step_size를 활용한 진짜 INT8 변환

핵심 아이디어:
1. LSQ 학습으로 최적화된 step_size를 그대로 사용
2. Weights를 실제 INT8로 저장 (4x 크기 감소)
3. PT2E quantization에 learned step_size 주입

수식:
- 양자화: w_int8 = round(clamp(w / step_size, -Qn, Qp))
- 역양자화: w_fp32 = w_int8 * step_size
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple, Optional
import copy


def extract_lsq_scales(qat_model: nn.Module) -> Dict[str, Dict[str, torch.Tensor]]:
    """
    LSQ QAT 모델에서 학습된 step_size (scale) 추출

    Returns:
        {
            'layer_name': {
                'weight_scale': tensor,  # per-channel [out_channels]
                'weight_qparams': (Q_N, Q_P),
                'act_scale': tensor,     # per-tensor scalar
                'act_qparams': (Q_N, Q_P),
            }
        }
    """
    from src.models.quantization import QuantizedConv2d, QuantizedLinear

    scales = {}

    for name, module in qat_model.named_modules():
        if isinstance(module, (QuantizedConv2d, QuantizedLinear)):
            layer_scales = {}

            # Weight quantizer
            wq = module.weight_quantizer
            if hasattr(wq, 'step_size') and wq.initialized:
                layer_scales['weight_scale'] = wq.step_size.data.clone()
                layer_scales['weight_qparams'] = (wq.Q_N, wq.Q_P)

            # Activation quantizer
            aq = module.act_quantizer
            if hasattr(aq, 'step_size') and aq.initialized:
                layer_scales['act_scale'] = aq.step_size.data.clone()
                layer_scales['act_qparams'] = (aq.Q_N, aq.Q_P)

            if layer_scales:
                scales[name] = layer_scales

    return scales


def quantize_weight_to_int8(
    weight: torch.Tensor,
    scale: torch.Tensor,
    Q_N: int = 128,
    Q_P: int = 127
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    FP32 weight를 INT8로 양자화 (LSQ step_size 사용)

    Args:
        weight: FP32 weights [out_ch, in_ch, H, W] or [out, in]
        scale: per-channel scale [out_ch] or scalar
        Q_N, Q_P: quantization bounds (default: signed 8-bit)

    Returns:
        (weight_int8, scale)
    """
    # Per-channel scale 확장
    if scale.dim() == 1 and weight.dim() == 4:
        scale_expanded = scale.view(-1, 1, 1, 1)
    elif scale.dim() == 1 and weight.dim() == 2:
        scale_expanded = scale.view(-1, 1)
    else:
        scale_expanded = scale

    # 양자화: round(clamp(w/s, -Qn, Qp))
    w_scaled = weight / scale_expanded
    w_clipped = torch.clamp(w_scaled, -Q_N, Q_P)
    w_int8 = torch.round(w_clipped).to(torch.int8)

    return w_int8, scale


class Int8Conv2d(nn.Module):
    """
    INT8 weights를 저장하는 Conv2d (activation quantization 포함)

    - weights: INT8 저장 (4x 크기 감소)
    - weight_scale: FP32 저장 (per-channel)
    - act_scale: FP32 저장 (per-tensor, activation quantization용)
    - forward: act_quant → dequantize_weight → conv2d
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        padding: int = 0,
        dilation: int = 1,
        groups: int = 1,
        bias: bool = False
    ):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups

        # INT8 weights (buffer, not parameter)
        self.register_buffer(
            'weight_int8',
            torch.zeros(out_channels, in_channels // groups, kernel_size, kernel_size, dtype=torch.int8)
        )

        # Per-channel weight scale
        self.register_buffer('weight_scale', torch.ones(out_channels))

        # Activation quantization parameters (unsigned [0, 255])
        self.register_buffer('act_scale', torch.tensor(0.0))
        self.has_act_quant: bool = False  # static Python bool for torch.export 호환
        self.act_qn: float = 0.0
        self.act_qp: float = 255.0

        # Optional bias (FP32)
        if bias:
            self.register_buffer('bias', torch.zeros(out_channels))
        else:
            self.register_buffer('bias', None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 1. Activation quantization: round(clip(x/s, Q_N, Q_P)) * s
        if self.has_act_quant:
            x = torch.round(torch.clamp(x / self.act_scale, -self.act_qn, self.act_qp)) * self.act_scale

        # 2. Dequantize weights: w_fp32 = w_int8 * scale
        scale = self.weight_scale.view(-1, 1, 1, 1)
        weight_fp32 = self.weight_int8.float() * scale

        return F.conv2d(
            x, weight_fp32, self.bias,
            self.stride, self.padding, self.dilation, self.groups
        )

    @classmethod
    def from_quantized_conv2d(cls, qconv, weight_scale: torch.Tensor, Q_N: int, Q_P: int,
                               act_scale: Optional[torch.Tensor] = None,
                               act_Q_N: int = 0, act_Q_P: int = 255):
        """QuantizedConv2d에서 변환 (activation scale 포함)"""
        conv = qconv.conv

        new_layer = cls(
            conv.in_channels, conv.out_channels,
            conv.kernel_size[0],
            stride=conv.stride[0],
            padding=conv.padding[0],
            dilation=conv.dilation[0],
            groups=conv.groups,
            bias=conv.bias is not None
        )

        # INT8로 양자화
        weight_int8, scale = quantize_weight_to_int8(
            conv.weight.data, weight_scale, Q_N, Q_P
        )

        new_layer.weight_int8.copy_(weight_int8)
        new_layer.weight_scale.copy_(scale)

        # Activation quantization parameters
        if act_scale is not None:
            new_layer.act_scale.copy_(act_scale.detach())
            new_layer.has_act_quant = True
            new_layer.act_qn = float(act_Q_N)
            new_layer.act_qp = float(act_Q_P)

        if conv.bias is not None:
            new_layer.bias = conv.bias.data.clone()

        return new_layer


class Int8Linear(nn.Module):
    """
    INT8 weights를 저장하는 Linear (activation quantization 포함)
    """

    def __init__(self, in_features: int, out_features: int, bias: bool = True):
        super().__init__()

        self.in_features = in_features
        self.out_features = out_features

        # INT8 weights
        self.register_buffer(
            'weight_int8',
            torch.zeros(out_features, in_features, dtype=torch.int8)
        )

        # Per-channel scale
        self.register_buffer('weight_scale', torch.ones(out_features))

        # Activation quantization parameters (unsigned [0, 255])
        self.register_buffer('act_scale', torch.tensor(0.0))
        self.has_act_quant: bool = False  # static Python bool for torch.export 호환
        self.act_qn: float = 0.0
        self.act_qp: float = 255.0

        # Optional bias
        if bias:
            self.register_buffer('bias', torch.zeros(out_features))
        else:
            self.register_buffer('bias', None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 1. Activation quantization: round(clip(x/s, Q_N, Q_P)) * s
        if self.has_act_quant:
            x = torch.round(torch.clamp(x / self.act_scale, -self.act_qn, self.act_qp)) * self.act_scale

        # 2. Dequantize weights
        scale = self.weight_scale.view(-1, 1)
        weight_fp32 = self.weight_int8.float() * scale

        return F.linear(x, weight_fp32, self.bias)

    @classmethod
    def from_quantized_linear(cls, qlinear, weight_scale: torch.Tensor, Q_N: int, Q_P: int,
                               act_scale: Optional[torch.Tensor] = None,
                               act_Q_N: int = 0, act_Q_P: int = 255):
        """QuantizedLinear에서 변환 (activation scale 포함)"""
        linear = qlinear.linear

        new_layer = cls(
            linear.in_features, linear.out_features,
            bias=linear.bias is not None
        )

        # INT8로 양자화
        weight_int8, scale = quantize_weight_to_int8(
            linear.weight.data, weight_scale, Q_N, Q_P
        )

        new_layer.weight_int8.copy_(weight_int8)
        new_layer.weight_scale.copy_(scale)

        # Activation quantization parameters
        if act_scale is not None:
            new_layer.act_scale.copy_(act_scale.detach())
            new_layer.has_act_quant = True
            new_layer.act_qn = float(act_Q_N)
            new_layer.act_qp = float(act_Q_P)

        if linear.bias is not None:
            new_layer.bias = linear.bias.data.clone()

        return new_layer


def convert_lsq_to_int8(qat_model: nn.Module) -> nn.Module:
    """
    LSQ QAT 모델을 INT8 저장 모델로 변환

    - QuantizedConv2d → Int8Conv2d
    - QuantizedLinear → Int8Linear
    - 학습된 step_size를 scale로 사용
    """
    from src.models.quantization import QuantizedConv2d, QuantizedLinear

    model = copy.deepcopy(qat_model)
    model.eval()

    # 먼저 step_size 초기화 (dummy forward)
    device = next(model.parameters()).device
    with torch.no_grad():
        dummy_input = torch.randn(1, 3, 224, 224, device=device)
        model.train()
        _ = model(dummy_input)
        model.eval()

    converted_count = 0
    act_quant_count = 0

    def replace_modules(parent_module: nn.Module, prefix: str = ''):
        nonlocal converted_count, act_quant_count

        for name, module in list(parent_module.named_children()):
            full_name = f"{prefix}.{name}" if prefix else name

            if isinstance(module, QuantizedConv2d):
                # 학습된 step_size 추출
                wq = module.weight_quantizer
                if wq.initialized:
                    weight_scale = wq.step_size.data
                    Q_N, Q_P = wq.Q_N, wq.Q_P

                    # Activation scale 추출
                    aq = module.act_quantizer
                    act_scale = None
                    act_Q_N, act_Q_P = 0, 255
                    if hasattr(aq, 'step_size') and aq.initialized:
                        act_scale = aq.step_size.data
                        act_Q_N, act_Q_P = aq.Q_N, aq.Q_P
                        act_quant_count += 1

                    # Int8Conv2d로 변환 (activation scale 포함)
                    int8_conv = Int8Conv2d.from_quantized_conv2d(
                        module, weight_scale, Q_N, Q_P,
                        act_scale=act_scale, act_Q_N=act_Q_N, act_Q_P=act_Q_P
                    )
                    setattr(parent_module, name, int8_conv)
                    converted_count += 1

            elif isinstance(module, QuantizedLinear):
                wq = module.weight_quantizer
                if wq.initialized:
                    weight_scale = wq.step_size.data
                    Q_N, Q_P = wq.Q_N, wq.Q_P

                    # Activation scale 추출
                    aq = module.act_quantizer
                    act_scale = None
                    act_Q_N, act_Q_P = 0, 255
                    if hasattr(aq, 'step_size') and aq.initialized:
                        act_scale = aq.step_size.data
                        act_Q_N, act_Q_P = aq.Q_N, aq.Q_P
                        act_quant_count += 1

                    int8_linear = Int8Linear.from_quantized_linear(
                        module, weight_scale, Q_N, Q_P,
                        act_scale=act_scale, act_Q_N=act_Q_N, act_Q_P=act_Q_P
                    )
                    setattr(parent_module, name, int8_linear)
                    converted_count += 1
            else:
                # 재귀적으로 자식 모듈 처리
                replace_modules(module, full_name)

    replace_modules(model)

    print(f"Converted {converted_count} layers to INT8")
    print(f"  - {act_quant_count} layers with activation quantization")

    return model


def get_model_size_breakdown(model: nn.Module) -> Dict[str, float]:
    """모델 크기 상세 분석"""
    int8_size = 0
    fp32_size = 0

    for name, param in model.named_parameters():
        fp32_size += param.numel() * 4  # FP32 = 4 bytes

    for name, buffer in model.named_buffers():
        if buffer.dtype == torch.int8:
            int8_size += buffer.numel() * 1  # INT8 = 1 byte
        elif buffer.dtype in [torch.float32, torch.float]:
            fp32_size += buffer.numel() * 4

    total_mb = (int8_size + fp32_size) / (1024 * 1024)

    return {
        'int8_mb': int8_size / (1024 * 1024),
        'fp32_mb': fp32_size / (1024 * 1024),
        'total_mb': total_mb
    }


def export_int8_to_executorch(
    int8_model: nn.Module,
    output_path: str,
    backend: str = 'xnnpack',
    verify: bool = False
) -> str:
    """
    INT8 저장 모델을 ExecuTorch로 export

    Note: 이 방식은 dequantize가 forward에서 발생하므로
          XNNPACK의 INT8 최적화를 사용하지 못함.
          하지만 learned step_size를 정확히 보존함.
    """
    import os
    from torch.export import export
    from executorch.exir import to_edge_transform_and_lower, EdgeCompileConfig

    print(f"\n{'='*60}")
    print("INT8 Storage Model Export (Learned step_size 사용)")
    print("="*60)

    int8_model.eval()
    example_input = (torch.randn(1, 3, 224, 224),)

    # 크기 분석
    size_info = get_model_size_breakdown(int8_model)
    print(f"Model size breakdown:")
    print(f"  INT8 weights: {size_info['int8_mb']:.2f} MB")
    print(f"  FP32 params:  {size_info['fp32_mb']:.2f} MB")
    print(f"  Total:        {size_info['total_mb']:.2f} MB")

    # Export
    print("\nStep 1: Exporting with torch.export...")
    exported = export(int8_model, example_input)

    # Backend 설정
    print(f"\nStep 2: Setting up {backend} backend...")
    partitioners = []

    if backend == 'xnnpack':
        try:
            from executorch.backends.xnnpack.partition.xnnpack_partitioner import XnnpackPartitioner
            partitioners.append(XnnpackPartitioner())
            print("  XNNPACK partitioner configured")
        except ImportError:
            print("  Warning: XNNPACK not available")

    # Edge 변환
    print("\nStep 3: Converting to edge program...")
    edge_config = EdgeCompileConfig(_check_ir_validity=False)

    try:
        edge = to_edge_transform_and_lower(
            exported,
            compile_config=edge_config,
            partitioner=partitioners if partitioners else None
        )
    except Exception as e:
        print(f"  Partitioner failed: {e}")
        print("  Using portable backend...")
        edge = to_edge_transform_and_lower(
            exported,
            compile_config=edge_config,
            partitioner=None
        )

    # ExecuTorch 생성
    print("\nStep 4: Generating ExecuTorch program...")
    et_program = edge.to_executorch()

    # 저장
    print(f"\nStep 5: Saving to {output_path}...")
    with open(output_path, 'wb') as f:
        f.write(et_program.buffer)

    file_size_mb = os.path.getsize(output_path) / (1024 * 1024)

    print(f"\n{'='*60}")
    print("Export Successful!")
    print(f"  Output: {output_path}")
    print(f"  Size: {file_size_mb:.2f} MB")
    print(f"  Method: INT8 storage + runtime dequantize")
    print(f"  Scales: Learned LSQ step_size (정확히 보존)")
    print("="*60)

    return output_path


# =============================================================================
# PT2E에 LSQ scales 주입하는 방식 (XNNPACK 최적화 활용)
# =============================================================================

class FixedScaleObserver(torch.ao.quantization.observer.ObserverBase):
    """
    고정된 scale을 반환하는 Observer (LSQ step_size 주입용)

    PT2E의 observer를 이것으로 교체하면 calibration 대신
    우리가 학습한 step_size를 사용함
    """

    def __init__(self, scale: float, zero_point: int = 0,
                 dtype=torch.qint8, qscheme=torch.per_tensor_affine):
        super().__init__(dtype)
        self.scale = scale
        self.zero_point = zero_point
        self.qscheme = qscheme

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x  # 통계 수집 안 함

    def calculate_qparams(self):
        return torch.tensor([self.scale]), torch.tensor([self.zero_point])


class FixedPerChannelObserver(torch.ao.quantization.observer.ObserverBase):
    """Per-channel 고정 scale observer"""

    def __init__(self, scales: torch.Tensor, zero_points: torch.Tensor = None,
                 dtype=torch.qint8, qscheme=torch.per_channel_affine, ch_axis: int = 0):
        super().__init__(dtype)
        self.scales = scales
        self.zero_points = zero_points if zero_points is not None else torch.zeros_like(scales, dtype=torch.int32)
        self.qscheme = qscheme
        self.ch_axis = ch_axis

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x

    def calculate_qparams(self):
        return self.scales, self.zero_points


def export_with_lsq_scales_pt2e(
    qat_model: nn.Module,
    output_path: str,
    config: dict = None,
    backend: str = 'xnnpack',
    calibration_samples: int = 500,
    verify: bool = False
) -> str:
    """
    QAT 모델을 PT2E로 export (native INT8 ops)

    핵심 원리:
    - QAT로 학습된 weights는 이미 양자화에 최적화되어 있음
    - PT2E는 BN을 Conv에 fuse함 (W_fused = W * gamma / sqrt(var + eps))
    - BN fusion 후 weights에 대해 calibration으로 올바른 scale을 찾음
    - LSQ scale을 직접 주입하면 BN fusion으로 인해 mismatch 발생

    방식:
    1. QAT 모델에서 raw weights + BN params를 base 모델로 복사 (pre-quant 없음)
    2. PT2E prepare (observers 삽입)
    3. 실제 COCO 데이터로 calibration (BN-fused weights에 맞는 scale 학습)
    4. PT2E convert → INT8
    5. ExecuTorch export

    장점: XNNPACK의 native INT8 연산 사용 + BN fusion 후 올바른 scale
    """
    import os
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
    print("PT2E Export with QAT Weights (Calibration-based)")
    print("="*60)

    # Step 1: Base 모델 생성 (raw QAT weights, pre-quant 없음)
    print("\nStep 1: Creating base model with raw QAT weights (no pre-quantization)...")
    print("  Note: Skipping pre-quantization to avoid BN fusion scale mismatch")
    base_model = create_base_model_from_qat(qat_model)
    base_model.eval()

    example_input = (torch.randn(1, 3, 224, 224),)

    # Step 2: PT2E 준비
    print("\nStep 2: Setting up PT2E quantization...")
    exported = export_for_training(base_model, example_input)
    gm = exported.module()

    quantizer = XNNPACKQuantizer()
    quant_config = get_symmetric_quantization_config(is_per_channel=True)
    quantizer.set_global(quant_config)

    prepared = prepare_pt2e(gm, quantizer)

    # Step 3: 실제 데이터로 calibration (BN-fused weights에 맞는 scale 학습)
    print(f"\nStep 3: Calibrating with {calibration_samples} COCO samples...")
    print("  (Calibration finds correct scales for BN-fused weights)")
    _calibrate_with_data(prepared, calibration_samples, config)

    # Step 4: INT8 변환
    print("\nStep 4: Converting to INT8...")
    quantized = convert_pt2e(prepared)

    # Step 5: Export
    print("\nStep 5: Exporting quantized model...")
    quantized_exported = export(quantized, example_input)

    # Step 6: Backend
    print(f"\nStep 6: Setting up {backend} backend...")
    partitioners = [XnnpackPartitioner()] if backend in ['xnnpack', 'nnapi'] else []

    # Step 7: Edge 변환
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
            quantized_exported,
            compile_config=edge_config,
            partitioner=None
        )

    # Step 8: 저장
    print("\nStep 8: Generating ExecuTorch program...")
    et_program = edge.to_executorch()

    print(f"\nSaving to {output_path}...")
    with open(output_path, 'wb') as f:
        f.write(et_program.buffer)

    file_size_mb = os.path.getsize(output_path) / (1024 * 1024)

    print(f"\n{'='*60}")
    print("PT2E Export Successful!")
    print(f"  Output: {output_path}")
    print(f"  Size: {file_size_mb:.2f} MB")
    print(f"  Method: PT2E with QAT-trained weights + calibration")
    print(f"  Backend: {backend} (native INT8 ops)")
    print("="*60)

    return output_path


def _calibrate_with_data(prepared_model, num_samples: int, config: dict = None):
    """실제 COCO 데이터로 calibration (observer 통계 수집)"""
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
                    if count % 50 == 0:
                        print(f"    Calibrated {count}/{num_samples} samples...")

            print(f"  Calibration completed with {count} COCO images")
            return

    except Exception as e:
        print(f"  Could not load COCO data: {e}")

    # Fallback: random data
    print("  Using random data for calibration (WARNING: less accurate)...")
    with torch.no_grad():
        for i in range(num_samples):
            prepared_model(torch.randn(1, 3, 224, 224))
            if (i + 1) % 50 == 0:
                print(f"    Calibrated {i+1}/{num_samples} samples...")
    print(f"  Calibration completed with {num_samples} random samples")


def _build_qat_to_base_key_map(qat_model: nn.Module) -> Dict[str, str]:
    """
    QAT 모델 레이어 이름 → base 모델 weight 키 매핑 생성

    예: 'layer1.0.conv1' (QuantizedConv2d) → 'layer1.0.conv1.weight' (base Conv2d)
    """
    from src.models.quantization import QuantizedConv2d, QuantizedLinear

    mapping = {}
    for name, module in qat_model.named_modules():
        if isinstance(module, QuantizedConv2d):
            # QAT: layer1.0.conv1 (QuantizedConv2d) → base: layer1.0.conv1 (Conv2d)
            mapping[name] = f"{name}.weight"
        elif isinstance(module, QuantizedLinear):
            mapping[name] = f"{name}.weight"

    return mapping


def create_base_model_from_qat(qat_model: nn.Module) -> nn.Module:
    """
    QAT 모델에서 base 모델 생성 (raw weights 복사, pre-quantization 없음)

    핵심:
    - PT2E가 BN fusion을 수행하므로, raw weights를 그대로 넘겨야 함
    - LSQ fake-quant를 적용하면 BN fusion 후 scale mismatch 발생
    - QAT로 학습된 weights는 이미 양자화에 최적화되어 있으므로
      calibration이 좋은 scale을 찾을 수 있음
    """
    from src.models.resnet import get_resnet
    from src.models.quantization import QuantizedConv2d, QuantizedLinear

    # 모델 아키텍처 감지
    has_conv3 = any('conv3' in name for name, _ in qat_model.named_modules())
    if has_conv3:
        model_name = 'resnet50'
    else:
        layer4_blocks = len([n for n in qat_model.state_dict().keys()
                            if n.startswith('layer4.') and '.conv1.conv.weight' in n])
        model_name = 'resnet34' if layer4_blocks >= 3 else 'resnet18'

    print(f"  Detected architecture: {model_name}")

    # Base 모델 생성
    base_model = get_resnet(model_name, num_classes=80, pretrained=False)
    base_state = base_model.state_dict()
    qat_state = qat_model.state_dict()

    # 명시적 키 매핑 생성: base_key → qat_key
    key_map = {}
    for base_key in base_state.keys():
        # 직접 매핑 시도
        if base_key in qat_state:
            key_map[base_key] = base_key
            continue

        # Conv2d: base 'layer1.0.conv1.weight' → QAT 'layer1.0.conv1.conv.weight'
        if '.weight' in base_key and 'bn' not in base_key and 'fc' not in base_key:
            parts = base_key.rsplit('.', 1)
            qat_conv_key = f"{parts[0]}.conv.{parts[1]}"
            if qat_conv_key in qat_state and base_state[base_key].shape == qat_state[qat_conv_key].shape:
                key_map[base_key] = qat_conv_key
                continue

        # Conv2d bias
        if '.bias' in base_key and 'bn' not in base_key and 'fc' not in base_key:
            parts = base_key.rsplit('.', 1)
            qat_conv_key = f"{parts[0]}.conv.{parts[1]}"
            if qat_conv_key in qat_state:
                key_map[base_key] = qat_conv_key
                continue

        # Linear: base 'fc.weight' → QAT 'fc.linear.weight'
        if base_key.startswith('fc.'):
            qat_fc_key = base_key.replace('fc.', 'fc.linear.', 1)
            if qat_fc_key in qat_state and base_state[base_key].shape == qat_state[qat_fc_key].shape:
                key_map[base_key] = qat_fc_key
                continue

    # Raw weights 복사 (pre-quantization 없음!)
    copied = 0
    for base_key, qat_key in key_map.items():
        if base_state[base_key].shape == qat_state[qat_key].shape:
            base_state[base_key] = qat_state[qat_key].clone()
            copied += 1

    # 매핑되지 않은 키 처리 (BN params 등은 직접 복사)
    for base_key in base_state.keys():
        if base_key not in key_map and base_key in qat_state:
            if base_state[base_key].shape == qat_state[base_key].shape:
                base_state[base_key] = qat_state[base_key].clone()
                copied += 1

    print(f"  Copied {copied} parameters (raw weights, no pre-quantization)")

    base_model.load_state_dict(base_state)
    return base_model


def create_base_model_with_lsq_weights(qat_model: nn.Module, lsq_scales: dict) -> nn.Module:
    """
    QAT 모델에서 base 모델 생성 (pre-quantized weights 포함)

    개선사항:
    - 명시적 키 매핑 (fuzzy matching 제거)
    - LSQ fake-quant를 정확히 적용
    - conv1/fc (LSQ 제외 레이어)는 원본 weights 유지
    """
    from src.models.resnet import get_resnet
    from src.models.quantization import QuantizedConv2d, QuantizedLinear

    # 모델 아키텍처 감지
    has_conv3 = any('conv3' in name for name, _ in qat_model.named_modules())
    if has_conv3:
        model_name = 'resnet50'
    else:
        layer4_blocks = len([n for n in qat_model.state_dict().keys()
                            if n.startswith('layer4.') and '.conv1.conv.weight' in n])
        model_name = 'resnet34' if layer4_blocks >= 3 else 'resnet18'

    print(f"  Detected architecture: {model_name}")

    # Base 모델 생성
    base_model = get_resnet(model_name, num_classes=80, pretrained=False)
    base_state = base_model.state_dict()
    qat_state = qat_model.state_dict()

    # 명시적 키 매핑 생성: QAT키 → base키
    key_map = {}  # base_key → qat_key
    for base_key in base_state.keys():
        # 직접 매핑 시도
        if base_key in qat_state:
            key_map[base_key] = base_key
            continue

        # Conv2d: base 'layer1.0.conv1.weight' → QAT 'layer1.0.conv1.conv.weight'
        if '.weight' in base_key and 'bn' not in base_key and 'fc' not in base_key:
            parts = base_key.rsplit('.', 1)
            qat_conv_key = f"{parts[0]}.conv.{parts[1]}"
            if qat_conv_key in qat_state and base_state[base_key].shape == qat_state[qat_conv_key].shape:
                key_map[base_key] = qat_conv_key
                continue

        # Conv2d bias
        if '.bias' in base_key and 'bn' not in base_key and 'fc' not in base_key:
            parts = base_key.rsplit('.', 1)
            qat_conv_key = f"{parts[0]}.conv.{parts[1]}"
            if qat_conv_key in qat_state:
                key_map[base_key] = qat_conv_key
                continue

        # Linear: base 'fc.weight' → QAT 'fc.linear.weight'
        if base_key.startswith('fc.'):
            qat_fc_key = base_key.replace('fc.', 'fc.linear.', 1)
            if qat_fc_key in qat_state and base_state[base_key].shape == qat_state[qat_fc_key].shape:
                key_map[base_key] = qat_fc_key
                continue

    # Weight 복사 + pre-quantization
    copied = 0
    quantized = 0

    for base_key, qat_key in key_map.items():
        weight = qat_state[qat_key].clone()

        # LSQ quantized 레이어인 경우 fake-quant 적용
        if 'weight' in base_key:
            # base_key에서 레이어 이름 추출: 'layer1.0.conv1.weight' → 'layer1.0.conv1'
            layer_name = base_key.rsplit('.', 1)[0]

            # 정확한 매칭으로 LSQ scale 찾기
            matched_scale = None
            for lsq_name, params in lsq_scales.items():
                # 정확한 매칭: layer1.0.conv1 == layer1.0.conv1
                if lsq_name == layer_name and 'weight_scale' in params:
                    matched_scale = params
                    break

            if matched_scale is not None:
                scale = matched_scale['weight_scale']
                Q_N, Q_P = matched_scale['weight_qparams']

                if scale.dim() == 1 and weight.dim() == 4:
                    scale_exp = scale.view(-1, 1, 1, 1)
                elif scale.dim() == 1 and weight.dim() == 2:
                    scale_exp = scale.view(-1, 1)
                else:
                    scale_exp = scale

                # LSQ fake quantization: round(clip(w/s, -Q_N, Q_P)) * s
                w_scaled = weight / scale_exp
                w_clipped = torch.clamp(w_scaled, -Q_N, Q_P)
                w_rounded = torch.round(w_clipped)
                weight = w_rounded * scale_exp
                quantized += 1

        base_state[base_key] = weight
        copied += 1

    # 매핑되지 않은 키 처리 (BN 등은 직접 복사)
    for base_key in base_state.keys():
        if base_key not in key_map and base_key in qat_state:
            if base_state[base_key].shape == qat_state[base_key].shape:
                base_state[base_key] = qat_state[base_key].clone()
                copied += 1

    print(f"  Copied {copied} parameters, pre-quantized {quantized} weight tensors")

    base_model.load_state_dict(base_state)
    return base_model


def inject_lsq_scales_to_weight_observers(prepared_model, lsq_scales: dict):
    """
    PT2E prepared 모델의 weight observer에 LSQ scales 주입

    개선사항:
    - Weight observer만 LSQ scale로 덮어쓰기 (activation은 calibration 유지)
    - Per-channel observer 지원
    - Ordered matching: LSQ scales 순서와 weight observer 순서 매칭
    """
    from torch.ao.quantization.observer import (
        MinMaxObserver,
        PerChannelMinMaxObserver,
        HistogramObserver,
        ObserverBase
    )

    # LSQ weight scales를 순서대로 수집 (per-channel)
    ordered_lsq_weight_scales = []
    for name, params in lsq_scales.items():
        if 'weight_scale' in params:
            ordered_lsq_weight_scales.append({
                'name': name,
                'scale': params['weight_scale'],
                'qparams': params['weight_qparams'],
            })

    # Per-channel weight observers 수집 (ordered)
    per_channel_observers = []
    for obs_name, module in prepared_model.named_modules():
        if isinstance(module, PerChannelMinMaxObserver):
            per_channel_observers.append((obs_name, module))

    print(f"  LSQ weight layers: {len(ordered_lsq_weight_scales)}")
    print(f"  PT2E per-channel observers: {len(per_channel_observers)}")

    # Per-channel observers는 weight observers임 (PT2E에서)
    # 네트워크 순서대로 매칭 시도
    injected = 0
    lsq_idx = 0

    for obs_name, observer in per_channel_observers:
        if lsq_idx >= len(ordered_lsq_weight_scales):
            break

        lsq_info = ordered_lsq_weight_scales[lsq_idx]
        scale = lsq_info['scale']  # per-channel: [out_channels]
        Q_N, Q_P = lsq_info['qparams']

        try:
            if hasattr(observer, 'min_val') and hasattr(observer, 'max_val'):
                num_channels = scale.numel()

                # Per-channel min/max 설정: symmetric이므로
                # max_val[c] = scale[c] * Q_P, min_val[c] = -scale[c] * Q_N
                max_val = scale * Q_P
                min_val = -scale * Q_N

                # Observer 크기가 맞는지 확인
                if observer.min_val.numel() == num_channels:
                    observer.min_val.copy_(min_val)
                    observer.max_val.copy_(max_val)
                    injected += 1
                    lsq_idx += 1
                elif observer.min_val.numel() == 0:
                    # 아직 초기화되지 않은 observer - 값 설정
                    observer.min_val = min_val.clone()
                    observer.max_val = max_val.clone()
                    injected += 1
                    lsq_idx += 1
                else:
                    # 채널 수 불일치 - 다음 LSQ scale과 매칭 시도
                    # conv1/fc 등 LSQ에서 제외된 레이어일 수 있음
                    continue

        except Exception as e:
            print(f"    Warning: Failed to inject scale to {obs_name}: {e}")
            continue

    # Scalar (per-tensor) observers에 LSQ activation scale 강제 주입
    #
    # 핵심 변환:
    #   LSQ unsigned [0, 255]: max_representable = 255 * lsq_scale
    #   XNNPACK symmetric signed [-128, 127]: scale = max_repr / 127
    #   → observer.min_val = -max_repr, observer.max_val = max_repr
    #
    act_injected = 0
    ordered_act_scales = []
    for name, params in lsq_scales.items():
        if 'act_scale' in params:
            ordered_act_scales.append({
                'name': name,
                'scale': params['act_scale'],
                'qparams': params['act_qparams'],
            })

    # Activation observers: HistogramObserver 또는 MinMaxObserver (per-channel 제외)
    act_observers = []
    for obs_name, module in prepared_model.named_modules():
        if isinstance(module, PerChannelMinMaxObserver):
            continue  # weight observer - skip
        if isinstance(module, (MinMaxObserver, HistogramObserver)):
            act_observers.append((obs_name, module))

    print(f"  LSQ activation layers: {len(ordered_act_scales)}")
    print(f"  PT2E activation observers: {len(act_observers)}")

    act_idx = 0
    for obs_name, observer in act_observers:
        if act_idx >= len(ordered_act_scales):
            break

        act_info = ordered_act_scales[act_idx]
        act_scale = act_info['scale']
        act_Q_N, act_Q_P = act_info['qparams']  # typically (0, 255)

        try:
            s = act_scale.item() if act_scale.dim() == 0 else act_scale.mean().item()

            # LSQ unsigned max representable value
            max_repr = float(act_Q_P) * s  # 255 * s

            # XNNPACK symmetric signed: observer min/max 강제 설정
            # symmetric observer: scale = max(|min_val|, |max_val|) / 127
            # → scale = max_repr / 127 = 255*s / 127
            if hasattr(observer, 'min_val') and hasattr(observer, 'max_val'):
                if observer.min_val.numel() >= 1:
                    observer.min_val.fill_(-max_repr)
                    observer.max_val.fill_(max_repr)
                    act_injected += 1
                elif observer.min_val.numel() == 0:
                    observer.min_val = torch.tensor([-max_repr])
                    observer.max_val = torch.tensor([max_repr])
                    act_injected += 1

            act_idx += 1
        except Exception as e:
            print(f"    Warning: Failed to inject act scale to {obs_name}: {e}")
            act_idx += 1
            continue

    print(f"  Injected LSQ scales to {injected} weight observers")
    print(f"  Injected LSQ scales to {act_injected}/{len(ordered_act_scales)} activation observers")


if __name__ == "__main__":
    # 테스트
    print("Testing INT8 Export Module...")

    # 더미 모델 테스트
    from src.models.resnet import resnet18
    from src.models.quantization import quantize_model

    # QAT 모델 생성
    model = resnet18(num_classes=80, pretrained=False)
    qat_model = quantize_model(model, bits=8, exclude_layers=['conv1', 'fc'])

    # 초기화
    qat_model.train()
    with torch.no_grad():
        _ = qat_model(torch.randn(2, 3, 224, 224))
    qat_model.eval()

    # LSQ scales 추출
    scales = extract_lsq_scales(qat_model)
    print(f"\nExtracted scales from {len(scales)} layers")

    # INT8 변환
    int8_model = convert_lsq_to_int8(qat_model)

    # 크기 비교
    size_info = get_model_size_breakdown(int8_model)
    print(f"\nModel size:")
    print(f"  INT8: {size_info['int8_mb']:.2f} MB")
    print(f"  FP32: {size_info['fp32_mb']:.2f} MB")
    print(f"  Total: {size_info['total_mb']:.2f} MB")

    # Forward 테스트
    with torch.no_grad():
        x = torch.randn(1, 3, 224, 224)
        y = int8_model(x)
        print(f"\nForward test: input {x.shape} → output {y.shape}")

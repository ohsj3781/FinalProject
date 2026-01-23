# 스마트폰을 위한 경량 ResNet 기반 사진 자동 태깅 시스템

On-Device AI 기반 실시간 사진 자동 태깅 시스템으로, LSQ(Learned Step Size Quantization) 양자화된 ResNet-18 모델을 ExecuTorch를 통해 모바일에 배포합니다.

## 프로젝트 개요

| 항목 | 내용 |
|------|------|
| **목표** | COCO 데이터셋 80개 카테고리를 인식하는 다중 레이블 이미지 분류 모델 개발 및 모바일 배포 |
| **모델** | ResNet-18 (Multi-label Classification) |
| **양자화** | LSQ 8-bit Quantization-Aware Training |
| **배포** | ExecuTorch (PyTorch Mobile) |
| **타겟 디바이스** | Galaxy S24+ (Exynos 2400, NPU 지원) |

## 현재 진행 상황

### 완료된 작업

- [x] ResNet-18 모델 구현 (다중 레이블 분류)
- [x] COCO 2017 데이터셋 로더 구현
- [x] 데이터 증강 파이프라인 구축
- [x] Full Precision (FP32) 학습 완료 (50 epochs)
- [x] LSQ Quantization-Aware Training (QAT) 완료 (30 epochs)
- [x] 모델 평가 및 성능 측정
- [x] ExecuTorch 모델 변환 (XNNPACK, NNAPI)

### 달성 성능

| 지표 | 목표 | 달성 | 상태 |
|------|------|------|------|
| mAP | > 60% | **66.72%** | ✅ 달성 |
| Precision | - | 73.56% | - |
| Recall | - | 70.02% | - |
| F1 Score | - | 68.54% | - |
| 모델 크기 (ExecuTorch) | < 5MB | 42.87 MB | ⚠️ 추가 최적화 필요 |

> **참고**: 현재 모델 크기가 목표보다 큽니다. 이는 LSQ가 "fake quantization"으로 학습 시에만 양자화를 시뮬레이션하기 때문입니다. 실제 INT8 모델로 변환하려면 추가적인 post-training quantization이 필요합니다.

### 남은 작업

- [ ] INT8 실제 양자화 적용 (모델 크기 ~11MB 목표)
- [ ] Android 앱 개발 (Kotlin + ExecuTorch Runtime)
- [ ] 모바일 추론 시간 측정 (CPU/NPU)
- [ ] 배터리 소모량 측정

## 프로젝트 구조

```
FinalProject/
├── configs/
│   └── config.yaml              # 학습/양자화/배포 설정
├── scripts/
│   ├── download_coco.py         # COCO 데이터셋 다운로드
│   ├── train.py                 # 학습 스크립트 (FP32 & QAT)
│   ├── evaluate.py              # 모델 평가
│   └── export_executorch.py     # ExecuTorch 변환
├── src/
│   ├── data/
│   │   ├── dataset.py           # COCO 다중 레이블 데이터셋
│   │   └── augmentation.py      # 데이터 증강 (Resize, Crop, ColorJitter)
│   ├── models/
│   │   ├── resnet.py            # ResNet-18 구현 (11.18M params)
│   │   └── quantization.py      # LSQ 양자화 모듈
│   ├── training/
│   │   ├── trainer.py           # Trainer 클래스 (AMP 지원)
│   │   └── loss.py              # 손실 함수 (BCE, Focal, Asymmetric)
│   ├── inference/
│   │   └── predictor.py         # PyTorch/ExecuTorch 추론
│   └── utils/
│       └── metrics.py           # 평가 지표 (mAP, F1 등)
├── checkpoints/
│   ├── fp32/                    # Full Precision 체크포인트
│   │   └── best_model.pth       # 최고 성능 FP32 모델
│   └── qat/                     # QAT 체크포인트
│       └── best_model.pth       # 최고 성능 QAT 모델
├── exported_models/
│   ├── resnet18_multilabel_qat_xnnpack.pte  # CPU용 ExecuTorch 모델
│   └── resnet18_multilabel_qat_nnapi.pte    # NPU용 ExecuTorch 모델
├── data/
│   └── coco/                    # COCO 2017 데이터셋 위치
├── logs/                        # TensorBoard 학습 로그
├── android/                     # Android 앱 프로젝트 (개발 예정)
├── reference/                   # 참고 논문 및 제안서
│   ├── paper/
│   │   ├── Deep Residual Learning for Image Recognition.pdf
│   │   ├── Learned Step Size Quantization.pdf
│   │   └── Quantization and Training of Neural Networks...pdf
│   └── proposal/
│       └── 연구논문작품 제안서.pdf
└── requirements.txt             # Python 의존성
```

## 설치 방법

### 1. 환경 설정

```bash
# 가상환경 생성 (권장)
python -m venv .venv
source .venv/bin/activate  # Linux/Mac
# .venv\Scripts\activate   # Windows

# 의존성 설치
pip install -r requirements.txt
```

### 2. COCO 데이터셋 다운로드

```bash
# 전체 데이터셋 (~19GB)
python scripts/download_coco.py

# 검증용 데이터만 (~1GB)
python scripts/download_coco.py --val-only
```

## 사용 방법

### 1. Full Precision 학습

```bash
python scripts/train.py --config configs/config.yaml
```

**학습 설정:**
- Epochs: 50
- Batch Size: 128
- Optimizer: SGD (lr=0.1, momentum=0.9, nesterov=True)
- Scheduler: Cosine Annealing
- Mixed Precision Training (AMP): 활성화

### 2. Quantization-Aware Training (QAT)

```bash
python scripts/train.py \
    --config configs/config.yaml \
    --qat \
    --checkpoint checkpoints/fp32/best_model.pth
```

**QAT 설정:**
- Epochs: 30
- Learning Rate: 0.01 (FP32의 1/10)
- Weight Decay: 5e-5 (FP32의 절반)
- 제외 레이어: conv1 (첫 번째 레이어), fc (마지막 레이어)

### 3. 모델 평가

```bash
# FP32 모델 평가
python scripts/evaluate.py --checkpoint checkpoints/fp32/best_model.pth

# QAT 모델 평가
python scripts/evaluate.py --checkpoint checkpoints/qat/best_model.pth --qat
```

### 4. ExecuTorch 변환 (모바일 배포용)

```bash
# XNNPACK 백엔드 (CPU)
python scripts/export_executorch.py \
    --checkpoint checkpoints/qat/best_model.pth \
    --qat \
    --backend xnnpack

# NNAPI 백엔드 (NPU)
python scripts/export_executorch.py \
    --checkpoint checkpoints/qat/best_model.pth \
    --qat \
    --backend nnapi
```

### 5. TensorBoard 로그 확인

```bash
tensorboard --logdir logs/
```

## 기술 스택

| 항목 | 기술 | 버전 |
|------|------|------|
| 학습 프레임워크 | PyTorch | >= 2.0.0 |
| 모바일 배포 | ExecuTorch | >= 1.0.0 |
| 양자화 라이브러리 | torchao | >= 0.4.0 |
| 데이터셋 | COCO 2017 | 80 카테고리 |
| Android 언어 | Kotlin | (예정) |
| 하드웨어 가속 | NNAPI | NPU 지원 |

## 모델 아키텍처

### ResNet-18 구조

```
Input (3, 224, 224)
    ↓
Conv1 (7x7, 64, stride=2) → BN → ReLU
    ↓
MaxPool (3x3, stride=2)
    ↓
Layer1: 2 × BasicBlock (64 channels)
    ↓
Layer2: 2 × BasicBlock (128 channels, stride=2)
    ↓
Layer3: 2 × BasicBlock (256 channels, stride=2)
    ↓
Layer4: 2 × BasicBlock (512 channels, stride=2)
    ↓
AdaptiveAvgPool (1, 1)
    ↓
FC (512 → 80) + Sigmoid
    ↓
Output: 80 class probabilities
```

### LSQ 양자화

```python
# 양자화 수식
v_bar = round(clip(v/s, -Q_N, Q_P))  # 정수 표현
v_hat = v_bar * s                     # 양자화된 실수 값

# 8-bit 설정
# Weights: Q_N = 128, Q_P = 127 (signed)
# Activations: Q_N = 0, Q_P = 255 (unsigned)
```

## 파이프라인 개요

```
┌─────────────────────────────────────────────────────────────────┐
│                    Training Pipeline (PC/Server)                 │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  COCO Dataset ──→ Data Augmentation ──→ ResNet-18 Training      │
│       │                                       │                  │
│       │                                       ↓                  │
│       │                              FP32 Model (mAP: ~67%)      │
│       │                                       │                  │
│       │                                       ↓                  │
│       └──────────────────→ LSQ QAT Training (30 epochs)          │
│                                       │                          │
│                                       ↓                          │
│                              QAT Model (mAP: 66.72%)             │
│                                       │                          │
│                                       ↓                          │
│                            ExecuTorch Export (.pte)              │
│                              ┌───────┴───────┐                   │
│                              ↓               ↓                   │
│                          XNNPACK          NNAPI                  │
│                           (CPU)           (NPU)                  │
└─────────────────────────────────────────────────────────────────┘
                                   ↓
┌─────────────────────────────────────────────────────────────────┐
│                    Inference Pipeline (Mobile)                   │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  Camera/Gallery ──→ Image Preprocessing ──→ ExecuTorch Runtime  │
│                           (224x224)               │              │
│                                                   ↓              │
│                                         Integer Inference        │
│                                               │                  │
│                                               ↓                  │
│                                    80 Category Predictions       │
│                                               │                  │
│                                               ↓                  │
│                                      Tag Generation              │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

## 주요 기능

### 1. 다중 레이블 분류
- 하나의 이미지에서 여러 객체/카테고리 동시 인식
- Sigmoid 활성화 함수로 각 클래스 독립적 확률 출력
- 임계값(threshold=0.5) 기반 태그 결정

### 2. 손실 함수 지원
- **BCEWithLogitsLoss**: 기본 이진 교차 엔트로피
- **FocalLoss**: 클래스 불균형 대응
- **AsymmetricLoss**: 다중 레이블 최적화

### 3. 학습 최적화
- **Mixed Precision Training (AMP)**: FP16으로 학습 속도 ~2배 향상
- **CuDNN Benchmark**: 입력 크기 고정 시 자동 최적화
- **Pin Memory**: CPU-GPU 데이터 전송 최적화

### 4. 양자화
- **LSQ (Learned Step Size Quantization)**: 학습 가능한 스텝 크기
- **Per-channel Quantization**: 가중치별 최적 양자화
- **첫/마지막 레이어 제외**: 정확도 유지

## 참고 문헌

1. **ResNet**: He et al., "Deep Residual Learning for Image Recognition" (CVPR 2016)
2. **LSQ**: Esser et al., "Learned Step Size Quantization" (ICLR 2020)
3. **Integer-Only Inference**: Jacob et al., "Quantization and Training of Neural Networks for Efficient Integer-Arithmetic-Only Inference" (CVPR 2018)

## 라이선스

이 프로젝트는 학술 연구 목적으로 개발되었습니다.

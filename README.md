# 스마트폰을 위한 경량 ResNet 기반 사진 자동 태깅 시스템

On-Device AI 기반 실시간 사진 자동 태깅 시스템으로, LSQ(Learned Step Size Quantization) 양자화된 ResNet 모델을 ExecuTorch를 통해 모바일에 배포합니다.

## 프로젝트 개요

| 항목 | 내용 |
|------|------|
| **목표** | COCO 데이터셋 80개 카테고리를 인식하는 다중 레이블 이미지 분류 모델 개발 및 모바일 배포 |
| **모델** | ResNet-18 / ResNet-34 / ResNet-50 (Multi-label Classification) |
| **양자화** | LSQ 8-bit QAT + INT8 Post-Training Quantization |
| **배포** | ExecuTorch (PyTorch Mobile) |
| **타겟 디바이스** | Galaxy S24+ (Exynos 2400, NPU 지원) |

### 지원 모델 비교

| 모델 | 파라미터 | FP32 크기 | INT8 크기 | 특징 |
|------|----------|-----------|-----------|------|
| ResNet-18 | 11.7M | ~45 MB | ~11 MB | 경량, 빠른 추론 |
| ResNet-34 | 21.8M | ~83 MB | ~21 MB | 중간 규모 |
| ResNet-50 | 23.7M | ~90 MB | ~24 MB | 높은 정확도 |

## 현재 진행 상황

### 완료된 작업

- [x] ResNet-18 모델 구현 (다중 레이블 분류)
- [x] COCO 2017 데이터셋 로더 구현
- [x] 데이터 증강 파이프라인 구축
- [x] Full Precision (FP32) 학습 완료 (50 epochs)
- [x] LSQ Quantization-Aware Training (QAT) 완료 (30 epochs)
- [x] 모델 평가 및 성능 측정
- [x] ExecuTorch 모델 변환 (XNNPACK, NNAPI)
- [x] **INT8 Post-Training Quantization 구현**
- [x] **ExecuTorch (.pte) 모델 평가 기능 구현**
- [x] **Android 벤치마크 앱 개발**

### 달성 성능 (FP32 모델)

> **Note**: 본 프로젝트의 mAP는 **Multi-label Classification mAP** (Precision-Recall AUC 기반)입니다.
> COCO Object Detection mAP (IoU@[0.5:0.95])와는 다른 지표입니다.

| 지표 | 목표 | 달성 | 상태 |
|------|------|------|------|
| mAP (Classification) | > 60% | **66.72%** | ✅ 달성 |
| Precision | - | 73.56% | - |
| Recall | - | 70.02% | - |
| F1 Score | - | 68.54% | - |
| 모델 크기 (QAT) | - | 43 MB | FP32 weights |
| **모델 크기 (INT8)** | < 15MB | **11 MB** | ✅ **75% 감소** |

### FP32 vs INT8 성능 비교

| 지표 | FP32 모델 | INT8 모델 | 차이 |
|------|-----------|-----------|------|
| **mAP (Classification)** | 66.72% | 63.33% | -3.39% |
| Precision | 73.56% | 72.23% | -1.33% |
| Recall | 70.02% | 70.89% | +0.87% |
| F1 Score | 68.54% | 68.18% | -0.36% |
| **모델 크기** | 43 MB | **11 MB** | **-75%** |
| 최적 Threshold | 0.50 | 0.25 | - |

> INT8 양자화로 mAP가 약 3.4% 감소했지만, 모델 크기는 75% 줄어들어 모바일 NPU에서 더 빠른 추론이 가능합니다.

### 모델 크기 비교

| 모델 타입 | 크기 | Classification mAP | 설명 |
|-----------|------|-------------------|------|
| QAT (XNNPACK) | 43 MB | 66.72% | Fake quantization, FP32 weights |
| QAT (NNAPI) | 43 MB | 66.72% | Fake quantization, FP32 weights |
| **INT8 (XNNPACK)** | **11 MB** | 63.33% | Real INT8 weights, CPU 최적화 |
| **INT8 (NNAPI)** | **11 MB** | 63.33% | Real INT8 weights, NPU 최적화 |

### 남은 작업

- [ ] 모바일 실제 추론 시간 측정 (CPU vs NPU)
- [ ] 배터리 소모량 측정
- [ ] mAP 80% 달성을 위한 추가 학습

## 프로젝트 구조

```
FinalProject/
├── configs/
│   └── config.yaml              # 학습/양자화/배포 설정
├── scripts/
│   ├── download_coco.py         # COCO 데이터셋 다운로드
│   ├── train.py                 # 학습 스크립트 (FP32 & QAT)
│   ├── evaluate.py              # 모델 평가
│   └── export_executorch.py     # ExecuTorch 변환 (INT8 지원)
├── src/
│   ├── data/
│   │   ├── dataset.py           # COCO 다중 레이블 데이터셋
│   │   └── augmentation.py      # 데이터 증강 (Resize, Crop, ColorJitter)
│   ├── models/
│   │   ├── resnet.py            # ResNet-18/34/50 구현
│   │   └── quantization.py      # LSQ 양자화 모듈
│   ├── training/
│   │   ├── trainer.py           # Trainer 클래스 (AMP 지원)
│   │   └── loss.py              # 손실 함수 (BCE, Focal, Asymmetric)
│   ├── inference/
│   │   └── predictor.py         # PyTorch/ExecuTorch 추론
│   └── utils/
│       └── metrics.py           # 평가 지표 (mAP, F1 등)
├── checkpoints/
│   ├── resnet18_fp32/           # ResNet-18 FP32 체크포인트
│   │   └── best_model.pth
│   ├── resnet18_qat/            # ResNet-18 QAT 체크포인트
│   │   └── best_model.pth
│   ├── resnet34_fp32/           # ResNet-34 FP32 체크포인트
│   ├── resnet50_fp32/           # ResNet-50 FP32 체크포인트
│   └── resnet50_qat/            # ResNet-50 QAT 체크포인트
├── exported_models/
│   ├── resnet18_multilabel_int8_xnnpack.pte  # ResNet-18 INT8 CPU용 (11MB)
│   ├── resnet18_multilabel_int8_nnapi.pte    # ResNet-18 INT8 NPU용 (11MB)
│   ├── resnet50_multilabel_int8_xnnpack.pte  # ResNet-50 INT8 CPU용 (24MB)
│   └── resnet50_multilabel_int8_nnapi.pte    # ResNet-50 INT8 NPU용 (24MB)
├── android/                     # Android 벤치마크 앱
│   ├── app/
│   │   └── src/main/
│   │       ├── java/.../MainActivity.kt      # 벤치마크 UI
│   │       ├── java/.../ModelBenchmark.kt    # ExecuTorch 추론
│   │       └── assets/                       # 모델 파일 위치
│   └── build.gradle.kts
├── data/
│   └── coco/                    # COCO 2017 데이터셋 위치
├── logs/                        # TensorBoard 학습 로그
├── reference/                   # 참고 논문 및 제안서
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

## 커맨드라인 옵션 요약

### train.py

| 옵션 | 설명 | 예시 |
|------|------|------|
| `--model` | 모델 선택 (resnet18, resnet34, resnet50) | `--model resnet50` |
| `--config` | 설정 파일 경로 | `--config configs/config.yaml` |
| `--qat` | QAT 학습 모드 활성화 | `--qat` |
| `--checkpoint` | QAT용 사전학습 체크포인트 | `--checkpoint checkpoints/resnet50_fp32/best_model.pth` |
| `--resume` | 학습 재개용 체크포인트 | `--resume checkpoints/resnet50_fp32/checkpoint_epoch_10.pth` |

### export_executorch.py

| 옵션 | 설명 | 예시 |
|------|------|------|
| `--model` | 모델 선택 | `--model resnet50` |
| `--checkpoint` | 체크포인트 경로 (필수) | `--checkpoint checkpoints/resnet50_fp32/best_model.pth` |
| `--int8` | INT8 PTQ 양자화 적용 | `--int8` |
| `--qat` | QAT 모델 export | `--qat` |
| `--backend` | ExecuTorch 백엔드 (xnnpack, nnapi) | `--backend xnnpack` |

### evaluate.py

| 옵션 | 설명 | 예시 |
|------|------|------|
| `--model` | 모델 선택 | `--model resnet50` |
| `--checkpoint` | PyTorch 체크포인트 경로 | `--checkpoint checkpoints/resnet50_fp32/best_model.pth` |
| `--pte` | ExecuTorch 모델 경로 | `--pte exported_models/resnet50_multilabel_int8_xnnpack.pte` |
| `--qat` | QAT 모델 평가 | `--qat` |

## 사용 방법

### 1. Full Precision 학습

```bash
# ResNet-18 학습 (기본값)
python scripts/train.py --config configs/config.yaml --model resnet18

# ResNet-34 학습
python scripts/train.py --config configs/config.yaml --model resnet34

# ResNet-50 학습 (높은 정확도)
python scripts/train.py --config configs/config.yaml --model resnet50

# 학습 재개
python scripts/train.py --config configs/config.yaml --model resnet50 \
    --resume checkpoints/resnet50_fp32/checkpoint_epoch_10.pth
```

**학습 설정:**
- Epochs: 50
- Batch Size: 128
- Optimizer: SGD (lr=0.1, momentum=0.9, nesterov=True)
- Scheduler: Cosine Annealing
- Mixed Precision Training (AMP): 활성화

**체크포인트 저장 위치:** `checkpoints/{model_name}_fp32/`

### 2. Quantization-Aware Training (QAT)

```bash
# ResNet-18 QAT
python scripts/train.py \
    --config configs/config.yaml \
    --model resnet18 \
    --qat \
    --checkpoint checkpoints/resnet18_fp32/best_model.pth

# ResNet-34 QAT
python scripts/train.py \
    --config configs/config.yaml \
    --model resnet34 \
    --qat \
    --checkpoint checkpoints/resnet34_fp32/best_model.pth

# ResNet-50 QAT
python scripts/train.py \
    --config configs/config.yaml \
    --model resnet50 \
    --qat \
    --checkpoint checkpoints/resnet50_fp32/best_model.pth
```

**QAT 설정:**
- Epochs: 30
- Learning Rate: 0.01 (FP32의 1/10)
- Weight Decay: 5e-5 (FP32의 절반)
- 제외 레이어: conv1 (첫 번째 레이어), fc (마지막 레이어)

**체크포인트 저장 위치:** `checkpoints/{model_name}_qat/`

### 3. 모델 평가

```bash
# FP32 모델 평가
python scripts/evaluate.py \
    --checkpoint checkpoints/resnet18_fp32/best_model.pth \
    --model resnet18

python scripts/evaluate.py \
    --checkpoint checkpoints/resnet34_fp32/best_model.pth \
    --model resnet34

python scripts/evaluate.py \
    --checkpoint checkpoints/resnet50_fp32/best_model.pth \
    --model resnet50

# QAT 모델 평가
python scripts/evaluate.py \
    --checkpoint checkpoints/resnet50_qat/best_model.pth \
    --model resnet50 \
    --qat
```

### 4. ExecuTorch 변환 (모바일 배포용)

#### QAT 모델 변환 (FP32 weights)

```bash
# ResNet-18 XNNPACK 백엔드 (CPU)
python scripts/export_executorch.py \
    --checkpoint checkpoints/resnet18_qat/best_model.pth \
    --model resnet18 \
    --qat \
    --backend xnnpack

# ResNet-50 NNAPI 백엔드 (NPU)
python scripts/export_executorch.py \
    --checkpoint checkpoints/resnet50_qat/best_model.pth \
    --model resnet50 \
    --qat \
    --backend nnapi
```

#### INT8 양자화 모델 변환 (권장)

```bash
# ResNet-18 INT8 XNNPACK (CPU 최적화)
python scripts/export_executorch.py \
    --checkpoint checkpoints/resnet18_fp32/best_model.pth \
    --model resnet18 \
    --int8 \
    --backend xnnpack \
    --calibration-samples 100

# ResNet-50 INT8 NNAPI (NPU 최적화)
python scripts/export_executorch.py \
    --checkpoint checkpoints/resnet50_fp32/best_model.pth \
    --model resnet50 \
    --int8 \
    --backend nnapi \
    --calibration-samples 100
```

**INT8 양자화 특징:**
- 모델 크기: ~75% 감소 (ResNet-18: 45MB → 11MB)
- NPU에서 최적화된 INT8 연산 사용
- Calibration 기반 정확한 양자화 범위 설정

**출력 파일명:** `exported_models/{model_name}_multilabel_{fp32|qat|int8}_{backend}.pte`

### 5. ExecuTorch 모델 평가

```bash
# INT8 모델 평가 (ExecuTorch)
python scripts/evaluate.py --pte exported_models/resnet18_multilabel_int8_xnnpack.pte
python scripts/evaluate.py --pte exported_models/resnet50_multilabel_int8_xnnpack.pte

# QAT 모델 평가 (ExecuTorch)
python scripts/evaluate.py --pte exported_models/resnet18_multilabel_qat_xnnpack.pte
```

### 6. Android 앱 빌드

```bash
cd android
./gradlew assembleDebug
```

**앱 기능:**
- ExecuTorch 모델 로딩 (Asset/External)
- COCO val2017 이미지 벤치마크
- CPU/NPU 백엔드 선택
- 추론 시간 측정 (총 시간, 이미지당 시간)

### 7. TensorBoard 로그 확인

```bash
tensorboard --logdir logs/
```

## 기술 스택

| 항목 | 기술 | 버전 |
|------|------|------|
| 학습 프레임워크 | PyTorch | >= 2.0.0 |
| 모바일 배포 | ExecuTorch | >= 1.0.0 |
| 양자화 라이브러리 | torchao | >= 0.14.0 |
| 데이터셋 | COCO 2017 | 80 카테고리 |
| Android 언어 | Kotlin | 1.9+ |
| 하드웨어 가속 | NNAPI / XNNPACK | NPU/CPU 지원 |

## 모델 아키텍처

### 지원 모델

| 모델 | 블록 타입 | 레이어 구성 | FC 입력 차원 |
|------|-----------|-------------|--------------|
| ResNet-18 | BasicBlock | [2, 2, 2, 2] | 512 |
| ResNet-34 | BasicBlock | [3, 4, 6, 3] | 512 |
| ResNet-50 | Bottleneck | [3, 4, 6, 3] | 2048 |

### ResNet 구조

```
Input (3, 224, 224)
    ↓
Conv1 (7x7, 64, stride=2) → BN → ReLU
    ↓
MaxPool (3x3, stride=2)
    ↓
Layer1: N × Block (64 channels)
    ↓
Layer2: N × Block (128 channels, stride=2)
    ↓
Layer3: N × Block (256 channels, stride=2)
    ↓
Layer4: N × Block (512 channels, stride=2)
    ↓
AdaptiveAvgPool (1, 1)
    ↓
FC (512/2048 → 80) + Sigmoid
    ↓
Output: 80 class probabilities
```

- **BasicBlock** (ResNet-18/34): Conv3x3 → BN → ReLU → Conv3x3 → BN → Skip
- **Bottleneck** (ResNet-50): Conv1x1 → BN → ReLU → Conv3x3 → BN → ReLU → Conv1x1 → BN → Skip

### 양자화 비교

| 방식 | 학습 | 모델 크기 | Classification mAP | 특징 |
|------|------|----------|-------------------|------|
| **LSQ QAT** | 학습 중 양자화 시뮬레이션 | 43 MB | 66.72% | Fake quantization, 높은 정확도 |
| **INT8 PTQ** | 학습 후 양자화 | **11 MB** | 63.33% | Real INT8 weights, 빠른 추론 |

## 파이프라인 개요

```
┌─────────────────────────────────────────────────────────────────┐
│                    Training Pipeline (PC/Server)                 │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  COCO Dataset ──→ Data Augmentation ──→ ResNet Training         │
│       │                                       │                  │
│       │                                       ↓                  │
│       │                              FP32 Model (mAP: ~67%)      │
│       │                                       │                  │
│       │                            ┌──────────┴──────────┐       │
│       │                            ↓                     ↓       │
│       │                      LSQ QAT Training      INT8 PTQ      │
│       │                            │                     │       │
│       │                            ↓                     ↓       │
│       │                   QAT Model (43MB)      INT8 Model (11MB)│
│       │                   mAP: 66.72%           mAP: 63.33%      │
│       │                            │                     │       │
│       │                            └──────────┬──────────┘       │
│       │                                       ↓                  │
│       │                            ExecuTorch Export (.pte)      │
│       │                              ┌───────┴───────┐           │
│       │                              ↓               ↓           │
│       │                          XNNPACK          NNAPI          │
│       │                           (CPU)           (NPU)          │
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
│                                      (INT8 optimized for NPU)    │
│                                               │                  │
│                                               ↓                  │
│                                    80 Category Predictions       │
│                                               │                  │
│                                               ↓                  │
│                                      Tag Generation              │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

> **mAP 지표 설명**: 위 다이어그램의 mAP는 Multi-label Classification mAP (PR-AUC 기반)입니다.

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
- **INT8 PTQ**: Post-Training Quantization으로 실제 INT8 변환
- **Per-tensor Quantization**: ExecuTorch 호환 양자화

### 5. Android 벤치마크 앱
- ExecuTorch 모델 로딩 및 추론
- COCO val2017 이미지 벤치마크
- CPU/NPU 백엔드 지원
- 디바이스 정보 표시 (NPU 감지)

## 참고 문헌

1. **ResNet**: He et al., "Deep Residual Learning for Image Recognition" (CVPR 2016)
2. **LSQ**: Esser et al., "Learned Step Size Quantization" (ICLR 2020)
3. **Integer-Only Inference**: Jacob et al., "Quantization and Training of Neural Networks for Efficient Integer-Arithmetic-Only Inference" (CVPR 2018)

## 라이선스

이 프로젝트는 학술 연구 목적으로 개발되었습니다.

# 스마트폰을 위한 경량 ResNet 기반 사진 자동 태깅 시스템

On-Device AI 기반 실시간 사진 자동 태깅 시스템으로, LSQ(Learned Step Size Quantization) 양자화된 ResNet-18 모델을 ExecuTorch를 통해 모바일에 배포합니다.

## 프로젝트 개요

- **목표**: COCO 데이터셋 80개 카테고리를 인식하는 다중 레이블 이미지 분류 모델 개발 및 모바일 배포
- **모델**: ResNet-18 (Multi-label Classification)
- **양자화**: LSQ 8-bit Quantization-Aware Training
- **배포**: ExecuTorch (PyTorch Mobile)
- **타겟 디바이스**: Galaxy S24+ (Exynos 2400, NPU 지원)

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
│   │   └── augmentation.py      # 데이터 증강
│   ├── models/
│   │   ├── resnet.py            # ResNet-18 구현
│   │   └── quantization.py      # LSQ 양자화 모듈
│   ├── training/
│   │   ├── trainer.py           # Trainer 클래스
│   │   └── loss.py              # 손실 함수 (BCE, Focal, Asymmetric)
│   ├── inference/
│   │   └── predictor.py         # PyTorch/ExecuTorch 추론
│   └── utils/
│       └── metrics.py           # 평가 지표 (mAP, F1 등)
├── data/
│   └── coco/                    # COCO 2017 데이터셋 위치
├── android/                     # Android 앱 프로젝트 (추후 개발)
├── notebooks/                   # 실험 노트북
├── reference/                   # 참고 논문 및 제안서
│   ├── paper/
│   │   ├── Deep Residual Learning for Image Recognition.pdf
│   │   ├── Learned Step Size Quantization.pdf
│   │   └── Quantization and Training of Neural Networks...pdf
│   └── proposal/
│       └── 연구논문작품 제안서.pdf
└── requirements.txt
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

### 2. Quantization-Aware Training (QAT)

```bash
python scripts/train.py \
    --config configs/config.yaml \
    --qat \
    --checkpoint checkpoints/fp32/best_model.pth
```

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

## 기술 스택

| 항목 | 기술 |
|------|------|
| 학습 프레임워크 | PyTorch |
| 양자화 | LSQ (Learned Step Size Quantization) |
| 모바일 배포 | ExecuTorch (PyTorch Mobile) |
| 데이터셋 | COCO 2017 (80 카테고리) |
| Android 언어 | Kotlin |
| 하드웨어 가속 | NNAPI (NPU) |

## 목표 성능

| 지표 | 목표 |
|------|------|
| mAP@0.5 | > 60% |
| 모델 크기 | < 5MB |
| CPU 추론 시간 | < 50ms |
| NPU 추론 시간 | < 20ms |
| 메모리 사용량 | < 100MB |

## 파이프라인 개요

```
┌─────────────────────────────────────────────────────────────┐
│                    Training (PC/Server)                      │
├─────────────────────────────────────────────────────────────┤
│  COCO Dataset → ResNet-18 학습 → LSQ QAT → ExecuTorch 변환  │
└─────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────┐
│                    Inference (Mobile)                        │
├─────────────────────────────────────────────────────────────┤
│  이미지 입력 → ExecuTorch Runtime → 태그 생성 (80 카테고리) │
└─────────────────────────────────────────────────────────────┘
```

## 참고 문헌

1. **ResNet**: He et al., "Deep Residual Learning for Image Recognition" (CVPR 2016)
2. **LSQ**: Esser et al., "Learned Step Size Quantization" (ICLR 2020)
3. **Integer-Only Inference**: Jacob et al., "Quantization and Training of Neural Networks for Efficient Integer-Arithmetic-Only Inference" (CVPR 2018)

## 라이선스

이 프로젝트는 학술 연구 목적으로 개발되었습니다.

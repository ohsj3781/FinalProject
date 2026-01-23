# AI Model Benchmark Application

ExecuTorch 모델의 추론 성능을 측정하는 Android 애플리케이션입니다.

## 기능

- COCO 2017 validation 데이터셋을 사용한 배치 추론
- 전체 추론 시간 측정
- 이미지당 평균 추론 시간 측정
- 실시간 진행 상황 표시
- Min/Max 추론 시간 통계
- FPS (초당 프레임 수) 계산

## 요구 사항

- Android 8.0 (API 26) 이상
- 저장소 접근 권한
- COCO val2017 이미지 (기기에 미리 복사 필요)

## 설정

### 1. 모델 파일 추가

`app/src/main/assets/` 폴더에 ExecuTorch 모델 파일을 추가합니다:

```
app/src/main/assets/model.pte
```

### 2. COCO 데이터셋 준비

COCO 2017 validation 이미지를 기기의 **내장 저장공간 Pictures 폴더**에 복사합니다:

```
내장 저장공간/Pictures/coco/val2017/
(실제 경로: /storage/emulated/0/Pictures/coco/val2017/)
```

PC에서 ADB를 사용하여 복사:

```bash
# PC에서 COCO val2017 이미지를 기기로 복사
adb push data/coco/val2017 /storage/emulated/0/Pictures/coco/val2017
```

또는 파일 관리자 앱을 사용하여 `Pictures/coco/val2017` 폴더를 만들고 이미지를 복사합니다.

### 3. 빌드 및 실행

```bash
# Android Studio에서 열기
# 또는 명령줄에서:
./gradlew assembleDebug
adb install app/build/outputs/apk/debug/app-debug.apk
```

## 사용 방법

1. 앱 실행
2. COCO 이미지 경로 확인 (기본값: 내장 저장공간 `Pictures/coco/val2017`)
3. "Start Benchmark" 버튼 클릭
4. 벤치마크 완료 후 결과 확인

## 출력 지표

| 지표 | 설명 |
|------|------|
| Total Time | 전체 벤치마크 소요 시간 |
| Inference Time | 순수 추론 시간 합계 |
| Avg per Image | 이미지당 평균 추론 시간 |
| Min Inference | 최소 추론 시간 |
| Max Inference | 최대 추론 시간 |
| FPS | 초당 처리 이미지 수 |

## 프로젝트 구조

```
android/
├── app/
│   ├── src/main/
│   │   ├── java/com/example/aibenchmark/
│   │   │   ├── MainActivity.kt      # 메인 UI
│   │   │   └── ModelBenchmark.kt    # 모델 추론 클래스
│   │   ├── res/
│   │   │   ├── layout/              # UI 레이아웃
│   │   │   └── values/              # 리소스 파일
│   │   ├── assets/                  # 모델 파일 위치
│   │   └── AndroidManifest.xml
│   └── build.gradle.kts
├── build.gradle.kts
├── settings.gradle.kts
└── gradle/
```

## 의존성

- ExecuTorch Android: `org.pytorch:executorch-android:0.4.0`
- AndroidX Core, AppCompat, Material Design
- Kotlin Coroutines

## 라이선스

학술 연구 목적으로 개발되었습니다.

# Compact Neural Appearance Models for Efficient Gaussian Splatting

[![Paper](https://img.shields.io/badge/Paper-arXiv%3A2609.05255-b31b1b.svg)](./doc/)
[![License: Apache-2.0](https://img.shields.io/badge/License-Apache--2.0-blue.svg)](./efficient-gaussian-appearance/LICENSE)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-EE4C2C?logo=PyTorch&logoColor=white)](https://pytorch.org/)
[![Three.js](https://img.shields.io/badge/Three.js-r177-049EF4?logo=threedotjs&logoColor=white)](https://threejs.org/)
[![WebGL 2.0](https://img.shields.io/badge/WebGL-2.0-990000?logo=WebGL&logoColor=white)](./efficient-gaussian-appearance/viewer/)

> **Compact Neural Appearance Models for Efficient Gaussian Splatting**  
> Official research implementation, standalone PyTorch modules, and interactive WebGL 3D viewer for next-generation efficient 3D Gaussian Splatting (3DGS).

---

## 📖 Overview (개요)

기존 3D Gaussian Splatting (3DGS)은 시점 의존적 외관(View-Dependent Appearance)을 표현하기 위해 3차 구면 조화 함수(Spherical Harmonics, SH)를 기본으로 사용합니다. 그러나 SH는 가우시안당 48개의 단정밀도 부동소수점 계수(**192바이트**)를 차지하여 VRAM 폭증, 대역폭 병목, 고주파 반사 표현 시 기하학적 왜곡을 초래합니다.

본 프로젝트는 Florian Hahlbohm et al. (2026)의 최신 논문 **"Compact Neural Appearance Models for Efficient Gaussian Splatting"**의 전체 파이프라인을 구현 및 제공합니다:
1. **8차원 잠재 특징(Latent Feature) + 3차원 베이스 색상**을 저장하여 가우시안당 외관 크기를 **단 28바이트**로 압축 (**기존 SH 대비 85.4% 메모리 절감**).
2. 씬 전체가 공유하는 **초소형 MLP (2층, 16개 뉴런, 816 파라미터)**가 시점 방향과 잠재 특징을 입력받아 실시간으로 고주파 반사광 잔차를 디코딩.
3. 표준 PyTorch 환경에서 즉시 실험할 수 있는 **독립형 모듈(`compact_appearance/`)** 제공.
4. GitHub Pages 및 Cloudflare Pages에 즉시 배포 가능한 **Three.js WebGL 3D 뷰어(`viewer/`)** 및 로컬 오프라인 고화질 모델 포함.

---

## 📁 Repository Structure (디렉터리 구조)

```
Efficient-Gaussian-Splatting/
├── doc/                                # 연구 논문 원문 PDF
│   ├── 2609.05255v1.pdf                # arXiv 프리프린트 판
│   └── Compact Neural Appearance...pdf # 프로젝트 공식 배포 판
│
├── compact_appearance/                 # 독립 실행 가능한 순수 PyTorch 외관 모델 패키지
│   ├── __init__.py                     # 모델 팩토리 함수 (create_appearance_model)
│   ├── appearance_base.py              # 통합 외관 기본 인터페이스 및 활성화 함수
│   ├── neural_appearance.py            # 컴팩트 뉴럴 모델 (SH 방향 인코더 + 주파수 인코더 + Shared MLP)
│   ├── spherical_models.py             # 비교 모델군 (SH, NASG, NASGabor, Spherical Voronoi)
│   └── benchmark_and_verify.py         # 메모리 및 반사광 피팅 검증 벤치마크
│
└── efficient-gaussian-appearance/      # CUDA 래스터라이저 및 NeRFICG 통합 프레임워크
    ├── Appearance/                     # 프레임워크용 외관 모듈
    ├── FasterGSVDACudaBackend/         # NVRTC 런타임 코드 생성 (RTC) CUDA 래스터라이저
    ├── viewer/                         # Three.js 기반 실시간 WebGL 3D 뷰어
    │   ├── index.html                  # 뷰어 진입점 (GitHub Pages / Cloudflare Pages 배포 가능)
    │   ├── main.js, utils.js, sorter.js
    │   ├── shaders/                    # WebGL2 GLSL 셰이더 (eval_neural.glsl 등)
    │   └── models/                     # 로컬 오프라인 3D 모델 (.ngsplat)
    ├── Model.py, Trainer.py, Renderer.py
    └── fastergsvda_*.yaml              # 모델별 학습 설정 파일
```

---

## 📊 Appearance Models Benchmark (모델 비교 실측표)

1,000,000개 가우시안 기준 메모리 및 고주파 반사광 피팅 성능 실측 결과:

| 모델 (Model) | 가우시안당 바이트 | 100만 개 파라미터 수 | 100만 개 저장 공간 | 반사광 피팅 PSNR | 특징 및 장단점 |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **None** (Diffuse) | **12 B** | 3,000,000 | 11.44 MB | - | 시점 무관 (기준선) |
| **SH** (기존 3DGS) | **192 B** | 48,000,000 | 183.11 MB | 19.14 dB | 대역 제한 기저, 높은 메모리/VRAM |
| **SV** (Spherical Voronoi) | **208 B** | 52,000,000 | 198.36 MB | - | 구면 사이트 소프트 보간 |
| **NASG** | **44 B** | 11,000,000 | 41.96 MB | - | 이방성 가우시안 로브 |
| **NASGabor** | **48 B** | 12,000,000 | 45.78 MB | 15.20 dB | NASG + 코사인 캐리어 주파수 |
| **Neural (제안 모델)** | **28 B** | **7,000,000** | **26.70 MB** | **33.96 dB** | **-85.4% 메모리 절감, +14.8 dB 고품질 복원** |

---

## 🚀 Quick Start (빠른 시작)

### 1. WebGL 3D 뷰어 로컬 구동 (Local WebGL Viewer)
별도의 빌드나 의존성 설치 없이 Python 기본 HTTP 서버로 즉시 3D 뷰어를 실행할 수 있습니다:

```bash
cd efficient-gaussian-appearance/viewer
python -m http.server 8080
```
브라우저에서 `http://localhost:8080`을 열면 갤러리가 로드되며, 다음 주소로 즉시 특정 3D 씬을 감상할 수 있습니다:
- **Chair (Neural 28B)**: `http://localhost:8080/?model=models/neural/chair.ngsplat`
- **Materials (고반사 메탈 구체 - Neural)**: `http://localhost:8080/?model=models/neural/materials.ngsplat`
- **Chair (기존 SH 192B)**: `http://localhost:8080/?model=models/sh/chair.ngsplat`

### 2. GitHub Pages / Cloudflare Pages 정적 배포
`efficient-gaussian-appearance/viewer` 폴더는 **100% 정적 웹 애플리케이션 (HTML5 + WebGL2 + Three.js ES Modules)**입니다:
- **GitHub Pages**: 저장소의 Pages 설정에서 `viewer` 디렉터리를 소스로 지정하면 즉시 글로벌 호스팅됩니다.
- **Cloudflare Pages**: 해당 저장소를 연결하고 빌드 출력 디렉터리를 `efficient-gaussian-appearance/viewer`로 지정하면 전 세계 초고속 CDN을 통해 서비스됩니다.

### 3. 독립형 PyTorch 모듈 검증 및 벤치마크
외부 복잡한 C++ 툴체인 없이 표준 PyTorch 환경에서 논문의 수식과 모델을 즉시 검증할 수 있습니다:

```bash
python -m compact_appearance.benchmark_and_verify
```

코드에서 직접 사용 예시:
```python
import torch
from compact_appearance import create_appearance_model

# 1. 10만 개 가우시안 대상 컴팩트 뉴럴 외관 모델 생성
model = create_appearance_model(
    model_type="NEURAL",
    num_primitives=100000,
    color_activation="relu",
    device="cuda"
)

# 2. 카메라 시점 방향 단위 벡터 (N, 3)로부터 색상 계산
directions = torch.randn(100000, 3, device="cuda")
directions = torch.nn.functional.normalize(directions, dim=-1)

colors = model(directions) # (100000, 3) 크기의 [0, 1] RGB 텐서
```

---

## 📜 Citation

```bibtex
@article{hahlbohm2026efficientgaussianappearance,
  title={Compact Neural Appearance Models for Efficient Gaussian Splatting},
  author={Hahlbohm, Florian and Condor, Jorge and Franke, Linus and Eisemann, Martin and Magnor, Marcus},
  journal={arXiv preprint arXiv:2609.05255},
  year={2026}
}
```

## 📄 License
This project is licensed under the Apache-2.0 License - see the [LICENSE](./efficient-gaussian-appearance/LICENSE) file for details.

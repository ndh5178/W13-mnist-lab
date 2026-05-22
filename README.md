# 과제 - 신경망을 이용한 손글씨 숫자 인식

## 1. 개요

본 과제는 **PyTorch, TensorFlow 등 외부 딥러닝 프레임워크를 사용하지 않고**, `NumPy`만으로 신경망의 핵심 구성 요소를 직접 구현하는 것을 목표로 합니다.

- **최종 목표**: MNIST 필기체 숫자 분류기 구현, **테스트 정확도 97% 이상** (최소 95% 이상)
- **참고 도서**: 『밑바닥부터 시작하는 딥러닝』 1~6장

---

## 2. 환경 설정

### 2.1 요구 사항

- **Python 3.11** (로컬 Conda 환경은 3.11로 통일)
- **허용 라이브러리**: `numpy`, `math`, `random`, `time`, `matplotlib`(시각화)
- **금지**: 허용 라이브러리 이외의 모든 라이브러리

### **2.2 Colab에서 사용할 때 (권장)**

1. 먼저 과제 템플릿을 본인 팀 저장소에 업로드합니다.
2. 브라우저에서 아래 주소로 이동해 노트북을 엽니다. `USERNAME`, 저장소명, 브랜치명은 본인 또는 팀 환경에 맞게 바꿉니다. (브랜치가 `main`이면 URL의 `master`를 `main`으로 변경)
  ```python
    https://colab.research.google.com/github/USERNAME/mnist-lab/blob/master/mnist_lab.ipynb
  ```
3. **런타임 설정**: 상단 메뉴 **런타임 → 런타임 유형 변경**에서 **Python 3**, CPU 또는 GPU를 선택합니다.
4. **첫 셀 실행**: 노트북 맨 위의 **「1. 환경설정」** 코드 셀을 먼저 실행합니다.

- **Colab에서만** 다음 두 가지를 입력합니다.
  - **GitHub 저장소 URL** (예: `github.com/USERNAME/mnist-lab.git`)
  - **GitHub Personal Access Token** (private 저장소인 경우)

5. 그 다음 셀부터 순서대로 실행하여 학습·평가를 진행합니다.

### 2.3 로컬에서 실행할 때 (Conda)

로컬에서는 **Conda**로 환경을 만들고 실행합니다. **Mac**은 **Miniforge**, **Windows**는 **Anaconda**를 사용하며, **Python 3.11**로 통일합니다.

#### Mac (Apple Silicon) — Miniforge

1. **Miniforge 설치** (Apple Silicon용, conda-forge 채널 기본)
  - 다운로드: [Miniforge - GitHub](https://github.com/conda-forge/miniforge#miniforge3)
  - Apple Silicon: `Miniforge3-macOS-arm64` 설치 파일 사용
  - 설치 후 Conda 위치(기본): `~/miniforge3` (홈 디렉터리 아래)
2. **터미널에서**:

```bash
# 저장소로 이동
cd mnist-lab

# Conda 환경 생성 (Python 3.11)
conda create -n mnist-nn python=3.11 -y

# 환경 활성화
conda activate mnist-nn

# 의존성 설치
pip install -r requirements.txt

# 테스트 실행 (선택)
pytest tests/ -v
```

- Miniforge가 PATH에 없으면: `~/miniforge3/bin/conda activate mnist-nn` 처럼 전체 경로로 실행

#### Windows — Anaconda

1. **Anaconda 설치**
  - 다운로드: [Anaconda Distribution](https://www.anaconda.com/download)
  - 설치 시 **"Add Anaconda to my PATH environment variable"** 옵션 권장 (체크 시 터미널에서 `conda` 바로 사용)
  - 설치 후 Conda 위치(기본): `C:\Users\<사용자명>\anaconda3` 또는 `C:\ProgramData\anaconda3`
2. **PowerShell** 또는 **명령 프롬프트(cmd)** 에서:

```bash
# 저장소로 이동
cd mnist-lab

# Conda 환경 생성 (Python 3.11)
conda create -n mnist-nn python=3.11 -y

# 환경 활성화
conda activate mnist-nn

# 의존성 설치
pip install -r requirements.txt

# 테스트 실행 (선택)
pytest tests/ -v
```

- PATH에 없으면: `C:\Users\<사용자명>\anaconda3\Scripts\conda.exe activate mnist-nn` 처럼 전체 경로로 실행
- **환경 비활성화**: `conda deactivate`

### 2.4 MNIST 데이터 (data 폴더)

- MNIST 데이터는 **`data/mnist.npz`**에 두고 사용합니다.
- **`load_mnist()`**는 이미 구현되어 있습니다.
  - `data/mnist.npz`가 있으면 해당 파일을 로드합니다.
  - 없으면 URL에서 다운로드한 뒤 `data/` 폴더에 저장한 후 로드합니다.
- 데이터를 미리 받으려면 프로젝트 루트에서 **`python download_mnist.py`**를 한 번 실행하면 됩니다.

---

## 3. 프로젝트 구조

```
mnist-lab/
├── .gitignore                     # data/mnist.npz, __pycache__ 등 제외
├── README.md                      # 이 파일 (과제 안내·환경)
├── REPORT.md                      # 제출용 보고서 (형식 예시)
├── requirements.txt               # numpy, matplotlib, pytest
├── download_mnist.py              # MNIST를 data/에 미리 다운로드 (선택)
├── mnist_lab.ipynb                # Colab/로컬용 노트북 (환경설정 → 데이터 로드 → 학습 → 평가)
├── data/                          # MNIST 데이터 (mnist.npz는 load_mnist() 또는 download_mnist.py로 생성)
├── src/
│   ├── __init__.py
│   ├── data.py                    # 데이터 로드
│   ├── activations.py             # ReLU, Softmax
│   ├── layers.py                  # Affine, BatchNorm, Dropout
│   ├── losses.py                  # cross_entropy_loss
│   ├── optimizers.py              # SGD, Adam
│   ├── network.py                 # NeuralNetwork
│   └── training.py                # train, evaluate, plot_loss_history
└── tests/
    ├── conftest.py                # 테스트 공통 import 경로 설정
    ├── test_relu.py               # ReLU 테스트
    ├── test_softmax.py            # Softmax 테스트
    ├── test_affine.py             # Affine 테스트
    ├── test_cross_entropy_loss.py # cross_entropy_loss 테스트
    ├── test_sgd.py                # SGD 테스트
    ├── test_adam.py               # Adam 테스트
    ├── test_neural_network.py     # NeuralNetwork 테스트
    ├── test_batchnorm.py          # BatchNorm 테스트
    ├── test_dropout.py            # Dropout 테스트
    ├── test_training.py           # train 테스트
    └── test_evaluate.py           # evaluate 테스트
```

`src/`에는 Python이 import하는 구현 파일만 둡니다. `mnist_lab.ipynb`는 학생이 가장 먼저 열어 실행 순서를 따라가는 안내서이므로 프로젝트 루트에 둡니다. 테스트와 노트북은 `from activations import ReLU`, `from network import NeuralNetwork`처럼 역할별 모듈을 직접 import합니다.

---

## 4. 제출물

- **팀별 제출**: **동작하는 소스코드** + **REPORT.md**
- **소스코드**: `src/` 아래 소스 전체를 zip으로 압축 
- **REPORT.md** 에 다음 구성을 포함할 것 (형식 예시: 저장소의 `REPORT.md` 참고):
  - **0. 반·팀원**: 반, 팀원 이름
  - **1. 실험 목적**: 과제 요약 (한두 문장)
  - **2. 모델 구조**: 입력/은닉층/출력, Affine·BatchNorm·ReLU·Dropout 구성
  - **3. 학습 설정**: 옵티마이저, 학습률, epochs, batch_size, Dropout 비율, BatchNorm momentum, 가중치 초기화
  - **4. 실험 환경**: Python·라이브러리, 학습 소요 시간
  - **5. 결과**: 테스트 정확도(%), 총 파라미터 수, 손실 커브 (그래프 또는 요약)
  - **6. 회고**: 수렴 여부, 과적합/과소적합, 구조·하이퍼파라미터 변경 시도와 결과


# MNIST Lab 코드 스터디 로그

## 진행 현황
- [x] `network.py` — 진행 중
- [ ] `training.py`

---

## network.py

### `from collections import OrderedDict`
- forward는 순서대로, backward는 reversed()로 뒤집어야 하므로 삽입 순서 보장이 필요
- Python 3.7+에선 일반 dict도 순서 보장이지만 의도 명시용으로 OrderedDict 사용

---

### `layer_sizes = [784, 512, 256, 10]`
- 784 = MNIST 이미지 28×28 픽셀을 1D로 펼친 것
- 512, 256 = 은닉층 뉴런 수
- 10 = 숫자 0~9 클래스 수

---

### 가중치 초기화 (He / Xavier)

```python
scale = np.sqrt(2.0 / in_dim) if idx < len(layer_sizes) - 1 else np.sqrt(1.0 / in_dim)
self.params[f"W{idx}"] = scale * np.random.randn(in_dim, out_dim)
```

**왜 초기화를 조정하나?**
- 가중치를 그냥 randn()으로만 쓰면, 레이어를 거칠수록 n개를 더하는 연산이 반복되어 분산이 누적으로 폭발하거나 소실됨
- Xavier/He는 레이어마다 분산을 일정하게 유지시켜주는 역할

**He 초기화 `√(2/n)` → 은닉층 (ReLU 뒤)**
- ReLU는 음수를 0으로 죽여서 뉴런 절반이 꺼짐 → 분산이 절반으로 줄어듦
- 2를 곱해서 "절반이 죽으니까 2배로 보정"

**Xavier 초기화 `√(1/n)` → 출력층 (Softmax 앞)**
- 출력층 뒤에는 레이어가 없으므로 분산 누적 문제는 없음
- 대신 W가 크면 Softmax 입력(로짓)이 커져서 확률이 한쪽으로 쏠림 → gradient 소실
- W를 작게 초기화해서 로짓을 고르게 만들어주는 것

> 정리: ReLU 은닉층 → He (앞으로 가는 분산 폭발 방지) / Softmax 출력층 → Xavier (로짓이 너무 커지는 것 방지)

**용어 정리**
- **로짓(logit)**: Softmax에 들어가기 직전의 날것 점수값. 어떤 값이든 가능 (음수 포함)
- **로그(log)**: 수학 함수. Cross Entropy 손실 계산할 때 등장. 로짓과 무관

---

### 레이어 쌓는 루프
- 은닉층 수만큼 `Affine → BatchNorm → ReLU → Dropout` 세트를 OrderedDict에 순서대로 쌓음
- 출력층 Affine은 루프 밖에 따로 → BatchNorm/ReLU/Dropout 필요 없기 때문
- `Affine(self.params["W1"], ...)` → 복사가 아니라 참조. optimizer가 params 업데이트하면 Affine.W도 자동으로 바뀜

### Affine이란?
- 완전연결층. `y = xW + b` 선형 변환
- W의 shape이 입출력 사이즈를 결정함 (784→512, 512→256, 256→10)
- Linear, Dense, FC 다 같은 말

### forward()
```python
def forward(self, x, train=True):
    out = x
    for layer in self.layers.values():
        if isinstance(layer, (BatchNorm, Dropout)):
            out = layer.forward(out, train=train)
        else:
            out = layer.forward(out)
    return self.softmax.forward(out)
```
- BatchNorm, Dropout만 `train=` 인자를 받음 → 학습/추론 때 동작이 다르기 때문
  - Dropout: 학습 때 랜덤으로 끔 / 추론 때 다 켬
  - BatchNorm: 학습 때 배치 통계 사용 / 추론 때 running 통계 사용
- Affine, ReLU는 항상 동작이 같아서 `train=` 불필요
- 마지막에 Softmax로 로짓 → 확률 변환

### backward()
```python
def backward(self, dout):
    dout = self.softmax.backward(dout)  # 그냥 통과 (gradient는 training.py에서 계산)
    for layer in reversed(list(self.layers.values())):
        dout = layer.backward(dout)
```
- 연쇄법칙(chain rule) 때문에 반드시 출력→입력 방향으로 계산
- Softmax backward는 껍데기 — training.py에서 gradient를 미리 계산해서 넘겨줌

### grads 저장
- backward 후 각 레이어에 흩어진 gradient를 `model.grads` 딕셔너리에 모음
- optimizer는 `model.grads`만 보기 때문
- ReLU, Dropout은 업데이트할 파라미터 자체가 없어서 grads에 없음

---

## training.py

## 다음에 이어서 볼 것
- `training.py` — 학습 루프 전체

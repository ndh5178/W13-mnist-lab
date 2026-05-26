# MNIST 손글씨 인식 과제 보고서

## 0. 반·팀원

| 항목 | 내용 |
| --- | --- |
| 반 | 301호 |
| 팀명 | 2조 |
| 팀원 | 김민철 |
| 팀원 | 김세민 |
| 팀원 | 남동현 |
| 팀원 | 박승현 |

---

## 1. 실험 목적

본 실험의 목적은 PyTorch, TensorFlow 같은 딥러닝 프레임워크 없이 NumPy만 사용하여 MNIST 손글씨 숫자 분류 신경망을 직접 구현하는 것이다.

구현 대상은 forward, loss, backward, optimizer update까지 이어지는 학습 흐름 전체이다. 입력 이미지는 28x28 픽셀을 펼친 784차원 벡터이며, 모델은 각 이미지가 0~9 중 어떤 숫자인지 확률로 예측한다.

목표 정확도는 최소 95% 이상, 권장 97% 이상이다.

---

## 2. 모델 구조

사용한 모델은 2개의 은닉층을 가진 다층 퍼셉트론(MLP)이다.

```text
입력 784
-> Affine(512)
-> BatchNorm
-> ReLU
-> Dropout
-> Affine(256)
-> BatchNorm
-> ReLU
-> Dropout
-> Affine(10)
-> Softmax
```

| 구분 | 내용 |
| --- | --- |
| 입력층 | 784차원, MNIST 28x28 이미지를 펼친 벡터 |
| 은닉층 1 | Affine(784, 512) -> BatchNorm -> ReLU -> Dropout |
| 은닉층 2 | Affine(512, 256) -> BatchNorm -> ReLU -> Dropout |
| 출력층 | Affine(256, 10) -> Softmax |
| 활성화 함수 | ReLU |
| 정규화 | BatchNorm 사용 |
| 규제 | Dropout 사용 |

BatchNorm은 각 은닉층의 Affine 출력 분포를 안정화하기 위해 사용했다. Dropout은 학습 중 일부 뉴런 출력을 0으로 만들어 과적합을 줄이기 위해 사용했다.

---

## 3. 학습 설정

| 항목 | 값 |
| --- | --- |
| optimizer | Adam |
| learning rate | 0.001 |
| epochs | 15 |
| batch size | 128 |
| dropout ratio | 0.2 |
| BatchNorm momentum | 0.9 |
| weight initialization | He initialization |
| loss function | Cross Entropy Loss |

가중치는 ReLU 계열 활성화 함수에 맞춰 He 초기화를 사용했다.

```python
W = np.random.randn(fan_in, fan_out) * np.sqrt(2.0 / fan_in)
```

학습 루프는 각 mini-batch마다 다음 순서로 진행했다.

```text
model.forward(x_batch, train=True)
-> cross_entropy_loss(y_pred, y_batch)
-> dout 생성
-> model.backward(dout)
-> optimizer.update(model.params, model.grads)
```

---

## 4. 실험 환경

| 항목 | 내용 |
| --- | --- |
| 실행 환경 | 로컬 Windows |
| Python | 3.11.15 |
| NumPy | 2.4.6 |
| 사용 라이브러리 | numpy, matplotlib |
| 실행 장치 | CPU |
| 학습 시간 | 15 epochs 기준 약 1127초, 약 18분 47초 |

과제 조건상 NumPy만 사용했기 때문에 GPU는 사용하지 않았다.

---

## 5. 결과

| 항목 | 값 |
| --- | --- |
| 최종 test accuracy | 98.31% |
| 최고 test accuracy | 98.47% at epoch 11 |
| 최종 train loss | 0.0171 |
| 총 파라미터 수 | 537,354 |

### Epoch별 결과

| epoch | train loss | test accuracy |
| --- | --- | --- |
| 1 | 0.2452 | 96.96% |
| 2 | 0.1085 | 97.47% |
| 3 | 0.0775 | 97.85% |
| 4 | 0.0612 | 98.25% |
| 5 | 0.0515 | 98.01% |
| 6 | 0.0415 | 98.17% |
| 7 | 0.0381 | 98.11% |
| 8 | 0.0324 | 98.24% |
| 9 | 0.0268 | 98.29% |
| 10 | 0.0284 | 98.38% |
| 11 | 0.0260 | 98.47% |
| 12 | 0.0214 | 98.31% |
| 13 | 0.0202 | 98.33% |
| 14 | 0.0202 | 98.41% |
| 15 | 0.0171 | 98.31% |

### Loss Curve 값

```text
[0.2452, 0.1085, 0.0775, 0.0612, 0.0515,
 0.0415, 0.0381, 0.0324, 0.0268, 0.0284,
 0.0260, 0.0214, 0.0202, 0.0202, 0.0171]
```

학습 loss는 전체적으로 감소했으며, test accuracy는 2 epoch부터 97%를 넘었다. 15 epoch 동안 최고 정확도는 98.47%였다.

---

## 6. 회고

학습 손실은 1 epoch의 0.2452에서 15 epoch의 0.0171까지 감소했다. 이를 통해 forward, loss, backward, optimizer update가 정상적으로 연결되어 있음을 확인했다.

BatchNorm을 사용하여 은닉층의 중간값 분포를 안정화했고, Dropout ratio 0.2를 적용하여 과적합을 줄이도록 구성했다. 최종 정확도는 98% 이상으로 목표였던 97%를 넘겼다.

다만 10 epoch 이후에는 test accuracy가 98.3~98.4% 근처에서 크게 오르지 않았다. train loss는 계속 감소했지만 test accuracy 향상 폭은 작았으므로, 더 높은 정확도를 목표로 한다면 learning rate 조정, dropout ratio 0.3 실험, 은닉층 크기 변경 등을 추가로 비교할 수 있다.


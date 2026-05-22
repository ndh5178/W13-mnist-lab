# -*- coding: utf-8 -*-
"""손실 함수 모음."""

import numpy as np


def cross_entropy_loss(y_pred, y_true):
    """
    Cross Entropy Error (배치 평균).
    y_pred: (batch_size, 10) 확률
    y_true: (batch_size,) 정수 레이블 0~9
    """
    # TODO: 정답 클래스 확률의 log 값을 이용해 batch 평균 cross entropy를 계산하세요.
    # 힌트: np.clip으로 log(0)을 피하고, np.arange(batch_size)로 정답 위치를 고릅니다.
    raise NotImplementedError("cross_entropy_loss를 구현하세요.")

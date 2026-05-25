# -*- coding: utf-8 -*-
"""손실 함수 모음."""

import numpy as np


def cross_entropy_loss(y_pred, y_true):
    """
    Cross Entropy Error (배치 평균).
    y_pred: (batch_size, 10) 확률
    y_true: (batch_size,) 정수 레이블 0~9
    """
    y_pred = np.clip(y_pred, 1e-7, 1.0)
    batch_size = y_pred.shape[0]
    target_probs = y_pred[np.arange(batch_size), y_true]
    return -np.mean(np.log(target_probs))

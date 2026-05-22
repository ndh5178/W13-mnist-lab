# -*- coding: utf-8 -*-
"""Cross entropy loss 단계 테스트."""

import numpy as np

from losses import cross_entropy_loss


class TestCrossEntropyLoss:
    """Step 4: cross_entropy_loss() 구현 후 실행."""

    def test_cross_entropy_loss_scalar(self):
        """cross_entropy_loss()는 배치 평균 손실 하나를 스칼라로 반환해야 한다."""
        y_pred = np.array([[0.1, 0.2, 0.7], [0.8, 0.1, 0.1]])
        y_true = np.array([2, 0])
        loss = cross_entropy_loss(y_pred, y_true)
        assert np.isscalar(loss) or loss.shape == ()
        assert loss > 0

    def test_cross_entropy_loss_perfect(self):
        """정답 클래스 확률이 1에 가까우면 cross entropy loss도 0에 가까워야 한다."""
        y_pred = np.array([[0.0, 0.0, 1.0], [1.0, 0.0, 0.0]])
        y_true = np.array([2, 0])
        loss = cross_entropy_loss(y_pred, y_true)
        assert loss < 0.01

# -*- coding: utf-8 -*-
"""Dropout layer 단계 테스트."""

import numpy as np

from layers import Dropout


class TestDropout:
    """Step 9: Dropout.forward(), Dropout.backward() 구현 후 실행."""

    def test_dropout_forward_train_shape(self):
        """학습 모드 Dropout.forward()가 입력 shape를 유지하는지 확인한다."""
        drop = Dropout(drop_ratio=0.5)
        x = np.random.randn(3, 4)
        out = drop.forward(x, train=True)
        assert out.shape == x.shape

    def test_dropout_forward_inference_scale(self):
        """추론 모드 Dropout.forward()가 평균 출력 크기에 맞게 scale하는지 확인한다."""
        drop = Dropout(drop_ratio=0.5)
        x = np.ones((2, 2))
        out = drop.forward(x, train=False)
        np.testing.assert_array_almost_equal(out, 0.5 * x)

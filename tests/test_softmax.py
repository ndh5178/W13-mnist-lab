# -*- coding: utf-8 -*-
"""Softmax 단계 테스트."""

import numpy as np

from activations import Softmax


class TestSoftmax:
    """Step 2: Softmax.forward(), Softmax.backward() 구현 후 실행."""

    def test_softmax_forward_sum_one(self):
        """Softmax 출력은 샘플마다 모든 클래스 확률의 합이 1이어야 한다."""
        softmax = Softmax()
        x = np.array([[1.0, 2.0, 3.0], [0.0, 1.0, 2.0]])
        out = softmax.forward(x)
        np.testing.assert_array_almost_equal(out.sum(axis=1), np.ones(2))

    def test_softmax_forward_non_negative(self):
        """Softmax 출력 확률은 항상 0 이상 1 이하 범위에 있어야 한다."""
        softmax = Softmax()
        x = np.random.randn(4, 10)
        out = softmax.forward(x)
        assert (out >= 0).all() and (out <= 1).all()

    def test_softmax_backward_shape(self):
        """Softmax.backward()는 입력 gradient와 같은 shape를 반환해야 한다."""
        softmax = Softmax()
        x = np.random.randn(3, 5)
        softmax.forward(x)
        dout = np.random.randn(3, 5)
        dx = softmax.backward(dout)
        assert dx.shape == dout.shape

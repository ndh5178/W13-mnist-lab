# -*- coding: utf-8 -*-
"""ReLU 단계 테스트.

실행 예:
    pytest tests/test_relu.py -v
    pytest tests/test_relu.py -v -k "backward"
"""

import numpy as np

from activations import ReLU


class TestReLU:
    """Step 1: ReLU.forward(), ReLU.backward() 구현 후 실행."""

    def test_relu_forward_positive(self):
        """양수 입력은 ReLU.forward()에서 값이 그대로 유지되어야 한다."""
        relu = ReLU()
        x = np.array([[1.0, 2.0], [3.0, 4.0]])
        out = relu.forward(x)
        np.testing.assert_array_almost_equal(out, x)

    def test_relu_forward_negative_zero(self):
        """음수와 0 입력은 ReLU.forward()에서 0으로 바뀌어야 한다."""
        relu = ReLU()
        x = np.array([[-1.0, 2.0], [0.0, -3.0]])
        out = relu.forward(x)
        expected = np.array([[0.0, 2.0], [0.0, 0.0]])
        np.testing.assert_array_almost_equal(out, expected)

    def test_relu_backward(self):
        """ReLU.backward()는 forward 때 양수였던 위치로만 gradient를 흘려야 한다."""
        relu = ReLU()
        x = np.array([[-1.0, 2.0], [0.0, -3.0]])
        relu.forward(x)
        dout = np.ones_like(x)
        dx = relu.backward(dout)
        # 음수/0 위치에서는 gradient가 0이어야 합니다.
        assert dx[0, 0] == 0 and dx[1, 0] == 0 and dx[1, 1] == 0
        assert dx[0, 1] == 1

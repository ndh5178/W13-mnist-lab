# -*- coding: utf-8 -*-
"""BatchNorm layer 단계 테스트."""

import numpy as np

from layers import BatchNorm


class TestBatchNorm:
    """Step 8: BatchNorm.forward(), BatchNorm.backward() 구현 후 실행."""

    def test_batchnorm_forward_shape(self):
        """BatchNorm.forward()가 입력과 같은 shape의 출력을 반환하는지 확인한다."""
        gamma = np.ones(4)
        beta = np.zeros(4)
        bn = BatchNorm(gamma, beta, momentum=0.0)
        x = np.random.randn(5, 4)
        out = bn.forward(x, train=True)
        assert out.shape == x.shape

    def test_batchnorm_backward_shape(self):
        """BatchNorm.backward()가 입력 x와 같은 shape의 dx를 반환하는지 확인한다."""
        gamma = np.ones(4)
        beta = np.zeros(4)
        bn = BatchNorm(gamma, beta, momentum=0.0)
        x = np.random.randn(5, 4)
        bn.forward(x, train=True)
        dout = np.random.randn(5, 4)
        dx = bn.backward(dout)
        assert dx.shape == x.shape

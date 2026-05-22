# -*- coding: utf-8 -*-
"""NeuralNetwork 조립 단계 테스트."""

import numpy as np
import pytest

from network import NeuralNetwork


class TestNeuralNetwork:
    """Step 7: NeuralNetwork __init__, forward, backward, params, grads 구현 후 실행."""

    @pytest.fixture
    def simple_model(self):
        """Affine, BatchNorm, ReLU, Dropout 구조가 동작하면 통과."""
        try:
            return NeuralNetwork(use_batchnorm=True, use_dropout=True)
        except TypeError:
            return NeuralNetwork()

    def test_neural_network_forward_shape(self, simple_model):
        """NeuralNetwork.forward()가 MNIST 배치를 받아 10개 클래스 확률을 반환하는지 확인한다."""
        x = np.random.randn(4, 784).astype(np.float32)
        out = simple_model.forward(x, train=True)
        assert out.shape == (4, 10)

    def test_neural_network_params_exist(self, simple_model):
        """NeuralNetwork가 optimizer에 넘길 params dict를 가지고 있는지 확인한다."""
        assert "params" in dir(simple_model)
        assert len(simple_model.params) >= 2

    def test_neural_network_backward_produces_grads(self, simple_model):
        """NeuralNetwork.backward() 후 params와 같은 개수의 grads가 만들어지는지 확인한다."""
        x = np.random.randn(2, 784).astype(np.float32)
        simple_model.forward(x, train=True)
        dout = np.random.randn(2, 10)
        simple_model.backward(dout)
        assert "grads" in dir(simple_model)
        assert len(simple_model.grads) == len(simple_model.params)

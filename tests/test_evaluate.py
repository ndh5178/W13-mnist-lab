# -*- coding: utf-8 -*-
"""evaluate() 제공 함수 동작 확인 테스트."""

import numpy as np
import pytest

from network import NeuralNetwork
from training import evaluate


class TestEvaluate:
    """evaluate()는 템플릿에서 제공되며, 모델 평가 흐름 확인용으로 실행한다."""

    def test_evaluate_returns_acc_and_params(self):
        """evaluate()가 정확도(0~100)와 양수 파라미터 개수를 반환하는지 확인한다."""
        try:
            model = NeuralNetwork(use_batchnorm=True, use_dropout=True)
        except (TypeError, NotImplementedError):
            pytest.skip("NeuralNetwork 미구현")
        x = np.random.randn(10, 784).astype(np.float32)
        y = np.random.randint(0, 10, 10)
        acc, n_params = evaluate(model, x, y)
        assert 0 <= acc <= 100
        assert n_params > 0

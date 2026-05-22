# -*- coding: utf-8 -*-
"""train() 학습 루프 단계 테스트."""

import numpy as np
import pytest

from network import NeuralNetwork
from optimizers import Adam
from training import train


class TestTrain:
    """Step 10: train() 구현 후 실행. 짧은 epoch로만 검증."""

    @pytest.fixture
    def tiny_data(self):
        """테스트용 소량 데이터. 네트워크나 파일 다운로드가 필요 없다."""
        np.random.seed(42)
        x = np.random.rand(256, 784).astype(np.float32)
        y = np.random.randint(0, 10, 256)
        return x, y

    def test_train_returns_loss_history(self, tiny_data):
        """train()이 1 epoch 학습을 수행하고 epoch별 loss history를 반환하는지 확인한다."""
        try:
            model = NeuralNetwork(use_batchnorm=True, use_dropout=True)
        except TypeError:
            model = NeuralNetwork()
        optimizer = Adam(lr=0.001)
        x_train, y_train = tiny_data
        history = train(model, optimizer, x_train, y_train, epochs=1, batch_size=64)
        assert isinstance(history, list)
        assert len(history) == 1
        assert history[0] >= 0

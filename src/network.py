# -*- coding: utf-8 -*-
"""
MNIST 분류용 신경망 조립 모듈.

개별 layer를 OrderedDict에 쌓아 forward/backward 순서를 명확히 유지합니다.
"""

from collections import OrderedDict

import numpy as np

from activations import ReLU, Softmax
from layers import Affine, BatchNorm, Dropout
from losses import cross_entropy_loss


class NeuralNetwork:
    """
    MNIST 분류용 신경망.
    입력 784 -> 은닉층(들) -> 출력 10 (Softmax).
    은닉층 구성: Affine -> BatchNorm -> ReLU -> Dropout (모두 필수)
    가중치 초기화: He 또는 Xavier 중 선택.
    """

    def __init__(self, use_batchnorm=True, use_dropout=True, dropout_ratio=0.2):
        """
        Args:
            use_batchnorm: 은닉층마다 BatchNorm을 넣을지 여부
            use_dropout: 은닉층마다 Dropout을 넣을지 여부
            dropout_ratio: Dropout에서 끌 뉴런 비율
        """
        # TODO: params dict를 만들고 Affine/BatchNorm/ReLU/Dropout layer를 순서대로 구성하세요.
        # 권장 구조: 784 -> 512 -> 256 -> 10
        # self.layers는 OrderedDict로 만들고, self.grads는 params와 같은 key를 갖게 합니다.
        self.use_batchnorm = use_batchnorm
        self.use_dropout = use_dropout
        self.params = {}
        self.params["W1"] = np.random.randn(784, 512) * np.sqrt(2.0 / 784)
        self.params["b1"] = np.zeros(512)
        self.params["W2"] = np.random.randn(512, 256) * np.sqrt(2.0 / 512)
        self.params["b2"] = np.zeros(256)
        self.params["W3"] = np.random.randn(256, 10) * np.sqrt(2.0 / 256)
        self.params["b3"] = np.zeros(10)

        if self.use_batchnorm:
            self.params["gamma1"] = np.ones(512)
            self.params["beta1"] = np.zeros(512)
            self.params["gamma2"] = np.ones(256)
            self.params["beta2"] = np.zeros(256)

        self.layers = OrderedDict()
        self.layers["Affine1"] = Affine(self.params["W1"], self.params["b1"])
        if self.use_batchnorm:
            self.layers["BatchNorm1"] = BatchNorm(self.params["gamma1"], self.params["beta1"])
        self.layers["ReLU1"] = ReLU()
        if self.use_dropout:
            self.layers["Dropout1"] = Dropout(dropout_ratio)
        self.layers["Affine2"] = Affine(self.params["W2"], self.params["b2"])
        if self.use_batchnorm:
            self.layers["BatchNorm2"] = BatchNorm(self.params["gamma2"], self.params["beta2"])
        self.layers["ReLU2"] = ReLU()
        if self.use_dropout:
            self.layers["Dropout2"] = Dropout(dropout_ratio)
        self.layers["Affine3"] = Affine(self.params["W3"], self.params["b3"])
        self.softmax = Softmax()

        self.grads = {key: np.zeros_like(value) for key, value in self.params.items()}

    def forward(self, x, train=True):
        """
        Args:
            x: (batch_size, 784) 정규화된 MNIST 이미지
            train: BatchNorm/Dropout의 학습 모드 여부

        Returns:
            (batch_size, 10) 각 숫자 클래스의 확률
        """
        # TODO: self.layers를 순서대로 통과시키고 마지막에 Softmax를 적용하세요.
        out = x
        for layer in self.layers.values():
            if isinstance(layer, (BatchNorm, Dropout)):
                out = layer.forward(out, train=train)
            else:
                out = layer.forward(out)
        out = self.softmax.forward(out)
        return out

    def backward(self, dout):
        """
        네트워크 전체 역전파를 수행하고 self.grads를 채웁니다.

        Args:
            dout: Softmax+CrossEntropy를 합친 출력층 gradient
        """
        # TODO: layer를 역순으로 통과시키고 Affine/BatchNorm의 gradient를 self.grads에 모으세요.
        dout = self.softmax.backward(dout)
        layers = list(self.layers.values())
        layers.reverse()
        for layer in layers:
            dout = layer.backward(dout)
        self.grads["W1"] = self.layers["Affine1"].dW
        self.grads["b1"] = self.layers["Affine1"].db
        self.grads["W2"] = self.layers["Affine2"].dW
        self.grads["b2"] = self.layers["Affine2"].db
        self.grads["W3"] = self.layers["Affine3"].dW
        self.grads["b3"] = self.layers["Affine3"].db
        if self.use_batchnorm:
            self.grads["gamma1"] = self.layers["BatchNorm1"].dgamma
            self.grads["beta1"] = self.layers["BatchNorm1"].dbeta
            self.grads["gamma2"] = self.layers["BatchNorm2"].dgamma
            self.grads["beta2"] = self.layers["BatchNorm2"].dbeta
        
    def loss(self, x, y):
        """현재 모델의 예측 확률을 만든 뒤 cross entropy loss를 반환합니다."""
        y_pred = self.forward(x, train=True)
        return cross_entropy_loss(y_pred, y)

    def predict(self, x):
        """추론 모드로 확률을 예측합니다. BatchNorm/Dropout은 train=False로 동작합니다."""
        return self.forward(x, train=False)

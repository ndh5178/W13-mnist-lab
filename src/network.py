# -*- coding: utf-8 -*-
"""Neural network assembly module for MNIST classification."""

from collections import OrderedDict

import numpy as np

from activations import ReLU, Softmax
from layers import Affine, BatchNorm, Dropout
from losses import cross_entropy_loss


class NeuralNetwork:
    """MNIST classifier: 784 -> 512 -> 256 -> 10."""

    def __init__(self, use_batchnorm=True, use_dropout=True, dropout_ratio=0.5):
        self.use_batchnorm = use_batchnorm
        self.use_dropout = use_dropout
        self.dropout_ratio = dropout_ratio

        layer_sizes = [784, 512, 256, 10]
        self.params = {}

        for idx in range(1, len(layer_sizes)):
            input_dim = layer_sizes[idx - 1]
            output_dim = layer_sizes[idx]
            self.params[f"W{idx}"] = np.random.randn(input_dim, output_dim) * np.sqrt(
                2.0 / input_dim
            )
            self.params[f"b{idx}"] = np.zeros(output_dim)

            if self.use_batchnorm and idx < len(layer_sizes) - 1:
                self.params[f"gamma{idx}"] = np.ones(output_dim)
                self.params[f"beta{idx}"] = np.zeros(output_dim)

        self.layers = OrderedDict()
        for idx in range(1, len(layer_sizes)):
            self.layers[f"Affine{idx}"] = Affine(
                self.params[f"W{idx}"], self.params[f"b{idx}"]
            )

            if idx < len(layer_sizes) - 1:
                if self.use_batchnorm:
                    self.layers[f"BatchNorm{idx}"] = BatchNorm(
                        self.params[f"gamma{idx}"], self.params[f"beta{idx}"]
                    )
                self.layers[f"ReLU{idx}"] = ReLU()
                if self.use_dropout:
                    self.layers[f"Dropout{idx}"] = Dropout(self.dropout_ratio)

        self.softmax = Softmax()
        self.grads = {key: np.zeros_like(value) for key, value in self.params.items()}

    def forward(self, x, train=True):
        out = x
        for layer in self.layers.values():
            if isinstance(layer, (BatchNorm, Dropout)):
                out = layer.forward(out, train=train)
            else:
                out = layer.forward(out)
        return self.softmax.forward(out)

    def backward(self, dout):
        dout = self.softmax.backward(dout)

        for layer in reversed(list(self.layers.values())):
            dout = layer.backward(dout)

        for idx in range(1, 4):
            affine = self.layers[f"Affine{idx}"]
            self.grads[f"W{idx}"] = affine.dW
            self.grads[f"b{idx}"] = affine.db

            if self.use_batchnorm and idx < 3:
                batchnorm = self.layers[f"BatchNorm{idx}"]
                self.grads[f"gamma{idx}"] = batchnorm.dgamma
                self.grads[f"beta{idx}"] = batchnorm.dbeta

        return dout

    def loss(self, x, y):
        y_pred = self.forward(x, train=True)
        return cross_entropy_loss(y_pred, y)

    def predict(self, x):
        return self.forward(x, train=False)

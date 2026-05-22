# -*- coding: utf-8 -*-
"""Affine layer step tests."""

import numpy as np

from layers import Affine


class TestAffine:
    """Step 3: run after implementing Affine methods."""

    def test_affine_forward_shape(self):
        """Check Affine.forward computes x @ W + b and correct output shape."""
        W = np.random.randn(4, 3)
        b = np.zeros(3)
        aff = Affine(W, b)
        x = np.random.randn(5, 4)
        out = aff.forward(x)
        assert out.shape == (5, 3)
        np.testing.assert_array_almost_equal(out, x @ W + b)

    def test_affine_backward_grad_shape(self):
        """Check Affine.backward returns dx, dW, db with correct shapes."""
        W = np.random.randn(4, 3)
        b = np.zeros(3)
        aff = Affine(W, b)
        x = np.random.randn(5, 4)
        aff.forward(x)
        dout = np.random.randn(5, 3)
        dx = aff.backward(dout)
        assert dx.shape == x.shape
        assert aff.dW.shape == W.shape
        assert aff.db.shape == b.shape

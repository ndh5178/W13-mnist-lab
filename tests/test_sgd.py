# -*- coding: utf-8 -*-
"""SGD optimizer 단계 테스트."""

import numpy as np

from optimizers import SGD


class TestSGD:
    """Step 5: SGD.update() 구현 후 실행."""

    def test_sgd_update_changes_params(self):
        """SGD.update()가 params를 gradient 반대 방향으로 lr만큼 갱신하는지 확인한다."""
        params = {"W": np.array([[1.0, 2.0], [3.0, 4.0]])}
        grads = {"W": np.ones((2, 2))}
        sgd = SGD(lr=0.1)
        sgd.update(params, grads)
        np.testing.assert_array_almost_equal(
            params["W"], np.array([[0.9, 1.9], [2.9, 3.9]])
        )

# -*- coding: utf-8 -*-
"""Adam optimizer 단계 테스트."""

import numpy as np

from optimizers import Adam


class TestAdam:
    """Step 6: Adam.update() 구현 후 실행."""

    def test_adam_update_changes_params(self):
        """Adam.update()가 내부 이동평균을 사용해 파라미터 값을 실제로 바꾸는지 확인한다."""
        params = {"W": np.ones((2, 2))}
        grads = {"W": np.ones((2, 2)) * 0.5}
        adam = Adam(lr=0.001)
        adam.update(params, grads)
        assert not np.allclose(params["W"], np.ones((2, 2)))

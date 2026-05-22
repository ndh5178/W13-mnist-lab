# -*- coding: utf-8 -*-
"""
MNIST 데이터를 data/mnist.npz에 미리 다운로드하는 스크립트.
프로젝트 루트에서 실행: python download_mnist.py
"""

import sys
from pathlib import Path

# 프로젝트 루트를 path에 추가
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from data import load_mnist

if __name__ == "__main__":
    print("MNIST 데이터를 data/ 폴더에 다운로드합니다...")
    (x_train, y_train), (x_test, y_test) = load_mnist()
    print(f"완료: train {x_train.shape}, test {x_test.shape}")

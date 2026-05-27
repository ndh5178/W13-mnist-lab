# -*- coding: utf-8 -*-
"""MNIST 학습 실행 파일."""

import sys
import os

# src 폴더 기준으로 data/ 경로 설정
sys.path.insert(0, os.path.dirname(__file__))
DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")

from data import load_mnist
from network import NeuralNetwork
from optimizers import Adam
from training import train, evaluate, plot_loss_history


def main():
    # 1. 데이터 로드 (없으면 자동 다운로드)
    print("데이터 로드 중...")
    (x_train, y_train), (x_test, y_test) = load_mnist(data_dir=DATA_DIR)
    print(f"  학습: {x_train.shape}, 테스트: {x_test.shape}")

    # 2. 모델 생성
    model = NeuralNetwork(use_batchnorm=True, use_dropout=True, dropout_ratio=0.5)

    # 3. Optimizer
    optimizer = Adam(lr=0.001)

    # 4. 학습
    print("\n학습 시작...")
    loss_history = train(
        model, optimizer,
        x_train, y_train,
        epochs=20,
        batch_size=128
    )

    # 5. 평가
    train_acc, total_params = evaluate(model, x_train, y_train)
    test_acc, _ = evaluate(model, x_test, y_test)
    print(f"\n학습 정확도: {train_acc:.2f}%")
    print(f"테스트 정확도: {test_acc:.2f}%")
    print(f"총 파라미터 수: {total_params:,}")

    # 6. Loss 곡선 출력
    plot_loss_history(loss_history)


if __name__ == "__main__":
    main()

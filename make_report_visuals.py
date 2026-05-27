# -*- coding: utf-8 -*-
"""Generate confusion matrix and misclassified example SVGs for the report."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from data import load_mnist  # noqa: E402
from network import NeuralNetwork  # noqa: E402
from optimizers import Adam  # noqa: E402
from training import train  # noqa: E402


def escape_xml(value: object) -> str:
    return (
        str(value)
        .replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )


def confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, n_classes=10) -> np.ndarray:
    matrix = np.zeros((n_classes, n_classes), dtype=int)
    for true, pred in zip(y_true, y_pred):
        matrix[int(true), int(pred)] += 1
    return matrix


def write_confusion_matrix_svg(path: Path, matrix: np.ndarray) -> None:
    cell = 42
    left = 92
    top = 70
    width = left + cell * 10 + 34
    height = top + cell * 10 + 88
    max_value = max(1, int(matrix.max()))

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        f'<text x="{width / 2}" y="28" text-anchor="middle" font-size="20" font-weight="700" fill="#111827">Confusion Matrix</text>',
        f'<text x="{width / 2}" y="{height - 16}" text-anchor="middle" font-size="13" fill="#374151">Predicted label</text>',
        f'<text x="22" y="{top + cell * 5}" transform="rotate(-90 22 {top + cell * 5})" text-anchor="middle" font-size="13" fill="#374151">True label</text>',
    ]

    for i in range(10):
        parts.append(
            f'<text x="{left + i * cell + cell / 2}" y="{top - 16}" text-anchor="middle" font-size="13" fill="#374151">{i}</text>'
        )
        parts.append(
            f'<text x="{left - 18}" y="{top + i * cell + cell / 2 + 5}" text-anchor="middle" font-size="13" fill="#374151">{i}</text>'
        )

    for r in range(10):
        for c in range(10):
            value = int(matrix[r, c])
            ratio = value / max_value
            blue = int(245 - ratio * 180)
            green = int(248 - ratio * 130)
            red = int(239 - ratio * 210)
            fill = f"rgb({red},{green},{blue})"
            text_color = "#ffffff" if ratio > 0.55 else "#111827"
            x = left + c * cell
            y = top + r * cell
            parts.append(
                f'<rect x="{x}" y="{y}" width="{cell}" height="{cell}" fill="{fill}" stroke="#e5e7eb"/>'
            )
            if value:
                parts.append(
                    f'<text x="{x + cell / 2}" y="{y + cell / 2 + 5}" text-anchor="middle" font-size="11" fill="{text_color}">{value}</text>'
                )

    parts.append("</svg>")
    path.write_text("\n".join(parts), encoding="utf-8")


def digit_to_cells(image: np.ndarray, x0: int, y0: int, scale: int) -> list[str]:
    cells = []
    image = image.reshape(28, 28)
    for y in range(28):
        for x in range(28):
            value = float(image[y, x])
            if value < 0.05:
                continue
            shade = int(255 - value * 235)
            fill = f"rgb({shade},{shade},{shade})"
            cells.append(
                f'<rect x="{x0 + x * scale}" y="{y0 + y * scale}" width="{scale}" height="{scale}" fill="{fill}"/>'
            )
    return cells


def write_misclassified_svg(
    path: Path,
    x_test: np.ndarray,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    max_items=8,
) -> None:
    wrong = np.where(y_true != y_pred)[0][:max_items]
    scale = 5
    img_size = 28 * scale
    gap = 28
    cols = 4
    rows = max(1, int(np.ceil(len(wrong) / cols)))
    width = cols * img_size + (cols + 1) * gap
    height = rows * (img_size + 54) + 72
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        f'<text x="{width / 2}" y="30" text-anchor="middle" font-size="20" font-weight="700" fill="#111827">Misclassified Examples</text>',
    ]

    if len(wrong) == 0:
        parts.append(
            f'<text x="{width / 2}" y="{height / 2}" text-anchor="middle" font-size="16" fill="#374151">No misclassified examples found.</text>'
        )
    else:
        for pos, idx in enumerate(wrong):
            row = pos // cols
            col = pos % cols
            x0 = gap + col * (img_size + gap)
            y0 = 58 + row * (img_size + 54)
            parts.append(
                f'<rect x="{x0 - 1}" y="{y0 - 1}" width="{img_size + 2}" height="{img_size + 2}" fill="#f9fafb" stroke="#d1d5db"/>'
            )
            parts.extend(digit_to_cells(x_test[idx], x0, y0, scale))
            label = f"true {int(y_true[idx])} / pred {int(y_pred[idx])}"
            parts.append(
                f'<text x="{x0 + img_size / 2}" y="{y0 + img_size + 24}" text-anchor="middle" font-size="13" fill="#111827">{escape_xml(label)}</text>'
            )

    parts.append("</svg>")
    path.write_text("\n".join(parts), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-size", type=int, default=10000)
    parser.add_argument("--test-size", type=int, default=2000)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--seed", type=int, default=123)
    args = parser.parse_args()

    np.random.seed(args.seed)
    (x_train, y_train), (x_test, y_test) = load_mnist(data_dir=str(ROOT / "data"))
    x_train = x_train[: args.train_size]
    y_train = y_train[: args.train_size]
    x_test = x_test[: args.test_size]
    y_test = y_test[: args.test_size]

    model = NeuralNetwork(use_batchnorm=True, use_dropout=True, dropout_ratio=0.5)
    train(model, Adam(lr=0.001), x_train, y_train, epochs=args.epochs, batch_size=args.batch_size)
    probs = model.predict(x_test)
    y_pred = np.argmax(probs, axis=1)
    acc = float(np.mean(y_pred == y_test) * 100)
    matrix = confusion_matrix(y_test, y_pred)

    assets = ROOT / "assets"
    os.makedirs(assets, exist_ok=True)
    write_confusion_matrix_svg(assets / "confusion_matrix.svg", matrix)
    write_misclassified_svg(assets / "misclassified_examples.svg", x_test, y_test, y_pred)
    print(f"accuracy={acc:.2f}%")
    print("wrote assets/confusion_matrix.svg")
    print("wrote assets/misclassified_examples.svg")


if __name__ == "__main__":
    main()

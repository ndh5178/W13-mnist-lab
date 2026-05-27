# -*- coding: utf-8 -*-
"""Run small MNIST experiments and generate report-ready SVG charts.

This script keeps dependencies light: NumPy is enough. It writes:
- assets/report_experiments.csv
- assets/loss_curve.svg
- assets/test_accuracy.svg
"""

from __future__ import annotations

import argparse
import csv
import os
import random
import sys
import time
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
sys.path.insert(0, str(SRC))

from data import load_mnist  # noqa: E402
from network import NeuralNetwork  # noqa: E402
from optimizers import Adam, SGD  # noqa: E402
from training import evaluate, train  # noqa: E402


EXPERIMENTS = [
    {
        "id": "baseline",
        "name": "기본 모델",
        "setting": "BN + Dropout + Adam",
        "use_batchnorm": True,
        "use_dropout": True,
        "optimizer": "adam",
        "lr": 0.001,
        "observation": "가장 안정적",
    },
    {
        "id": "no_batchnorm",
        "name": "BatchNorm 제거",
        "setting": "Dropout + Adam",
        "use_batchnorm": False,
        "use_dropout": True,
        "optimizer": "adam",
        "lr": 0.001,
        "observation": "loss 변동 증가",
    },
    {
        "id": "no_dropout",
        "name": "Dropout 제거",
        "setting": "BN + Adam",
        "use_batchnorm": True,
        "use_dropout": False,
        "optimizer": "adam",
        "lr": 0.001,
        "observation": "과적합 가능성 증가",
    },
    {
        "id": "sgd",
        "name": "SGD 사용",
        "setting": "BN + Dropout + SGD",
        "use_batchnorm": True,
        "use_dropout": True,
        "optimizer": "sgd",
        "lr": 0.05,
        "observation": "수렴 속도 느림",
    },
]


def make_optimizer(name: str, lr: float):
    if name == "adam":
        return Adam(lr=lr)
    if name == "sgd":
        return SGD(lr=lr)
    raise ValueError(f"unknown optimizer: {name}")


def escape_xml(value: object) -> str:
    return (
        str(value)
        .replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )


def write_loss_curve(path: Path, losses: list[float], title: str) -> None:
    width, height = 760, 420
    left, right, top, bottom = 72, 32, 48, 64
    plot_w = width - left - right
    plot_h = height - top - bottom

    y_min = min(losses)
    y_max = max(losses)
    if y_max == y_min:
        y_max += 1.0
        y_min -= 1.0

    points = []
    for idx, loss in enumerate(losses):
        x = left + (idx / max(1, len(losses) - 1)) * plot_w
        y = top + (1 - (loss - y_min) / (y_max - y_min)) * plot_h
        points.append(f"{x:.2f},{y:.2f}")

    circles = []
    labels = []
    for idx, loss in enumerate(losses):
        x_str, y_str = points[idx].split(",")
        circles.append(
            f'<circle cx="{x_str}" cy="{y_str}" r="4" fill="#2563eb" />'
        )
        labels.append(
            f'<text x="{x_str}" y="{float(y_str) - 10:.2f}" '
            f'text-anchor="middle" font-size="11" fill="#374151">{loss:.3f}</text>'
        )

    x_ticks = []
    for idx in range(len(losses)):
        x = left + (idx / max(1, len(losses) - 1)) * plot_w
        x_ticks.append(
            f'<text x="{x:.2f}" y="{height - 30}" text-anchor="middle" '
            f'font-size="12" fill="#4b5563">{idx + 1}</text>'
        )

    svg = f"""<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">
  <rect width="100%" height="100%" fill="#ffffff"/>
  <text x="{width / 2}" y="28" text-anchor="middle" font-size="20" font-weight="700" fill="#111827">{escape_xml(title)}</text>
  <line x1="{left}" y1="{top}" x2="{left}" y2="{height - bottom}" stroke="#9ca3af"/>
  <line x1="{left}" y1="{height - bottom}" x2="{width - right}" y2="{height - bottom}" stroke="#9ca3af"/>
  <text x="20" y="{height / 2}" transform="rotate(-90 20 {height / 2})" text-anchor="middle" font-size="13" fill="#374151">Loss</text>
  <text x="{width / 2}" y="{height - 8}" text-anchor="middle" font-size="13" fill="#374151">Epoch</text>
  <text x="{left - 10}" y="{top + 4}" text-anchor="end" font-size="12" fill="#4b5563">{y_max:.3f}</text>
  <text x="{left - 10}" y="{height - bottom + 4}" text-anchor="end" font-size="12" fill="#4b5563">{y_min:.3f}</text>
  <polyline points="{' '.join(points)}" fill="none" stroke="#2563eb" stroke-width="3" stroke-linejoin="round" stroke-linecap="round"/>
  {''.join(circles)}
  {''.join(labels)}
  {''.join(x_ticks)}
</svg>
"""
    path.write_text(svg, encoding="utf-8")


def write_accuracy_bar(path: Path, rows: list[dict[str, object]]) -> None:
    width, height = 840, 440
    left, right, top, bottom = 80, 40, 52, 96
    plot_w = width - left - right
    plot_h = height - top - bottom
    max_acc = max(100.0, max(float(row["test_acc"]) for row in rows))
    bar_gap = 26
    bar_w = (plot_w - bar_gap * (len(rows) - 1)) / len(rows)

    bars = []
    for idx, row in enumerate(rows):
        acc = float(row["test_acc"])
        x = left + idx * (bar_w + bar_gap)
        h = (acc / max_acc) * plot_h
        y = top + plot_h - h
        label = escape_xml(row["name"])
        bars.append(
            f'<rect x="{x:.2f}" y="{y:.2f}" width="{bar_w:.2f}" height="{h:.2f}" rx="4" fill="#0f766e"/>'
            f'<text x="{x + bar_w / 2:.2f}" y="{y - 8:.2f}" text-anchor="middle" font-size="13" font-weight="700" fill="#111827">{acc:.2f}%</text>'
            f'<text x="{x + bar_w / 2:.2f}" y="{height - 60}" text-anchor="middle" font-size="12" fill="#374151">{label}</text>'
        )

    svg = f"""<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">
  <rect width="100%" height="100%" fill="#ffffff"/>
  <text x="{width / 2}" y="30" text-anchor="middle" font-size="20" font-weight="700" fill="#111827">Test Accuracy by Experiment</text>
  <line x1="{left}" y1="{top}" x2="{left}" y2="{height - bottom}" stroke="#9ca3af"/>
  <line x1="{left}" y1="{height - bottom}" x2="{width - right}" y2="{height - bottom}" stroke="#9ca3af"/>
  <text x="22" y="{height / 2}" transform="rotate(-90 22 {height / 2})" text-anchor="middle" font-size="13" fill="#374151">Accuracy (%)</text>
  <text x="{left - 10}" y="{top + 4}" text-anchor="end" font-size="12" fill="#4b5563">100</text>
  <text x="{left - 10}" y="{height - bottom + 4}" text-anchor="end" font-size="12" fill="#4b5563">0</text>
  {''.join(bars)}
</svg>
"""
    path.write_text(svg, encoding="utf-8")


def run(args: argparse.Namespace) -> list[dict[str, object]]:
    np.random.seed(args.seed)
    random.seed(args.seed)

    (x_train, y_train), (x_test, y_test) = load_mnist(data_dir=str(ROOT / "data"))
    x_train = x_train[: args.train_size]
    y_train = y_train[: args.train_size]
    x_test = x_test[: args.test_size]
    y_test = y_test[: args.test_size]

    experiments = EXPERIMENTS
    if args.only:
        experiments = [
            config
            for config in EXPERIMENTS
            if str(config["id"]) == args.only
            or str(config["name"]) == args.only
            or str(config["optimizer"]) == args.only
        ]
        if not experiments:
            raise ValueError(f"unknown experiment: {args.only}")

    rows = []
    for idx, config in enumerate(experiments, start=1):
        np.random.seed(args.seed + idx)
        random.seed(args.seed + idx)

        print(f"[{idx}/{len(experiments)}] {config['name']} training...", flush=True)
        model = NeuralNetwork(
            use_batchnorm=bool(config["use_batchnorm"]),
            use_dropout=bool(config["use_dropout"]),
            dropout_ratio=args.dropout_ratio,
        )
        optimizer = make_optimizer(str(config["optimizer"]), float(config["lr"]))

        started = time.perf_counter()
        losses = train(
            model,
            optimizer,
            x_train,
            y_train,
            epochs=args.epochs,
            batch_size=args.batch_size,
        )
        elapsed = time.perf_counter() - started
        train_acc, param_count = evaluate(model, x_train, y_train)
        test_acc, _ = evaluate(model, x_test, y_test)

        rows.append(
            {
                "name": config["name"],
                "setting": config["setting"],
                "optimizer": config["optimizer"],
                "lr": config["lr"],
                "train_acc": train_acc,
                "test_acc": test_acc,
                "final_loss": losses[-1],
                "losses": losses,
                "param_count": param_count,
                "seconds": elapsed,
                "observation": config["observation"],
            }
        )
        print(
            f"  loss={losses[-1]:.4f}, train_acc={train_acc:.2f}%, "
            f"test_acc={test_acc:.2f}%, time={elapsed:.1f}s",
            flush=True,
        )

    return rows


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "name",
                "setting",
                "optimizer",
                "lr",
                "train_acc",
                "test_acc",
                "final_loss",
                "param_count",
                "seconds",
                "observation",
                "losses",
            ],
        )
        writer.writeheader()
        for row in rows:
            item = dict(row)
            item["losses"] = "|".join(f"{loss:.6f}" for loss in row["losses"])
            writer.writerow(item)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--train-size", type=int, default=10000)
    parser.add_argument("--test-size", type=int, default=2000)
    parser.add_argument("--dropout-ratio", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--only", default=None, help="Run one experiment id, e.g. baseline.")
    parser.add_argument(
        "--output-prefix",
        default="",
        help="Prefix for generated asset filenames.",
    )
    args = parser.parse_args()

    assets_dir = ROOT / "assets"
    os.makedirs(assets_dir, exist_ok=True)

    rows = run(args)
    prefix = args.output_prefix
    write_csv(assets_dir / f"{prefix}report_experiments.csv", rows)
    write_loss_curve(
        assets_dir / f"{prefix}loss_curve.svg",
        list(rows[0]["losses"]),
        "Baseline Training Loss",
    )
    write_accuracy_bar(assets_dir / f"{prefix}test_accuracy.svg", rows)

    print("\nSummary")
    print("| Experiment | Test Accuracy | Final Loss |")
    print("|---|---:|---:|")
    for row in rows:
        print(f"| {row['name']} | {row['test_acc']:.2f}% | {row['final_loss']:.4f} |")
    print(f"\nWrote outputs to {assets_dir}")


if __name__ == "__main__":
    main()

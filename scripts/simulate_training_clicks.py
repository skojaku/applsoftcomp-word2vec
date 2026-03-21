#!/usr/bin/env python3
"""
Simulate marimo Add (+) / Subtract (−) clicks with the same math as neural_network_training.py.

Usage (from repo root):
    uv run --with numpy --with pillow scripts/simulate_training_clicks.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image


def load_bundle(repo_root: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, int]:
    base = repo_root / "data" / "afhq_catdog"
    man_path = base / "manifest.json"
    if not man_path.is_file():
        print(f"Missing {man_path}; run: uv run scripts/prepare_afhq_subset.py", file=sys.stderr)
        sys.exit(1)
    man = json.loads(man_path.read_text(encoding="utf-8"))
    grid = int(man["grid_size"])

    def load_split(entries: list) -> tuple[np.ndarray, np.ndarray]:
        xs, ys = [], []
        for e in entries:
            p = base / str(e["file"])
            arr = np.asarray(Image.open(p).convert("L"), dtype=np.float32) / 255.0
            xs.append(arr.reshape(-1))
            ys.append(int(e["label"]))
        return np.stack(xs), np.array(ys, dtype=np.int64)

    tx, ty = load_split(man["train"])
    ex, ey = load_split(man["eval"])
    return tx, ty, ex, ey, grid


def eval_acc(w: np.ndarray, b: float, ex: np.ndarray, ey: np.ndarray) -> float:
    scores = ex @ w.astype(np.float64) + b
    preds = (scores > 0).astype(np.int64)
    return float(np.mean(preds == ey))


def step_add(w: np.ndarray, b: float, x: np.ndarray, true: int, eta_big: float, eta_sm: float) -> tuple[np.ndarray, float]:
    y = 1.0 if true == 1 else -1.0
    s = float(np.dot(w, x) + b)
    pred = 1 if s > 0 else 0
    eta = eta_big if pred != true else eta_sm
    w = w.astype(np.float32, copy=True)
    w += eta * y * x
    b = b + eta * y
    return w, b


def step_subtract(w: np.ndarray, b: float, x: np.ndarray, true: int, eta_big: float, eta_sm: float) -> tuple[np.ndarray, float]:
    y = 1.0 if true == 1 else -1.0
    s = float(np.dot(w, x) + b)
    pred = 1 if s > 0 else 0
    eta = eta_big if pred != true else eta_sm
    w = w.astype(np.float32, copy=True)
    w -= eta * y * x
    b = b - eta * y
    return w, b


def main() -> None:
    repo = Path(__file__).resolve().parent.parent
    tx, ty, ex, ey, grid = load_bundle(repo)
    n_train = len(tx)
    flat = grid * grid
    eta_big = 18.0 / flat
    eta_sm = 3.5 / flat

    w = np.zeros(flat, dtype=np.float32)
    b = 0.0
    baseline = eval_acc(w, b, ex, ey)
    print(f"Train: {n_train}  Eval: {len(ex)}  Grid: {grid}×{grid}")
    print(f"Baseline eval accuracy (w=0, b=0): {baseline:.1%}\n")

    # --- Add-only: cycle 0,1,...,n_train-1 like the notebook ---
    def run_add_only(max_clicks: int) -> list[tuple[int, float]]:
        w2 = np.zeros(flat, dtype=np.float32)
        b2 = 0.0
        hist: list[tuple[int, float]] = []
        idx = 0
        for click in range(1, max_clicks + 1):
            x = tx[idx].astype(np.float32, copy=False)
            true = int(ty[idx])
            w2, b2 = step_add(w2, b2, x, true, eta_big, eta_sm)
            hist.append((click, eval_acc(w2, b2, ex, ey)))
            idx = (idx + 1) % n_train
        return hist

    max_c = 5000
    hist = run_add_only(max_c)
    accs = [a for _, a in hist]

    # First strict increase over baseline
    first_up = next((c for c, a in hist if a > baseline + 1e-9), None)
    print("**Add (+) only** (same order as notebook: advance image each click)")
    if first_up is None:
        print(f"  No increase above baseline in {max_c} clicks.")
    else:
        print(f"  First eval accuracy strictly above baseline: click #{first_up} → {hist[first_up - 1][1]:.1%}")

    # First time we hit +5% absolute vs baseline (if ever)
    target5 = baseline + 0.05
    hit5 = next((c for c, a in hist if a >= target5), None)
    if hit5 is not None:
        print(f"  First time ≥ baseline + 5% ({target5:.1%}): click #{hit5} → {hist[hit5 - 1][1]:.1%}")

    for goal in (0.60, 0.65, 0.70, 0.75, 0.80):
        hit = next((c for c, a in hist if a >= goal), None)
        if hit is not None:
            print(f"  First time ≥ {goal:.0%}: click #{hit} → {hist[hit - 1][1]:.1%}")

    best_c, best_a = max(hist, key=lambda t: t[1])
    print(f"  Best in first {max_c} clicks: {best_a:.1%} at click #{best_c}")

    # Snapshots every full epoch
    print("\n  End-of-epoch eval accuracy (Add only):")
    for ep in range(1, min(21, max_c // n_train + 1)):
        c = ep * n_train
        if c <= len(hist):
            print(f"    epoch {ep:2d} (click {c:4d}): {hist[c - 1][1]:.1%}")

    # --- Subtract-only from zeros (usually hurts) ---
    w3 = np.zeros(flat, dtype=np.float32)
    b3 = 0.0
    idx = 0
    print("\n**Subtract (−) only** from w=0, b=0 (first 5 clicks eval acc):")
    for click in range(1, 6):
        x = tx[idx].astype(np.float32, copy=False)
        true = int(ty[idx])
        w3, b3 = step_subtract(w3, b3, x, true, eta_big, eta_sm)
        print(f"  click {click}: {eval_acc(w3, b3, ex, ey):.1%}")
        idx = (idx + 1) % n_train

    # --- Add then Subtract pairs (net zero per pair if same image — notebook advances, so not exact) ---
    print("\n**Alternating Add / Subtract** (same index cycle; odd clicks Add, even Subtract):")
    w4 = np.zeros(flat, dtype=np.float32)
    b4 = 0.0
    idx = 0
    for click in range(1, min(201, max_c + 1)):
        x = tx[idx].astype(np.float32, copy=False)
        true = int(ty[idx])
        if click % 2 == 1:
            w4, b4 = step_add(w4, b4, x, true, eta_big, eta_sm)
        else:
            w4, b4 = step_subtract(w4, b4, x, true, eta_big, eta_sm)
        idx = (idx + 1) % n_train
        if click in (10, 50, 100, 180, 200):
            print(f"  click {click}: {eval_acc(w4, b4, ex, ey):.1%}")


if __name__ == "__main__":
    main()

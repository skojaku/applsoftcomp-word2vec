# /// script
# requires-python = ">=3.10"
# dependencies = [
#   "datasets>=2.16.0",
#   "numpy>=1.24",
#   "Pillow>=10.0",
# ]
# ///
"""
One-shot download: AFHQ (cat + dog only) → resized grayscale PNGs + manifest.json.

Run from the project root (parent of `scripts/`):

    uv run scripts/prepare_afhq_subset.py

Or:  python scripts/prepare_afhq_subset.py   (with datasets + Pillow + NumPy installed)

First run downloads the Hugging Face dataset into the HF cache (~729 MB), then writes only
220 small PNGs under data/afhq_catdog/ for the Marimo notebook to read.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from datasets import load_dataset
from PIL import Image


def _rgb_to_gray_grid(pil_img: Image.Image, grid: int) -> np.ndarray:
    rgb = pil_img.convert("RGB").resize((grid, grid), Image.Resampling.BILINEAR)
    arr = np.asarray(rgb, dtype=np.float32) / 255.0
    return (0.299 * arr[..., 0] + 0.587 * arr[..., 1] + 0.114 * arr[..., 2]).astype(
        np.float32
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Export AFHQ cat/dog subset for neural_network_training.py")
    parser.add_argument("--grid", type=int, default=64, help="Square side length after resize")
    parser.add_argument("--n-train", type=int, default=180)
    parser.add_argument("--n-eval", type=int, default=40)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Output directory (default: <repo>/data/afhq_catdog)",
    )
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parent.parent
    out_dir = args.out if args.out is not None else repo_root / "data" / "afhq_catdog"
    img_train = out_dir / "images" / "train"
    img_eval = out_dir / "images" / "eval"
    img_train.mkdir(parents=True, exist_ok=True)
    img_eval.mkdir(parents=True, exist_ok=True)

    need = args.n_train + args.n_eval
    print("Loading huggan/AFHQ from Hugging Face (may download on first run)...", file=sys.stderr)
    ds = load_dataset("huggan/AFHQ", split="train")
    cd = ds.filter(lambda ex: ex["label"] < 2)
    cd = cd.shuffle(seed=args.seed)
    if len(cd) < need:
        raise SystemExit(f"Need at least {need} cat+dog images, got {len(cd)}")
    subset = cd.select(range(need))

    train_entries: list[dict[str, object]] = []
    eval_entries: list[dict[str, object]] = []

    for i in range(need):
        ex = subset[i]
        gray = _rgb_to_gray_grid(ex["image"], args.grid)
        label = int(ex["label"])
        u8 = (np.clip(gray, 0.0, 1.0) * 255.0).astype(np.uint8)

        if i < args.n_train:
            rel = f"images/train/{i:04d}.png"
            path = img_train / f"{i:04d}.png"
            split_name = "train"
            train_entries.append({"file": rel, "label": label})
        else:
            j = i - args.n_train
            rel = f"images/eval/{j:04d}.png"
            path = img_eval / f"{j:04d}.png"
            split_name = "eval"
            eval_entries.append({"file": rel, "label": label})

        Image.fromarray(u8, mode="L").save(path, optimize=True)
        if (i + 1) % 50 == 0 or i + 1 == need:
            print(f"  wrote {i + 1}/{need} ({split_name}) …", file=sys.stderr)

    manifest = {
        "version": 1,
        "source": "huggan/AFHQ (cat and dog classes only; wild excluded)",
        "grid_size": args.grid,
        "class_names": ["Cat", "Dog"],
        "seed": args.seed,
        "train": train_entries,
        "eval": eval_entries,
    }
    man_path = out_dir / "manifest.json"
    man_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    print(f"Done. Manifest: {man_path}", file=sys.stderr)
    print(f"Images: {out_dir / 'images'}", file=sys.stderr)


if __name__ == "__main__":
    main()

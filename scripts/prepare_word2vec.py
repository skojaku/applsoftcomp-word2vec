# /// script
# requires-python = ">=3.10"
# dependencies = [
#   "gensim>=4.3.0",
#   "numpy>=1.24",
#   "scikit-learn>=1.3",
# ]
# ///
"""
Download word2vec-google-news-300, keep as many frequent words as possible,
PCA-compress to retain a target variance, and save as a compact .npz for the
teaching notebook.

Run from the project root:

    uv run scripts/prepare_word2vec.py

Options:
    --min-variance  Minimum variance to retain (default: 0.8 = 80%)
    --max-size      Maximum output file size in MB (default: 150)
    --out           Output directory (default: data/word2vec)

First run downloads the model via gensim (~1.7 GB compressed, ~3.6 GB on disk).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Prepare a compact word2vec subset for teaching"
    )
    parser.add_argument(
        "--min-variance",
        type=float,
        default=0.80,
        help="Minimum variance to retain via PCA (default: 0.80)",
    )
    parser.add_argument(
        "--max-size",
        type=float,
        default=150.0,
        help="Maximum output size in MB (default: 150)",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Output directory (default: <repo>/data/word2vec)",
    )
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parent.parent
    out_dir = args.out if args.out is not None else repo_root / "data" / "word2vec"
    out_dir.mkdir(parents=True, exist_ok=True)

    # --- 1. Load the full model ---
    import gensim.downloader as api

    print("Loading word2vec-google-news-300 (may download on first run)...", file=sys.stderr)
    model = api.load("word2vec-google-news-300")

    # --- 2. Determine PCA dimensions for target variance ---
    from sklearn.decomposition import PCA

    # Fit full PCA on a sample to find the right number of components
    sample_n = min(100_000, len(model.key_to_index))
    sample_vectors = model.vectors[:sample_n]

    print(f"Fitting PCA on top {sample_n:,} words to find target dimensions...", file=sys.stderr)
    pca_full = PCA(n_components=300, random_state=42)
    pca_full.fit(sample_vectors)
    cumvar = np.cumsum(pca_full.explained_variance_ratio_)
    target_dim = int(np.searchsorted(cumvar, args.min_variance) + 1)
    print(f"Need {target_dim} dims for {args.min_variance:.0%} variance", file=sys.stderr)

    # --- 3. Maximize vocabulary within size budget ---
    # Estimate: uncompressed size ≈ n_words * n_dims * 4 + n_words * avg_word_len
    # npz compression typically gives ~0.85x ratio for float32 embeddings
    budget_bytes = args.max_size * 1024 * 1024
    compression_ratio = 0.98  # npz compression is minimal for float32 embeddings
    avg_word_bytes = 10
    bytes_per_float = 2  # float16
    max_words = int(budget_bytes / (compression_ratio * (target_dim * bytes_per_float + avg_word_bytes)))
    max_words = min(max_words, len(model.key_to_index))
    print(f"Estimated max words within {args.max_size:.0f} MB: ~{max_words:,}", file=sys.stderr)

    # --- 4. Extract top-N words ---
    words = model.index_to_key[:max_words]
    vectors = model.vectors[:max_words]
    print(f"Kept {max_words:,} words out of {len(model.key_to_index):,}", file=sys.stderr)

    # --- 5. PCA compress ---
    print(f"Running PCA: {vectors.shape[1]}d → {target_dim}d ...", file=sys.stderr)
    pca = PCA(n_components=target_dim, random_state=42)
    vectors_reduced = pca.fit_transform(vectors).astype(np.float16)

    variance_kept = pca.explained_variance_ratio_.sum()
    print(f"Variance retained: {variance_kept:.1%}", file=sys.stderr)

    # --- 6. Save ---
    out_path = out_dir / "word2vec-compact.npz"
    np.savez_compressed(
        out_path,
        words=np.array(words, dtype=str),
        vectors=vectors_reduced,
    )

    size_mb = out_path.stat().st_size / (1024 * 1024)
    print(f"Saved: {out_path} ({size_mb:.1f} MB)", file=sys.stderr)
    print(f"  {len(words):,} words × {target_dim} dimensions", file=sys.stderr)

    # If we overshot the budget, warn
    if size_mb > args.max_size:
        print(f"WARNING: Output ({size_mb:.1f} MB) exceeds budget ({args.max_size:.0f} MB).", file=sys.stderr)
        print("Re-run with fewer words or lower --min-variance.", file=sys.stderr)


if __name__ == "__main__":
    main()

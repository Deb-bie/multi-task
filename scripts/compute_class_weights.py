"""
scripts/compute_class_weights.py — Compute inverse-frequency class weights
for the segmentation loss from the training split.

Why this matters
----------------
Organ sizes in SynthRAD2025 span two orders of magnitude.  For abdomen the
pancreas occupies ~0.5 % of foreground voxels, while the liver occupies ~40 %.
Without per-class weighting, the CE gradient from the pancreas is ~80× weaker
than from the liver, causing the network to largely ignore small organs.

Strategy: median-frequency balancing (Eigen & Fergus, 2015)
------------------------------------------------------------
For each class c:
    freq_c   = total voxels of class c / total foreground voxels
    weight_c = median(freq_1 … freq_C) / freq_c

Background (class 0) always receives weight 0.05 regardless of its frequency,
because background gradients would otherwise overwhelm the foreground signal.

The weights are printed as a JSON list suitable for copy-paste into an
anatomy config file under the ``CLASS_WEIGHTS`` key.

Usage
-----
    python scripts/compute_class_weights.py \\
        --split_dir   splits/ \\
        --anatomy     abdomen \\
        --num_classes 6

    # Using environment variables:
    ANATOMY=thorax python scripts/compute_class_weights.py
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional


# ---------------------------------------------------------------------------
# NIfTI loader (numpy only, no full dataset stack required)
# ---------------------------------------------------------------------------

def _load_seg_np(seg_path: str):
    """Load a segmentation NIfTI as a flat integer NumPy array.

    Args:
        seg_path: Absolute path to the ``.nii.gz`` segmentation file.

    Returns:
        1-D int64 NumPy array of all voxel labels.
    """
    try:
        import SimpleITK as sitk
    except ImportError as exc:
        raise ImportError(
            "SimpleITK is required.  Install with: pip install SimpleITK==2.3.1"
        ) from exc

    import numpy as np
    img = sitk.ReadImage(seg_path)
    arr = sitk.GetArrayFromImage(img)          # (Z, Y, X)
    return arr.astype("int64").ravel()


# ---------------------------------------------------------------------------
# Weight computation
# ---------------------------------------------------------------------------

def compute_weights(
    split_dir:   str,
    anatomy:     str,
    num_classes: int,
    bg_weight:   float = 0.05,
) -> List[float]:
    """Compute median-frequency balancing weights from the training split.

    Args:
        split_dir:   Directory containing ``{anatomy}_split.json``.
        anatomy:     Anatomical region name.
        num_classes: Total number of classes including background (class 0).
        bg_weight:   Fixed weight for background class (default 0.05).

    Returns:
        List of ``num_classes`` floats — one per class — suitable for
        ``CLASS_WEIGHTS`` in the anatomy config.

    Raises:
        FileNotFoundError: If the split JSON is missing.
        ValueError:        If no training entries are found.
    """
    import numpy as np

    json_path = Path(split_dir) / f"{anatomy}_split.json"
    if not json_path.exists():
        raise FileNotFoundError(
            f"Split JSON not found: {json_path}\n"
            f"Run scripts/prepare_splits.py first."
        )

    with open(json_path) as fh:
        data = json.load(fh)

    train_entries = data.get("train", [])
    if not train_entries:
        raise ValueError(f"No training entries found in {json_path}")

    print(f"Computing class weights from {len(train_entries)} training patients …")

    # Accumulate voxel counts per class
    counts = np.zeros(num_classes, dtype=np.int64)
    for entry in train_entries:
        seg_path = entry.get("seg_path", "")
        if not seg_path or not Path(seg_path).exists():
            print(f"  [WARNING] Seg file not found, skipping: {seg_path}")
            continue

        labels = _load_seg_np(seg_path)
        for c in range(num_classes):
            counts[c] += int((labels == c).sum())

    total_fg = counts[1:].sum()       # exclude background from frequency base
    print(f"\nVoxel counts per class (background={counts[0]:,}):")
    for c in range(1, num_classes):
        pct = 100.0 * counts[c] / max(total_fg, 1)
        print(f"  class {c}: {counts[c]:>12,}  ({pct:.2f}%)")

    # Median-frequency balancing
    fg_freqs = counts[1:].astype(np.float64) / max(total_fg, 1)  # (num_classes-1,)
    median_freq = float(np.median(fg_freqs[fg_freqs > 0]))

    weights: List[float] = [bg_weight]          # class 0 = background
    for c in range(1, num_classes):
        freq_c = fg_freqs[c - 1]
        if freq_c > 0:
            w = round(float(median_freq / freq_c), 4)
        else:
            w = round(float(median_freq / 1e-6), 4)   # absent class → large weight
        weights.append(w)

    return weights


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Compute inverse-frequency class weights from training segmentation labels."
        )
    )
    p.add_argument(
        "--split_dir",
        default=os.environ.get("SPLIT_DIR", "splits/"),
        help="Directory containing *_split.json files (default: splits/).",
    )
    p.add_argument(
        "--anatomy",
        default=os.environ.get("ANATOMY"),
        choices=["head_neck", "thorax", "abdomen"],
        help="Anatomical region.",
    )
    p.add_argument(
        "--num_classes",
        type=int,
        default=6,
        help="Total number of classes including background (default: 6).",
    )
    p.add_argument(
        "--bg_weight",
        type=float,
        default=0.05,
        help=(
            "Fixed weight for background class 0 (default: 0.05).  "
            "Background is excluded from median-frequency balancing."
        ),
    )
    p.add_argument(
        "--update_config",
        default=None,
        metavar="CONFIG_PATH",
        help=(
            "If provided, write CLASS_WEIGHTS directly into this JSON config file "
            "(e.g. configs/anatomy/abdomen.json).  The file is updated in-place."
        ),
    )
    args = p.parse_args()
    if args.anatomy is None:
        print("[ERROR] --anatomy is required (or set ANATOMY env var).", file=sys.stderr)
        p.print_usage(sys.stderr)
        sys.exit(2)
    return args


def main() -> None:
    args = parse_args()

    weights = compute_weights(
        split_dir   = args.split_dir,
        anatomy     = args.anatomy,
        num_classes = args.num_classes,
        bg_weight   = args.bg_weight,
    )

    print(f"\nMedian-frequency CLASS_WEIGHTS for {args.anatomy}:")
    print(f"  {weights}")
    print(f'\nCopy into configs/anatomy/{args.anatomy}.json:')
    print(f'  "CLASS_WEIGHTS": {json.dumps(weights)}')

    if args.update_config:
        cfg_path = Path(args.update_config)
        if not cfg_path.exists():
            print(f"[ERROR] Config file not found: {cfg_path}", file=sys.stderr)
            sys.exit(1)
        with open(cfg_path) as fh:
            cfg = json.load(fh)
        cfg["CLASS_WEIGHTS"] = weights
        with open(cfg_path, "w") as fh:
            json.dump(cfg, fh, indent=2)
        print(f"\nWrote CLASS_WEIGHTS to: {cfg_path}")


if __name__ == "__main__":
    main()

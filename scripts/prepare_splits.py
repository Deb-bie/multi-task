"""
scripts/prepare_splits.py — Generate stratified train/val/test split JSON files.

Scans a SynthRAD2025 anatomy directory for patient sub-folders, verifies that
the four required files (``mr.nii.gz``, ``ct.nii.gz``, ``seg.nii.gz``,
``mask.nii.gz``) all exist, then randomly partitions the patients into:

    70 % train  |  14 % val  |  16 % test

matching the proportions reported in the SynthRAD2025 paper.

Output
------
``{SPLIT_DIR}/{ANATOMY}_split.json`` with the structure::

    {
        "train": [
            {
                "patient_id": "...",
                "mr_path":    "/absolute/path/to/mr.nii.gz",
                "ct_path":    "/absolute/path/to/ct.nii.gz",
                "seg_path":   "/absolute/path/to/seg.nii.gz",
                "mask_path":  "/absolute/path/to/mask.nii.gz"
            },
            ...
        ],
        "val": [...],
        "test": [...]
    }

Usage
-----
    python scripts/prepare_splits.py \\
        --data_root /data/Task1 \\
        --anatomy   head_neck \\
        --split_dir splits/ \\
        --seed      42

    # When the on-disk directory name differs from the anatomy label
    # (e.g. SynthRAD2025 uses HN / TH / AB), pass --data_dir explicitly:
    python scripts/prepare_splits.py \\
        --data_dir  /data/Task1/HN \\
        --anatomy   head_neck \\
        --split_dir splits/ \\
        --seed      42

    # Or via environment variables:
    DATA_ROOT=/data/Task1 ANATOMY=thorax python scripts/prepare_splits.py

All of DATA_ROOT, ANATOMY, SPLIT_DIR, and SEED may also be passed as
environment variables (command-line arguments take precedence).
--data_dir takes precedence over --data_root when both are supplied.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


# ---------------------------------------------------------------------------
# Expected file names within each patient directory
# ---------------------------------------------------------------------------
# SynthRAD2025 stores MR, CT, and body-mask in MetaImage (.mha) format.
# Segmentation labels (seg.nii.gz) are produced separately by TotalSegmentator
# + merge_seg_labels.py and are treated as optional here — patients without a
# seg file are still included in the split (seg_path will be null) so that
# synthesis training can proceed while segmentation labels are being generated.
_REQUIRED_FILES: List[str] = [
    "mr.mha",
    "ct.mha",
    "mask.mha",
]

# Searched in order; first match wins.  Set to [] to skip seg entirely.
_SEG_CANDIDATES: List[str] = ["seg.nii.gz", "seg.mha"]

# Split ratios (must sum to 1.0)
_TRAIN_RATIO: float = 0.70
_VAL_RATIO:   float = 0.14
# test = 1 - train - val  ≈  0.16


# ---------------------------------------------------------------------------
# Patient discovery
# ---------------------------------------------------------------------------

def _discover_patients(
    anat_dir: Path,
) -> Tuple[List[Dict[str, Any]], List[str]]:
    """Scan the anatomy directory and collect valid patient entries.

    A patient directory is valid when **all four** required NIfTI files exist.
    Directories where any file is missing are excluded and logged.

    Args:
        anat_dir: Direct path to the anatomy directory (e.g. /data/Task1/HN).

    Returns:
        Tuple ``(valid_entries, missing_entries)`` where each entry in
        *valid_entries* is a dict with keys
        ``patient_id``, ``mr_path``, ``ct_path``, ``seg_path``, ``mask_path``;
        and *missing_entries* is a list of patient IDs where at least one file
        was absent.
    """
    if not anat_dir.is_dir():
        raise FileNotFoundError(f"Anatomy directory not found: {anat_dir}")

    patient_dirs = sorted(
        p for p in anat_dir.iterdir() if p.is_dir()
    )

    valid: List[Dict[str, Any]] = []
    missing: List[str] = []
    no_seg: List[str] = []

    for pdir in patient_dirs:
        pid = pdir.name

        # Check required files (.mha)
        absent = [f for f in _REQUIRED_FILES if not (pdir / f).exists()]
        if absent:
            missing.append(f"{pid}: missing {absent}")
            continue

        # Find seg file (optional — first candidate that exists wins)
        seg_path: Optional[str] = None
        for candidate in _SEG_CANDIDATES:
            if (pdir / candidate).exists():
                seg_path = str(pdir / candidate)
                break
        if seg_path is None:
            no_seg.append(pid)

        valid.append({
            "patient_id": pid,
            "mr_path":    str(pdir / "mr.mha"),
            "ct_path":    str(pdir / "ct.mha"),
            "mask_path":  str(pdir / "mask.mha"),
            "seg_path":   seg_path,   # None if not yet generated
        })

    if no_seg:
        print(
            f"\n[INFO] {len(no_seg)} patient(s) have no seg file yet "
            f"(seg_path=null in split JSON). "
            f"Run scripts/run_totalsegmentator.sh then "
            f"scripts/merge_seg_labels.py to generate them.\n"
            f"  First few: {no_seg[:5]}"
        )

    return valid, missing


# ---------------------------------------------------------------------------
# Stratified split
# ---------------------------------------------------------------------------

def _split_patients(
    patients: List[Dict[str, Any]],
    train_ratio: float,
    val_ratio:   float,
    seed:        int,
) -> Dict[str, List[Dict[str, Any]]]:
    """Randomly partition *patients* into train / val / test subsets.

    The split is reproducible for a fixed *seed*.  Rounding is applied such
    that ``len(train) + len(val) + len(test) == len(patients)`` exactly (any
    rounding remainder goes into the test set).

    Args:
        patients:    List of patient entry dicts (already verified as valid).
        train_ratio: Fraction for training (e.g. ``0.70``).
        val_ratio:   Fraction for validation (e.g. ``0.14``).
        seed:        Random seed for reproducibility.

    Returns:
        Dict with keys ``"train"``, ``"val"``, ``"test"``.
    """
    rng = random.Random(seed)
    shuffled = patients.copy()
    rng.shuffle(shuffled)

    n       = len(shuffled)
    n_train = int(round(n * train_ratio))
    n_val   = int(round(n * val_ratio))
    # test gets everything that remains (handles rounding)
    n_test  = n - n_train - n_val

    if n_test < 0:
        # Edge case: very small dataset; bump val down by one
        n_val  = max(0, n_val + n_test)
        n_test = n - n_train - n_val

    return {
        "train": shuffled[:n_train],
        "val":   shuffled[n_train : n_train + n_val],
        "test":  shuffled[n_train + n_val :],
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def prepare_splits(
    anatomy:    str,
    split_dir:  str,
    seed:       int,
    data_root:  Optional[str] = None,
    data_dir:   Optional[str] = None,
) -> None:
    """Discover patients, split, verify paths, and write the JSON file.

    Either *data_root* or *data_dir* must be supplied:

    * **data_root** – root of the dataset; anatomy sub-directory is appended
      automatically (``data_root / anatomy``).
    * **data_dir** – direct path to the anatomy directory.  Takes precedence
      over *data_root* when both are provided.  Use this when the on-disk
      directory name differs from the anatomy label (e.g. SynthRAD2025 stores
      head-and-neck data under ``HN/`` rather than ``head_neck/``).

    Args:
        anatomy:   Anatomical region label used for naming the output JSON
                   (``"head_neck"``, ``"thorax"``, ``"abdomen"``).
        split_dir: Directory where the split JSON will be written.
        seed:      Random seed for the shuffle.
        data_root: Root directory of the SynthRAD2025 dataset (optional if
                   *data_dir* is given).
        data_dir:  Direct path to the anatomy directory (takes precedence over
                   *data_root*).
    """
    split_dir_p = Path(split_dir)
    split_dir_p.mkdir(parents=True, exist_ok=True)

    # Resolve the anatomy directory
    if data_dir is not None:
        anat_dir = Path(data_dir).resolve()
    elif data_root is not None:
        anat_dir = Path(data_root).resolve() / anatomy
    else:
        raise ValueError("Either --data_root or --data_dir must be supplied.")

    print(f"Scanning: {anat_dir}")

    valid_patients, missing = _discover_patients(anat_dir)

    if missing:
        print(f"\n[WARNING] {len(missing)} patient(s) excluded (missing files):")
        for m in missing:
            print(f"  {m}")

    if not valid_patients:
        print(f"[ERROR] No valid patients found for anatomy={anatomy}.")
        sys.exit(1)

    splits = _split_patients(valid_patients, _TRAIN_RATIO, _VAL_RATIO, seed)

    n_train = len(splits["train"])
    n_val   = len(splits["val"])
    n_test  = len(splits["test"])
    n_total = n_train + n_val + n_test

    print(f"\nSplit sizes for {anatomy} (seed={seed}):")
    print(f"  Total valid : {n_total}")
    print(f"  Train       : {n_train}  ({100 * n_train / n_total:.1f}%)")
    print(f"  Val         : {n_val}  ({100 * n_val   / n_total:.1f}%)")
    print(f"  Test        : {n_test}  ({100 * n_test  / n_total:.1f}%)")

    out_path = split_dir_p / f"{anatomy}_split.json"
    with open(out_path, "w") as fh:
        json.dump(splits, fh, indent=2)
    print(f"\nSplit JSON written to: {out_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    """Parse command-line arguments, falling back to environment variables.

    Returns:
        Parsed argument namespace.
    """
    p = argparse.ArgumentParser(
        description=(
            "Generate stratified train/val/test split JSON files for SynthRAD2025."
        )
    )
    p.add_argument(
        "--data_root",
        default=os.environ.get("DATA_ROOT"),
        help="Root directory of the SynthRAD2025 dataset "
             "(or set DATA_ROOT env var). "
             "The anatomy sub-directory is appended automatically. "
             "Ignored when --data_dir is supplied.",
    )
    p.add_argument(
        "--data_dir",
        default=os.environ.get("DATA_DIR"),
        help="Direct path to the anatomy directory "
             "(e.g. /data/Task1/HN for head_neck). "
             "Takes precedence over --data_root when both are supplied.",
    )
    p.add_argument(
        "--anatomy",
        default=os.environ.get("ANATOMY"),
        choices=["head_neck", "thorax", "abdomen"],
        help="Anatomical region label used for output JSON naming "
             "(or set ANATOMY env var).",
    )
    p.add_argument(
        "--split_dir",
        default=os.environ.get("SPLIT_DIR", "splits/"),
        help="Output directory for split JSON files (default: splits/).",
    )
    p.add_argument(
        "--seed",
        type=int,
        default=int(os.environ.get("SEED", "42")),
        help="Random seed (default: 42).",
    )
    args = p.parse_args()

    # Validate required arguments
    errors = []
    if args.data_root is None and args.data_dir is None:
        errors.append(
            "Either --data_root or --data_dir is required "
            "(or set DATA_ROOT / DATA_DIR env var)."
        )
    if args.anatomy is None:
        errors.append("--anatomy is required (or set ANATOMY env var).")
    if errors:
        for e in errors:
            print(f"[ERROR] {e}", file=sys.stderr)
        p.print_usage(sys.stderr)
        sys.exit(2)

    return args


def main() -> None:
    """Entry point for the split generator."""
    args = parse_args()
    prepare_splits(
        anatomy   = args.anatomy,
        split_dir = args.split_dir,
        seed      = args.seed,
        data_root = args.data_root,
        data_dir  = args.data_dir,
    )


if __name__ == "__main__":
    main()

"""
scripts/merge_seg_labels.py — Merge per-organ TotalSegmentator masks into a
single integer-labelled segmentation volume and write it into each patient's
data directory as ``seg.nii.gz``.

Background
----------
``run_totalsegmentator.sh`` saves one ``.nii.gz`` per organ under::

    OUTPUT_DIR/{ANATOMY}/{patient_id}/seg/{class_name}.nii.gz

This script reads those individual binary masks, assigns each the integer label
defined in the anatomy's class mapping, merges them into a single volume, and
copies the result to::

    DATA_ROOT/{anatomy_dir}/{patient_id}/seg.nii.gz

so that ``prepare_splits.py`` can find it alongside ``mr.mha`` / ``ct.mha``.

Class mappings match ``configs/{anatomy}.json → CLASS_WEIGHTS`` ordering
(index 0 = background, index 1..N = foreground organs in the order listed).

Anatomy → class name → integer label:
  head_neck : brainstem=1 parotid_L=2 parotid_R=3 mandible=4 spinal_cord=5
  thorax    : lung_L=1   lung_R=2    heart=3      spinal_cord=4 esophagus=5
  abdomen   : liver=1    kidney_L=2  kidney_R=3   spleen=4      pancreas=5

Usage
-----
    python scripts/merge_seg_labels.py \\
        --anatomy    head_neck \\
        --data_dir   /data/Task1/Task1/HN \\
        --seg_dir    /data/segs/head_neck \\
        --overwrite               # optional: re-merge even if seg.nii.gz exists

    # Batch all three anatomies:
    for ANAT in head_neck thorax abdomen; do
        python scripts/merge_seg_labels.py \\
            --anatomy  ${ANAT} \\
            --data_dir /data/Task1/Task1/${DIR} \\
            --seg_dir  /data/segs/${ANAT}
    done
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np

try:
    import SimpleITK as sitk
except ImportError as exc:
    raise ImportError(
        "SimpleITK is required.  Install with: pip install SimpleITK==2.3.1"
    ) from exc


# ---------------------------------------------------------------------------
# Per-anatomy label maps: class_name → integer label
# Keys must match the file stems produced by run_totalsegmentator.sh
# ---------------------------------------------------------------------------
_LABEL_MAPS: Dict[str, Dict[str, int]] = {
    "head_neck": {
        "brainstem":    1,
        "parotid_L":    2,
        "parotid_R":    3,
        "mandible":     4,
        "spinal_cord":  5,
    },
    "thorax": {
        "lung_L":       1,
        "lung_R":       2,
        "heart":        3,
        "spinal_cord":  4,
        "esophagus":    5,
    },
    "abdomen": {
        "liver":        1,
        "kidney_L":     2,
        "kidney_R":     3,
        "spleen":       4,
        "pancreas":     5,
    },
}


# ---------------------------------------------------------------------------
# Core merge function
# ---------------------------------------------------------------------------

def merge_patient(
    patient_data_dir: Path,
    patient_seg_dir:  Path,
    label_map:        Dict[str, int],
    overwrite:        bool = False,
) -> str:
    """Merge per-organ masks for one patient.

    Args:
        patient_data_dir: The patient's data folder (contains mr.mha, ct.mha…).
        patient_seg_dir:  The patient's seg folder (contains {class}.nii.gz).
        label_map:        Mapping from class name → integer label.
        overwrite:        If False, skip patients that already have seg.nii.gz.

    Returns:
        Status string: ``"merged"``, ``"skipped"``, or ``"failed:<reason>"``.
    """
    out_path = patient_data_dir / "seg.nii.gz"

    if out_path.exists() and not overwrite:
        return "skipped"

    if not patient_seg_dir.is_dir():
        return f"failed:seg dir missing ({patient_seg_dir})"

    # Load reference image (CT) for shape / spacing / origin / direction
    ref_path = patient_data_dir / "ct.mha"
    if not ref_path.exists():
        return f"failed:ct.mha missing"

    ref_img = sitk.ReadImage(str(ref_path))
    ref_arr = sitk.GetArrayFromImage(ref_img)           # (Z, Y, X)
    merged  = np.zeros(ref_arr.shape, dtype=np.int16)

    for class_name, label in sorted(label_map.items(), key=lambda x: x[1]):
        mask_path = patient_seg_dir / f"{class_name}.nii.gz"
        if not mask_path.exists():
            # Non-fatal — organ might be absent for this patient
            continue

        mask_img = sitk.ReadImage(str(mask_path))
        mask_arr = sitk.GetArrayFromImage(mask_img)

        if mask_arr.shape != merged.shape:
            # Resample mask to reference space
            resampler = sitk.ResampleImageFilter()
            resampler.SetReferenceImage(ref_img)
            resampler.SetInterpolator(sitk.sitkNearestNeighbor)
            resampler.SetDefaultPixelValue(0)
            mask_img = resampler.Execute(mask_img)
            mask_arr = sitk.GetArrayFromImage(mask_img)

        # Later labels overwrite earlier ones (higher label = higher priority)
        merged[mask_arr > 0.5] = label

    # Write with same geometry as reference CT
    seg_img = sitk.GetImageFromArray(merged)
    seg_img.CopyInformation(ref_img)
    sitk.WriteImage(seg_img, str(out_path))
    return "merged"


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    p = argparse.ArgumentParser(
        description=(
            "Merge per-organ TotalSegmentator masks into a single seg.nii.gz "
            "per patient and copy it to the patient data directory."
        )
    )
    p.add_argument(
        "--anatomy",
        required=True,
        choices=list(_LABEL_MAPS.keys()),
        help="Anatomical region (determines class-to-label mapping).",
    )
    p.add_argument(
        "--data_dir",
        required=True,
        help=(
            "Directory containing patient sub-folders with mr.mha / ct.mha. "
            "e.g. /data/Task1/Task1/HN"
        ),
    )
    p.add_argument(
        "--seg_dir",
        required=True,
        help=(
            "Root of the TotalSegmentator output for this anatomy. "
            "Expects {seg_dir}/{patient_id}/seg/{class_name}.nii.gz. "
            "e.g. /data/segs/head_neck"
        ),
    )
    p.add_argument(
        "--overwrite",
        action="store_true",
        default=False,
        help="Re-merge even if seg.nii.gz already exists in patient dir.",
    )
    args = p.parse_args()

    data_dir = Path(args.data_dir).resolve()
    seg_dir  = Path(args.seg_dir).resolve()
    label_map = _LABEL_MAPS[args.anatomy]

    if not data_dir.is_dir():
        print(f"[ERROR] data_dir not found: {data_dir}", file=sys.stderr)
        sys.exit(1)

    patient_dirs: List[Path] = sorted(p for p in data_dir.iterdir() if p.is_dir())
    if not patient_dirs:
        print(f"[ERROR] No patient sub-directories found in {data_dir}", file=sys.stderr)
        sys.exit(1)

    print(f"Anatomy  : {args.anatomy}")
    print(f"Data dir : {data_dir}")
    print(f"Seg dir  : {seg_dir}")
    print(f"Patients : {len(patient_dirs)}")
    print(f"Classes  : {label_map}")
    print()

    n_merged  = 0
    n_skipped = 0
    n_failed  = 0

    for pdir in patient_dirs:
        pid = pdir.name
        patient_seg_dir = seg_dir / pid / "seg"
        status = merge_patient(pdir, patient_seg_dir, label_map, args.overwrite)

        if status == "merged":
            print(f"  [OK]      {pid}")
            n_merged += 1
        elif status == "skipped":
            print(f"  [SKIP]    {pid}  (seg.nii.gz exists; use --overwrite to redo)")
            n_skipped += 1
        else:
            print(f"  [FAIL]    {pid}  — {status}")
            n_failed += 1

    print()
    print(f"Done.  merged={n_merged}  skipped={n_skipped}  failed={n_failed}")

    if n_failed > 0:
        sys.exit(1)


if __name__ == "__main__":
    main()

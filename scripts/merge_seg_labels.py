#!/usr/bin/env python3
"""
Merge per-organ TotalSegmentator masks into a single seg.nii.gz.
Assumes directory structure: seg_root / anatomy / patient_id / seg/*.nii.gz
"""

import argparse
import sys
from pathlib import Path
import numpy as np
import SimpleITK as sitk

LABEL_MAPS = {
    "head_neck": {
        "brainstem": 1, "parotid_L": 2, "parotid_R": 3, "mandible": 4, "spinal_cord": 5,
    },
    "thorax": {
        "lung_L": 1, "lung_R": 2, "heart": 3, "spinal_cord": 4, "esophagus": 5,
    },
    "abdomen": {
        "liver": 1, "kidney_L": 2, "kidney_R": 3, "spleen": 4, "pancreas": 5,
    },
}

def merge_patient(data_dir, seg_root, anatomy, patient_id, overwrite=False):
    out_path = data_dir / patient_id / "seg.nii.gz"
    if out_path.exists() and not overwrite:
        return "skipped"

    seg_dir = seg_root / anatomy / patient_id / "seg"
    if not seg_dir.is_dir():
        return f"failed: {seg_dir} not found"

    ref_path = data_dir / patient_id / "ct.mha"
    if not ref_path.exists():
        return "failed: ct.mha missing"

    ref_img = sitk.ReadImage(str(ref_path))
    ref_arr = sitk.GetArrayFromImage(ref_img)
    merged = np.zeros(ref_arr.shape, dtype=np.int16)

    label_map = LABEL_MAPS[anatomy]
    for class_name, label in label_map.items():
        mask_path = seg_dir / f"{class_name}.nii.gz"
        if not mask_path.exists():
            continue
        mask_img = sitk.ReadImage(str(mask_path))
        mask_arr = sitk.GetArrayFromImage(mask_img)
        if mask_arr.shape != merged.shape:
            resampler = sitk.ResampleImageFilter()
            resampler.SetReferenceImage(ref_img)
            resampler.SetInterpolator(sitk.sitkNearestNeighbor)
            resampler.SetDefaultPixelValue(0)
            mask_img = resampler.Execute(mask_img)
            mask_arr = sitk.GetArrayFromImage(mask_img)
        merged[mask_arr > 0.5] = label

    seg_img = sitk.GetImageFromArray(merged)
    seg_img.CopyInformation(ref_img)
    sitk.WriteImage(seg_img, str(out_path))
    return "merged"

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--anatomy", required=True, choices=LABEL_MAPS.keys())
    p.add_argument("--data_dir", required=True, help="Root containing patient subfolders with ct.mha")
    p.add_argument("--seg_root", required=True, help="Root where TotalSegmentator saved outputs (e.g., /data/seg_outputs)")
    p.add_argument("--overwrite", action="store_true")
    args = p.parse_args()

    data_root = Path(args.data_dir)
    seg_root = Path(args.seg_root)
    anatomy = args.anatomy

    patients = [p for p in data_root.iterdir() if p.is_dir()]
    print(f"Anatomy: {anatomy}, Patients: {len(patients)}")

    merged = skipped = failed = 0
    for pdir in patients:
        pid = pdir.name
        status = merge_patient(data_root, seg_root, anatomy, pid, args.overwrite)
        if status == "merged":
            print(f"  [OK] {pid}")
            merged += 1
        elif status == "skipped":
            print(f"  [SKIP] {pid}")
            skipped += 1
        else:
            print(f"  [FAIL] {pid} - {status}")
            failed += 1

    print(f"Done: merged={merged}, skipped={skipped}, failed={failed}")
    sys.exit(1 if failed else 0)

if __name__ == "__main__":
    main()
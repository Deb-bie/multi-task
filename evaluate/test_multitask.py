"""
evaluate/test_multitask.py — Test-set evaluation for the multi-task CycleGAN.

Loads ``best_model.pth`` for a given anatomy and ablation configuration, runs
inference on the held-out test split, saves per-patient NIfTI volumes, and
writes a CSV of per-patient metrics.  A summary (mean ± std) is printed to
stdout at the end.

HU-equivalent MAE
------------------
CT images are stored normalised to [-1, 1] using:
    HU = (normalised + 1) / 2 * (HU_max - HU_min) + HU_min
         with HU_min=-1000, HU_max=3000
The reported ``mr2ct_mae`` is always in Hounsfield Units.  MRI MAE
(``ct2mr_mae``) is reported in normalised units (no HU conversion).

NIfTI output
------------
Synthetic volumes are assembled from 2.5-D axial slices and saved as
``float32`` NIfTI files.  The affine and header are copied from the
**source MRI** volume so that the output is properly registered in
scanner space.

Usage::

    python evaluate/test_multitask.py \\
        --anatomy       head_neck \\
        --ablation_name plus_anatomy_consistency \\
        --data_root     /data/synthrad2025 \\
        --split_dir     splits/ \\
        --checkpoint    checkpoints/plus_anatomy_consistency/head_neck/best_model.pth \\
        --config        configs/base_config.json \\
        --output_dir    .

"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch

# Project root on sys.path so relative imports work when called as a script
_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.dataset import _normalize_ct, _normalize_mr
from src.metrics import (
    dice_per_class,
    masked_mae,
    masked_psnr,
    masked_ssim,
    summary_stats,
)
from src.models.multitask_cyclegan import MultitaskCycleGAN


# ---------------------------------------------------------------------------
# NIfTI helpers (SimpleITK)
# ---------------------------------------------------------------------------

def _sitk_import():
    """Lazy import of SimpleITK with a helpful error message."""
    try:
        import SimpleITK as sitk
        return sitk
    except ImportError as exc:
        raise ImportError(
            "SimpleITK is required for NIfTI I/O. "
            "Install with: pip install SimpleITK"
        ) from exc


def _load_nifti_as_array(path: Path):
    """Load a NIfTI volume to a float32 NumPy array plus the SimpleITK image.

    Args:
        path: Path to the ``.nii.gz`` file.

    Returns:
        Tuple ``(array_zyx, sitk_image)``.  The array has axes (Z, Y, X).
    """
    sitk = _sitk_import()
    img  = sitk.ReadImage(str(path))
    arr  = sitk.GetArrayFromImage(img).astype(np.float32)  # (Z, Y, X)
    return arr, img


def _save_nifti(
    array: np.ndarray,
    reference_sitk,
    out_path: Path,
) -> None:
    """Save a float32 array as NIfTI, copying affine/header from *reference_sitk*.

    Args:
        array:          Float32 NumPy array with axes (Z, Y, X).
        reference_sitk: SimpleITK image whose spatial metadata is copied.
        out_path:       Output path (parent directories must exist).
    """
    sitk = _sitk_import()
    out_img = sitk.GetImageFromArray(array.astype(np.float32))
    out_img.CopyInformation(reference_sitk)
    sitk.WriteImage(out_img, str(out_path))


# ---------------------------------------------------------------------------
# Dataset helpers
# ---------------------------------------------------------------------------

def _load_split(split_dir: Path, anatomy: str) -> List[Dict[str, str]]:
    """Load the test subset from ``{split_dir}/{anatomy}_split.json``.

    Args:
        split_dir: Directory containing ``*_split.json`` files.
        anatomy:   Anatomical region name.

    Returns:
        List of patient entry dicts with keys
        ``patient_id``, ``mr_path``, ``ct_path``, ``seg_path``, ``mask_path``.
    """
    json_path = Path(split_dir) / f"{anatomy}_split.json"
    if not json_path.exists():
        raise FileNotFoundError(f"Split file not found: {json_path}")

    with open(json_path) as fh:
        data = json.load(fh)

    test_entries = data.get("test", [])
    if not test_entries:
        raise ValueError(f"No test entries found in {json_path}")
    return test_entries


# ---------------------------------------------------------------------------
# Inference helpers
# ---------------------------------------------------------------------------

def _slices_to_3ch_tensor(volume_zyx: np.ndarray) -> torch.Tensor:
    """Replicate a single-channel volume slice-by-slice into 3-channel tensors.

    The model ingests 2.5-D slices of shape ``(1, 3, H, W)``.  We replicate
    the grayscale channel three times (RGB-like) so it matches the
    SharedEncoder's expected input.

    Args:
        volume_zyx: Float32 array ``(Z, H, W)`` in normalised [-1, 1].

    Returns:
        Tensor of shape ``(Z, 3, H, W)``.
    """
    t = torch.from_numpy(volume_zyx)          # (Z, H, W)
    return t.unsqueeze(1).expand(-1, 3, -1, -1)  # (Z, 3, H, W)


# @torch.no_grad()
# def _run_patient_inference(
#     model:      MultitaskCycleGAN,
#     mr_arr:     np.ndarray,
#     ct_arr:     np.ndarray,
#     mask_arr:   np.ndarray,
#     device:     torch.device,
#     batch_size: int = 8,
#     image_size: int = 256,
# ) -> Dict[str, np.ndarray]:
#     """Run slice-by-slice inference for a single patient.

#     Slices are resized to ``image_size`` × ``image_size`` before the forward
#     pass (matching the training resolution) and resized back to the native
#     patient resolution before returning, so NIfTI outputs preserve the
#     original voxel grid.

#     Args:
#         model:      Loaded MultitaskCycleGAN in eval mode.
#         mr_arr:     MRI volume ``(Z, H, W)`` in [-1, 1].
#         ct_arr:     CT volume ``(Z, H, W)`` in [-1, 1].
#         mask_arr:   Binary foreground mask ``(Z, H, W)``.
#         device:     Compute device.
#         batch_size: Number of slices per forward pass.
#         image_size: Spatial resolution the model was trained at (default 256).

#     Returns:
#         Dict with keys ``fake_CT``, ``fake_MR``, ``seg_pred`` — each a
#         float32 NumPy array with axes (Z, H, W) at the *native* resolution.
#         ``seg_pred`` contains integer argmax class indices.
#     """
#     import torch.nn.functional as F

#     model.eval()
#     Z, H_nat, W_nat = mr_arr.shape
#     need_resize = (H_nat != image_size or W_nat != image_size)

#     # Outputs are accumulated at native resolution
#     fake_ct_slices  = np.zeros_like(mr_arr)
#     fake_mr_slices  = np.zeros_like(ct_arr)
#     seg_pred_slices = np.zeros(mr_arr.shape, dtype=np.int16)

#     mr_tensor = _slices_to_3ch_tensor(mr_arr)  # (Z, 3, H, W)
#     ct_tensor = _slices_to_3ch_tensor(ct_arr)

#     for start in range(0, Z, batch_size):
#         end  = min(start + batch_size, Z)
#         mr_b = mr_tensor[start:end].to(device)   # (B, 3, H, W)
#         ct_b = ct_tensor[start:end].to(device)

#         # Resize to training resolution if native size differs
#         if need_resize:
#             mr_b = F.interpolate(mr_b, size=(image_size, image_size),
#                                  mode="bilinear", align_corners=False)
#             ct_b = F.interpolate(ct_b, size=(image_size, image_size),
#                                  mode="bilinear", align_corners=False)

#         out = model(mr_b, ct_b)

#         fake_ct = out["fake_CT"]   # (B, 3, image_size, image_size)
#         fake_mr = out["fake_MR"]
#         seg_logits = out["seg_real_MR"]   # (B, C, image_size, image_size)

#         # Resize outputs back to native resolution
#         if need_resize:
#             fake_ct    = F.interpolate(fake_ct,    size=(H_nat, W_nat),
#                                        mode="bilinear", align_corners=False)
#             fake_mr    = F.interpolate(fake_mr,    size=(H_nat, W_nat),
#                                        mode="bilinear", align_corners=False)
#             seg_logits = F.interpolate(seg_logits, size=(H_nat, W_nat),
#                                        mode="bilinear", align_corners=False)

#         fake_ct_slices[start:end]  = fake_ct.mean(dim=1).cpu().numpy()
#         fake_mr_slices[start:end]  = fake_mr.mean(dim=1).cpu().numpy()
#         seg_pred_slices[start:end] = (
#             seg_logits.argmax(dim=1).cpu().numpy().astype(np.int16)
#         )

#     return {
#         "fake_CT":  fake_ct_slices,
#         "fake_MR":  fake_mr_slices,
#         "seg_pred": seg_pred_slices.astype(np.float32),
#     }







@torch.no_grad()
def _run_patient_inference(
    model:      MultitaskCycleGAN,
    mr_arr:     np.ndarray,
    ct_arr:     np.ndarray,
    mask_arr:   np.ndarray,
    device:     torch.device,
    batch_size: int = 8,
    image_size: int = 256,
) -> Dict[str, np.ndarray]:
    """Run slice-by-slice inference for a single patient."""
    import torch.nn.functional as F

    model.eval()
    Z, H_nat, W_nat = mr_arr.shape
    need_resize = (H_nat != image_size or W_nat != image_size)

    fake_ct_slices  = np.zeros_like(mr_arr)
    fake_mr_slices  = np.zeros_like(ct_arr)
    seg_pred_slices = np.zeros(mr_arr.shape, dtype=np.int16)

    mr_tensor = _slices_to_3ch_tensor(mr_arr)
    ct_tensor = _slices_to_3ch_tensor(ct_arr)

    for start in range(0, Z, batch_size):
        end = min(start + batch_size, Z)
        mr_b = mr_tensor[start:end].to(device)
        ct_b = ct_tensor[start:end].to(device)

        if need_resize:
            mr_b = F.interpolate(mr_b, size=(image_size, image_size), mode="bilinear", align_corners=False)
            ct_b = F.interpolate(ct_b, size=(image_size, image_size), mode="bilinear", align_corners=False)

        out = model(mr_b, ct_b)

        fake_ct = out["fake_CT"]   # (B, 3, H, W)
        fake_mr = out["fake_MR"]

        # === DEBUG + FIX ===
        print(f"  [Debug] fake_CT range: {fake_ct.min():.4f} ~ {fake_ct.max():.4f} | mean={fake_ct.mean():.4f}")

        # Force correct range + collapse channels safely
        fake_ct_1ch = torch.clamp(fake_ct.mean(dim=1), -1.0, 1.0)
        fake_mr_1ch = torch.clamp(fake_mr.mean(dim=1), -1.0, 1.0)

        if need_resize:
            fake_ct_1ch = F.interpolate(fake_ct_1ch.unsqueeze(1), size=(H_nat, W_nat),
                                       mode="bilinear", align_corners=False).squeeze(1)
            fake_mr_1ch = F.interpolate(fake_mr_1ch.unsqueeze(1), size=(H_nat, W_nat),
                                       mode="bilinear", align_corners=False).squeeze(1)

        fake_ct_slices[start:end] = fake_ct_1ch.cpu().numpy()
        fake_mr_slices[start:end] = fake_mr_1ch.cpu().numpy()

        # Segmentation part stays the same
        seg_logits = out["seg_real_MR"]
        if need_resize:
            seg_logits = F.interpolate(seg_logits, size=(H_nat, W_nat),
                                       mode="bilinear", align_corners=False)
        seg_pred_slices[start:end] = seg_logits.argmax(dim=1).cpu().numpy().astype(np.int16)

    return {
        "fake_CT":  fake_ct_slices,
        "fake_MR":  fake_mr_slices,
        "seg_pred": seg_pred_slices.astype(np.float32),
    }









# ---------------------------------------------------------------------------
# Per-patient metrics
# ---------------------------------------------------------------------------

def _compute_patient_metrics(
    mr_arr:       np.ndarray,
    ct_arr:       np.ndarray,
    fake_ct_arr:  np.ndarray,
    fake_mr_arr:  np.ndarray,
    seg_pred_arr: np.ndarray,
    seg_gt_arr:   np.ndarray,
    mask_arr:     np.ndarray,
    num_classes:  int,
    device:       torch.device,
) -> Dict[str, float]:
    """Compute all synthesis and segmentation metrics for one patient.

    Metrics are computed over the full 3-D volume by treating the slice
    dimension as the batch dimension.

    Args:
        mr_arr:       Real MRI, ``(Z, H, W)`` in [-1, 1].
        ct_arr:       Real CT, ``(Z, H, W)`` in [-1, 1].
        fake_ct_arr:  Synthetic CT, ``(Z, H, W)`` in [-1, 1].
        fake_mr_arr:  Synthetic MRI, ``(Z, H, W)`` in [-1, 1].
        seg_pred_arr: Predicted seg class indices, ``(Z, H, W)`` float32.
        seg_gt_arr:   Ground-truth seg labels, ``(Z, H, W)``.
        mask_arr:     Binary foreground mask, ``(Z, H, W)``.
        num_classes:  Total number of segmentation classes.
        device:       Compute device.

    Returns:
        Dict of metric key → float value.  MAE for MR→CT is in HU.
    """
    def _np_to_tensor_4d(arr: np.ndarray) -> torch.Tensor:
        """(Z, H, W) → (Z, 1, H, W) float tensor."""
        return torch.from_numpy(arr).unsqueeze(1).to(device)

    real_ct_t   = _np_to_tensor_4d(ct_arr)
    fake_ct_t   = _np_to_tensor_4d(fake_ct_arr)
    real_mr_t   = _np_to_tensor_4d(mr_arr)
    fake_mr_t   = _np_to_tensor_4d(fake_mr_arr)
    mask_t      = _np_to_tensor_4d(mask_arr)

    # MR→CT synthesis metrics
    mr2ct_ssim = masked_ssim(fake_ct_t, real_ct_t, mask_t, data_range=2.0)
    mr2ct_psnr = masked_psnr(fake_ct_t, real_ct_t, mask_t, data_range=2.0)
    mr2ct_mae  = masked_mae(fake_ct_t, real_ct_t, mask_t, is_ct=True)

    # CT→MR synthesis metrics (MRI: normalised units, not HU)
    ct2mr_ssim = masked_ssim(fake_mr_t, real_mr_t, mask_t, data_range=2.0)
    ct2mr_psnr = masked_psnr(fake_mr_t, real_mr_t, mask_t, data_range=2.0)
    ct2mr_mae  = masked_mae(fake_mr_t, real_mr_t, mask_t, is_ct=False)

    # Segmentation Dice per class.
    # Convert integer prediction map → one-hot logit proxy (Z, C, H, W) using
    # np.eye indexing, which is correct and vectorised.  The previous
    # per-slice advanced-indexing approach (seg_logits_np[z, seg_pred_int[z],
    # arange(H)[:,None], arange(W)]) mis-used a 2-D class index as a flat
    # channel selector, producing silently wrong assignments.
    seg_pred_int = seg_pred_arr.astype(np.int64)          # (Z, H, W)
    # np.eye(C)[pred] → (Z, H, W, C), then transpose to (Z, C, H, W)
    seg_logits_np = (
        np.eye(num_classes, dtype=np.float32)[seg_pred_int]  # (Z, H, W, C)
        .transpose(0, 3, 1, 2)                                # (Z, C, H, W)
    )

    seg_logits_t = torch.from_numpy(seg_logits_np).to(device)   # (Z, C, H, W)
    seg_gt_t     = torch.from_numpy(seg_gt_arr.astype(np.int64)).to(device)  # (Z, H, W)

    dpc = dice_per_class(seg_logits_t, seg_gt_t, num_classes=num_classes)
    mean_dice = float(np.mean(dpc)) if dpc else 0.0

    metrics: Dict[str, float] = {
        "mr2ct_ssim": mr2ct_ssim,
        "mr2ct_psnr": mr2ct_psnr,
        "mr2ct_mae":  mr2ct_mae,   # HU
        "ct2mr_ssim": ct2mr_ssim,
        "ct2mr_psnr": ct2mr_psnr,
        "ct2mr_mae":  ct2mr_mae,   # normalised units
        "mean_dice":  mean_dice,
    }
    for i, d in enumerate(dpc):
        metrics[f"dice_class_{i + 1}"] = d  # class 0 is background, skipped

    return metrics


# ---------------------------------------------------------------------------
# CSV helpers
# ---------------------------------------------------------------------------

def _write_csv(csv_path: Path, rows: List[Dict[str, Any]]) -> None:
    """Write all per-patient rows to *csv_path*.

    The header is derived from the union of all row keys (preserving insertion
    order of the first row).

    Args:
        csv_path: Destination CSV file path.
        rows:     List of per-patient metric dicts.
    """
    if not rows:
        return
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys())
    with open(csv_path, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


# ---------------------------------------------------------------------------
# Summary printer
# ---------------------------------------------------------------------------

_SUMMARY_COLS = [
    ("mr2ct_ssim", "MR→CT SSIM"),
    ("mr2ct_psnr", "MR→CT PSNR"),
    ("mr2ct_mae",  "MR→CT MAE (HU)"),
    ("ct2mr_ssim", "CT→MR SSIM"),
    ("ct2mr_psnr", "CT→MR PSNR"),
    ("ct2mr_mae",  "CT→MR MAE (norm)"),
    ("mean_dice",  "Mean Dice"),
]


def _print_summary(rows: List[Dict[str, Any]], anatomy: str, ablation_name: str) -> None:
    """Print mean ± std for each metric across all test patients.

    Args:
        rows:          List of per-patient metric dicts.
        anatomy:       Anatomical region name (for display).
        ablation_name: Ablation configuration name (for display).
    """
    print(f"\n{'═' * 70}")
    print(f"  Test results — {anatomy.upper()}  |  ablation: {ablation_name}")
    print(f"  n = {len(rows)} patients")
    print(f"{'═' * 70}")

    for key, label in _SUMMARY_COLS:
        vals = [r[key] for r in rows if key in r and r[key] is not None]
        if not vals:
            print(f"  {label:<22}  N/A")
            continue
        st = summary_stats(vals)
        print(f"  {label:<22}  {st['mean']:.4f} ± {st['std']:.4f}"
              f"  [median={st['median']:.4f}, "
              f"IQR={st['p25']:.4f}–{st['p75']:.4f}]")

    print(f"{'═' * 70}\n")


# ---------------------------------------------------------------------------
# Main evaluation loop
# ---------------------------------------------------------------------------

def evaluate(
    anatomy:          str,
    ablation_name:    str,
    data_root:        str,
    split_dir:        str,
    checkpoint:       str,
    config:           str,
    output_dir:       str,
    batch_size:       int = 8,
    device_str:       str = "cuda",
    checkpoint_type:  str = "best_model",
) -> None:
    """Run full test-set evaluation for one anatomy + ablation combination.

    Steps:
    1. Load config and checkpoint.
    2. Build and restore ``MultitaskCycleGAN``.
    3. Iterate over test patients from the split file.
    4. For each patient: load volumes → inference → save NIfTIs → metrics.
    5. Write per-patient CSV and print summary.

    Args:
        anatomy:          Anatomical region (``"head_neck"``, ``"thorax"``,
                          ``"abdomen"``).
        ablation_name:    Ablation config name (used to locate checkpoint and
                          name output files).
        data_root:        Root directory of the SynthRAD2025 dataset.
        split_dir:        Directory containing ``*_split.json`` files.
        checkpoint:       Path to the checkpoint file.  When *checkpoint_type*
                          is ``"best_seg_model"``, this should point to
                          ``best_seg_model.pth``; otherwise ``best_model.pth``.
        config:           Path to a merged config JSON (base + anatomy overlays).
        output_dir:       Root output directory.
        batch_size:       Number of slices per forward pass.
        device_str:       PyTorch device string (``"cuda"`` or ``"cpu"``).
        checkpoint_type:  Which checkpoint to report on load:
                          ``"best_model"``     — synthesis-optimised (default),
                          ``"best_seg_model"`` — segmentation-optimised.
    """
    output_dir_p  = Path(output_dir)
    results_dir   = output_dir_p / "results"
    outputs_root  = output_dir_p / "outputs" / ablation_name
    results_dir.mkdir(parents=True, exist_ok=True)

    # ── Load config ──────────────────────────────────────────────────────────
    with open(config) as fh:
        cfg: Dict[str, Any] = json.load(fh)

    # Normalise anatomy-config keys (accept both cases)
    num_classes     = cfg.get("num_classes",    cfg.get("NUM_CLASSES",    6))
    shared_encoder  = cfg.get("shared_encoder", cfg.get("SHARED_ENCODER", True))
    organ_names     = cfg.get("organ_names",    cfg.get("ORGAN_NAMES",
                      ["background"] + [f"class_{i}" for i in range(1, num_classes)]))

    # ── Build model ───────────────────────────────────────────────────────────
    device = torch.device(device_str if torch.cuda.is_available() else "cpu")

    model = MultitaskCycleGAN(
        num_seg_classes=num_classes,
        shared_encoder=shared_encoder,
    ).to(device)

    ckpt = torch.load(checkpoint, map_location=device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    print(f"Loaded checkpoint from: {checkpoint}  [{checkpoint_type}]")
    if checkpoint_type == "best_seg_model":
        best_val = ckpt.get("best_dice", "N/A")
        metric_label = "Best Dice"
    else:
        best_val = ckpt.get("best_ssim", "N/A")
        metric_label = "Best SSIM"
    val_str = f"{best_val:.4f}" if isinstance(best_val, float) else str(best_val)
    print(f"{metric_label} recorded at training: {val_str}  "
          f"(epoch {ckpt.get('epoch', '?')})")

    # ── Load test split ───────────────────────────────────────────────────────
    test_entries = _load_split(Path(split_dir), anatomy)
    print(f"Test patients: {len(test_entries)}")

    # ── Per-patient loop ──────────────────────────────────────────────────────
    all_rows: List[Dict[str, Any]] = []

    for entry in test_entries:
        patient_id = entry["patient_id"]
        mr_path    = Path(entry["mr_path"])
        ct_path    = Path(entry["ct_path"])
        seg_path   = Path(entry["seg_path"])
        mask_path  = Path(entry["mask_path"])

        print(f"  Processing: {patient_id} ...", end="", flush=True)

        # Load volumes
        mr_arr,   mr_sitk   = _load_nifti_as_array(mr_path)
        ct_arr,   _         = _load_nifti_as_array(ct_path)
        mask_arr, _         = _load_nifti_as_array(mask_path)
        seg_arr,  _         = _load_nifti_as_array(seg_path)

        mask_arr = (mask_arr > 0.5).astype(np.float32)
        seg_arr  = seg_arr.astype(np.int64)

        # Normalise to [-1, 1] — model was trained on normalised inputs.
        # _normalize_mr uses per-volume masked 99th-percentile clipping;
        # _normalize_ct clips HU to [-1000, 3000] then scales to [-1, 1].
        # This must happen AFTER mask binarisation so the MR normalisation
        # can correctly identify foreground voxels.
        mr_arr = _normalize_mr(mr_arr, mask_arr)
        ct_arr = _normalize_ct(ct_arr)

        # Inference
        preds = _run_patient_inference(
            model, mr_arr, ct_arr, mask_arr, device, batch_size,
            image_size=cfg.get("IMAGE_SIZE", 256),
        )

        # Save NIfTI outputs
        patient_out = outputs_root / anatomy / patient_id
        patient_out.mkdir(parents=True, exist_ok=True)

        _save_nifti(preds["fake_CT"],  mr_sitk, patient_out / "synthetic_ct.nii.gz")
        _save_nifti(preds["fake_MR"],  mr_sitk, patient_out / "synthetic_mr.nii.gz")
        _save_nifti(preds["seg_pred"], mr_sitk, patient_out / "seg_pred.nii.gz")

        # Compute metrics
        metrics = _compute_patient_metrics(
            mr_arr        = mr_arr,
            ct_arr        = ct_arr,
            fake_ct_arr   = preds["fake_CT"],
            fake_mr_arr   = preds["fake_MR"],
            seg_pred_arr  = preds["seg_pred"],
            seg_gt_arr    = seg_arr,
            mask_arr      = mask_arr,
            num_classes   = num_classes,
            device        = device,
        )

        row: Dict[str, Any] = {
            "patient_id":    patient_id,
            "anatomy":       anatomy,
            "ablation_name": ablation_name,
        }
        row.update(metrics)
        all_rows.append(row)

        print(f"  SSIM={metrics['mr2ct_ssim']:.4f}  "
              f"MAE={metrics['mr2ct_mae']:.1f} HU  "
              f"Dice={metrics['mean_dice']:.4f}")

    # ── Write CSV ─────────────────────────────────────────────────────────────
    csv_path = results_dir / f"{ablation_name}_{anatomy}_test_results.csv"
    _write_csv(csv_path, all_rows)
    print(f"\nPer-patient results saved to: {csv_path}")

    # ── Print summary ─────────────────────────────────────────────────────────
    _print_summary(all_rows, anatomy, ablation_name)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for the test evaluator.

    Returns:
        Parsed argument namespace.
    """
    p = argparse.ArgumentParser(
        description="Test-set evaluation for the multi-task CycleGAN."
    )
    p.add_argument(
        "--anatomy",
        required=True,
        choices=["head_neck", "thorax", "abdomen"],
        help="Anatomical region.",
    )
    p.add_argument(
        "--ablation_name",
        required=True,
        help="Ablation configuration name (used for output file naming).",
    )
    p.add_argument(
        "--data_root",
        required=True,
        help="Root directory of the SynthRAD2025 dataset.",
    )
    p.add_argument(
        "--split_dir",
        required=True,
        help="Directory containing *_split.json files.",
    )
    p.add_argument(
        "--checkpoint",
        required=True,
        help="Path to the checkpoint file (best_model.pth or best_seg_model.pth).",
    )
    p.add_argument(
        "--checkpoint_type",
        default="best_model",
        choices=["best_model", "best_seg_model"],
        help=(
            "Which checkpoint type to evaluate.  "
            "'best_model' (default) uses the synthesis-SSIM-optimised checkpoint; "
            "'best_seg_model' uses the Dice-optimised checkpoint.  "
            "Pass the matching file to --checkpoint."
        ),
    )
    p.add_argument(
        "--config",
        required=True,
        help="Path to merged config JSON (base + anatomy overrides).",
    )
    p.add_argument(
        "--output_dir",
        default=str(Path(__file__).resolve().parents[1]),
        help="Root output directory for NIfTIs and CSV.",
    )
    p.add_argument(
        "--batch_size",
        type=int,
        default=8,
        help="Number of slices per forward pass (default: 8).",
    )
    p.add_argument(
        "--device",
        default="cuda",
        help="PyTorch device string (default: cuda).",
    )
    return p.parse_args()


def main() -> None:
    """Entry point for the test evaluator."""
    args = parse_args()
    evaluate(
        anatomy          = args.anatomy,
        ablation_name    = args.ablation_name,
        data_root        = args.data_root,
        split_dir        = args.split_dir,
        checkpoint       = args.checkpoint,
        config           = args.config,
        output_dir       = args.output_dir,
        batch_size       = args.batch_size,
        device_str       = args.device,
        checkpoint_type  = args.checkpoint_type,
    )


if __name__ == "__main__":
    main()

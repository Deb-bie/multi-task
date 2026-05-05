"""
src/metrics.py — Centralised metric functions for synthesis and segmentation.

All functions operate on **PyTorch tensors** and return Python floats (or lists
of floats).  They are designed to be called from both the validation loop
(``train/validate.py``) and the test evaluator (``evaluate/test_multitask.py``).

Coordinate conventions
-----------------------
* Synthesis images are in the normalised range **[-1, 1]**.
* Masks are binary float tensors broadcastable to image shape.
* Segmentation logits have shape ``(B, C, H, W)``; labels ``(B, H, W)``.

HU-equivalent MAE
------------------
The SynthRAD2025 CT volumes are normalised to [-1, 1] using:

    normalised = (HU - HU_min) / (HU_max - HU_min) * 2 - 1

with HU_min = -1000 and HU_max = 3000 (4000 HU range).

To recover Hounsfield Units from a normalised value:

    HU = (normalised + 1) / 2 * (HU_max - HU_min) + HU_min
       = (normalised + 1) / 2 * 4000 - 1000

``masked_mae`` applies this conversion before computing the absolute error so
that the returned value is always in **HU** (not normalised units).  All test
CSVs and summary tables report MAE in HU.
"""

from __future__ import annotations

import math
from typing import List, Optional

import numpy as np
import torch
import torch.nn.functional as F

# ---------------------------------------------------------------------------
# HU conversion constants (CT normalisation: HU_min=-1000, HU_max=3000)
# ---------------------------------------------------------------------------
_HU_MIN: float = -1000.0
_HU_MAX: float = 3000.0
_HU_RANGE: float = _HU_MAX - _HU_MIN  # 4000 HU


def _to_hu(normalised: torch.Tensor) -> torch.Tensor:
    """Convert a CT image in [-1, 1] to Hounsfield Units.

    Args:
        normalised: Tensor with values in ``[-1, 1]``.

    Returns:
        Tensor in Hounsfield Units (float range ``[HU_min, HU_max]``).
    """
    return (normalised + 1.0) / 2.0 * _HU_RANGE + _HU_MIN


# ---------------------------------------------------------------------------
# masked_ssim
# ---------------------------------------------------------------------------

def masked_ssim(
    pred:       torch.Tensor,
    target:     torch.Tensor,
    mask:       torch.Tensor,
    data_range: float = 2.0,
) -> float:
    """Compute SSIM restricted to the foreground mask region.

    Uses ``torchmetrics.functional.structural_similarity_index_measure`` on
    pixels selected by *mask*.  The mask is applied by zeroing background
    pixels; SSIM is then computed over the full spatial extent so that the
    structural context of the foreground is preserved.

    Args:
        pred:       Predicted image, shape ``(B, C, H, W)``, range ``[-1, 1]``.
        target:     Ground-truth image, same shape as *pred*.
        mask:       Binary foreground mask, broadcastable to ``(B, 1, H, W)``.
        data_range: Value range of the images (default ``2.0`` for ``[-1, 1]``).

    Returns:
        Mean SSIM across the batch as a Python float.
    """
    try:
        from torchmetrics.functional import structural_similarity_index_measure as ssim_fn
    except ImportError as exc:
        raise ImportError(
            "torchmetrics is required for masked_ssim. "
            "Install with: pip install torchmetrics"
        ) from exc

    if mask.dim() == 3:
        mask = mask.unsqueeze(1)  # (B,1,H,W)
    mask = mask.to(pred.dtype).expand_as(pred)

    pred_m   = pred   * mask
    target_m = target * mask

    val = ssim_fn(pred_m, target_m, data_range=data_range)
    return float(val)


# ---------------------------------------------------------------------------
# masked_psnr
# ---------------------------------------------------------------------------

def masked_psnr(
    pred:       torch.Tensor,
    target:     torch.Tensor,
    mask:       torch.Tensor,
    data_range: float = 2.0,
) -> float:
    """Compute PSNR restricted to the foreground mask region.

    MSE is averaged only over foreground voxels (where mask > 0).

    Args:
        pred:       Predicted image, shape ``(B, C, H, W)``, range ``[-1, 1]``.
        target:     Ground-truth image, same shape as *pred*.
        mask:       Binary foreground mask, broadcastable to ``(B, 1, H, W)``.
        data_range: Value range of the images (default ``2.0`` for ``[-1, 1]``).

    Returns:
        PSNR in dB as a Python float.  Returns ``0.0`` if mask is empty.
    """
    if mask.dim() == 3:
        mask = mask.unsqueeze(1)
    mask = mask.to(pred.dtype).expand_as(pred)

    diff_sq = ((pred - target) ** 2) * mask
    fg_sum  = mask.sum().clamp(min=1.0)
    mse     = diff_sq.sum() / fg_sum

    if mse == 0.0:
        return float("inf")

    psnr = 10.0 * math.log10(data_range ** 2 / mse.item())
    return psnr


# ---------------------------------------------------------------------------
# masked_mae  (returns HU-equivalent for CT; raw [-1,1] for MRI)
# ---------------------------------------------------------------------------

def masked_rmse(
    pred:   torch.Tensor,
    target: torch.Tensor,
    mask:   torch.Tensor,
    is_ct:  bool = True,
) -> float:
    """Root-mean-square error over foreground pixels.

    When *is_ct* is ``True``, both tensors are first converted from [-1, 1]
    to Hounsfield Units so the returned value is in **HU**.

    Args:
        pred:   Predicted image, shape ``(B, C, H, W)``, range ``[-1, 1]``.
        target: Ground-truth image, same shape as *pred*.
        mask:   Binary foreground mask, broadcastable to ``(B, 1, H, W)``.
        is_ct:  If ``True``, convert to HU before computing RMSE.

    Returns:
        Foreground RMSE (HU if ``is_ct=True``, normalised if ``is_ct=False``).
    """
    if mask.dim() == 3:
        mask = mask.unsqueeze(1)
    mask = mask.to(pred.dtype).expand_as(pred)

    if is_ct:
        pred   = _to_hu(pred)
        target = _to_hu(target)

    sq_diff = ((pred - target) ** 2) * mask
    fg_sum  = mask.sum().clamp(min=1.0)
    return float(torch.sqrt(sq_diff.sum() / fg_sum))


def masked_mae(
    pred:       torch.Tensor,
    target:     torch.Tensor,
    mask:       torch.Tensor,
    is_ct:      bool = True,
) -> float:
    """Compute mean absolute error restricted to the foreground mask.

    When *is_ct* is ``True`` (default), both *pred* and *target* are first
    converted from normalised [-1, 1] to Hounsfield Units using:

        HU = (normalised + 1) / 2 * 4000 - 1000

    so that the returned MAE is in **HU** (clinically interpretable units).
    For MRI synthesis (``is_ct=False``), the error is returned in normalised
    units since HU conversion is meaningless for MRI intensities.

    Args:
        pred:   Predicted image, shape ``(B, C, H, W)``, range ``[-1, 1]``.
        target: Ground-truth image, same shape as *pred*.
        mask:   Binary foreground mask, broadcastable to ``(B, 1, H, W)``.
        is_ct:  If ``True``, convert to HU before computing MAE.

    Returns:
        Foreground MAE (HU if ``is_ct=True``, normalised if ``is_ct=False``).
    """
    if mask.dim() == 3:
        mask = mask.unsqueeze(1)
    mask = mask.to(pred.dtype).expand_as(pred)

    if is_ct:
        pred   = _to_hu(pred)
        target = _to_hu(target)

    abs_diff = (pred - target).abs() * mask
    fg_sum   = mask.sum().clamp(min=1.0)
    return float(abs_diff.sum() / fg_sum)


# ---------------------------------------------------------------------------
# dice_per_class
# ---------------------------------------------------------------------------

def dice_per_class(
    pred_logits:   torch.Tensor,
    target_labels: torch.Tensor,
    num_classes:   int   = 6,
    smooth:        float = 1e-5,
) -> List[float]:
    """Compute mean Dice score per foreground class across a batch.

    Background (class 0) is **excluded** from the returned list.

    Args:
        pred_logits:   Raw logits, shape ``(B, C, H, W)``.
        target_labels: Integer ground-truth labels, shape ``(B, H, W)``.
        num_classes:   Total number of classes including background.
        smooth:        Laplace smoothing term to avoid division by zero.

    Returns:
        List of ``num_classes - 1`` floats — mean Dice per foreground class
        (classes 1 … ``num_classes - 1``) averaged over the batch.
        Classes absent from *both* pred and target in a given image contribute
        ``1.0`` (vacuous truth), consistent with nnU-Net convention.
    """
    B = pred_logits.shape[0]
    probs      = F.softmax(pred_logits, dim=1)           # (B, C, H, W)
    pred_hard  = probs.argmax(dim=1)                      # (B, H, W)

    scores: List[float] = []
    for cls in range(1, num_classes):  # skip background (0)
        pred_bin   = (pred_hard   == cls).float()         # (B, H, W)
        target_bin = (target_labels == cls).float()       # (B, H, W)

        # Compute per-sample Dice, then average across batch
        inter = (pred_bin * target_bin).sum(dim=(1, 2))   # (B,)
        denom = pred_bin.sum(dim=(1, 2)) + target_bin.sum(dim=(1, 2))  # (B,)

        # Vacuous Dice = 1 when both pred and target have zero foreground
        dice_per_sample = torch.where(
            denom > 0,
            (2.0 * inter + smooth) / (denom + smooth),
            torch.ones_like(inter),
        )
        scores.append(float(dice_per_sample.mean()))

    return scores


# ---------------------------------------------------------------------------
# iou_per_class
# ---------------------------------------------------------------------------

def iou_per_class(
    pred_logits:   torch.Tensor,
    target_labels: torch.Tensor,
    num_classes:   int   = 6,
    smooth:        float = 1e-5,
) -> List[float]:
    """Compute mean IoU (Jaccard index) per foreground class across a batch.

    Background (class 0) is **excluded** from the returned list.
    IoU and Dice are monotonically related: ``IoU = Dice / (2 - Dice)``.
    IoU is numerically lower than Dice and is the standard metric in
    segmentation benchmarks (PASCAL VOC, COCO, nnU-Net leaderboards).

    Args:
        pred_logits:   Raw logits, shape ``(B, C, H, W)``.
        target_labels: Integer ground-truth labels, shape ``(B, H, W)``.
        num_classes:   Total number of classes including background.
        smooth:        Laplace smoothing term to avoid division by zero.

    Returns:
        List of ``num_classes - 1`` floats — mean IoU per foreground class
        (classes 1 … ``num_classes - 1``) averaged over the batch.
        Vacuous IoU (both pred and target absent) = 1.0, consistent with
        nnU-Net convention.
    """
    B = pred_logits.shape[0]
    pred_hard = pred_logits.argmax(dim=1)   # (B, H, W)

    scores: List[float] = []
    for cls in range(1, num_classes):
        pred_bin   = (pred_hard    == cls).float()
        target_bin = (target_labels == cls).float()

        inter = (pred_bin * target_bin).sum(dim=(1, 2))          # (B,)
        union = pred_bin.sum(dim=(1, 2)) + target_bin.sum(dim=(1, 2)) - inter  # (B,)

        iou_per_sample = torch.where(
            union > 0,
            (inter + smooth) / (union + smooth),
            torch.ones_like(inter),   # vacuous IoU = 1.0
        )
        scores.append(float(iou_per_sample.mean()))

    return scores


# ---------------------------------------------------------------------------
# hd95_per_class  (test-time only — too slow for per-epoch validation)
# ---------------------------------------------------------------------------

def hd95_per_class(
    pred_logits:   torch.Tensor,
    target_labels: torch.Tensor,
    num_classes:   int = 6,
    voxel_spacing: float = 1.0,
) -> List[Optional[float]]:
    """95th-percentile Hausdorff Distance per foreground class.

    Measures boundary accuracy in physical units (mm if *voxel_spacing* is in
    mm).  HD95 is preferred over plain HD because it is robust to single
    outlier voxels.

    **This function operates on CPU numpy arrays via scipy and is intended for
    test-time evaluation only.**  Do not call it inside the per-epoch
    validation loop — use :func:`iou_per_class` and :func:`dice_per_class`
    there instead.

    Args:
        pred_logits:   Raw logits, shape ``(B, C, H, W)``.  Converted to
                       hard predictions internally.
        target_labels: Integer ground-truth labels, shape ``(B, H, W)``.
        num_classes:   Total number of classes including background.
        voxel_spacing: Isotropic voxel size in mm (default 1.0 = pixel units).

    Returns:
        List of ``num_classes - 1`` values — mean HD95 per foreground class
        across the batch, in units of *voxel_spacing*.  Returns ``None`` for
        classes that are absent from **both** prediction and target in every
        image in the batch (vacuous case).
    """
    try:
        from scipy.ndimage import distance_transform_edt
    except ImportError as exc:
        raise ImportError(
            "scipy is required for hd95_per_class. "
            "Install with: pip install scipy"
        ) from exc

    pred_np   = pred_logits.argmax(dim=1).cpu().numpy()   # (B, H, W)
    target_np = target_labels.cpu().numpy()                # (B, H, W)
    B = pred_np.shape[0]

    results: List[Optional[float]] = []

    for cls in range(1, num_classes):
        batch_hd95: List[float] = []

        for b in range(B):
            pred_bin   = (pred_np[b]   == cls).astype(np.uint8)
            target_bin = (target_np[b] == cls).astype(np.uint8)

            # Skip if both are empty (vacuous case)
            if pred_bin.sum() == 0 and target_bin.sum() == 0:
                continue

            # If only one is empty, HD95 = max possible distance (penalty)
            if pred_bin.sum() == 0 or target_bin.sum() == 0:
                h, w = pred_bin.shape
                batch_hd95.append(float(np.sqrt(h ** 2 + w ** 2)) * voxel_spacing)
                continue

            # Distance from each pred-surface voxel to nearest target voxel
            dist_target = distance_transform_edt(1 - target_bin) * voxel_spacing
            dist_pred   = distance_transform_edt(1 - pred_bin)   * voxel_spacing

            # Surface voxels = foreground pixels adjacent to background
            pred_surface   = pred_bin   & (distance_transform_edt(pred_bin)   == 1)
            target_surface = target_bin & (distance_transform_edt(target_bin) == 1)

            # Directed distances
            d_p2t = dist_target[pred_surface   > 0]
            d_t2p = dist_pred  [target_surface > 0]

            if len(d_p2t) == 0 or len(d_t2p) == 0:
                continue

            all_d = np.concatenate([d_p2t, d_t2p])
            batch_hd95.append(float(np.percentile(all_d, 95)))

        results.append(float(np.mean(batch_hd95)) if batch_hd95 else None)

    return results


# ---------------------------------------------------------------------------
# summary_stats
# ---------------------------------------------------------------------------

def summary_stats(values: List[float]) -> dict:
    """Compute descriptive statistics for a list of scalar values.

    Args:
        values: Non-empty list of floats.

    Returns:
        Dictionary with keys: ``mean``, ``std``, ``median``, ``p25``, ``p75``.
        Returns all-zero dict if *values* is empty.
    """
    if not values:
        return {"mean": 0.0, "std": 0.0, "median": 0.0, "p25": 0.0, "p75": 0.0}

    t = torch.tensor(values, dtype=torch.float64)
    n = t.numel()

    mean   = float(t.mean())
    std    = float(t.std(correction=1)) if n > 1 else 0.0
    sorted_t = t.sort().values
    median = float(sorted_t[n // 2]) if n % 2 == 1 else float(
        (sorted_t[n // 2 - 1] + sorted_t[n // 2]) / 2.0
    )
    p25 = float(sorted_t[max(0, int(0.25 * n) - 1)])
    p75 = float(sorted_t[min(n - 1, int(0.75 * n))])

    return {
        "mean":   mean,
        "std":    std,
        "median": median,
        "p25":    p25,
        "p75":    p75,
    }

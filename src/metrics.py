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
from typing import List

import torch # type: ignore
import torch.nn.functional as F # type: ignore

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

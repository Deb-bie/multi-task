"""
Validation loop for the multi-task CycleGAN.

Computes synthesis quality metrics (SSIM, MAE, PSNR) and segmentation Dice
over the validation split, using **only foreground (body-mask) voxels** for
all synthesis metrics.

Metric conventions
------------------
- SSIM: ``data_range=2.0`` because images are normalised to ``[-1, 1]``.
- MAE / PSNR: computed over foreground pixels only (pixels where ``mask > 0``).
- Dice: hard-argmax predictions vs. ground-truth ``seg_labels`` from the
  dataloader; per-class Dice averaged over the validation set.

Dataloader batch keys expected
-------------------------------
    ``"mr"``   – real MRI  ``(B, 3, 256, 256)``
    ``"ct"``   – real CT   ``(B, 3, 256, 256)``
    ``"mask"`` – body mask ``(B, 1, 256, 256)`` binary float
    ``"seg"``  – organ labels ``(B, 256, 256)`` long, values in ``[0, num_classes)``
"""

from __future__ import annotations

from collections import defaultdict
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torchmetrics.functional import (
    structural_similarity_index_measure as _ssim_fn,
)


# ---------------------------------------------------------------------------
# Per-batch metric helpers
# ---------------------------------------------------------------------------

def _masked_ssim(
    pred: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    data_range: float = 2.0,
) -> float:
    """SSIM computed on masked images (background zeroed out).

    Args:
        pred:       ``(B, 3, H, W)`` predicted image in ``[-1, 1]``.
        target:     ``(B, 3, H, W)`` ground-truth image in ``[-1, 1]``.
        mask:       ``(B, 1, H, W)`` binary mask (1 = foreground).
        data_range: Range of pixel values (2.0 for ``[-1, 1]``).

    Returns:
        Scalar SSIM value for this batch.
    """
    fg = mask.float()
    return _ssim_fn(pred * fg, target * fg, data_range=data_range).item()


def _masked_mae(
    pred: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
) -> float:
    """Mean absolute error over foreground pixels only.

    Args:
        pred:   ``(B, 3, H, W)`` predicted image.
        target: ``(B, 3, H, W)`` ground-truth image.
        mask:   ``(B, 1, H, W)`` binary mask (1 = foreground).

    Returns:
        Scalar masked MAE.
    """
    fg = mask.float().expand_as(pred)   # (B, 3, H, W)
    n_fg = fg.sum().clamp(min=1.0)
    return ((pred - target).abs() * fg).sum().item() / n_fg.item()


def _masked_psnr(
    pred: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    data_range: float = 2.0,
) -> float:
    """Peak signal-to-noise ratio over foreground pixels only.

    PSNR = 10 · log₁₀( data_range² / MSE_foreground )

    Args:
        pred:       ``(B, 3, H, W)`` predicted image.
        target:     ``(B, 3, H, W)`` ground-truth image.
        mask:       ``(B, 1, H, W)`` binary mask (1 = foreground).
        data_range: Range of pixel values (2.0 for ``[-1, 1]``).

    Returns:
        Scalar masked PSNR in dB.
    """
    fg = mask.float().expand_as(pred)
    n_fg = fg.sum().clamp(min=1.0)
    mse = (((pred - target) ** 2) * fg).sum() / n_fg
    # Clamp MSE to avoid log(0)
    psnr = 10.0 * torch.log10((data_range ** 2) / mse.clamp(min=1e-10))
    return psnr.item()


def _dice_per_class(
    pred_logits: torch.Tensor,
    targets: torch.Tensor,
    num_classes: int,
    smooth: float = 1e-5,
) -> list[float]:
    """Hard Dice per **foreground** organ class (background class 0 excluded).

    Matches the convention in :func:`src.metrics.dice_per_class`: returns a
    list of length ``num_classes - 1`` covering classes 1 … num_classes-1.
    This ensures that training-time validation CSVs and test-time evaluation
    CSVs use identical column names (``dice_class_1`` … ``dice_class_N``).

    Vacuous Dice (both pred and target absent for class *c* in this batch) is
    set to 1.0, consistent with nnU-Net convention and
    :func:`src.metrics.dice_per_class`.

    Args:
        pred_logits: Raw segmentation logits ``(B, C, H, W)``.
        targets:     Integer class labels  ``(B, H, W)`` in ``[0, C)``.
        num_classes: Total number of classes (including background at index 0).
        smooth:      Laplace smoothing term (default 1e-5).

    Returns:
        List of ``num_classes - 1`` Dice scores ``∈ [0, 1]`` for classes
        1 … num_classes-1 (background excluded).
    """
    pred_cls = pred_logits.argmax(dim=1)   # (B, H, W)

    dice_scores: list[float] = []
    for c in range(1, num_classes):        # skip class 0 (background)
        pred_c   = (pred_cls == c).float()
        target_c = (targets   == c).float()
        intersection = (pred_c * target_c).sum()
        union        = pred_c.sum() + target_c.sum()
        if union < smooth:
            # Vacuous Dice: neither predicted nor present → 1.0
            dice_scores.append(1.0)
        else:
            dice = (2.0 * intersection + smooth) / (union + smooth)
            dice_scores.append(dice.item())

    return dice_scores


# ---------------------------------------------------------------------------
# Main validation function
# ---------------------------------------------------------------------------

def validate(
    model: Any,
    val_loader: Any,
    device: torch.device,
    num_classes: int = 6,
) -> dict[str, Any]:
    """Run one full validation pass and return aggregated metrics.

    The model is set to ``eval()`` mode for the duration of this function and
    restored to ``train()`` mode before returning.

    Args:
        model:       :class:`~src.models.MultitaskCycleGAN` instance.
        val_loader:  DataLoader whose batches contain keys
                     ``"mr"``, ``"ct"``, ``"mask"``, ``"seg"``.
        device:      Torch device (CUDA or CPU).
        num_classes: Number of segmentation classes including background
                     (default 6).

    Returns:
        Dictionary with the following keys:

        Synthesis metrics (masked, foreground only):
            ``mr2ct_ssim``, ``ct2mr_ssim`` – mean SSIM (data_range=2.0)
            ``mr2ct_mae``,  ``ct2mr_mae``  – mean MAE
            ``mr2ct_psnr``, ``ct2mr_psnr`` – mean PSNR (dB)

        Segmentation metrics (MRI branch — primary, GT-supervised):
            ``mean_dice``        – mean Dice across foreground classes (bg excluded)
            ``dice_per_class``   – list of per-class Dice, length num_classes-1
                                   (background excluded; index 0 = class 1)
            ``dice_class_{i}``   – individual Dice for class ``i`` (1-indexed,
                                   matching test CSV convention; for CSV logging)

        Segmentation metrics (CT branch — diagnostic, cross-modal proxy):
            ``ct_seg_mean_dice`` – mean Dice of seg_real_CT vs the MRI-space GT
                                   labels.  Measures how well the shared encoder
                                   generalises to CT without direct CT supervision.
                                   Low values here indicate that the CT→MRI anatomy
                                   consistency loss is working from a weak signal.
            ``ct_seg_class_{i}`` – per-class CT-branch Dice (1-indexed).
    """
    model.eval()

    accum: dict[str, list[float]] = defaultdict(list)
    # Per-class Dice accumulators (background excluded, length = num_classes-1)
    class_dice_accum:    list[list[float]] = []   # MRI branch
    ct_class_dice_accum: list[list[float]] = []   # CT  branch

    with torch.no_grad():
        for batch in val_loader:
            real_MR    = batch["mr"].to(device)    # (B, 3, 256, 256)
            real_CT    = batch["ct"].to(device)    # (B, 3, 256, 256)
            mask       = batch["mask"].to(device)  # (B, 1, 256, 256)
            seg_labels = batch["seg"].to(device)   # (B, 256, 256) long

            outs = model(real_MR, real_CT)

            fake_CT: torch.Tensor = outs["fake_CT"]
            fake_MR: torch.Tensor = outs["fake_MR"]

            # ── Synthesis metrics (all masked, foreground only) ────────
            accum["mr2ct_ssim"].append(
                _masked_ssim(fake_CT, real_CT, mask, data_range=2.0)
            )
            accum["ct2mr_ssim"].append(
                _masked_ssim(fake_MR, real_MR, mask, data_range=2.0)
            )
            accum["mr2ct_mae"].append(_masked_mae(fake_CT, real_CT, mask))
            accum["ct2mr_mae"].append(_masked_mae(fake_MR, real_MR, mask))
            accum["mr2ct_psnr"].append(
                _masked_psnr(fake_CT, real_CT, mask, data_range=2.0)
            )
            accum["ct2mr_psnr"].append(
                _masked_psnr(fake_MR, real_MR, mask, data_range=2.0)
            )

            # ── MRI segmentation (primary — GT-supervised) ────────────
            mr_per_class = _dice_per_class(
                outs["seg_real_MR"], seg_labels, num_classes
            )
            class_dice_accum.append(mr_per_class)

            # ── CT segmentation (diagnostic — cross-modal proxy) ──────
            # Evaluated against MRI-space GT labels.  No direct CT ground
            # truth exists in the default pipeline, so this measures the
            # shared encoder's cross-modal generalisation quality, which
            # determines the strength of the CT→MRI anatomy consistency
            # signal.  Track over epochs: should improve as training
            # progresses even without direct CT supervision.
            ct_per_class = _dice_per_class(
                outs["seg_real_CT"], seg_labels, num_classes
            )
            ct_class_dice_accum.append(ct_per_class)

    model.train()

    # ── Aggregate synthesis metrics ───────────────────────────────────────
    result: dict[str, Any] = {k: float(np.mean(v)) for k, v in accum.items()}

    # ── MRI segmentation metrics ──────────────────────────────────────────
    # Shape: (num_batches, num_classes-1) → mean along batch axis
    mr_dice_array  = np.array(class_dice_accum)           # (N, num_classes-1)
    dice_per_class = mr_dice_array.mean(axis=0).tolist()  # length num_classes-1

    mean_dice = float(np.mean(dice_per_class)) if dice_per_class else 0.0

    result["mean_dice"]      = mean_dice
    result["dice_per_class"] = dice_per_class

    # 1-indexed flat entries for CSV serialisation (matches test_multitask.py)
    for i, d in enumerate(dice_per_class):
        result[f"dice_class_{i + 1}"] = float(d)

    # ── CT segmentation metrics (diagnostic) ─────────────────────────────
    ct_dice_array    = np.array(ct_class_dice_accum)          # (N, num_classes-1)
    ct_dice_per_cls  = ct_dice_array.mean(axis=0).tolist()

    ct_mean_dice = float(np.mean(ct_dice_per_cls)) if ct_dice_per_cls else 0.0

    result["ct_seg_mean_dice"] = ct_mean_dice
    result["ct_seg_per_class"] = ct_dice_per_cls

    for i, d in enumerate(ct_dice_per_cls):
        result[f"ct_seg_class_{i + 1}"] = float(d)

    return result

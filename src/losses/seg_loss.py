"""
Segmentation loss for the multi-task CycleGAN framework.

Provides:
    DiceLoss
        Soft multi-class Dice computed over softmax probabilities.
        Smooth=1e-5 prevents division by zero on empty classes.

    SegLoss
        Weighted combination of :class:`DiceLoss` and
        ``nn.CrossEntropyLoss`` for stable multi-class segmentation.
        Default weighting: 0.5 Dice + 0.5 CE.

Both losses accept an optional binary *mask* (foreground body mask) so that
background voxels outside the patient body are excluded from gradient
computation — consistent with the project-wide constraint that all losses
are computed inside the body mask.

Target format
-------------
``targets`` must be a ``torch.long`` tensor of class indices with shape
``(B, H, W)``; values in ``[0, num_classes)``.  The Dice branch converts
these to one-hot internally.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Soft Dice loss
# ---------------------------------------------------------------------------

class DiceLoss(nn.Module):
    """Soft multi-class Dice loss.

    Computes the mean (1 − Dice) across all classes and the batch.
    Predictions are softmax-normalised before Dice computation, so raw
    logits should be passed in.

    The *smooth* term is added to both numerator and denominator to avoid
    zero-division on empty foreground classes (common with small organs like
    the pancreas).

    Args:
        smooth: Laplace smoothing constant (default 1e-5).
    """

    def __init__(self, smooth: float = 1e-5) -> None:
        super().__init__()
        self.smooth = smooth

    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Compute soft Dice loss.

        Args:
            logits:  Raw class logits of shape ``(B, C, H, W)``.
            targets: Integer class labels of shape ``(B, H, W)`` with
                     values in ``[0, C)``.
            mask:    Optional binary body mask ``(B, 1, H, W)`` or
                     ``(B, H, W)`` (1 = foreground, 0 = background).
                     When supplied, masked-out voxels do not contribute to
                     the Dice numerator or denominator.

        Returns:
            Scalar Dice loss ``∈ [0, 2]`` (typical range ``[0, 1]``).
        """
        B, C, H, W = logits.shape

        # Softmax over class dimension
        probs = F.softmax(logits, dim=1)  # (B, C, H, W)

        # One-hot encode targets: (B, H, W) → (B, C, H, W)
        targets_oh = (
            F.one_hot(targets, num_classes=C)   # (B, H, W, C)
            .permute(0, 3, 1, 2)                # (B, C, H, W)
            .float()
        )

        # Apply spatial mask if provided
        if mask is not None:
            # Normalise mask to (B, 1, H, W)
            if mask.dim() == 3:
                mask = mask.unsqueeze(1)
            mask = mask.float()
            probs      = probs      * mask
            targets_oh = targets_oh * mask

        # Dice per (batch, class): sum over H, W
        intersection = (probs * targets_oh).sum(dim=(2, 3))          # (B, C)
        cardinality  = probs.sum(dim=(2, 3)) + targets_oh.sum(dim=(2, 3))  # (B, C)

        dice_per_class = (2.0 * intersection + self.smooth) / (cardinality + self.smooth)
        return 1.0 - dice_per_class.mean()


# ---------------------------------------------------------------------------
# Combined segmentation loss
# ---------------------------------------------------------------------------

class SegLoss(nn.Module):
    """Combined Dice + Cross-Entropy segmentation loss with optional class weighting.

    The Dice term handles class imbalance well (especially for small organs
    like the pancreas or esophagus) while Cross-Entropy provides dense
    per-pixel gradient signal for fast convergence.

    **Class weighting** is strongly recommended for SynthRAD2025 anatomies
    because organ sizes span two orders of magnitude (liver ~40 % of
    foreground voxels vs pancreas ~0.5 %).  Without weights, the CE gradient
    from small organs is 80× weaker than from large organs.

    Recommended weighting strategy: inverse-square-root of class frequency,
    computed from the training set and stored in the anatomy config under
    ``CLASS_WEIGHTS``.  Example for abdomen::

        # Approximate median-frequency balancing weights
        # [background, liver, kidney_L, kidney_R, spleen, pancreas]
        "CLASS_WEIGHTS": [0.05, 0.5, 1.0, 1.0, 0.8, 3.0]

    Set ``CLASS_WEIGHTS`` to ``null`` in the config (or omit it) to use
    uniform weights (original behaviour).

    Args:
        dice_weight:   Weight for the Dice loss term (default 0.5).
        ce_weight:     Weight for the Cross-Entropy loss term (default 0.5).
        smooth:        Smoothing constant forwarded to :class:`DiceLoss`
                       (default 1e-5).
        ignore_index:  Class index to ignore in CE (default -1 = none).
        class_weights: Optional 1-D float tensor of per-class CE weights,
                       length ``num_classes``.  ``None`` = uniform weights.
    """

    def __init__(
        self,
        dice_weight:   float = 0.5,
        ce_weight:     float = 0.5,
        smooth:        float = 1e-5,
        ignore_index:  int   = -1,
        class_weights: Optional[torch.Tensor] = None,
    ) -> None:
        super().__init__()
        self.dice_weight  = dice_weight
        self.ce_weight    = ce_weight
        self.dice_loss    = DiceLoss(smooth=smooth)
        self.ce_loss      = nn.CrossEntropyLoss(
            weight=class_weights,   # None → uniform; tensor → per-class weights
            ignore_index=ignore_index,
            reduction="none",       # keep spatial dims so we can apply mask
        )

    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Compute weighted Dice + CE segmentation loss.

        Args:
            logits:  Raw class logits ``(B, C, H, W)``.
            targets: Integer class labels ``(B, H, W)`` in ``[0, C)``.
            mask:    Optional binary body mask ``(B, 1, H, W)`` or
                     ``(B, H, W)`` (1 = foreground).  Background voxels
                     are excluded from *both* the Dice and CE terms.

        Returns:
            Scalar combined loss.
        """
        # Dice component (mask applied internally)
        loss_dice = self.dice_loss(logits, targets, mask=mask)

        # Cross-entropy component
        ce_map = self.ce_loss(logits, targets)  # (B, H, W)

        if mask is not None:
            fg_mask = mask.float()
            if fg_mask.dim() == 4:
                fg_mask = fg_mask.squeeze(1)    # (B, H, W)
            num_fg = fg_mask.sum().clamp(min=1.0)
            loss_ce = (ce_map * fg_mask).sum() / num_fg
        else:
            loss_ce = ce_map.mean()

        return self.dice_weight * loss_dice + self.ce_weight * loss_ce

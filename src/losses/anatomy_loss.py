"""
Anatomy consistency loss for the multi-task CycleGAN framework.

Enforces that a synthetic image preserves the same organ boundaries as its
source modality.  Applied **in both synthesis directions**:

* MRI→CT:  ``seg_fake_CT`` should agree with ``seg_real_MR`` (source MRI).
* CT→MRI:  ``seg_fake_MR`` should agree with ``seg_real_CT`` (source CT).

Loss formulation
----------------
1. Convert the *source* segmentation logits to pseudo-labels via ``argmax``
   and **detach** from the computation graph — treating the source-modality
   segmentation as a fixed supervisory signal for this loss term.
2. Compute the soft Dice loss between the *fake* segmentation logits and
   the pseudo-labels from step 1.

The detach prevents the anatomy loss from back-propagating into the
segmentation network's parameters via the source branch, ensuring that
the loss only drives the synthesis path (G or F, through E) to produce
organ-faithful images.

Bidirectionality note
---------------------
The CT→MRI anatomy loss uses ``seg_real_CT`` as pseudo-labels.  Since
``seg_real_CT`` is produced by the shared encoder applied to real CT but
supervised only indirectly (no per-patient CT-space ground-truth labels in
the default pipeline), it relies on the segmentation network's learned
cross-modal generalisation.  The signal is still meaningful: it enforces
that ``fake_MR`` (synthetic MRI produced from real CT) can be segmented in
the same way as the input CT, creating a self-consistency constraint that
tightens as training progresses.

Reference: anatomy consistency is commonly used in cross-modality synthesis
(e.g. Chen et al., "Anatomy-Regularized Representation Learning", MICCAI 2019).
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .seg_loss import DiceLoss


# ---------------------------------------------------------------------------
# Anatomy Consistency Loss
# ---------------------------------------------------------------------------

class AnatomyConsistencyLoss(nn.Module):
    """Dice loss that drives a synthetic image's segmentation to match its source.

    Designed to be called **twice per training step** — once for each
    synthesis direction:

    MRI→CT direction::

        L_anatomy_mr2ct = AnatomyConsistencyLoss()(seg_fake_CT, seg_real_CT, mask)

    CT→MRI direction::

        L_anatomy_ct2mr = AnatomyConsistencyLoss()(seg_fake_MR, seg_real_CT, mask)

    Two target modes are supported:

    **Hard targets** (default, ``soft_targets=False``)::

        target = argmax(seg_source).detach()          # (B, H, W) long
        L = DiceLoss(seg_fake, target)

    **Soft targets** (``soft_targets=True``)::

        teacher = softmax(seg_source / T).detach()    # (B, C, H, W) float
        student = softmax(seg_fake    / T)
        L = 1 - mean_over_classes_and_spatial(
                    (2 * student * teacher + ε) / (student + teacher + ε)
                )

    Soft targets preserve the teacher's uncertainty rather than collapsing
    it to a one-hot label.  Particularly useful in early training when the
    segmentation network is not yet confident.

    Args:
        smooth:       Smoothing constant (default 1e-5).
        soft_targets: If ``True``, use soft Dice against the teacher's
                      class probability distribution.  If ``False`` (default),
                      use hard argmax pseudo-labels.
        temperature:  Temperature for softening the teacher distribution when
                      ``soft_targets=True``.  Values > 1 produce softer
                      distributions; ``T=1`` equals standard softmax.
    """

    def __init__(
        self,
        smooth:       float = 1e-5,
        soft_targets: bool  = False,
        temperature:  float = 2.0,
    ) -> None:
        super().__init__()
        self.dice        = DiceLoss(smooth=smooth)
        self.smooth      = smooth
        self.soft_targets = soft_targets
        self.temperature  = max(temperature, 1e-3)   # guard against division by zero

    # ------------------------------------------------------------------
    # Private: soft Dice against probability tensors
    # ------------------------------------------------------------------

    def _soft_dice(
        self,
        pred_probs:    torch.Tensor,   # (B, C, H, W)  after softmax
        target_probs:  torch.Tensor,   # (B, C, H, W)  teacher distribution, detached
        mask:          Optional[torch.Tensor],
    ) -> torch.Tensor:
        """Compute mean soft Dice between two probability distributions.

        Applies the body mask by zeroing out background-region voxels before
        summing, so that empty padding does not inflate the Dice numerator.

        Args:
            pred_probs:   Student class probabilities ``(B, C, H, W)``.
            target_probs: Teacher class probabilities ``(B, C, H, W)``, detached.
            mask:         Binary body mask ``(B, 1, H, W)`` or ``None``.

        Returns:
            Scalar soft Dice loss in ``[0, 1]``.
        """
        if mask is not None:
            # Broadcast mask (B,1,H,W) → (B,C,H,W)
            m = mask.float().expand_as(pred_probs)
            pred_probs   = pred_probs   * m
            target_probs = target_probs * m

        # Flatten spatial dims: (B, C, H*W)
        p = pred_probs.flatten(start_dim=2)
        t = target_probs.flatten(start_dim=2)

        intersection = (p * t).sum(dim=2)                      # (B, C)
        denom        = p.sum(dim=2) + t.sum(dim=2)             # (B, C)

        soft_dice_per_class = (2.0 * intersection + self.smooth) / (denom + self.smooth)
        # Mean over classes (all C) then over batch
        return 1.0 - soft_dice_per_class.mean()

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(
        self,
        seg_fake:   torch.Tensor,
        seg_source: torch.Tensor,
        mask:       Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Compute anatomy consistency loss for one synthesis direction.

        Args:
            seg_fake:   Segmentation logits for the *synthesised* image,
                        shape ``(B, C, H, W)``.  Gradients flow through
                        this tensor back into the synthesis generator.
            seg_source: Segmentation logits for the *real source* image,
                        shape ``(B, C, H, W)``.  **Detached internally** —
                        treated as a fixed pseudo-label target.
            mask:       Optional binary body mask, broadcastable to
                        ``(B, 1, H, W)`` (1 = foreground).

        Returns:
            Scalar anatomy consistency loss (hard or soft Dice).
        """
        if self.soft_targets:
            # ── Soft target mode ─────────────────────────────────────────
            # Teacher: temperature-scaled softmax of source logits (detached).
            # Student: softmax of fake logits (temperature=1 for student,
            #          matching standard Dice semantics on the pred side).
            with torch.no_grad():
                teacher_probs = F.softmax(
                    seg_source.detach() / self.temperature, dim=1
                )
            student_probs = F.softmax(seg_fake, dim=1)
            return self._soft_dice(student_probs, teacher_probs, mask)

        else:
            # ── Hard target mode (original behaviour) ─────────────────────
            # Convert source segmentation logits to hard pseudo-labels.
            with torch.no_grad():
                pseudo_labels = seg_source.detach().argmax(dim=1)  # (B, H, W) long

            return self.dice(seg_fake, pseudo_labels, mask=mask)

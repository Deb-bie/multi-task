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

from .seg_loss import DiceLoss


# ---------------------------------------------------------------------------
# Anatomy Consistency Loss
# ---------------------------------------------------------------------------

class AnatomyConsistencyLoss(nn.Module):
    """Dice loss that drives a synthetic image's segmentation to match its source.

    Designed to be called **twice per training step** — once for each
    synthesis direction:

    MRI→CT direction::

        L_anatomy_mr2ct = AnatomyConsistencyLoss()(seg_fake_CT, seg_real_MR, mask)

    CT→MRI direction::

        L_anatomy_ct2mr = AnatomyConsistencyLoss()(seg_fake_MR, seg_real_CT, mask)

    Formally for each call::

        L = DiceLoss(seg_fake, argmax(seg_source).detach())

    Args:
        smooth: Smoothing constant for the underlying
                :class:`~src.losses.seg_loss.DiceLoss` (default 1e-5).
    """

    def __init__(self, smooth: float = 1e-5) -> None:
        super().__init__()
        self.dice = DiceLoss(smooth=smooth)

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
                        Pass ``seg_real_MR`` for MRI→CT, or
                        ``seg_real_CT`` for CT→MRI.
            mask:       Optional binary body mask, broadcastable to
                        ``(B, 1, H, W)`` (1 = foreground).

        Returns:
            Scalar anatomy consistency Dice loss.

        Note:
            ``seg_source`` is detached before ``argmax``, so this loss does
            **not** affect the segmentation network's parameters via the
            source branch.  Only the synthesis path receives gradients.
        """
        # Convert source segmentation logits to hard pseudo-labels (detached).
        with torch.no_grad():
            pseudo_labels = seg_source.detach().argmax(dim=1)  # (B, H, W) long

        return self.dice(seg_fake, pseudo_labels, mask=mask)

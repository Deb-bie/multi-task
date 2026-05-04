"""
VGG16-based perceptual loss for the multi-task CycleGAN.

Extracts intermediate feature maps from a frozen VGG16 network (trained on
ImageNet) and minimises the L1 distance between predicted and target feature
activations.  This encourages the synthesised image to preserve high-level
perceptual structure (tissue texture, boundary sharpness) that pixel-level
L1 loss alone cannot capture.

Critical implementation detail
-------------------------------
Input images are in ``[-1, 1]``.  Before passing to VGG16 they are:
    1. Rescaled to ``[0, 1]``   via  ``x = (x + 1) / 2``
    2. Normalised to ImageNet statistics channel-wise:
         mean = [0.485, 0.456, 0.406]
         std  = [0.229, 0.224, 0.225]

**This normalisation step is mandatory**.  Skipping it (or using the wrong
order) causes the VGG features to lie far outside their training distribution,
inflating the perceptual loss and degrading synthesis quality.  In the
original paired CycleGAN this step was absent, which contributed to MRI→CT
SSIM dropping from 0.85 to 0.70 when LAMBDA_PERCEPTUAL was too large.  The
corrected value is LAMBDA_PERCEPTUAL = 1.

Masking
-------
An optional binary mask ``(B, 1, H, W)`` can be supplied so that background
(non-body) voxels do not contribute to the perceptual loss.  The mask is
applied to both ``pred`` and ``target`` *before* VGG feature extraction.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import vgg16, VGG16_Weights


class PerceptualLoss(nn.Module):
    """Masked VGG16 perceptual loss with ImageNet normalisation.

    Uses the first 16 layers of VGG16 (up to ``relu3_3``) as the feature
    extractor.  All VGG weights are frozen; only the L1 distance between
    activation maps is differentiable.

    Args:
        vgg_layers: Number of VGG16 feature layers to use (default 16,
                    capturing conv1 through conv3_3).
        reduction:  ``'mean'`` or ``'sum'`` passed to the internal L1 loss
                    (default ``'mean'``).

    Example::

        criterion = PerceptualLoss().to(device)
        loss = criterion(fake_MR, real_MR, mask)
    """

    # ImageNet statistics (RGB order)
    _IMAGENET_MEAN: list[float] = [0.485, 0.456, 0.406]
    _IMAGENET_STD:  list[float] = [0.229, 0.224, 0.225]

    def __init__(self, vgg_layers: int = 16, reduction: str = "mean") -> None:
        super().__init__()

        # ── Frozen VGG16 feature extractor ──────────────────────────────
        vgg = vgg16(weights=VGG16_Weights.IMAGENET1K_V1)
        self.vgg = vgg.features[:vgg_layers].eval()
        for p in self.vgg.parameters():
            p.requires_grad = False

        # ── ImageNet normalisation buffers ───────────────────────────────
        mean = torch.tensor(self._IMAGENET_MEAN).view(1, 3, 1, 1)
        std  = torch.tensor(self._IMAGENET_STD ).view(1, 3, 1, 1)
        self.register_buffer("mean", mean)
        self.register_buffer("std",  std)

        self.reduction = reduction

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _preprocess(self, x: torch.Tensor) -> torch.Tensor:
        """Map ``[-1, 1]`` → ImageNet-normalised ``[0, 1]``-based tensor.

        Steps:
            1. ``x = (x + 1) / 2``     — rescale to [0, 1]
            2. ``x = (x - mean) / std`` — ImageNet normalisation

        Args:
            x: Image tensor of shape ``(B, 3, H, W)`` in ``[-1, 1]``.

        Returns:
            Preprocessed tensor ready for VGG16 input.
        """
        x = (x + 1.0) / 2.0                     # [0, 1]
        x = (x - self.mean) / self.std           # ImageNet normalised
        return x

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Compute perceptual loss between *pred* and *target*.

        Args:
            pred:   Synthesised image ``(B, 3, H, W)`` in ``[-1, 1]``.
            target: Ground-truth image ``(B, 3, H, W)`` in ``[-1, 1]``.
            mask:   Optional binary body mask ``(B, 1, H, W)``
                    (1 = foreground, 0 = background).  Applied before
                    feature extraction so that background pixels do not
                    pollute the perceptual features.

        Returns:
            Scalar perceptual L1 loss.
        """
        # Apply spatial mask (background → 0)
        if mask is not None:
            fg = mask.float()
            if fg.dim() == 3:
                fg = fg.unsqueeze(1)       # ensure (B, 1, H, W)
            # Broadcast mask across 3 channels
            pred   = pred   * fg
            target = target * fg

        # Normalise to ImageNet space
        pred_n   = self._preprocess(pred)
        target_n = self._preprocess(target)

        # Extract VGG features (no gradient through VGG weights)
        with torch.no_grad():
            feats_target = self.vgg(target_n)
        feats_pred = self.vgg(pred_n)          # gradient flows through pred only

        return F.l1_loss(feats_pred, feats_target.detach(), reduction=self.reduction)

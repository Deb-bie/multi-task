"""
Discriminator architectures for the multi-task CycleGAN.

Classes
-------
PatchGANDiscriminator
    Standard 70×70 PatchGAN (Isola et al., 2017).  Used as D_CT to
    discriminate real vs. synthesised CT images.

MultiScaleDiscriminator
    Two PatchGAN discriminators operating at the original resolution and at
    2× downsampled resolution.  Used as D_MR to capture both coarse tissue
    boundaries and fine MRI texture.

Both classes are ported directly from the existing paired CycleGAN codebase
(``src/models.py``) so that the new multi-task package is self-contained.
"""

from __future__ import annotations

import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# PatchGAN discriminator
# ---------------------------------------------------------------------------

class PatchGANDiscriminator(nn.Module):
    """70×70 PatchGAN discriminator (single input modality).

    Outputs a spatial grid of real/fake predictions rather than a single
    scalar, allowing the discriminator to model local image statistics.

    Architecture (default ``in_ch=3``, ``features=64``):
        Conv(4, 64, stride=2)  → LeakyReLU
        Conv(64, 128, stride=2) → InstanceNorm → LeakyReLU
        Conv(128, 256, stride=2) → InstanceNorm → LeakyReLU
        Conv(256, 512, stride=1) → InstanceNorm → LeakyReLU
        Conv(512, 1,  stride=1)

    Args:
        in_ch:    Number of input channels (3 for single-modality CycleGAN,
                  6 for paired Pix2Pix where input+target are concatenated).
        features: Base filter count (default 64).
    """

    def __init__(self, in_ch: int = 3, features: int = 64) -> None:
        super().__init__()
        self.model = nn.Sequential(
            # Layer 1 – no normalisation on first conv (standard practice)
            nn.Conv2d(in_ch, features, kernel_size=4, stride=2, padding=1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # Layer 2
            nn.Conv2d(features, features * 2, kernel_size=4, stride=2, padding=1, bias=False),
            nn.InstanceNorm2d(features * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # Layer 3
            nn.Conv2d(features * 2, features * 4, kernel_size=4, stride=2, padding=1, bias=False),
            nn.InstanceNorm2d(features * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # Layer 4 – stride 1
            nn.Conv2d(features * 4, features * 8, kernel_size=4, stride=1, padding=1, bias=False),
            nn.InstanceNorm2d(features * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # Output – stride 1, no activation (raw logits for LSGAN)
            nn.Conv2d(features * 8, 1, kernel_size=4, stride=1, padding=1, bias=False),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return patch-level real/fake logit map.

        Args:
            x: Image tensor of shape ``(B, in_ch, H, W)``.

        Returns:
            Logit map of shape ``(B, 1, H', W')``.
        """
        return self.model(x)


# ---------------------------------------------------------------------------
# Multi-scale discriminator
# ---------------------------------------------------------------------------

class MultiScaleDiscriminator(nn.Module):
    """Two-scale PatchGAN discriminator for MRI synthesis.

    Runs two independent PatchGAN discriminators:
    - ``D1`` operates on the original-resolution image.
    - ``D2`` operates on a 2× average-pooled (half-resolution) image.

    The dual-scale design helps capture both global contrast/anatomy
    (coarse scale) and fine texture detail (full scale), which is
    especially important for CT→MRI synthesis.

    Args:
        in_ch:    Number of input channels (default 3).
        features: Base filter count passed to each PatchGANDiscriminator
                  (default 64).
    """

    def __init__(self, in_ch: int = 3, features: int = 64) -> None:
        super().__init__()
        self.D1 = PatchGANDiscriminator(in_ch=in_ch, features=features)
        self.D2 = PatchGANDiscriminator(in_ch=in_ch, features=features)
        self.downsample = nn.AvgPool2d(kernel_size=2)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Return logit maps at two scales.

        Args:
            x: Image tensor of shape ``(B, C, H, W)``.

        Returns:
            Tuple ``(pred_full, pred_half)`` — logit tensors at full and
            half spatial resolution respectively.
        """
        pred_full = self.D1(x)
        pred_half = self.D2(self.downsample(x))
        return pred_full, pred_half

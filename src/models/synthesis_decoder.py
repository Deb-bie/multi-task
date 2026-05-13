"""
Synthesis decoder for the multi-task CycleGAN framework.

Converts the shared encoder's 256-channel 64×64 bottleneck representation
back to a full-resolution (256×256) synthesised image.

Architecture summary (input 256×64×64 → output 3×256×256):

    res×3  : 3 task-specific ResidualBlocks at 256 ch / 64×64
    up1    : ConvTranspose 3×3 stride-2 → 128 ch / 128×128 → InstanceNorm → ReLU
    up2    : ConvTranspose 3×3 stride-2 →  64 ch / 256×256 → InstanceNorm → ReLU
    head   : ReflectionPad(3) → Conv 7×7 → Tanh

Two independent :class:`SynthesisDecoder` instances are created:
- **G** (MRI → CT synthesis)
- **F** (CT → MRI synthesis)

Each is initialised separately via :func:`~src.models.utils.init_weights`.
"""

from __future__ import annotations

import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# Building block
# ---------------------------------------------------------------------------

class ResidualBlock(nn.Module):
    """Task-specific residual block with reflection padding and InstanceNorm.

    Maintains spatial dimensions; no downsampling.

    Args:
        channels: Number of input *and* output feature channels.
    """

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(channels, channels, kernel_size=3, bias=False),
            nn.InstanceNorm2d(channels, affine=False),
            nn.ReLU(inplace=True),
            nn.ReflectionPad2d(1),
            nn.Conv2d(channels, channels, kernel_size=3, bias=False),
            nn.InstanceNorm2d(channels, affine=False),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with residual shortcut.

        Args:
            x: Feature tensor of shape ``(B, C, H, W)``.

        Returns:
            Tensor of the same shape as *x*.
        """
        return x + self.block(x)


# ---------------------------------------------------------------------------
# Synthesis Decoder
# ---------------------------------------------------------------------------

class SynthesisDecoder(nn.Module):
    """Task-specific image synthesis decoder.

    Accepts the shared encoder's bottleneck feature map and produces a
    synthesised image in the target modality.  Used for both the
    **G** (MRI→CT) and **F** (CT→MRI) generators, each instantiated and
    weight-initialised independently.

    Input  : ``(B, in_channels, 64, 64)``  – encoder bottleneck.
    Output : ``(B, out_channels, 256, 256)`` – synthesised image in [-1, 1].

    Args:
        in_channels:  Number of channels from the shared encoder (default 256).
        out_channels: Number of output image channels (default 3 for 2.5D).
        base_filters: Sets the channel schedule for the upsample stages.
                      After up1 the width is ``base_filters × 2`` (128);
                      after up2 it is ``base_filters`` (64).  Default 64.
    """

    def __init__(
        self,
        in_channels: int = 256,
        out_channels: int = 3,
        base_filters: int = 64,
    ) -> None:
        super().__init__()

        f = base_filters  # shorthand

        # ── 3 task-specific residual blocks at 256 ch / 64×64 ────────────
        self.res_blocks = nn.Sequential(
            ResidualBlock(in_channels),
            ResidualBlock(in_channels),
            ResidualBlock(in_channels),
        )

        # ── Upsample 1: 64×64 → 128×128, 256 → 128 ch ───────────────────
        self.up1 = nn.Sequential(
            nn.ConvTranspose2d(
                in_channels, f * 2,
                kernel_size=3, stride=2, padding=1, output_padding=1,
                bias=False,
            ),
            nn.InstanceNorm2d(f * 2, affine=False),
            nn.ReLU(inplace=True),
        )

        # ── Upsample 2: 128×128 → 256×256, 128 → 64 ch ──────────────────
        self.up2 = nn.Sequential(
            nn.ConvTranspose2d(
                f * 2, f,
                kernel_size=3, stride=2, padding=1, output_padding=1,
                bias=False,
            ),
            nn.InstanceNorm2d(f, affine=False),
            nn.ReLU(inplace=True),
        )

        # ── Output head: 64 ch → out_channels, Tanh ─────────────────────
        self.out_head = nn.Sequential(
            nn.ReflectionPad2d(3),
            nn.Conv2d(f, out_channels, kernel_size=7),
            nn.Tanh(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Decode a bottleneck feature map to a synthesised image.

        Args:
            x: Feature tensor of shape ``(B, 256, 64, 64)`` from
               :class:`~src.models.shared_encoder.SharedEncoder`.

        Returns:
            Synthesised image tensor of shape ``(B, 3, 256, 256)``
            with pixel values in ``[-1, 1]``.
        """
        x = self.res_blocks(x)  # (B, 256, 64,  64)
        x = self.up1(x)         # (B, 128, 128, 128)
        x = self.up2(x)         # (B,  64, 256, 256)
        x = self.out_head(x)    # (B,   3, 256, 256) ∈ [-1, 1]
        return x

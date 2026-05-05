"""
Shared encoder for the multi-task CycleGAN framework.

The encoder is used by **both** synthesis decoders (G and F) and the
segmentation decoder (S), allowing the three tasks to share low-level
and mid-level feature representations.

Architecture summary (input 3×256×256 → output 256×64×64):

    stem   : ReflectionPad2d(3) → Conv 7×7 → InstanceNorm → ReLU
    down1  : Conv 3×3 stride-2 → InstanceNorm → ReLU      [→ skip1, 128ch, 128×128]
    down2  : Conv 3×3 stride-2 → InstanceNorm → ReLU      [→ skip2, 256ch,  64×64]
    res×6  : 6 × ResidualBlock (reflection-padded, InstanceNorm)

Skip connections ``self.skip1`` and ``self.skip2`` are written as instance
attributes on each forward pass so that the :class:`SegDecoder` can read
them immediately after calling the encoder.
"""

from __future__ import annotations

import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# Building block
# ---------------------------------------------------------------------------

class ResidualBlock(nn.Module):
    """Residual block with reflection padding and InstanceNorm.

    Maintains spatial dimensions; no downsampling.  Each block applies:
        ReflectionPad → Conv 3×3 → InstanceNorm → ReLU
        ReflectionPad → Conv 3×3 → InstanceNorm
    and adds the input as a residual (skip) connection.

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
# Shared Encoder
# ---------------------------------------------------------------------------

class SharedEncoder(nn.Module):
    """Shared convolutional encoder for multi-task CycleGAN.

    Encodes a 2.5D medical image slice (3 adjacent slices stacked as
    channels) into a compact feature representation at 1/4 spatial
    resolution.  Intermediate activations are cached as ``self.skip1``
    and ``self.skip2`` after each forward call for use by
    :class:`~src.models.seg_decoder.SegDecoder`.

    Input  : ``(B, 3, 256, 256)`` – three-channel 2.5D MRI *or* CT slice.
    Output : ``(B, 256, 64, 64)`` – feature map at 1/4 resolution.

    Side-effects (set after every :meth:`forward` call):
        self.skip1 : ``(B, 128, 128, 128)`` – after first downsampling stage.
        self.skip2 : ``(B, 256,  64,  64)`` – after second downsampling stage,
                     before the residual blocks.

    Args:
        in_channels:  Number of input image channels (default 3 for 2.5D).
        base_filters: Number of filters in the stem conv (default 64).
                      Doubles at each downsampling stage so that channels
                      reach ``base_filters × 4 = 256`` at the bottleneck.
        n_res_blocks: Number of residual blocks at the bottleneck (default 6).
    """

    def __init__(
        self,
        in_channels: int = 3,
        base_filters: int = 64,
        n_res_blocks: int = 6,
    ) -> None:
        super().__init__()

        f = base_filters  # shorthand

        # ── Stem: 256×256, in_ch → f ────────────────────────────────────
        self.stem = nn.Sequential(
            nn.ReflectionPad2d(3),
            nn.Conv2d(in_channels, f, kernel_size=7, bias=False),
            nn.InstanceNorm2d(f, affine=False),
            nn.ReLU(inplace=True),
        )

        # ── Downsampling stage 1: 256×256 → 128×128, f → 2f ─────────────
        self.down1 = nn.Sequential(
            nn.Conv2d(f, f * 2, kernel_size=3, stride=2, padding=1, bias=False),
            nn.InstanceNorm2d(f * 2, affine=False),
            nn.ReLU(inplace=True),
        )

        # ── Downsampling stage 2: 128×128 → 64×64, 2f → 4f ─────────────
        self.down2 = nn.Sequential(
            nn.Conv2d(f * 2, f * 4, kernel_size=3, stride=2, padding=1, bias=False),
            nn.InstanceNorm2d(f * 4, affine=False),
            nn.ReLU(inplace=True),
        )

        # ── Bottleneck residual blocks: 64×64, 4f → 4f ──────────────────
        self.res_blocks = nn.Sequential(
            *[ResidualBlock(f * 4) for _ in range(n_res_blocks)]
        )

        # Public skip-connection tensors (populated during forward)
        self.skip1: torch.Tensor | None = None  # (B, 128, 128, 128)
        self.skip2: torch.Tensor | None = None  # (B, 256,  64,  64)

        # Expose channel dimensions so downstream modules can query them
        # without hard-coding magic numbers.
        self.skip1_channels: int = f * 2   # 128
        self.skip2_channels: int = f * 4   # 256
        self.out_channels: int   = f * 4   # 256

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Encode an image to a compact feature map and cache skip features.

        Args:
            x: Input tensor of shape ``(B, 3, 256, 256)``.

        Returns:
            Feature tensor of shape ``(B, 256, 64, 64)``.

        Side-effects:
            Sets ``self.skip1`` (128×128, 128 ch) and ``self.skip2``
            (64×64, 256 ch).  These references remain valid until the next
            :meth:`forward` call overwrites them, so callers should capture
            them with::

                feat  = encoder(img)
                skip1 = encoder.skip1   # save reference before next call
                skip2 = encoder.skip2
        """
        x = self.stem(x)        # (B,  64, 256, 256)
        x = self.down1(x)       # (B, 128, 128, 128)
        self.skip1 = x          # ← captured for SegDecoder
        x = self.down2(x)       # (B, 256,  64,  64)
        self.skip2 = x          # ← captured for SegDecoder
        x = self.res_blocks(x)  # (B, 256,  64,  64)
        return x

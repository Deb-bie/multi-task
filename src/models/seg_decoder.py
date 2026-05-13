"""
Segmentation decoder for the multi-task CycleGAN framework.

Performs organ segmentation using a U-Net–style decoder that consumes the
shared encoder's bottleneck feature map and its intermediate skip
connections (``skip1``, ``skip2``).

Architecture summary (bottleneck 256×64×64 → logits N_cls×256×256):

    UpBlock-1  [fusion at 64×64 ]  enc_out + skip2 → 256 ch  (stride-1 ConvTranspose)
    UpBlock-2  [upsample 64→128 ]  fused  + skip1  → 128 ch  (stride-2 ConvTranspose)
    UpBlock-3  [upsample 128→256]  no encoder skip  →  64 ch  (stride-2 ConvTranspose)
    UpBlock-4  [refine  at 256  ]  no encoder skip  →  32 ch  (stride-1 ConvTranspose)
    Classifier : Conv 1×1 → num_classes logits  (softmax applied externally)

Skip connection design
----------------------
``skip2`` (256 ch, 64×64) is captured *before* the 6 bottleneck residual
blocks in :class:`~src.models.shared_encoder.SharedEncoder`.  Fusing it with
the post-residual encoder output at the same spatial resolution gives the
decoder access to features that have not yet been strongly task-conditioned,
which helps preserve anatomical detail.

``skip1`` (128 ch, 128×128) is injected at the first true upsample step
(64→128), the spatial level at which its resolution exactly matches.
"""

from __future__ import annotations

from typing import List, Optional, Tuple

import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# UpBlock
# ---------------------------------------------------------------------------

class UpBlock(nn.Module):
    """Single decoder block: ConvTranspose → (optional concat skip) → Conv → IN → ReLU.

    The ConvTranspose performs spatial upsampling when ``stride=2`` or acts
    as a learned projection when ``stride=1`` (no change in resolution).

    After transposition the feature map is optionally concatenated with an
    encoder skip tensor, then refined by a 3×3 convolution followed by
    InstanceNorm and ReLU.

    Args:
        in_ch:         Input channels entering the ConvTranspose.
        transpose_out_ch: Channels output by the ConvTranspose (before concat).
        skip_ch:       Channels of the encoder skip to concatenate (0 = no skip).
        out_ch:        Channels after the refinement convolution.
        stride:        Stride for the ConvTranspose (2 = upsample ×2, 1 = same size).
    """

    def __init__(
        self,
        in_ch: int,
        transpose_out_ch: int,
        skip_ch: int,
        out_ch: int,
        stride: int = 2,
    ) -> None:
        super().__init__()

        output_padding = 1 if stride == 2 else 0

        self.up = nn.ConvTranspose2d(
            in_ch, transpose_out_ch,
            kernel_size=3, stride=stride, padding=1,
            output_padding=output_padding,
            bias=False,
        )

        # After the ConvTranspose the skip (if any) is concatenated, so
        # the refinement conv sees ``transpose_out_ch + skip_ch`` channels.
        refine_in = transpose_out_ch + skip_ch
        self.conv = nn.Sequential(
            nn.Conv2d(refine_in, out_ch, kernel_size=3, padding=1, bias=False),
            nn.InstanceNorm2d(out_ch, affine=False),
            nn.ReLU(inplace=True),
        )

    def forward(
        self,
        x: torch.Tensor,
        skip: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Upsample *x*, optionally fuse *skip*, then refine.

        Args:
            x:    Input tensor of shape ``(B, in_ch, H, W)``.
            skip: Optional encoder skip of shape
                  ``(B, skip_ch, H', W')`` where ``H'``, ``W'`` match the
                  post-transpose spatial size.  Pass ``None`` for blocks
                  without an encoder skip connection.

        Returns:
            Refined tensor of shape ``(B, out_ch, H', W')``.
        """
        x = self.up(x)
        if skip is not None:
            x = torch.cat([x, skip], dim=1)
        return self.conv(x)


# ---------------------------------------------------------------------------
# Segmentation Decoder
# ---------------------------------------------------------------------------

class SegDecoder(nn.Module):
    """U-Net–style segmentation decoder with four UpBlocks.

    Accepts the encoder bottleneck plus the two intermediate skip connections
    stored by :class:`~src.models.shared_encoder.SharedEncoder` and produces
    per-pixel class logits at the original input resolution.

    **No softmax is applied here** – use ``torch.softmax`` or
    ``F.cross_entropy`` (which expects raw logits) externally.

    UpBlock channel schedule (default ``enc_channels=256``,
    ``skip1_channels=128``, ``skip2_channels=256``, ``num_classes=6``):

    ===== ========= ============================== ======= ============
    Block  stride    input                          concat  output
    ===== ========= ============================== ======= ============
    1      1 (same)  enc_out (256 ch, 64×64)        skip2   256 ch, 64×64
    2      2 (×2)    UpBlock-1 out (256 ch, 64×64)  skip1   128 ch, 128×128
    3      2 (×2)    UpBlock-2 out (128 ch, 128×128) —       64 ch, 256×256
    4      1 (same)  UpBlock-3 out  (64 ch, 256×256) —       32 ch, 256×256
    ===== ========= ============================== ======= ============

    Classifier: Conv 1×1, 32 → ``num_classes``

    Args:
        enc_channels:        Encoder bottleneck channels (default 256).
        skip1_channels:      Channels of ``encoder.skip1`` (default 128).
        skip2_channels:      Channels of ``encoder.skip2`` (default 256).
        num_classes:         Number of output organ segmentation classes
                             (including background, default 6).
        use_deep_supervision: If ``True`` (default), attach auxiliary
                             classifier heads at UpBlock-2 output (128×128)
                             and UpBlock-3 output (256×256 pre-refinement).
                             In **training mode** ``forward()`` returns
                             ``(main_logits, [aux_logits_128, aux_logits_256])``.
                             In **eval mode** the list is always empty,
                             keeping the inference interface clean.
    """

    def __init__(
        self,
        enc_channels: int = 256,
        skip1_channels: int = 128,
        skip2_channels: int = 256,
        num_classes: int = 6,
        use_deep_supervision: bool = True,
    ) -> None:
        super().__init__()
        self.use_deep_supervision = use_deep_supervision

        # ── UpBlock-1: fuse encoder output with skip2 at 64×64 ───────────
        self.up1 = UpBlock(
            in_ch=enc_channels,
            transpose_out_ch=enc_channels,          # 256
            skip_ch=skip2_channels,                  # 256 → cat → 512
            out_ch=enc_channels,                     # 256
            stride=1,
        )

        # ── UpBlock-2: upsample 64×64 → 128×128, fuse skip1 ─────────────
        self.up2 = UpBlock(
            in_ch=enc_channels,
            transpose_out_ch=enc_channels // 2,      # 128
            skip_ch=skip1_channels,                  # 128 → cat → 256
            out_ch=enc_channels // 2,                # 128
            stride=2,
        )

        # ── UpBlock-3: upsample 128×128 → 256×256, no encoder skip ───────
        self.up3 = UpBlock(
            in_ch=enc_channels // 2,
            transpose_out_ch=enc_channels // 4,      # 64
            skip_ch=0,
            out_ch=enc_channels // 4,                # 64
            stride=2,
        )

        # ── UpBlock-4: refine at 256×256, no encoder skip ─────────────────
        self.up4 = UpBlock(
            in_ch=enc_channels // 4,
            transpose_out_ch=enc_channels // 8,      # 32
            skip_ch=0,
            out_ch=enc_channels // 8,                # 32
            stride=1,
        )

        # ── Main classifier: 32 → num_classes logits at 256×256 ──────────
        self.classifier = nn.Conv2d(enc_channels // 8, num_classes, kernel_size=1)

        # ── Auxiliary classifier heads (deep supervision) ─────────────────
        # aux_cls_2: attached after UpBlock-2 output (128 ch, 128×128)
        # aux_cls_3: attached after UpBlock-3 output  (64 ch, 256×256)
        # Both use 1×1 convolutions to produce logits without adding spatial
        # parameters, keeping the auxiliary path lightweight.
        if use_deep_supervision:
            self.aux_cls_2 = nn.Conv2d(enc_channels // 2, num_classes, kernel_size=1)
            self.aux_cls_3 = nn.Conv2d(enc_channels // 4, num_classes, kernel_size=1)

    def forward(
        self,
        enc_out: torch.Tensor,
        skip1: torch.Tensor,
        skip2: torch.Tensor,
    ) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """Decode encoder features to per-pixel segmentation logits.

        Args:
            enc_out: Encoder bottleneck ``(B, 256, 64, 64)``
                     — output of the 6 residual blocks.
            skip1:   Encoder skip ``(B, 128, 128, 128)``
                     — captured after the first downsampling stage.
            skip2:   Encoder skip ``(B, 256, 64, 64)``
                     — captured after the second downsampling stage.

        Returns:
            ``(main_logits, aux_logits_list)`` where:

            * ``main_logits`` — shape ``(B, num_classes, 256, 256)``.
              Pass directly to :class:`~src.losses.seg_loss.SegLoss` or
              apply ``torch.softmax`` for probabilities.
            * ``aux_logits_list`` — list of auxiliary logit tensors for
              deep supervision.  **Non-empty only in training mode** and
              only when ``use_deep_supervision=True``:

              - ``[0]``: 128×128 output after UpBlock-2
              - ``[1]``: 256×256 output after UpBlock-3 (pre-refinement)

              The list is **empty** in eval mode, so downstream code that
              calls ``model.eval()`` can safely ignore it.
        """
        x2 = self.up1(enc_out, skip2)  # (B, 256,  64,  64)
        x3 = self.up2(x2, skip1)       # (B, 128, 128, 128)
        x4 = self.up3(x3, None)        # (B,  64, 256, 256)
        x5 = self.up4(x4, None)        # (B,  32, 256, 256)
        main_logits = self.classifier(x5)  # (B, num_classes, 256, 256)

        aux_list: List[torch.Tensor] = []
        if self.use_deep_supervision and self.training:
            aux_list = [
                self.aux_cls_2(x3),   # (B, num_classes, 128, 128)
                self.aux_cls_3(x4),   # (B, num_classes, 256, 256)
            ]

        return main_logits, aux_list

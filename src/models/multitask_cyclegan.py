"""
Multi-task CycleGAN: joint MRI↔CT synthesis and organ segmentation.

Assembles the shared encoder, two synthesis decoders, a segmentation decoder,
and two discriminators into a single ``nn.Module``.  The :meth:`forward`
method performs **one complete multi-task forward pass** and returns a
dictionary of all intermediate outputs needed to compute every loss term.

Component summary
-----------------
- **E**    – :class:`~src.models.shared_encoder.SharedEncoder`
             (shared between both synthesis directions and segmentation)
- **G**    – :class:`~src.models.synthesis_decoder.SynthesisDecoder`
             (MRI → CT)
- **F**    – :class:`~src.models.synthesis_decoder.SynthesisDecoder`
             (CT → MRI)
- **S**    – :class:`~src.models.seg_decoder.SegDecoder`
             (organ segmentation, shared across domains)
- **D_CT** – :class:`~src.models.discriminators.PatchGANDiscriminator`
             (real vs. fake CT)
- **D_MR** – :class:`~src.models.discriminators.MultiScaleDiscriminator`
             (real vs. fake MRI, two scales)

All sub-modules are weight-initialised with normal(0, 0.02) via
:func:`~src.models.utils.init_weights`.

Forward-pass outputs
--------------------
``forward(real_MR, real_CT)`` returns a :class:`dict` with keys:

=================== =============================== ============================
Key                 Description                     Shape / Type
=================== =============================== ============================
fake_CT             G(E(MRI))                       (B, 3, 256, 256)
fake_MR             F(E(CT))                        (B, 3, 256, 256)
cycle_MR            F(E(G(E(MRI))))                 (B, 3, 256, 256)
cycle_CT            G(E(F(E(CT))))                  (B, 3, 256, 256)
idt_CT              G(E(CT))  – identity CT→CT      (B, 3, 256, 256)
idt_MR              F(E(MRI)) – identity MRI→MRI    (B, 3, 256, 256)
seg_real_MR         S(E(MRI))  – main logits        (B, C, 256, 256)
seg_fake_CT         S(E(fake_CT)) – main logits     (B, C, 256, 256)
seg_real_CT         S(E(CT))   – main logits        (B, C, 256, 256)
seg_fake_MR         S(E(fake_MR)) – main logits     (B, C, 256, 256)
seg_aux_real_MR     Deep-supervision aux logits     List[Tensor] (empty in eval)
=================== =============================== ============================

Usage in the training loop
--------------------------
After calling ``forward``, the training script should call
``D_CT.forward`` and ``D_MR.forward`` separately (using detached fakes for
the discriminator update step) because the discriminator forward passes are
not included here.  This keeps the forward method device-/AMP-agnostic and
easy to unit-test without a discriminator.
"""

from __future__ import annotations

from typing import Dict

import torch
import torch.nn as nn

from .shared_encoder import SharedEncoder
from .synthesis_decoder import SynthesisDecoder
from .seg_decoder import SegDecoder
from .discriminators import PatchGANDiscriminator, MultiScaleDiscriminator
from .utils import init_weights


# ---------------------------------------------------------------------------
# Convenience type alias
# ---------------------------------------------------------------------------

from typing import Dict, List, Union

ForwardOutputs = Dict[str, Union[torch.Tensor, List[torch.Tensor]]]


# ---------------------------------------------------------------------------
# Multi-task CycleGAN
# ---------------------------------------------------------------------------

class MultitaskCycleGAN(nn.Module):
    """Joint MRI↔CT synthesis and organ segmentation model.

    Combines a shared encoder with modality-specific synthesis decoders and
    a single segmentation decoder.  The encoder is shared across both
    synthesis directions and segmentation to promote disentangled,
    anatomically consistent representations.

    Args:
        in_channels:       Image input channels (default 3 for 2.5D).
        out_channels:      Synthesised image channels (default 3).
        base_filters:      Base filter count for the encoder / decoders
                           (default 64 → bottleneck 256 ch).
        n_res_blocks_enc:  Residual blocks in the shared encoder (default 6).
        n_res_blocks_dec:  Task-specific residual blocks in each synthesis
                           decoder (always 3, fixed by architecture).
        num_seg_classes:   Number of organ classes including background
                           (default 6, same for all anatomical regions).
        disc_features:     Base filter count for both discriminators (default 64).

    Example::

        model = MultitaskCycleGAN(num_seg_classes=6)
        outs  = model(real_MR, real_CT)
        fake_CT      = outs["fake_CT"]
        seg_real_MR  = outs["seg_real_MR"]
    """

    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 3,
        base_filters: int = 64,
        n_res_blocks_enc: int = 6,
        num_seg_classes: int = 6,
        disc_features: int = 64,
        shared_encoder: bool = True,
    ) -> None:
        super().__init__()

        self._shared_encoder = shared_encoder

        if shared_encoder:
            # ── Shared encoder (default) ─────────────────────────────────
            self.E = SharedEncoder(
                in_channels=in_channels,
                base_filters=base_filters,
                n_res_blocks=n_res_blocks_enc,
            )
            enc_ch   = self.E.out_channels
            skip1_ch = self.E.skip1_channels
            skip2_ch = self.E.skip2_channels
        else:
            # ── Separate per-domain encoders (architecture ablation) ──────
            # E_MR processes MRI inputs; E_CT processes CT inputs.
            # Neither encoder sees the other domain during the forward pass,
            # so all cross-modal alignment must emerge from the cycle loss alone.
            self.E_MR = SharedEncoder(
                in_channels=in_channels,
                base_filters=base_filters,
                n_res_blocks=n_res_blocks_enc,
            )
            self.E_CT = SharedEncoder(
                in_channels=in_channels,
                base_filters=base_filters,
                n_res_blocks=n_res_blocks_enc,
            )
            enc_ch   = self.E_MR.out_channels
            skip1_ch = self.E_MR.skip1_channels
            skip2_ch = self.E_MR.skip2_channels

        # ── Synthesis decoder G: MRI → CT ────────────────────────────────
        self.G = SynthesisDecoder(
            in_channels=enc_ch,
            out_channels=out_channels,
            base_filters=base_filters,
        )

        # ── Synthesis decoder F: CT → MRI ────────────────────────────────
        self.F = SynthesisDecoder(
            in_channels=enc_ch,
            out_channels=out_channels,
            base_filters=base_filters,
        )

        # ── Segmentation decoder S ───────────────────────────────────────
        self.S = SegDecoder(
            enc_channels=enc_ch,
            skip1_channels=skip1_ch,
            skip2_channels=skip2_ch,
            num_classes=num_seg_classes,
        )

        # ── Discriminator D_CT: PatchGAN for CT images ───────────────────
        self.D_CT = PatchGANDiscriminator(in_ch=in_channels, features=disc_features)

        # ── Discriminator D_MR: MultiScale for MRI images ───────────────
        self.D_MR = MultiScaleDiscriminator(in_ch=in_channels, features=disc_features)

        # ── Initialise all weights ───────────────────────────────────────
        init_weights(self)

    # ------------------------------------------------------------------
    # Private helper: encode + capture skips + segment
    # ------------------------------------------------------------------

    def _encode_and_segment(
        self,
        img:    torch.Tensor,
        domain: str = "shared",
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Encode *img* with the appropriate encoder and capture skip tensors.

        When ``shared_encoder=True`` (default), ``self.E`` is used for all
        inputs regardless of *domain*.  When ``shared_encoder=False``
        (separate-encoder ablation), *domain* selects ``self.E_MR`` for
        ``"mr"`` inputs and ``self.E_CT`` for ``"ct"`` inputs.  Synthesised
        images always use the target domain's encoder so that the cycle path
        is self-consistent.

        Args:
            img:    Image tensor ``(B, 3, 256, 256)``.
            domain: ``"mr"`` | ``"ct"`` | ``"shared"`` (ignored when
                    ``shared_encoder=True``).

        Returns:
            Tuple ``(feat, skip1, skip2)`` where *feat* is the bottleneck
            ``(B, enc_ch, H/4, W/4)`` and *skip1*, *skip2* are intermediate
            activations used by the segmentation decoder's skip connections.
        """
        if self._shared_encoder:
            enc = self.E
        elif domain == "mr":
            enc = self.E_MR
        else:                  # "ct" or "shared" fall back to E_CT
            enc = self.E_CT

        feat  = enc(img)
        skip1 = enc.skip1
        skip2 = enc.skip2
        return feat, skip1, skip2

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(
        self,
        real_MR: torch.Tensor,
        real_CT: torch.Tensor,
    ) -> ForwardOutputs:
        """Full multi-task forward pass.

        Performs four encoder calls (real MRI, real CT, fake CT, fake MRI)
        and reuses the cached bottleneck features for identity synthesis,
        avoiding two additional encoder passes.

        Args:
            real_MR: Real MRI batch  ``(B, 3, 256, 256)`` in ``[-1, 1]``.
            real_CT: Real CT batch   ``(B, 3, 256, 256)`` in ``[-1, 1]``.

        Returns:
            Dictionary with ten entries – see module docstring for the full
            table of keys, descriptions, and shapes.

        Note:
            Discriminator forward passes are **not** included here.  Call
            ``model.D_CT(real_CT)`` / ``model.D_CT(fake_CT.detach())`` etc.
            separately inside the training loop.
        """
        # ── (1) Encode real MRI ──────────────────────────────────────────
        feat_real_MR, skip1_real_MR, skip2_real_MR = self._encode_and_segment(real_MR, "mr")

        # ── (2) Synthesise fake CT and identity MRI ──────────────────────
        fake_CT = self.G(feat_real_MR)
        idt_MR  = self.F(feat_real_MR)

        # ── (3) Segment real MRI ─────────────────────────────────────────
        seg_real_MR, seg_aux_real_MR = self.S(feat_real_MR, skip1_real_MR, skip2_real_MR)

        # ── (4) Encode real CT ───────────────────────────────────────────
        feat_real_CT, skip1_real_CT, skip2_real_CT = self._encode_and_segment(real_CT, "ct")

        # ── (5) Synthesise fake MRI and identity CT ──────────────────────
        fake_MR = self.F(feat_real_CT)
        idt_CT  = self.G(feat_real_CT)

        # ── (6) Segment real CT ──────────────────────────────────────────
        seg_real_CT, _aux_real_CT = self.S(feat_real_CT, skip1_real_CT, skip2_real_CT)

        # ── (7) Encode fake CT → cycle MRI ──────────────────────────────
        # fake_CT is a CT-domain image → use the CT encoder
        feat_fake_CT, skip1_fake_CT, skip2_fake_CT = self._encode_and_segment(fake_CT, "ct")
        cycle_MR    = self.F(feat_fake_CT)
        seg_fake_CT, _aux_fake_CT = self.S(feat_fake_CT, skip1_fake_CT, skip2_fake_CT)

        # ── (8) Encode fake MRI → cycle CT ──────────────────────────────
        # fake_MR is an MRI-domain image → use the MR encoder
        feat_fake_MR, skip1_fake_MR, skip2_fake_MR = self._encode_and_segment(fake_MR, "mr")
        cycle_CT    = self.G(feat_fake_MR)
        seg_fake_MR, _aux_fake_MR = self.S(feat_fake_MR, skip1_fake_MR, skip2_fake_MR)

        return {
            # ── Synthesis outputs ────────────────────────────────────────
            "fake_CT":  fake_CT,
            "fake_MR":  fake_MR,
            "cycle_MR": cycle_MR,
            "cycle_CT": cycle_CT,
            "idt_CT":   idt_CT,
            "idt_MR":   idt_MR,
            # ── Segmentation main logits (no softmax) ────────────────────
            "seg_real_MR": seg_real_MR,  # S(E(MR))        (B, C, 256, 256)
            "seg_fake_CT": seg_fake_CT,  # S(E(fake_CT))   (B, C, 256, 256)
            "seg_real_CT": seg_real_CT,  # S(E(CT))        (B, C, 256, 256)
            "seg_fake_MR": seg_fake_MR,  # S(E(fake_MR))   (B, C, 256, 256)
            # ── Deep-supervision auxiliary logits (empty in eval mode) ───
            # List[Tensor]: [aux_128x128, aux_256x256_pre_refine]
            # Non-empty only in training mode and when use_deep_supervision=True.
            "seg_aux_real_MR": seg_aux_real_MR,
        }

    # ------------------------------------------------------------------
    # Parameter group helpers (used by the training script)
    # ------------------------------------------------------------------

    def generator_parameters(self):
        """Return an iterator over all generator (E / E_MR+E_CT, G, F, S) parameters.

        Handles both the shared-encoder (default) and separate-encoder
        (``shared_encoder=False``) configurations automatically.

        Used to build the generator optimiser::

            opt_G = torch.optim.Adam(model.generator_parameters(), lr=2e-4)
        """
        if self._shared_encoder:
            enc_params = list(self.E.parameters())
        else:
            enc_params = list(self.E_MR.parameters()) + list(self.E_CT.parameters())

        return (
            enc_params
            + list(self.G.parameters())
            + list(self.F.parameters())
            + list(self.S.parameters())
        )

    def discriminator_parameters(self):
        """Return an iterator over all discriminator (D_CT, D_MR) parameters.

        Used to build the discriminator optimiser::

            opt_D = torch.optim.Adam(model.discriminator_parameters(), lr=2e-4)
        """
        return (
            list(self.D_CT.parameters())
            + list(self.D_MR.parameters())
        )

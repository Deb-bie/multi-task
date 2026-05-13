"""
Main training script for the multi-task CycleGAN (Phase 2).

Trains joint MRI↔CT synthesis and organ segmentation on SynthRAD2025 data.

Usage
-----
Single anatomy run::

    python train/train_multitask.py \\
        --anatomy head_neck \\
        --data_root /data/synthrad2025 \\
        --split_dir splits/ \\
        --config configs/base_config.json \\
        --output_dir .

Ablation run (called by evaluate/ablation_runner.py)::

    python train/train_multitask.py \\
        --anatomy abdomen \\
        --data_root /data/synthrad2025 \\
        --split_dir splits/ \\
        --config /tmp/ablation_cfg.json \\
        --output_dir . \\
        --ablation_name no_perceptual

Config layering
---------------
1. ``--config`` (base_config.json or ablation-merged config)
2. ``configs/anatomy/{anatomy}.json`` overlaid on top (if present)
3. Command-line ``--ablation_name`` tags checkpoint/log paths only.

Loss terms (generator)
-----------------------
L_adv_CT   LSGAN(D_CT(fake_CT), 1)
L_adv_MR   LSGAN(D_MR_scale1(fake_MR), 1) + LSGAN(D_MR_scale2(fake_MR), 1)
L_cycle    λ_cyc  · [L1(cyc_MR·m, MR·m) + L1(cyc_CT·m, CT·m)]
L_identity λ_idt  · [L1(idt_CT·m, CT·m) + L1(idt_MR·m, MR·m)]
L_paired   λ_p_mc · L1(fCT·m, CT·m) + λ_p_cm · L1(fMR·m, MR·m)
L_perc     λ_perc · [VGG(fake_CT, real_CT, mask) + VGG(fake_MR, real_MR, mask)]
L_seg      λ_seg  · seg_weight · SegLoss(seg_real_CT, seg_labels, mask)
L_anatomy  seg_weight · [λ_anat    · Dice(seg_fake_CT, argmax(seg_real_CT).detach())
                        + λ_anat_c2m · Dice(seg_fake_MR, argmax(seg_real_CT).detach())]

Discriminator bug fix
---------------------
Original code: D_MR scale-1 and scale-2 had mismatched coefficients.
Fix applied here: both scales get 0.5 (equal weighting).
"""

from __future__ import annotations

import argparse
import csv
import gc
import json
import os
import random
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.cuda.amp import GradScaler, autocast
from torch.optim.lr_scheduler import LambdaLR
from tqdm import tqdm

# ── Project imports ──────────────────────────────────────────────────────────
_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.dataset import make_dataloader, load_split      # noqa: E402  (written in Phase 3)
from src.losses import (                                   # noqa: E402
    GANLoss,
    PerceptualLoss,
    SegLoss,
    AnatomyConsistencyLoss,
)
from src.models import MultitaskCycleGAN, ImageBuffer     # noqa: E402
from train.validate import validate                        # noqa: E402


# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------

def set_seed(seed: int) -> None:
    """Fix all random seeds for full reproducibility.

    Args:
        seed: Seed value (42 per project spec).
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ---------------------------------------------------------------------------
# LR schedule
# ---------------------------------------------------------------------------

def make_lambda_rule(decay_epoch: int, total_epochs: int):
    """Return a LambdaLR rule: flat until *decay_epoch*, then linear → 0.

    PyTorch's LambdaLR passes ``last_epoch`` (0-indexed call count) to the
    lambda.  After the scheduler is created, ``last_epoch`` starts at 0 and
    increments by 1 each time ``scheduler.step()`` is called.

    Args:
        decay_epoch:  Epoch at which decay begins (e.g. 50).
        total_epochs: Final training epoch (e.g. 100).

    Returns:
        Callable ``(last_epoch: int) -> float`` multiplier in ``[0, 1]``.
    """
    decay_steps = max(total_epochs - decay_epoch, 1)

    def rule(last_epoch: int) -> float:
        if last_epoch < decay_epoch:
            return 1.0
        return max(0.0, 1.0 - (last_epoch - decay_epoch) / decay_steps)

    return rule


# ---------------------------------------------------------------------------
# Loss function factory
# ---------------------------------------------------------------------------

def build_losses(
    device: torch.device,
    config: Optional[Dict[str, Any]] = None,
) -> Dict[str, torch.nn.Module]:
    """Instantiate and return all loss modules on *device*.

    Reads ``CLASS_WEIGHTS`` from *config* (if provided) and passes it to
    :class:`~src.losses.seg_loss.SegLoss` as per-class CE weights.
    ``CLASS_WEIGHTS`` should be a list of ``num_classes`` floats.
    Set to ``null`` or omit to use uniform weights (original behaviour).

    Args:
        device: Torch device for all loss modules.
        config: Optional merged config dict.  If ``None``, all losses use
                their default (uniform-weight) settings.

    Returns:
        Dict with keys ``"gan"``, ``"perceptual"``, ``"seg"``, ``"anatomy"``.
    """
    class_weights: Optional[torch.Tensor] = None
    if config is not None:
        raw_w = config.get("CLASS_WEIGHTS", None)
        if raw_w is not None:
            class_weights = torch.tensor(raw_w, dtype=torch.float32, device=device)

    # Only instantiate VGG16 when the perceptual loss will actually be used.
    # When LAMBDA_PERCEPTUAL=0 (e.g. thorax memory budget), skipping this
    # saves ~550 MB of VRAM and avoids the checkpoint download entirely.
    lambda_perc = config.get("LAMBDA_PERCEPTUAL", 1) if config is not None else 1
    perc_loss = PerceptualLoss().to(device) if lambda_perc > 0 else None

    return {
        "gan":        GANLoss().to(device),
        "perceptual": perc_loss,
        "seg":        SegLoss(class_weights=class_weights).to(device),
        "anatomy":    AnatomyConsistencyLoss().to(device),
    }


# ---------------------------------------------------------------------------
# Single training step
# ---------------------------------------------------------------------------

def train_step(
    model:       MultitaskCycleGAN,
    opt_G:       torch.optim.Optimizer,
    opt_D:       torch.optim.Optimizer,
    batch:       Dict[str, torch.Tensor],
    config:      Dict[str, Any],
    device:      torch.device,
    scaler_G:    GradScaler,
    scaler_D:    GradScaler,
    buf_CT:      ImageBuffer,
    buf_MR:      ImageBuffer,
    losses_dict: Dict[str, torch.nn.Module],
    epoch:       int,
    seg_weight:  float,
) -> Optional[Dict[str, float]]:
    """Execute one forward/backward pass for generators and discriminators.

    Generator loss terms
    --------------------
    1. ``L_adv_CT``   adversarial (MRI→CT generator vs. D_CT)
    2. ``L_adv_MR``   adversarial (CT→MRI generator vs. D_MR, both scales summed)
    3. ``L_cycle``    cycle-consistency (masked L1)
    4. ``L_identity`` identity regularisation (masked L1)
    5. ``L_paired``   paired supervision (masked L1, separate λ per direction)
    6. ``L_perc``     VGG16 perceptual loss on fake_MR (masked, ImageNet-normalised)
    7. ``L_seg``      segmentation loss on real-CT features vs CT-space labels (warmup-weighted)
    8. ``L_anatomy``  anatomy consistency anchored to seg_real_CT (warmup-weighted)

    Discriminator bug fix
    ---------------------
    Original code applied mismatched coefficients to D_MR scale-1 and scale-2.
    This function applies **0.5 to both scales**, giving equal gradient signal
    from both spatial resolutions of the MRI discriminator.

    Args:
        model:       :class:`~src.models.MultitaskCycleGAN` in train mode.
        opt_G:       Generator optimiser (E, G, F, S parameters).
        opt_D:       Discriminator optimiser (D_CT, D_MR parameters).
        batch:       Dataloader batch with keys ``"mr"``, ``"ct"``,
                     ``"mask"``, ``"seg"``.
        config:      Merged config dict (base + anatomy overrides).
        device:      Torch device.
        scaler_G:    AMP ``GradScaler`` for the generator step.
        scaler_D:    AMP ``GradScaler`` for the discriminator step.
        buf_CT:      Replay buffer for fake CT images (size 50).
        buf_MR:      Replay buffer for fake MRI images (size 50).
        losses_dict: Pre-built loss modules from :func:`build_losses`.
        epoch:       Current epoch number (1-indexed), used for logging.
        seg_weight:  Segmentation loss weight from warmup schedule,
                     ``min(1.0, (epoch-1) / SEG_WARMUP_EPOCHS)``.

    Returns:
        Dict with keys ``loss_G_total``, ``loss_D_total``, ``loss_paired``,
        ``loss_cycle``, ``loss_seg``, ``loss_anatomy``, or ``None`` if a
        NaN/Inf was detected (batch is skipped silently).
    """
    # ── Unpack batch ──────────────────────────────────────────────────────
    real_MR    = batch["mr"].to(device)    # (B, 3, 256, 256)
    real_CT    = batch["ct"].to(device)    # (B, 3, 256, 256)
    mask       = batch["mask"].to(device)  # (B, 1, 256, 256)  broadcast-ready
    seg_labels = batch["seg"].to(device)   # (B, 256, 256)  long

    # Guard against corrupted input tensors
    if not (torch.isfinite(real_MR).all() and torch.isfinite(real_CT).all()):
        return None

    gan_loss  = losses_dict["gan"]
    perc_loss = losses_dict["perceptual"]
    seg_loss  = losses_dict["seg"]
    anat_loss = losses_dict["anatomy"]

    # ══════════════════════════════════════════════════════════════════════
    # GENERATOR STEP
    # ══════════════════════════════════════════════════════════════════════
    opt_G.zero_grad(set_to_none=True)

    with autocast():
        # Full multi-task forward pass
        outs = model(real_MR, real_CT)

        # (a) Adversarial losses ─────────────────────────────────────────
        L_adv_CT = gan_loss(model.D_CT(outs["fake_CT"]), is_real=True)

        p1_fake, p2_fake = model.D_MR(outs["fake_MR"])
        L_adv_MR = (                                     # sum both scales
            gan_loss(p1_fake, is_real=True)
            + gan_loss(p2_fake, is_real=True)
        )

        # (b) Cycle-consistency (masked) ─────────────────────────────────
        L_cycle = (
            F.l1_loss(outs["cycle_MR"] * mask, real_MR * mask)
            + F.l1_loss(outs["cycle_CT"] * mask, real_CT * mask)
        ) * config["LAMBDA_CYCLE"]

        # (c) Identity (masked) ──────────────────────────────────────────
        L_identity = (
            F.l1_loss(outs["idt_CT"] * mask, real_CT * mask)
            + F.l1_loss(outs["idt_MR"] * mask, real_MR * mask)
        ) * config["LAMBDA_IDENTITY"]

        # (d) Paired supervision (masked, asymmetric λ) ──────────────────
        L_paired = (
            F.l1_loss(outs["fake_CT"] * mask, real_CT * mask)
            * config["LAMBDA_PAIRED_MR2CT"]
            + F.l1_loss(outs["fake_MR"] * mask, real_MR * mask)
            * config["LAMBDA_PAIRED_CT2MR"]
        )

        # (e) Perceptual — both synthesis directions ──────────────────────
        # Applied symmetrically so neither generator is favoured.
        # VGG16 feature differences are meaningful even for CT (out-of-
        # distribution for ImageNet), providing structural regularisation
        # for both decoders equally.
        # perc_loss is None when LAMBDA_PERCEPTUAL=0 (e.g. thorax memory
        # budget), in which case we skip VGG entirely.
        if perc_loss is not None and config["LAMBDA_PERCEPTUAL"] > 0:
            L_perc = (
                perc_loss(outs["fake_CT"], real_CT, mask)   # MRI→CT direction
                + perc_loss(outs["fake_MR"], real_MR, mask) # CT→MRI direction
            ) * config["LAMBDA_PERCEPTUAL"]
        else:
            L_perc = torch.zeros(1, device=device)

        # (f) Segmentation (warmup-weighted, foreground-masked) ───────────
        # TotalSegmentator labels come from ct.mha → they are in CT space.
        # Supervise seg_real_CT directly; seg_real_MR learns via anatomy loss.
        L_seg = (
            seg_loss(outs["seg_real_CT"], seg_labels, mask)
            * config["LAMBDA_SEG"]
            * seg_weight
        )

        # (f2) Deep supervision auxiliary losses ──────────────────────────
        # Auxiliary heads at 128×128 and 256×256 (pre-refinement) propagate
        # segmentation gradients deeper into the shared encoder earlier in
        # training.  Targets are downsampled with nearest-neighbour to
        # preserve integer class labels.  Weights decay with depth:
        # aux_128 gets 0.2 × L_seg weight, aux_256 gets 0.4 × L_seg weight.
        aux_logits_list = outs.get("seg_aux_real_CT", [])
        if aux_logits_list:
            ds_weights = config.get("DEEP_SUPERVISION_WEIGHTS", [0.2, 0.4])
            for aux_logits, ds_w in zip(aux_logits_list, ds_weights):
                H_aux, W_aux = aux_logits.shape[-2:]
                H_lbl, W_lbl = seg_labels.shape[-2:]
                if H_aux != H_lbl or W_aux != W_lbl:
                    # Downsample labels to auxiliary spatial size
                    tgt_ds = F.interpolate(
                        seg_labels.float().unsqueeze(1),
                        size=(H_aux, W_aux),
                        mode="nearest",
                    ).squeeze(1).long()
                    msk_ds = (
                        F.interpolate(
                            mask.float()
                            if mask.dim() == 4
                            else mask.float().unsqueeze(1),
                            size=(H_aux, W_aux),
                            mode="nearest",
                        )
                        if mask is not None else None
                    )
                else:
                    tgt_ds, msk_ds = seg_labels, mask
                L_seg = L_seg + ds_w * (
                    seg_loss(aux_logits, tgt_ds, msk_ds)
                    * config["LAMBDA_SEG"]
                    * seg_weight
                )

        # (g) Anatomy consistency — both synthesis directions ─────────────
        #
        # seg_real_CT is directly supervised by seg_labels (CT-space labels).
        # Both fake outputs are anchored to it so that:
        #
        # MRI→CT:  seg_fake_CT must match seg_real_CT (the supervised branch)
        #   Ensures G produces a CT whose anatomy matches the labelled CT.
        #
        # CT→MRI:  seg_fake_MR must match seg_real_CT
        #   Ensures F produces an MRI whose anatomy matches the input CT.
        #   seg_real_MR is left as an unsupervised cross-modal branch that
        #   learns solely through the GAN / cycle objectives.
        L_anatomy = (
            anat_loss(outs["seg_fake_CT"], outs["seg_real_CT"], mask)
            * config["LAMBDA_ANATOMY"]
            * seg_weight
            + anat_loss(outs["seg_fake_MR"], outs["seg_real_CT"], mask)
            * config.get("LAMBDA_ANATOMY_CT2MR", config["LAMBDA_ANATOMY"])
            * seg_weight
        )

        loss_G = (
            L_adv_CT + L_adv_MR
            + L_cycle + L_identity + L_paired
            + L_perc + L_seg + L_anatomy
        )

    # NaN / Inf guard
    if not torch.isfinite(loss_G):
        opt_G.zero_grad(set_to_none=True)
        return None

    scaler_G.scale(loss_G).backward()
    scaler_G.unscale_(opt_G)
    torch.nn.utils.clip_grad_norm_(
        model.generator_parameters(), max_norm=1.0
    )
    scaler_G.step(opt_G)
    scaler_G.update()

    # ══════════════════════════════════════════════════════════════════════
    # DISCRIMINATOR STEP
    # ══════════════════════════════════════════════════════════════════════
    # Push current fakes into buffers AFTER generator backward so that
    # we never accidentally hold live computation graphs in the buffer.
    fake_CT_buf = buf_CT.push_and_pop(outs["fake_CT"].detach())
    fake_MR_buf = buf_MR.push_and_pop(outs["fake_MR"].detach())

    opt_D.zero_grad(set_to_none=True)

    with autocast():
        # D_CT ────────────────────────────────────────────────────────────
        loss_D_CT = 0.5 * (
            gan_loss(model.D_CT(real_CT),         is_real=True)
            + gan_loss(model.D_CT(fake_CT_buf),   is_real=False)
        )

        # D_MR (MultiScale) ── BUG FIX: both scales get 0.5 ──────────────
        p1_real, p2_real = model.D_MR(real_MR)
        p1_fake_d, p2_fake_d = model.D_MR(fake_MR_buf)

        loss_D_MR = (
            # scale-1  (full resolution)  — coefficient 0.5
            0.5 * (
                gan_loss(p1_real,   is_real=True)
                + gan_loss(p1_fake_d, is_real=False)
            )
            # scale-2  (half resolution)  — coefficient 0.5  [bug fix applied]
            + 0.5 * (
                gan_loss(p2_real,   is_real=True)
                + gan_loss(p2_fake_d, is_real=False)
            )
        )

        loss_D = loss_D_CT + loss_D_MR

    # NaN / Inf guard
    if not torch.isfinite(loss_D):
        opt_D.zero_grad(set_to_none=True)
        return None

    scaler_D.scale(loss_D).backward()
    scaler_D.unscale_(opt_D)
    torch.nn.utils.clip_grad_norm_(
        model.discriminator_parameters(), max_norm=1.0
    )
    scaler_D.step(opt_D)
    scaler_D.update()

    return {
        "loss_G_total": loss_G.item(),
        "loss_D_total": loss_D.item(),
        "loss_paired":  L_paired.item(),
        "loss_cycle":   L_cycle.item(),
        "loss_seg":     L_seg.item(),
        "loss_anatomy": L_anatomy.item(),
    }


# ---------------------------------------------------------------------------
# Checkpoint helpers
# ---------------------------------------------------------------------------

def find_latest_checkpoint(ckpt_dir: Path) -> Optional[Path]:
    """Scan *ckpt_dir* for ``epoch_NNN.pth`` files and return the latest one.

    Args:
        ckpt_dir: Directory to scan.

    Returns:
        Path to the checkpoint with the highest epoch number, or ``None``
        if the directory is empty or contains no matching files.
    """
    candidates = sorted(ckpt_dir.glob("epoch_*.pth"))
    return candidates[-1] if candidates else None


def save_checkpoint(
    ckpt_dir:   Path,
    epoch:      int,
    model:      MultitaskCycleGAN,
    opt_G:      torch.optim.Optimizer,
    opt_D:      torch.optim.Optimizer,
    sched_G:    LambdaLR,
    sched_D:    LambdaLR,
    scaler_G:   GradScaler,
    scaler_D:   GradScaler,
    val_metrics: Dict[str, Any],
    best_ssim:  float,
    config:     Dict[str, Any],
    best_dice:  float = -float("inf"),
) -> None:
    """Save a full training checkpoint to *ckpt_dir/epoch_NNN.pth*.

    Saves model weights, optimiser states, scheduler states, AMP scaler
    states, best SSIM, validation metrics, and the active config so that
    training can be resumed exactly.

    Args:
        ckpt_dir:    Directory in which to write the checkpoint file.
        epoch:       Current epoch (used in filename).
        model:       Multi-task CycleGAN module.
        opt_G:       Generator optimiser.
        opt_D:       Discriminator optimiser.
        sched_G:     Generator LR scheduler.
        sched_D:     Discriminator LR scheduler.
        scaler_G:    Generator AMP scaler.
        scaler_D:    Discriminator AMP scaler.
        val_metrics: Validation metric dict from :func:`~train.validate.validate`.
        best_ssim:   Best mean SSIM seen so far (for display).
        config:      Full merged config dict (serialised into checkpoint).
    """
    ckpt_path = ckpt_dir / f"epoch_{epoch:03d}.pth"
    torch.save(
        {
            "epoch":        epoch,
            "model_state":  model.state_dict(),
            "opt_G_state":  opt_G.state_dict(),
            "opt_D_state":  opt_D.state_dict(),
            "sched_G_state": sched_G.state_dict(),
            "sched_D_state": sched_D.state_dict(),
            "scaler_G_state": scaler_G.state_dict(),
            "scaler_D_state": scaler_D.state_dict(),
            "val_metrics":  val_metrics,
            "best_ssim":    best_ssim,
            "best_dice":    best_dice,
            "config":       config,
        },
        ckpt_path,
    )


def save_best_model(
    ckpt_dir:    Path,
    epoch:       int,
    model:       MultitaskCycleGAN,
    val_metrics: Dict[str, Any],
    best_ssim:   float,
) -> None:
    """Save ``best_model.pth`` (model weights + metrics only, no optimiser state).

    Args:
        ckpt_dir:    Checkpoint directory.
        epoch:       Epoch at which the best model was achieved.
        model:       Multi-task CycleGAN module.
        val_metrics: Validation metrics at this epoch.
        best_ssim:   Updated best mean SSIM value.
    """
    torch.save(
        {
            "epoch":       epoch,
            "model_state": model.state_dict(),
            "val_metrics": val_metrics,
            "best_ssim":   best_ssim,
        },
        ckpt_dir / "best_model.pth",
    )


def save_best_seg_model(
    ckpt_dir:    Path,
    epoch:       int,
    model:       MultitaskCycleGAN,
    val_metrics: Dict[str, Any],
    best_dice:   float,
) -> None:
    """Save ``best_seg_model.pth`` when mean Dice improves.

    Tracked independently of ``best_model.pth`` (which optimises synthesis
    SSIM).  This ensures the best segmentation checkpoint is always available
    for organ-at-risk evaluation even if synthesis quality is not at its peak.

    Args:
        ckpt_dir:    Checkpoint directory.
        epoch:       Epoch at which the best Dice was achieved.
        model:       Multi-task CycleGAN module.
        val_metrics: Validation metrics at this epoch.
        best_dice:   Updated best mean Dice value.
    """
    torch.save(
        {
            "epoch":       epoch,
            "model_state": model.state_dict(),
            "val_metrics": val_metrics,
            "best_dice":   best_dice,
        },
        ckpt_dir / "best_seg_model.pth",
    )


# ---------------------------------------------------------------------------
# CSV logging
# ---------------------------------------------------------------------------

def append_csv_row(csv_path: Path, row: Dict[str, Any]) -> None:
    """Append a single row dict to a CSV file, writing the header if new.

    Args:
        csv_path: Path to the CSV log file.
        row:      Dict mapping column names to scalar values.
    """
    fieldnames = list(row.keys())
    write_header = not csv_path.exists()
    with open(csv_path, "a", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        writer.writerow({k: (f"{v:.6f}" if isinstance(v, float) else v)
                         for k, v in row.items()})


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for the multi-task training script.

    Returns:
        Parsed argument namespace.
    """
    p = argparse.ArgumentParser(
        description="Multi-task CycleGAN training (MRI↔CT synthesis + segmentation)."
    )
    p.add_argument(
        "--anatomy",
        required=True,
        choices=["head_neck", "thorax", "abdomen"],
        help="Anatomical region to train on.",
    )
    p.add_argument(
        "--data_root",
        required=True,
        help="Root directory of the SynthRAD2025 dataset.",
    )
    p.add_argument(
        "--split_dir",
        required=True,
        help="Directory containing train/val/test split JSON files.",
    )
    p.add_argument(
        "--config",
        default=str(_ROOT / "configs" / "base_config.json"),
        help="Path to the base (or ablation-merged) config JSON.",
    )
    p.add_argument(
        "--output_dir",
        default=str(_ROOT),
        help="Root output directory (checkpoints/, logs/ written here).",
    )
    p.add_argument(
        "--ablation_name",
        default=None,
        help="Optional ablation tag; sets checkpoint subdir to "
             "checkpoints/{ablation_name}/{anatomy}/.",
    )
    return p.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    """Entry point: load config, build model, run training loop."""
    args = parse_args()
    output_dir = Path(args.output_dir)

    # ── Config: base + anatomy overlay ────────────────────────────────────
    with open(args.config) as fh:
        config: Dict[str, Any] = json.load(fh)

    anatomy_cfg_path = _ROOT / "configs" / "anatomy" / f"{args.anatomy}.json"
    if anatomy_cfg_path.exists():
        with open(anatomy_cfg_path) as fh:
            anatomy_cfg = json.load(fh)
        # Remove comment keys before merging
        anatomy_cfg = {k: v for k, v in anatomy_cfg.items()
                       if not k.startswith("_")}
        config.update(anatomy_cfg)

    # Normalise case: prefer lowercase keys from anatomy config
    num_classes = config.get("num_classes", config.get("NUM_CLASSES", 6))
    organ_names = config.get(
        "organ_names",
        config.get("ORGAN_NAMES", [f"class_{i}" for i in range(num_classes)]),
    )
    config["num_classes"]  = num_classes
    config["organ_names"]  = organ_names

    # ── Seed ──────────────────────────────────────────────────────────────
    set_seed(config["SEED"])

    # ── Device ────────────────────────────────────────────────────────────
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device  : {device}")
    print(f"Anatomy : {args.anatomy}  |  num_classes: {num_classes}")
    print(f"Epochs  : {config['EPOCHS']}  |  DECAY_EPOCH: {config['DECAY_EPOCH']}")
    print(f"λ_seg={config['LAMBDA_SEG']}  λ_anatomy={config['LAMBDA_ANATOMY']}  "
          f"λ_perc={config['LAMBDA_PERCEPTUAL']}  warmup={config['SEG_WARMUP_EPOCHS']}")

    # ── Paths ──────────────────────────────────────────────────────────────
    tag = args.ablation_name if args.ablation_name else "multitask"
    ckpt_dir = output_dir / "checkpoints" / tag / args.anatomy
    logs_dir = output_dir / "logs"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)

    csv_name = f"{tag}_{args.anatomy}.csv"
    csv_path = logs_dir / csv_name

    # ── Data ──────────────────────────────────────────────────────────────
    split_data   = load_split(args.split_dir, args.anatomy)
    # Slice indices are cached to /data/splits/index_cache/ so rebuilding on
    # restart is instant instead of reading hundreds of .mha headers again.
    # index_cache_dir = str(Path(args.split_dir) / "index_cache")

    train_loader = make_dataloader(
        split_data["train"], args.anatomy, args.data_root, "train",
        config["BATCH_SIZE"], config["NUM_WORKERS"], config["IMAGE_SIZE"],
        # index_cache_dir=index_cache_dir,
    )
    val_loader = make_dataloader(
        split_data["val"], args.anatomy, args.data_root, "val",
        config["BATCH_SIZE"], config["NUM_WORKERS"], config["IMAGE_SIZE"],
        # index_cache_dir=index_cache_dir,
    )

    # ── Model ──────────────────────────────────────────────────────────────
    model = MultitaskCycleGAN(num_seg_classes=num_classes).to(device)

    # ── Optimisers ─────────────────────────────────────────────────────────
    opt_G = optim.Adam(
        model.generator_parameters(),
        lr=config["LR"],
        betas=(config["BETA1"], config["BETA2"]),
    )
    opt_D = optim.Adam(
        model.discriminator_parameters(),
        lr=config.get("LR_D", config["LR"] * 0.5),
        betas=(config["BETA1"], config["BETA2"]),
    )

    # ── LR schedulers ──────────────────────────────────────────────────────
    lambda_rule = make_lambda_rule(config["DECAY_EPOCH"], config["EPOCHS"])
    sched_G = LambdaLR(opt_G, lr_lambda=lambda_rule)
    sched_D = LambdaLR(opt_D, lr_lambda=lambda_rule)

    # ── AMP scalers ─────────────────────────────────────────────────────────
    scaler_G = GradScaler(enabled=device.type == "cuda")
    scaler_D = GradScaler(enabled=device.type == "cuda")

    # ── Replay buffers (size 50 per spec) ──────────────────────────────────
    buf_CT = ImageBuffer(max_size=50)
    buf_MR = ImageBuffer(max_size=50)

    # ── Loss modules ────────────────────────────────────────────────────────
    losses_dict = build_losses(device, config)

    # ── Resume from checkpoint ──────────────────────────────────────────────
    start_epoch = 1
    best_ssim   = -float("inf")
    best_dice   = -float("inf")

    latest_ckpt = find_latest_checkpoint(ckpt_dir)
    if latest_ckpt is not None:
        print(f"Resuming from checkpoint: {latest_ckpt.name}")
        ckpt = torch.load(latest_ckpt, map_location=device)
        model.load_state_dict(ckpt["model_state"])
        opt_G.load_state_dict(ckpt["opt_G_state"])
        opt_D.load_state_dict(ckpt["opt_D_state"])
        sched_G.load_state_dict(ckpt["sched_G_state"])
        sched_D.load_state_dict(ckpt["sched_D_state"])
        scaler_G.load_state_dict(ckpt["scaler_G_state"])
        scaler_D.load_state_dict(ckpt["scaler_D_state"])
        best_ssim   = ckpt.get("best_ssim", -float("inf"))
        best_dice   = ckpt.get("best_dice",  -float("inf"))
        start_epoch = ckpt["epoch"] + 1
        print(f"  → Resumed at epoch {start_epoch}  |  "
              f"best_ssim={best_ssim:.4f}  best_dice={best_dice:.4f}")

    # ══════════════════════════════════════════════════════════════════════
    # TRAINING LOOP
    # ══════════════════════════════════════════════════════════════════════
    for epoch in range(start_epoch, config["EPOCHS"] + 1):

        # Segmentation warmup weight (b in spec)
        # min(1.0, (epoch-1) / SEG_WARMUP_EPOCHS)
        warmup_denom = max(config["SEG_WARMUP_EPOCHS"], 1)
        seg_weight = min(1.0, (epoch - 1) / warmup_denom)

        model.train()
        epoch_losses: Dict[str, list] = defaultdict(list)
        skipped = 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch:03d}/{config['EPOCHS']}", leave=False)
        for batch in pbar:
            step = train_step(
                model, opt_G, opt_D, batch, config, device,
                scaler_G, scaler_D, buf_CT, buf_MR,
                losses_dict, epoch, seg_weight,
            )
            if step is None:
                skipped += 1
                continue
            for k, v in step.items():
                epoch_losses[k].append(v)
            pbar.set_postfix(
                G=f"{step['loss_G_total']:.3f}",
                D=f"{step['loss_D_total']:.3f}",
                seg=f"{step['loss_seg']:.3f}",
            )

        # Advance LR schedulers (one step per epoch)
        sched_G.step()
        sched_D.step()

        # Mean losses for this epoch (empty list → 0.0)
        mean_losses = {
            k: float(np.mean(v)) if v else 0.0
            for k, v in epoch_losses.items()
        }

        # ── Validation ────────────────────────────────────────────────────
        val_metrics = validate(model, val_loader, device, num_classes=num_classes)
        mean_ssim   = 0.5 * (val_metrics["mr2ct_ssim"] + val_metrics["ct2mr_ssim"])

        # Print epoch summary
        print(
            f"[{epoch:03d}/{config['EPOCHS']}]  "
            f"G={mean_losses.get('loss_G_total', 0):.4f}  "
            f"D={mean_losses.get('loss_D_total', 0):.4f}  "
            f"seg={mean_losses.get('loss_seg', 0):.4f}  "
            f"skipped={skipped}"
        )
        print(
            f"  synthesis  "
            f"MR→CT  SSIM={val_metrics['mr2ct_ssim']:.4f}  "
            f"MAE={val_metrics['mr2ct_mae']:.4f}  "
            f"PSNR={val_metrics['mr2ct_psnr']:.2f}dB  "
            f"RMSE={val_metrics['mr2ct_rmse']:.4f}"
        )
        print(
            f"             "
            f"CT→MR  SSIM={val_metrics['ct2mr_ssim']:.4f}  "
            f"MAE={val_metrics['ct2mr_mae']:.4f}  "
            f"PSNR={val_metrics['ct2mr_psnr']:.2f}dB  "
            f"RMSE={val_metrics['ct2mr_rmse']:.4f}"
        )
        print(
            f"  seg        "
            f"Dice={val_metrics['mean_dice']:.4f}  "
            f"IoU={val_metrics['mean_iou']:.4f}"
        )

        # ── Periodic checkpoint (every CHECKPOINT_FREQ epochs) ────────────
        if epoch % config["CHECKPOINT_FREQ"] == 0:
            save_checkpoint(
                ckpt_dir, epoch, model,
                opt_G, opt_D, sched_G, sched_D,
                scaler_G, scaler_D,
                val_metrics, best_ssim, config,
                best_dice=best_dice,
            )
            print(f"  → Checkpoint saved: epoch_{epoch:03d}.pth")

        # ── Best synthesis model (ALL FOUR conditions must hold) ─────────
        # (a) mean_ssim > previous best
        # (b) mr2ct_ssim >= MR2CT_SSIM_FLOOR — MRI→CT cannot collapse
        # (c) ct2mr_ssim >= CT2MR_SSIM_FLOOR — CT→MRI cannot collapse
        # (d) mean_dice  >= MEAN_DICE_FLOOR   — segmentation cannot collapse
        #     Set MEAN_DICE_FLOOR=0.0 in config to disable (default).
        mr2ct_floor = config.get("MR2CT_SSIM_FLOOR", 0.82)
        ct2mr_floor = config.get("CT2MR_SSIM_FLOOR", 0.75)
        dice_floor  = config.get("MEAN_DICE_FLOOR",  0.0)
        if (mean_ssim > best_ssim
                and val_metrics["mr2ct_ssim"] >= mr2ct_floor
                and val_metrics["ct2mr_ssim"] >= ct2mr_floor
                and val_metrics["mean_dice"]  >= dice_floor):
            best_ssim = mean_ssim
            save_best_model(ckpt_dir, epoch, model, val_metrics, best_ssim)
            print(f"  → New best_model  mean_ssim={best_ssim:.4f}  "
                  f"mr2ct={val_metrics['mr2ct_ssim']:.4f}  "
                  f"ct2mr={val_metrics['ct2mr_ssim']:.4f}  "
                  f"dice={val_metrics['mean_dice']:.4f}")

        # ── Best segmentation model (tracked independently of SSIM) ──────
        # Saved whenever mean Dice improves, regardless of synthesis quality.
        # Useful for organ-at-risk delineation evaluation separately from
        # image synthesis evaluation.
        cur_dice = val_metrics["mean_dice"]
        if cur_dice > best_dice:
            best_dice = cur_dice
            save_best_seg_model(ckpt_dir, epoch, model, val_metrics, best_dice)
            print(f"  → New best_seg_model  mean_dice={best_dice:.4f}")

        # ── CSV logging ───────────────────────────────────────────────────
        csv_row: Dict[str, Any] = {"epoch": epoch}
        csv_row.update(mean_losses)
        # Add zeros for any loss keys that had no batches
        for lk in ("loss_G_total", "loss_D_total", "loss_paired",
                    "loss_cycle", "loss_seg", "loss_anatomy"):
            csv_row.setdefault(lk, 0.0)
        csv_row.update(val_metrics)
        # Remove nested list (dice_per_class) before writing — individual
        # dice_class_{i} columns are already in val_metrics
        csv_row.pop("dice_per_class", None)
        append_csv_row(csv_path, csv_row)

        # Memory housekeeping
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()

    print(f"\nTraining complete.  Best mean SSIM: {best_ssim:.4f}  "
          f"Best mean Dice: {best_dice:.4f}")
    print(f"CSV log : {csv_path}")
    print(f"Checkpoints: {ckpt_dir}")


if __name__ == "__main__":
    import sys, traceback
    try:
        main()
    except Exception:
        # Print full traceback and flush immediately so it appears in kubectl
        # logs even when stdout/stderr are piped (buffered by default).
        traceback.print_exc()
        sys.stdout.flush()
        sys.stderr.flush()
        sys.exit(1)

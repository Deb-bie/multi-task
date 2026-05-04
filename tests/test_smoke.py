"""
tests/test_smoke.py — Smoke tests for the multi-task CycleGAN codebase.

These tests verify:
  1. Model forward pass produces expected output shapes.
  2. All loss modules are callable and return finite scalars.
  3. Metric functions produce values in expected ranges.
  4. Dataset normalisation helpers are invertible / correct.
  5. ImageBuffer push/pop preserves tensor shape.
  6. Validate function returns the expected metric keys.

Run with:
    pytest tests/test_smoke.py -v

No GPU or real data required — all tests use random tensors.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
import torch
import torch.nn.functional as F

# ── Make the project root importable ────────────────────────────────────────
_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

B  = 2          # batch size
C  = 3          # image channels (2.5D)
H  = 256
W  = 256
NC = 6          # num seg classes

DEVICE = torch.device("cpu")


@pytest.fixture(scope="module")
def model():
    from src.models.multitask_cyclegan import MultitaskCycleGAN
    m = MultitaskCycleGAN(
        in_channels=C,
        out_channels=C,
        base_filters=16,        # small for test speed
        n_res_blocks_enc=2,
        num_seg_classes=NC,
        disc_features=16,
    )
    m.eval()
    return m


@pytest.fixture(scope="module")
def real_batch():
    torch.manual_seed(0)
    return {
        "mr":   torch.randn(B, C, H, W),
        "ct":   torch.randn(B, C, H, W),
        "mask": (torch.rand(B, 1, H, W) > 0.3).float(),
        "seg":  torch.randint(0, NC, (B, H, W)),
    }


# ---------------------------------------------------------------------------
# 1. Model forward pass — output shapes
# ---------------------------------------------------------------------------

class TestModelForward:
    def test_fake_ct_shape(self, model, real_batch):
        with torch.no_grad():
            outs = model(real_batch["mr"], real_batch["ct"])
        assert outs["fake_CT"].shape == (B, C, H, W), "fake_CT shape mismatch"

    def test_fake_mr_shape(self, model, real_batch):
        with torch.no_grad():
            outs = model(real_batch["mr"], real_batch["ct"])
        assert outs["fake_MR"].shape == (B, C, H, W), "fake_MR shape mismatch"

    def test_cycle_shapes(self, model, real_batch):
        with torch.no_grad():
            outs = model(real_batch["mr"], real_batch["ct"])
        assert outs["cycle_MR"].shape == (B, C, H, W)
        assert outs["cycle_CT"].shape == (B, C, H, W)

    def test_identity_shapes(self, model, real_batch):
        with torch.no_grad():
            outs = model(real_batch["mr"], real_batch["ct"])
        assert outs["idt_CT"].shape == (B, C, H, W)
        assert outs["idt_MR"].shape == (B, C, H, W)

    def test_seg_logit_shapes(self, model, real_batch):
        with torch.no_grad():
            outs = model(real_batch["mr"], real_batch["ct"])
        for key in ("seg_real_MR", "seg_fake_CT", "seg_real_CT", "seg_fake_MR"):
            assert outs[key].shape == (B, NC, H, W), f"{key} shape mismatch"

    def test_seg_aux_empty_in_eval(self, model, real_batch):
        """Auxiliary logits list must be empty in eval mode."""
        model.eval()
        with torch.no_grad():
            outs = model(real_batch["mr"], real_batch["ct"])
        assert outs["seg_aux_real_MR"] == [], "aux list should be empty in eval"

    def test_seg_aux_nonempty_in_train(self, model, real_batch):
        """Auxiliary logits list must be non-empty in training mode."""
        model.train()
        with torch.no_grad():
            outs = model(real_batch["mr"], real_batch["ct"])
        assert len(outs["seg_aux_real_MR"]) == 2, "expected 2 aux heads in train mode"
        model.eval()   # restore

    def test_output_range_tanh(self, model, real_batch):
        """Synthesised images must lie in [-1, 1] (Tanh output)."""
        with torch.no_grad():
            outs = model(real_batch["mr"], real_batch["ct"])
        for key in ("fake_CT", "fake_MR"):
            assert outs[key].min() >= -1.0 - 1e-5
            assert outs[key].max() <=  1.0 + 1e-5


# ---------------------------------------------------------------------------
# 2. Loss modules
# ---------------------------------------------------------------------------

class TestLossModules:
    def test_gan_loss_real(self):
        from src.losses.gan_loss import GANLoss
        criterion = GANLoss()
        pred = torch.randn(B, 1, 30, 30)
        loss = criterion(pred, is_real=True)
        assert loss.item() >= 0.0 and torch.isfinite(loss)

    def test_gan_loss_fake(self):
        from src.losses.gan_loss import GANLoss
        criterion = GANLoss()
        pred = torch.randn(B, 1, 30, 30)
        loss = criterion(pred, is_real=False)
        assert loss.item() >= 0.0 and torch.isfinite(loss)

    def test_dice_loss(self):
        from src.losses.seg_loss import DiceLoss
        criterion = DiceLoss()
        logits  = torch.randn(B, NC, H, W)
        targets = torch.randint(0, NC, (B, H, W))
        loss = criterion(logits, targets)
        assert 0.0 <= loss.item() <= 2.0 and torch.isfinite(loss)

    def test_seg_loss_uniform_weights(self):
        from src.losses.seg_loss import SegLoss
        criterion = SegLoss()
        logits  = torch.randn(B, NC, H, W)
        targets = torch.randint(0, NC, (B, H, W))
        mask    = (torch.rand(B, H, W) > 0.3).float()
        loss = criterion(logits, targets, mask=mask)
        assert torch.isfinite(loss) and loss.item() >= 0.0

    def test_seg_loss_class_weights(self):
        from src.losses.seg_loss import SegLoss
        w = torch.tensor([0.05, 0.3, 1.0, 1.0, 0.7, 6.0])
        criterion = SegLoss(class_weights=w)
        logits  = torch.randn(B, NC, H, W)
        targets = torch.randint(0, NC, (B, H, W))
        loss = criterion(logits, targets)
        assert torch.isfinite(loss) and loss.item() >= 0.0

    def test_anatomy_loss(self):
        from src.losses.anatomy_loss import AnatomyConsistencyLoss
        criterion = AnatomyConsistencyLoss()
        seg_fake   = torch.randn(B, NC, H, W)
        seg_source = torch.randn(B, NC, H, W)
        loss = criterion(seg_fake, seg_source)
        assert torch.isfinite(loss) and loss.item() >= 0.0

    def test_perceptual_loss(self):
        from src.losses.perceptual_loss import PerceptualLoss
        criterion = PerceptualLoss()
        pred   = torch.rand(B, C, H, W) * 2 - 1   # [-1, 1]
        target = torch.rand(B, C, H, W) * 2 - 1
        loss = criterion(pred, target)
        assert torch.isfinite(loss) and loss.item() >= 0.0


# ---------------------------------------------------------------------------
# 3. Metric functions
# ---------------------------------------------------------------------------

class TestMetrics:
    def test_masked_ssim_range(self):
        from src.metrics import masked_ssim
        pred   = torch.rand(B, 1, H, W)
        target = torch.rand(B, 1, H, W)
        mask   = torch.ones(B, 1, H, W)
        val = masked_ssim(pred, target, mask, data_range=1.0)
        assert 0.0 <= val <= 1.0, f"SSIM out of range: {val}"

    def test_masked_mae_perfect(self):
        from src.metrics import masked_mae
        img  = torch.rand(B, 1, H, W)
        mask = torch.ones(B, 1, H, W)
        val = masked_mae(img, img, mask, is_ct=False)
        assert abs(val) < 1e-5, f"MAE for identical images should be 0, got {val}"

    def test_masked_mae_hu_scale(self):
        from src.metrics import masked_mae
        # Normalised difference of 1.0 → HU difference of 2000
        pred   = torch.ones(B, 1, H, W)
        target = torch.zeros(B, 1, H, W)
        mask   = torch.ones(B, 1, H, W)
        val = masked_mae(pred, target, mask, is_ct=True)
        assert abs(val - 2000.0) < 1.0, f"Expected ~2000 HU, got {val}"

    def test_dice_per_class_length(self):
        from src.metrics import dice_per_class
        logits  = torch.randn(B, NC, H, W)
        targets = torch.randint(0, NC, (B, H, W))
        scores = dice_per_class(logits, targets, num_classes=NC)
        assert len(scores) == NC - 1, "Expected NC-1 scores (bg excluded)"

    def test_dice_perfect_prediction(self):
        from src.metrics import dice_per_class
        targets = torch.zeros(1, H, W, dtype=torch.long)
        targets[0, :H//2, :] = 1   # half the image is class 1
        # Perfect logits: high score for each pixel's true class
        logits = torch.full((1, NC, H, W), -10.0)
        logits[0, 0, H//2:, :] = 10.0
        logits[0, 1, :H//2, :] = 10.0
        scores = dice_per_class(logits, targets, num_classes=NC)
        assert abs(scores[0] - 1.0) < 1e-3, f"Perfect Dice should be 1.0, got {scores[0]}"


# ---------------------------------------------------------------------------
# 4. CT normalisation round-trip
# ---------------------------------------------------------------------------

class TestNormalisation:
    def test_ct_norm_range(self):
        from src.dataset import _normalize_ct
        # Input HU spanning [-1000, 3000]
        arr = np.linspace(-1000, 3000, 100, dtype=np.float32).reshape(1, 10, 10)
        norm = _normalize_ct(arr)
        assert norm.min() >= -1.0 - 1e-5
        assert norm.max() <=  1.0 + 1e-5

    def test_ct_norm_minus1000(self):
        from src.dataset import _normalize_ct
        arr = np.full((1, 4, 4), -1000.0, dtype=np.float32)
        assert np.allclose(_normalize_ct(arr), -1.0)

    def test_ct_norm_3000(self):
        from src.dataset import _normalize_ct
        arr = np.full((1, 4, 4), 3000.0, dtype=np.float32)
        assert np.allclose(_normalize_ct(arr), 1.0)

    def test_mr_norm_range(self):
        from src.dataset import _normalize_mr
        arr  = np.abs(np.random.randn(10, 64, 64).astype(np.float32)) * 500
        mask = np.ones((10, 64, 64), dtype=np.float32)
        norm = _normalize_mr(arr, mask)
        assert norm.min() >= -1.0 - 1e-5
        assert norm.max() <=  1.0 + 1e-5


# ---------------------------------------------------------------------------
# 5. ImageBuffer
# ---------------------------------------------------------------------------

class TestImageBuffer:
    def test_push_pop_shape(self):
        from src.models.utils import ImageBuffer
        buf = ImageBuffer(max_size=10)
        imgs = torch.randn(B, C, H, W)
        out  = buf.push_and_pop(imgs)
        assert out.shape == imgs.shape

    def test_buffer_fills(self):
        from src.models.utils import ImageBuffer
        buf = ImageBuffer(max_size=5)
        for _ in range(10):
            imgs = torch.randn(1, C, 8, 8)
            buf.push_and_pop(imgs)
        assert len(buf.buffer) == 5


# ---------------------------------------------------------------------------
# 6. Validate returns expected keys
# ---------------------------------------------------------------------------

class TestValidate:
    def test_validate_keys(self, model, real_batch):
        from train.validate import validate
        from torch.utils.data import DataLoader, TensorDataset

        # Build a tiny DataLoader from the fixture batch
        ds = TensorDataset(
            real_batch["mr"], real_batch["ct"],
            real_batch["mask"], real_batch["seg"],
        )

        class _DictLoader:
            """Wraps TensorDataset to yield dicts (matching validate()'s expectation)."""
            def __init__(self, batch):
                self.batch = batch
            def __iter__(self):
                yield self.batch
            def __len__(self):
                return 1

        metrics = validate(model, _DictLoader(real_batch), DEVICE, num_classes=NC)

        required = {
            "mr2ct_ssim", "ct2mr_ssim",
            "mr2ct_mae",  "ct2mr_mae",
            "mr2ct_psnr", "ct2mr_psnr",
            "mean_dice",
            "ct_seg_mean_dice",
            "dice_class_1",
            "ct_seg_class_1",
        }
        for key in required:
            assert key in metrics, f"validate() missing key: {key}"

    def test_validate_dice_nonnegative(self, model, real_batch):
        from train.validate import validate

        class _DictLoader:
            def __init__(self, b): self.b = b
            def __iter__(self): yield self.b
            def __len__(self): return 1

        metrics = validate(model, _DictLoader(real_batch), DEVICE, num_classes=NC)
        assert metrics["mean_dice"] >= 0.0
        assert metrics["ct_seg_mean_dice"] >= 0.0

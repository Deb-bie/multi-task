"""
LSGAN loss wrapper for the multi-task CycleGAN.

Least-squares GAN (Mao et al., 2017) replaces the binary cross-entropy GAN
objective with an MSE loss against one-hot targets.  This provides smoother
gradients, avoids vanishing gradients in the discriminator, and is more
stable than vanilla GAN loss in medical image synthesis tasks.

The loss is used for both the generator step (labels = 1) and the
discriminator step (real labels = 1, fake labels = 0).
"""

from __future__ import annotations

import torch
import torch.nn as nn


class GANLoss(nn.Module):
    """LSGAN MSE loss with dynamic target creation.

    Targets are allocated on the same device and dtype as the discriminator
    prediction, so no manual device management is required at call sites.

    Args:
        real_label:  Scalar label used for real images (default 1.0).
        fake_label:  Scalar label used for fake images (default 0.0).

    Example::

        criterion = GANLoss()
        # Generator step: fool discriminator (predict real for fakes)
        loss_G = criterion(D(fake), is_real=True)
        # Discriminator step
        loss_D = 0.5 * (criterion(D(real), True) + criterion(D(fake.detach()), False))
    """

    def __init__(self, real_label: float = 1.0, fake_label: float = 0.0) -> None:
        super().__init__()
        self.register_buffer("real_label", torch.tensor(real_label))
        self.register_buffer("fake_label", torch.tensor(fake_label))
        self.loss = nn.MSELoss()

    def _get_target(self, prediction: torch.Tensor, is_real: bool) -> torch.Tensor:
        """Return a target tensor matching *prediction* shape, device, and dtype."""
        label = self.real_label if is_real else self.fake_label
        return label.expand_as(prediction)

    def forward(self, prediction: torch.Tensor, is_real: bool) -> torch.Tensor:
        """Compute LSGAN loss.

        Args:
            prediction: Raw discriminator output of shape ``(B, 1, H', W')``
                        (PatchGAN) or ``(B, 1)`` (global).  No sigmoid
                        should be applied beforehand.
            is_real:    If ``True``, target is ``real_label``; otherwise
                        ``fake_label``.

        Returns:
            Scalar MSE loss.
        """
        target = self._get_target(prediction, is_real)
        return self.loss(prediction, target)

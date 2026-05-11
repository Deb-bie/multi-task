"""
Shared model utilities for the multi-task CycleGAN codebase.

Provides:
    init_weights  – Initialises Conv / ConvTranspose / Norm layers with a
                    normal distribution (mean=0, std=0.02), matching the
                    original paired CycleGAN convention.
    ImageBuffer   – Replay buffer (size 50) for discriminator training
                    stability; randomly swaps incoming images with stored
                    ones to decorrelate consecutive batches.
"""

from __future__ import annotations

import random
import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# Weight initialisation
# ---------------------------------------------------------------------------

def init_weights(model: nn.Module, mean: float = 0.0, std: float = 0.02) -> None:
    """Initialise all Conv / ConvTranspose / Norm parameters in *model*.

    Convolutional weights are drawn from N(mean, std²).
    Normalisation weights (scale) are drawn from N(1, std²) to start near
    the identity mapping; biases are zeroed.

    Args:
        model: Any ``nn.Module``; applied recursively to every sub-module.
        mean:  Mean of the normal distribution for Conv weights (default 0).
        std:   Standard deviation for all weight initialisations (default 0.02).
    """
    for m in model.modules():
        if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
            nn.init.normal_(m.weight.data, mean=mean, std=std)
            if m.bias is not None:
                nn.init.constant_(m.bias.data, 0.0)
        elif isinstance(m, (nn.BatchNorm2d, nn.InstanceNorm2d)):
            if m.weight is not None:
                nn.init.normal_(m.weight.data, mean=1.0, std=std)
            if m.bias is not None:
                nn.init.constant_(m.bias.data, 0.0)


# ---------------------------------------------------------------------------
# Image replay buffer
# ---------------------------------------------------------------------------

class ImageBuffer:
    """Discriminator replay buffer (Shrivastava et al., 2017).

    Stores up to *max_size* previously generated images.  On each call to
    :meth:`push_and_pop`, a random half of the returned batch comes from the
    buffer (replacing a stored image) and the other half is the current batch.
    This decorrelates consecutive discriminator updates and stabilises GAN
    training.

    Args:
        max_size: Maximum number of images to keep in the buffer (default 50).
    """

    def __init__(self, max_size: int = 50) -> None:
        self.max_size = max_size
        self.buffer: list[torch.Tensor] = []

    def push_and_pop(self, images: torch.Tensor) -> torch.Tensor:
        """Push *images* into the buffer and return a mixed batch.

        Args:
            images: Tensor of shape ``(B, C, H, W)``.

        Returns:
            Tensor of shape ``(B, C, H, W)`` mixing current and buffered images.
        """
        result: list[torch.Tensor] = []
        for image in images:
            image = image.unsqueeze(0)  # (1, C, H, W)
            if len(self.buffer) < self.max_size:
                self.buffer.append(image)
                result.append(image)
            else:
                if random.random() > 0.5:
                    idx = random.randint(0, self.max_size - 1)
                    tmp = self.buffer[idx].clone()
                    self.buffer[idx] = image
                    result.append(tmp)
                else:
                    result.append(image)
        return torch.cat(result, dim=0)

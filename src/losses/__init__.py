"""src.losses – loss function package for multi-task CycleGAN."""
from .gan_loss       import GANLoss
from .perceptual_loss import PerceptualLoss
from .seg_loss       import DiceLoss, SegLoss
from .anatomy_loss   import AnatomyConsistencyLoss

__all__ = ["GANLoss", "PerceptualLoss", "DiceLoss", "SegLoss", "AnatomyConsistencyLoss"]

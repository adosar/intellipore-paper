r"""
This module contains a copy of the model architecture from the intellipore
branch of the RetNeXt repository: https://github.com/adosar/retnext

The code is included here so that the architecture can be inspected
without requiring RetNeXt to be installed as an external dependency.

The source code was copied from:
https://github.com/adosar/retnext/blob/intellipore/src/retnext/modules.py
"""

import torch
from torch import nn


def conv3d_block(in_channels: int, out_channels: int, **kwargs):
    r"""
    Return a block of the form Conv -> BatchNorm -> ReLU.

    Examples
    --------
    >>> block = conv3d_block(4, 8, kernel_size=2, bias=False, padding_mode='circular')
    >>> block[0].kernel_size
    (2, 2, 2)
    >>> block[0].padding_mode
    'circular'
    >>> block[0].bias
    """
    return nn.Sequential(
            nn.Conv3d(in_channels, out_channels, **kwargs),
            nn.BatchNorm3d(out_channels),
            nn.ReLU()
            )


class IntelliPore(nn.Module):
    r"""Backbone for multi-task, multi-domain pretraining on porous materials."""
    def __init__(
            self,
            in_channels: int = 1,
            n_outputs: int | None = None,
            ):
        super().__init__()
        self.backbone = nn.Sequential(
                conv3d_block(in_channels, 32, kernel_size=3, padding='same'),
                conv3d_block(32, 32, kernel_size=3, padding='same'),
                nn.MaxPool3d(kernel_size=2), # 1st pooling layer
                conv3d_block(32, 64, kernel_size=3, padding='same'),
                conv3d_block(64, 64, kernel_size=3, padding='same'),
                nn.MaxPool3d(kernel_size=2),  # 2nd pooling layer
                conv3d_block(64, 128, kernel_size=3),
                conv3d_block(128, 128, kernel_size=3),
                nn.AdaptiveAvgPool3d(1),
                nn.Flatten()
                )
        self.head = torch.nn.Identity() if n_outputs is None else nn.Linear(128, n_outputs)

    def forward(self, x):
        return self.head(self.backbone(x))

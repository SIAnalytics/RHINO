# Copyright (c) SI Analytics. All rights reserved.
# Licensed under the CC BY-NC 4.0 License. See LICENSE file in the project root for full license information.
#
# Copyright (c) OpenMMLab. All rights reserved.
# Licensed under the Apache License, Version 2.0. See LICENSE file in the mmrotate repository for full license information.
from torch import nn


class RotationInvariantPooling(nn.Module):
    """Rotating invariant pooling module.

    Args:
        nInputPlane (int): The number of Input plane.
        nOrientation (int, optional): The number of oriented channels.
    """

    def __init__(self, nInputPlane, nOrientation=8):
        super(RotationInvariantPooling, self).__init__()
        self.nInputPlane = nInputPlane
        self.nOrientation = nOrientation

    def forward(self, x):
        """Forward function."""
        N, c, h, w = x.size()
        x = x.view(N, -1, self.nOrientation, h, w)
        x, _ = x.max(dim=2, keepdim=False)
        return x

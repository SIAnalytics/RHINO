# Copyright (c) SI Analytics. All rights reserved.
# Licensed under the CC BY-NC 4.0 License. See LICENSE file in the project root for full license information.
#
# Copyright (c) OpenMMLab. All rights reserved.
# Licensed under the Apache License, Version 2.0. See LICENSE file in the mmrotate repository for full license information.
from .bbox_overlaps import fake_rbbox_overlaps, rbbox_overlaps
from .box_converters import (hbox2qbox, hbox2rbox, qbox2hbox, qbox2rbox,
                             rbox2hbox, rbox2qbox)
from .quadri_boxes import QuadriBoxes
from .rotated_boxes import RotatedBoxes
from .transforms import distance2obb, gaussian2bbox, gt2gaussian, norm_angle

__all__ = [
    'QuadriBoxes', 'RotatedBoxes', 'hbox2rbox', 'hbox2qbox', 'rbox2hbox',
    'rbox2qbox', 'qbox2hbox', 'qbox2rbox', 'gaussian2bbox', 'gt2gaussian',
    'norm_angle', 'rbbox_overlaps', 'fake_rbbox_overlaps', 'distance2obb'
]

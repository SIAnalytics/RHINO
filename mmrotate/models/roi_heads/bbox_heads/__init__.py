# Copyright (c) SI Analytics. All rights reserved.
# Licensed under the CC BY-NC 4.0 License. See LICENSE file in the project root for full license information.
#
# Copyright (c) OpenMMLab. All rights reserved.
# Licensed under the Apache License, Version 2.0. See LICENSE file in the mmrotate repository for full license information.
from .convfc_rbbox_head import RotatedShared2FCBBoxHead
from .gv_bbox_head import GVBBoxHead

__all__ = ['RotatedShared2FCBBoxHead', 'GVBBoxHead']

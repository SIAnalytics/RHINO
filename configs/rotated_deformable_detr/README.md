# Rotated Deformable DETR

> [Deformable DETR: Deformable Transformers for End-to-End Object Detection](https://arxiv.org/abs/2010.04159)

<!-- [ALGORITHM] -->

## Abstract

DETR has been recently proposed to eliminate the need for many hand-designed components in object detection while demonstrating good performance. However, it suffers from slow convergence and limited feature spatial resolution, due to the limitation of Transformer attention modules in processing image feature maps. To mitigate these issues, we proposed Deformable DETR, whose attention modules only attend to a small set of key sampling points around a reference. Deformable DETR can achieve better performance than DETR (especially on small objects) with 10 times less training epochs. Extensive experiments on the COCO benchmark demonstrate the effectiveness of our approach.

<div align=center>
<img src=""/>
</div>

## Results and Models

DOTA1.0

|         Backbone         |  mAP  | Angle | lr schd | Mem (GB) | Inf Time (fps) |  Aug  | Batch Size |                                                                        Configs                                                                         |         Download         |
| :----------------------: | :---: | :---: | :-----: | :------: | :------------: | :---: | :--------: | :----------------------------------------------------------------------------------------------------------------------------------------------------: | :----------------------: |
| ResNet50 (1024,1024,200) | 68.50 | le90  |   1x    |          |                |   -   |     32     |                 [rotated_deformable-detr_r50_16xb2-50e_dota](../rotated_deformable_detr/rotated_deformable-detr_r50_16xb2-50e_dota.py)                 | [model](<>) \| [log](<>) |
| ResNet50 (1024,1024,200) | 73.38 | le90  |   1x    |          |                | MS+RR |     32     |           [rotated_deformable-detr_r50_16xb2-50e_dota_ms_rr](../rotated_deformable_detr/rotated_deformable-detr_r50_16xb2-50e_dota_ms_rr.py)           | [model](<>) \| [log](<>) |
| ResNet50 (1024,1024,200) | 70.33 | le90  |   1x    |          |                |   -   |     32     | [rotated_deformable-detr_refine_twostage_r50_16xb2-50e_dota](../rotated_deformable_detr/rotated_deformable-detr_refine_twostage_r50_16xb2-50e_dota.py) | [model](<>) \| [log](<>) |

Notes:

- `MS` means multiple scale image split.
- `RR` means random rotation.

# Copyright (c) OpenMMLab. All rights reserved.
import torch
from mmdet.models.dense_heads import DeformableDETRHead
from mmdet.models.utils import multi_apply
from mmdet.utils import ConfigType, InstanceList, OptInstanceList, reduce_mean
from mmengine.structures import InstanceData
from torch import Tensor
from typing import Dict, List, Tuple

from mmrotate.registry import MODELS, TASK_UTILS
from .rhino_head import RHINOHead


@MODELS.register_module()
class RHINOPositiveHungarianHead(RHINOHead):
    """RHINOPositiveHungarianHead.

    Origin loss for denosing query
        - neg dn : reconstruction loss (classification(background))
        - pos dn :
            - for each gt,
              apply reconstruction loss (classification, regression)
    Postivie hungarian loss for denosing query
        - neg dn : reconstruction loss (classification(background))
        - pos dn :
            - hungarian matching with object query
            - if hungarian matching result same with for each gt
                apply reconstruction loss (classification, regression)
              else apply reconstruction loss (classification(background))
    """

    def __init__(self,
                 *args,
                 train_cfg: ConfigType = dict(
                     assigner=dict(
                         type='mmdet.HungarianAssigner',
                         match_costs=[
                             dict(type='mmdet.FocalLossCost', weight=2.0),
                             dict(
                                 type='CenterL1Cost',
                                 weight=5.0,
                                 box_format='xywha'),
                             dict(
                                 type='GDCost',
                                 loss_type='kld',
                                 fun='log1p',
                                 tau=1,
                                 sqrt=False,
                                 weight=2.0)
                         ]),
                     dn_assigner=dict(
                         type='DNGroupHungarianAssigner',
                         match_costs=[
                             dict(type='mmdet.FocalLossCost', weight=2.0),
                             dict(
                                 type='CenterL1Cost',
                                 weight=5.0,
                                 box_format='xywha'),
                             dict(
                                 type='GDCost',
                                 loss_type='kld',
                                 fun='log1p',
                                 tau=1,
                                 sqrt=False,
                                 weight=2.0)
                         ])),
                 **kwargs) -> None:
        super().__init__(*args, train_cfg=train_cfg, **kwargs)
        dn_assigner = train_cfg['dn_assigner']
        self.dn_assigner = TASK_UTILS.build(dn_assigner)

    def loss_by_feat(
        self,
        all_layers_cls_scores: Tensor,
        all_layers_bbox_preds: Tensor,
        enc_cls_scores: Tensor,
        enc_bbox_preds: Tensor,
        batch_gt_instances: InstanceList,
        batch_img_metas: List[dict],
        dn_meta: Dict[str, int],
        batch_gt_instances_ignore: OptInstanceList = None
    ) -> Dict[str, Tensor]:
        """Loss function.

        Args:
            all_layers_cls_scores (Tensor): Classification scores of all
                decoder layers, has shape (num_decoder_layers, bs,
                num_queries_total, cls_out_channels), where
                `num_queries_total` is the sum of `num_denoising_queries`
                and `num_matching_queries`.
            all_layers_bbox_preds (Tensor): Regression outputs of all decoder
                layers. Each is a 5D-tensor with normalized coordinate format
                (cx, cy, w, h, a) and has shape (num_decoder_layers, bs,
                num_queries_total, 5).
            enc_cls_scores (Tensor): The score of each point on encode
                feature map, has shape (bs, num_feat_points, cls_out_channels).
            enc_bbox_preds (Tensor): The proposal generate from the encode
                feature map, has shape (bs, num_feat_points, 5) with the last
                dimension arranged as (cx, cy, w, h, a).
            batch_gt_instances (list[:obj:`InstanceData`]): Batch of
                gt_instance. It usually includes ``bboxes`` and ``labels``
                attributes.
            batch_img_metas (list[dict]): Meta information of each image, e.g.,
                image size, scaling factor, etc.
            dn_meta (Dict[str, int]): The dictionary saves information about
                group collation, including 'num_denoising_queries' and
                'num_denoising_groups'. It will be used for split outputs of
                denoising and matching parts and loss calculation.
            batch_gt_instances_ignore (list[:obj:`InstanceData`], optional):
                Batch of gt_instances_ignore. It includes ``bboxes`` attribute
                data that is ignored during training and testing.
                Defaults to None.

        Returns:
            dict[str, Tensor]: A dictionary of loss components.
        """
        # extract denoising and matching part of outputs
        (all_layers_matching_cls_scores, all_layers_matching_bbox_preds,
         all_layers_denoising_cls_scores, all_layers_denoising_bbox_preds) = \
            self.split_outputs(
                all_layers_cls_scores, all_layers_bbox_preds, dn_meta)
        loss_dict = super(DeformableDETRHead, self).loss_by_feat(
            all_layers_matching_cls_scores, all_layers_matching_bbox_preds,
            batch_gt_instances, batch_img_metas, batch_gt_instances_ignore)
        # NOTE DETRHead.loss_by_feat but not DeformableDETRHead.loss_by_feat
        # is called, because the encoder loss calculations are different
        # between DINO and DeformableDETR.

        # loss of proposal generated from encode feature map.
        if enc_cls_scores is not None:
            # NOTE The enc_loss calculation of the DINO is
            # different from that of Deformable DETR.
            enc_loss_cls, enc_losses_bbox, enc_losses_iou = \
                self.loss_by_feat_single(
                    enc_cls_scores, enc_bbox_preds,
                    batch_gt_instances=batch_gt_instances,
                    batch_img_metas=batch_img_metas)
            loss_dict['enc_loss_cls'] = enc_loss_cls
            loss_dict['enc_loss_bbox'] = enc_losses_bbox
            loss_dict['enc_loss_iou'] = enc_losses_iou

        if all_layers_denoising_cls_scores is not None:
            # calculate denoising loss from all decoder layers
            dn_losses_cls, dn_losses_bbox, dn_losses_iou = self.loss_dn(
                all_layers_denoising_cls_scores,
                all_layers_denoising_bbox_preds,
                all_layers_matching_cls_scores,
                all_layers_matching_bbox_preds,
                batch_gt_instances=batch_gt_instances,
                batch_img_metas=batch_img_metas,
                dn_meta=dn_meta)
            # collate denoising loss
            loss_dict['dn_loss_cls'] = dn_losses_cls[-1]
            loss_dict['dn_loss_bbox'] = dn_losses_bbox[-1]
            loss_dict['dn_loss_iou'] = dn_losses_iou[-1]
            for num_dec_layer, (loss_cls_i, loss_bbox_i, loss_iou_i) in \
                    enumerate(zip(dn_losses_cls[:-1], dn_losses_bbox[:-1],
                                  dn_losses_iou[:-1])):
                loss_dict[f'd{num_dec_layer}.dn_loss_cls'] = loss_cls_i
                loss_dict[f'd{num_dec_layer}.dn_loss_bbox'] = loss_bbox_i
                loss_dict[f'd{num_dec_layer}.dn_loss_iou'] = loss_iou_i
        return loss_dict

    def loss_dn(self, all_layers_denoising_cls_scores: Tensor,
                all_layers_denoising_bbox_preds: Tensor,
                all_layers_matching_cls_scores: Tensor,
                all_layers_matching_bbox_preds: Tensor,
                batch_gt_instances: InstanceList, batch_img_metas: List[dict],
                dn_meta: Dict[str, int]) -> Tuple[List[Tensor]]:
        """Calculate denoising loss.

        Args:
            all_layers_denoising_cls_scores (Tensor): Classification scores of
                all decoder layers in denoising part, has shape (
                num_decoder_layers, bs, num_denoising_queries,
                cls_out_channels).
            all_layers_denoising_bbox_preds (Tensor): Regression outputs of all
                decoder layers in denoising part. Each is a 5D-tensor with
                normalized coordinate format (cx, cy, w, h, a) and has shape
                (num_decoder_layers, bs, num_denoising_queries, 5).
            all_layers_matching_cls_scores (Tensor): Classification scores of
                all decoder layers in matching part, has shape (
                num_decoder_layers, bs, num_queries,
                cls_out_channels).
            all_layers_matching_bbox_preds (Tensor): Regression outputs of all
                decoder layers in matching part. Each is a 5D-tensor with
                normalized coordinate format (cx, cy, w, h, a) and has shape
                (num_decoder_layers, bs, num_queries, 5).
            batch_gt_instances (list[:obj:`InstanceData`]): Batch of
                gt_instance. It usually includes ``bboxes`` and ``labels``
                attributes.
            batch_img_metas (list[dict]): Meta information of each image, e.g.,
                image size, scaling factor, etc.
            dn_meta (Dict[str, int]): The dictionary saves information about
              group collation, including 'num_denoising_queries' and
              'num_denoising_groups'. It will be used for split outputs of
              denoising and matching parts and loss calculation.

        Returns:
            Tuple[List[Tensor]]: The loss_dn_cls, loss_dn_bbox, and loss_dn_iou
            of each decoder layers.
        """
        return multi_apply(
            self._loss_dn_single,
            all_layers_denoising_cls_scores,
            all_layers_denoising_bbox_preds,
            all_layers_matching_cls_scores,
            all_layers_matching_bbox_preds,
            batch_gt_instances=batch_gt_instances,
            batch_img_metas=batch_img_metas,
            dn_meta=dn_meta)

    def _loss_dn_single(self, dn_cls_scores: Tensor, dn_bbox_preds: Tensor,
                        mtc_cls_scores: Tensor, mtc_bbox_preds: Tensor,
                        batch_gt_instances: InstanceList,
                        batch_img_metas: List[dict],
                        dn_meta: Dict[str, int]) -> Tuple[Tensor]:
        """Denoising loss for outputs from a single decoder layer.

        Args:
            dn_cls_scores (Tensor): Classification scores of a single decoder
                layer in denoising part, has shape (bs, num_denoising_queries,
                cls_out_channels).
            dn_bbox_preds (Tensor): Regression outputs of a single decoder
                layer in denoising part. Each is a 5D-tensor with normalized
                coordinate format (cx, cy, w, h, a) and has shape
                (bs, num_denoising_queries, 5).
            mtc_cls_scores (Tensor): Classification scores of a single decoder
                layer in matching part, has shape (bs, num_queries,
                cls_out_channels).
            mtc_cls_scores (Tensor): Regression outputs of a single decoder
                layer in matching part. Each is a 5D-tensor with normalized
                coordinate format (cx, cy, w, h, a) and has shape
                (bs, num_queries, 5).
            batch_gt_instances (list[:obj:`InstanceData`]): Batch of
                gt_instance. It usually includes ``bboxes`` and ``labels``
                attributes.
            batch_img_metas (list[dict]): Meta information of each image, e.g.,
                image size, scaling factor, etc.
            dn_meta (Dict[str, int]): The dictionary saves information about
              group collation, including 'num_denoising_queries' and
              'num_denoising_groups'. It will be used for split outputs of
              denoising and matching parts and loss calculation.

        Returns:
            Tuple[Tensor]: A tuple including `loss_cls`, `loss_box` and
            `loss_iou`.
        """
        cls_reg_targets = self.get_dn_targets(dn_cls_scores, dn_bbox_preds,
                                              mtc_cls_scores, mtc_bbox_preds,
                                              batch_gt_instances,
                                              batch_img_metas, dn_meta)
        (labels_list, label_weights_list, bbox_targets_list, bbox_weights_list,
         num_total_pos, num_total_neg) = cls_reg_targets
        labels = torch.cat(labels_list, 0)
        label_weights = torch.cat(label_weights_list, 0)
        bbox_targets = torch.cat(bbox_targets_list, 0)
        bbox_weights = torch.cat(bbox_weights_list, 0)

        # classification loss
        cls_scores = dn_cls_scores.reshape(-1, self.cls_out_channels)
        # construct weighted avg_factor to match with the official DETR repo
        cls_avg_factor = \
            num_total_pos * 1.0 + num_total_neg * self.bg_cls_weight
        if self.sync_cls_avg_factor:
            cls_avg_factor = reduce_mean(
                cls_scores.new_tensor([cls_avg_factor]))
        cls_avg_factor = max(cls_avg_factor, 1)

        if len(cls_scores) > 0:
            loss_cls = self.loss_cls(
                cls_scores, labels, label_weights, avg_factor=cls_avg_factor)
        else:
            loss_cls = torch.zeros(
                1, dtype=cls_scores.dtype, device=cls_scores.device)

        # Compute the average number of gt boxes across all gpus, for
        # normalization purposes
        num_total_pos = loss_cls.new_tensor([num_total_pos])
        num_total_pos = torch.clamp(reduce_mean(num_total_pos), min=1).item()

        # construct factors used for rescale bboxes
        factors = []
        for img_meta, bbox_pred in zip(batch_img_metas, dn_bbox_preds):
            img_h, img_w = img_meta['img_shape']
            factor = bbox_pred.new_tensor(
                [img_w, img_h, img_w, img_h,
                 self.angle_factor]).unsqueeze(0).repeat(bbox_pred.size(0), 1)
            factors.append(factor)
        factors = torch.cat(factors)

        # DETR regress the relative position of boxes (cxcywh) in the image,
        # thus the learning target is normalized by the image size. So here
        # we need to re-scale them for calculating IoU loss
        bbox_preds = dn_bbox_preds.reshape(-1, 5)
        bboxes = bbox_preds * factors
        bboxes_gt = bbox_targets * factors

        # regression IoU loss, defaultly GIoU loss
        loss_iou = self.loss_iou(
            bboxes, bboxes_gt, bbox_weights, avg_factor=num_total_pos)

        # regression L1 loss
        loss_bbox = self.loss_bbox(
            bbox_preds, bbox_targets, bbox_weights, avg_factor=num_total_pos)
        return loss_cls, loss_bbox, loss_iou

    def get_dn_targets(self, dn_cls_scores, dn_bbox_preds, mtc_cls_scores,
                       mtc_bbox_preds, batch_gt_instances: InstanceList,
                       batch_img_metas: dict, dn_meta: Dict[str,
                                                            int]) -> tuple:
        """Get targets in denoising part for a batch of images.

        Args:
            dn_cls_scores (Tensor): Classification scores of a single decoder
                layer in denoising part, has shape (bs, num_denoising_queries,
                cls_out_channels).
            dn_bbox_preds (Tensor): Regression outputs of a single decoder
                layer in denoising part. Each is a 5D-tensor with normalized
                coordinate format (cx, cy, w, h, a) and has shape
                (bs, num_denoising_queries, 5).
            mtc_cls_scores (Tensor): Classification scores of a single decoder
                layer in matching part, has shape (bs, num_queries,
                cls_out_channels).
            mtc_cls_scores (Tensor): Regression outputs of a single decoder
                layer in matching part. Each is a 5D-tensor with normalized
                coordinate format (cx, cy, w, h, a) and has shape
                (bs, num_queries, 5).
            batch_gt_instances (list[:obj:`InstanceData`]): Batch of
                gt_instance. It usually includes ``bboxes`` and ``labels``
                attributes.
            batch_img_metas (list[dict]): Meta information of each image, e.g.,
                image size, scaling factor, etc.
            dn_meta (Dict[str, int]): The dictionary saves information about
              group collation, including 'num_denoising_queries' and
              'num_denoising_groups'. It will be used for split outputs of
              denoising and matching parts and loss calculation.

        Returns:
            tuple: a tuple containing the following targets.

            - labels_list (list[Tensor]): Labels for all images.
            - label_weights_list (list[Tensor]): Label weights for all images.
            - bbox_targets_list (list[Tensor]): BBox targets for all images.
            - bbox_weights_list (list[Tensor]): BBox weights for all images.
            - num_total_pos (int): Number of positive samples in all images.
            - num_total_neg (int): Number of negative samples in all images.
        """
        num_imgs = dn_cls_scores.size(0)
        dn_cls_scores_list = [dn_cls_scores[i] for i in range(num_imgs)]
        dn_bbox_preds_list = [dn_bbox_preds[i] for i in range(num_imgs)]
        mtc_cls_scores_list = [mtc_cls_scores[i] for i in range(num_imgs)]
        mtc_bbox_preds_list = [mtc_bbox_preds[i] for i in range(num_imgs)]
        (labels_list, label_weights_list, bbox_targets_list, bbox_weights_list,
         pos_inds_list, neg_inds_list) = multi_apply(
             self._get_dn_targets_single,
             dn_cls_scores_list,
             dn_bbox_preds_list,
             mtc_cls_scores_list,
             mtc_bbox_preds_list,
             batch_gt_instances,
             batch_img_metas,
             dn_meta=dn_meta)
        num_total_pos = sum((inds.numel() for inds in pos_inds_list))
        num_total_neg = sum((inds.numel() for inds in neg_inds_list))
        return (labels_list, label_weights_list, bbox_targets_list,
                bbox_weights_list, num_total_pos, num_total_neg)

    def _get_dn_targets_single(self, dn_cls_score, dn_bbox_pred, mtc_cls_score,
                               mtc_bbox_pred, gt_instances: InstanceData,
                               img_meta: dict, dn_meta: Dict[str,
                                                             int]) -> tuple:
        """Get targets in denoising part for one image.

        Args:
            dn_cls_scores (Tensor): Classification scores of a single decoder
                layer in denoising part, has shape (num_denoising_queries,
                cls_out_channels).
            dn_bbox_preds (Tensor): Regression outputs of a single decoder
                layer in denoising part. Each is a 5D-tensor with normalized
                coordinate format (cx, cy, w, h, a) and has shape
                (num_denoising_queries, 5).
            mtc_cls_scores (Tensor): Classification scores of a single decoder
                layer in matching part, has shape (num_queries,
                cls_out_channels).
            mtc_cls_scores (Tensor): Regression outputs of a single decoder
                layer in matching part. Each is a 5D-tensor with normalized
                coordinate format (cx, cy, w, h, a) and has shape
                (num_queries, 5).
            gt_instances (:obj:`InstanceData`): Ground truth of instance
                annotations. It should includes ``bboxes`` and ``labels``
                attributes.
            img_meta (dict): Meta information for one image.
            dn_meta (Dict[str, int]): The dictionary saves information about
              group collation, including 'num_denoising_queries' and
              'num_denoising_groups'. It will be used for split outputs of
              denoising and matching parts and loss calculation.

        Returns:
            tuple[Tensor]: a tuple containing the following for one image.

            - labels (Tensor): Labels of each image.
            - label_weights (Tensor]): Label weights of each image.
            - bbox_targets (Tensor): BBox targets of each image.
            - bbox_weights (Tensor): BBox weights of each image.
            - pos_inds (Tensor): Sampled positive indices for each image.
            - neg_inds (Tensor): Sampled negative indices for each image.
        """
        img_h, img_w = img_meta['img_shape']
        # regularize GT boxes
        gt_instances.bboxes.regularize_boxes(**self.angle_cfg)
        gt_bboxes = gt_instances.bboxes.tensor
        factor = gt_bboxes.new_tensor(
            [img_w, img_h, img_w, img_h, self.angle_factor]).unsqueeze(0)

        gt_labels = gt_instances.labels

        num_groups = dn_meta['num_denoising_groups']
        num_denoising_queries = dn_meta['num_denoising_queries']
        num_queries_each_group = int(num_denoising_queries / num_groups)
        device = gt_bboxes.device
        if len(gt_labels) > 0:
            t = torch.arange(len(gt_labels), dtype=torch.long, device=device)
            t = t.unsqueeze(0).repeat(num_groups, 1)
            pos_assigned_gt_inds = t.flatten()
            pos_inds = torch.arange(
                num_groups, dtype=torch.long, device=device)
            pos_inds = pos_inds.unsqueeze(1) * num_queries_each_group + t
            pos_inds = pos_inds.flatten()

            mtc_bbox_pred_scaled = mtc_bbox_pred * factor
            pred_instances = InstanceData(
                scores=mtc_cls_score, bboxes=mtc_bbox_pred_scaled)

            pos_dn_cls_score = dn_cls_score[pos_inds]
            pos_dn_bbox_pred_scaled = dn_bbox_pred[pos_inds] * factor

            pos_dn_instances = InstanceData(
                scores=pos_dn_cls_score, bboxes=pos_dn_bbox_pred_scaled)

            dn_assign_result = self.dn_assigner.assign(
                pred_instances=pred_instances,
                dn_instances=pos_dn_instances,
                gt_instances=gt_instances,
                dn_meta=dn_meta,
                img_meta=img_meta,
            )
            group_assign = pos_assigned_gt_inds == (
                dn_assign_result.gt_inds - 1)
        else:
            group_assign = pos_inds = pos_assigned_gt_inds = \
                gt_bboxes.new_tensor([], dtype=torch.long)

        neg_inds = pos_inds + num_queries_each_group // 2

        # label targets
        labels = gt_bboxes.new_full((num_denoising_queries, ),
                                    self.num_classes,
                                    dtype=torch.long)
        neg_inds = torch.cat([pos_inds[~group_assign], neg_inds], dim=0)
        pos_inds = pos_inds[group_assign]
        pos_assigned_gt_inds = pos_assigned_gt_inds[group_assign]
        labels[pos_inds] = gt_labels[pos_assigned_gt_inds]
        label_weights = gt_bboxes.new_ones(num_denoising_queries)

        # bbox targets  4->5
        bbox_targets = torch.zeros(num_denoising_queries, 5, device=device)
        bbox_weights = torch.zeros(num_denoising_queries, 5, device=device)
        bbox_weights[pos_inds] = 1.0

        # DETR regress the relative position of boxes (cxcywh) in the image.
        # Thus the learning target should be normalized by the image size, also
        # the box format should be converted from defaultly x1y1x2y2 to cxcywh.

        gt_bboxes_normalized = gt_bboxes / factor
        gt_bboxes_targets = gt_bboxes_normalized
        bbox_targets[pos_inds] = gt_bboxes_targets[pos_assigned_gt_inds]

        return (labels, label_weights, bbox_targets, bbox_weights, pos_inds,
                neg_inds)

_base_ = './rhino-4scale_r50_2xb2-12e_dior.py'

max_epochs = 36
train_cfg = dict(
    type='EpochBasedTrainLoop', max_epochs=max_epochs, val_interval=12)
param_scheduler = [
    dict(
        type='MultiStepLR',
        begin=0,
        end=max_epochs,
        by_epoch=True,
        milestones=[27, 33],
        gamma=0.1)
]

costs = [
    dict(type='mmdet.FocalLossCost', weight=2.0),
    dict(type='HausdorffCost', weight=5.0, box_format='xywha'),
    dict(
        type='GDCost',
        loss_type='kld',
        fun='log1p',
        tau=1,
        sqrt=False,
        weight=5.0)
]

model = dict(
    version='v2',
    bbox_head=dict(
        type='RHINOPositiveHungarianClassificationHead',
        loss_iou=dict(
            type='GDLoss',
            loss_type='kld',
            fun='log1p',
            tau=1,
            sqrt=False,
            loss_weight=5.0)),
    dn_cfg=dict(group_cfg=dict(max_num_groups=30)),
    train_cfg=dict(
        assigner=dict(match_costs=costs),
        dn_assigner=dict(type='DNGroupHungarianAssigner', match_costs=costs),
    ))

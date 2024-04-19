_base_ = './rhino-4scale_swint_8xb2-36e_fix_dota.py'

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

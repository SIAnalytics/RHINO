_base_ = './rhino-4scale_r50_8xb2-12e_dotav2.py'

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

depths = [2, 2, 6, 2]

pretrained = 'https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_tiny_patch4_window7_224.pth'  # noqa
model = dict(
    version='v2',
    backbone=dict(
        _delete_=True,
        type='mmdet.SwinTransformer',
        embed_dims=96,
        depths=depths,
        num_heads=[3, 6, 12, 24],
        window_size=7,
        mlp_ratio=4,
        qkv_bias=True,
        qk_scale=None,
        drop_rate=0.,
        attn_drop_rate=0.,
        drop_path_rate=0.2,
        patch_norm=True,
        out_indices=(1, 2, 3),
        with_cp=False,
        convert_weights=True,
        init_cfg=dict(type='Pretrained', checkpoint=pretrained)),
    neck=dict(
        _delete_=True,
        type='mmdet.ChannelMapper',
        in_channels=[192, 384, 768],
        out_channels=256,
        num_outs=4),
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

# optimizer
optim_wrapper = dict(
    type='OptimWrapper',
    paramwise_cfg=dict(
        custom_keys={
            'absolute_pos_embed': dict(decay_mult=0.),
            'relative_position_bias_table': dict(decay_mult=0.),
            'norm': dict(decay_mult=0.)
        }))

_base_ = [
    './ocrnet_swin-t_512x1024_40k_dfuc2022.py'
]

norm_cfg = dict(type='SyncBN', requires_grad=True)
checkpoint_file = 'https://download.openmmlab.com/mmsegmentation/v0.5/pretrain/swin/swin_base_patch4_window12_384_22k_20220317-e5c09f74.pth'  # noqa
model = dict(
    backbone=dict(
        init_cfg=dict(type='Pretrained', checkpoint=checkpoint_file),
        embed_dims=128,
        depths=[2, 2, 18, 2],
        num_heads=[4, 8, 16, 32]),
        # [128, 256, 512, 1024],
      decode_head=[
        dict(
            type='FCNHead',
            in_channels=[128, 256, 512, 1024],
            channels=sum([128, 256, 512, 1024]),
            in_index=(0, 1, 2, 3),
            input_transform='resize_concat',
            kernel_size=1,
            num_convs=1,
            concat_input=False,
            dropout_ratio=-1,
            num_classes=2,
            norm_cfg=norm_cfg,
            align_corners=False,
            loss_decode=dict(
                type='CrossEntropyLoss', use_sigmoid=False, loss_weight=0.4)),
        dict(
            type='OCRHead',
            in_channels=[128, 256, 512, 1024],
            in_index=(0, 1, 2, 3),
            input_transform='resize_concat',
            channels=512,
            ocr_channels=256,
            dropout_ratio=-1,
            num_classes=2,
            norm_cfg=norm_cfg,
            align_corners=False,
            loss_decode=dict(
                type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0))
    ])
    

# By default, models are trained on 2 GPUs with 4 images per GPU
data = dict(
    samples_per_gpu=4,
    workers_per_gpu=4)

# runtime settings
runner = dict(type='IterBasedRunner', max_iters=80000)
evaluation = dict(interval=2000, metric='mDice', pre_eval=True)
checkpoint_config = dict(by_epoch=False, interval=2000)


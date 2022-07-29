_base_ = [
    '../_base_/models/ocrnet_convnext.py', '../_base_/datasets/dfuc.py',
    '../_base_/default_runtime.py', '../_base_/schedules/schedule_40k.py'
]
norm_cfg = dict(type='SyncBN', requires_grad=True)
checkpoint_file = 'https://download.openmmlab.com/mmclassification/v0/convnext/downstream/convnext-xlarge_3rdparty_in21k_20220301-08aa5ddc.pth'  # noqa
model = dict(
    backbone=dict(
        type='mmcls.ConvNeXt',
        arch='xlarge',
        out_indices=[0, 1, 2, 3],
        drop_path_rate=0.4,
        layer_scale_init_value=1.0,
        gap_before_final_norm=False,
        init_cfg=dict(
            type='Pretrained', checkpoint=checkpoint_file,
            prefix='backbone.')),
   decode_head=[
        dict(
            type='FCNHead',
            in_channels=[256, 512, 1024, 2048],
            channels=sum([256, 512, 1024, 2048]),
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
            type='OCRHead_EDGE',
            in_channels=[256, 512, 1024, 2048],
            in_index=(0, 1, 2, 3),
            input_transform='resize_concat',
            channels=512,
            ocr_channels=256,
            dropout_ratio=-1,
            num_classes=2,
            norm_cfg=norm_cfg,
            align_corners=False,
            loss_decode=[
                dict(type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0),
                dict(type='CrossEntropyLoss', use_edge=True, loss_weight=20.0, loss_name='loss_edgece')          
    ] )
    ])
    

optimizer = dict(
    constructor='LearningRateDecayOptimizerConstructor',
    _delete_=True,
    type='AdamW',
    lr=0.00008,
    betas=(0.9, 0.999),
    weight_decay=0.05,
    paramwise_cfg={
        'decay_rate': 0.9,
        'decay_type': 'stage_wise',
        'num_layers': 12
    })

lr_config = dict(
    _delete_=True,
    policy='poly',
    warmup='linear',
    warmup_iters=1500,
    warmup_ratio=1e-6,
    power=1.0,
    min_lr=0.0,
    by_epoch=False)






# By default, models are trained on 2 GPUs with 8 images per GPU
data = dict(
    samples_per_gpu=4,
    workers_per_gpu=2,
        train=dict(
        img_dir='images/images',
        ann_dir='annotations/n_labels'),
    val=dict(
        img_dir='images/images',
        ann_dir='annotations/n_labels'),
    test=dict(
         img_dir='images/val',
        ann_dir='annotations/n_val'))


# runtime settings
runner = dict(type='IterBasedRunner', max_iters=80000)
evaluation = dict(interval=1000, metric=['mDice', 'mIoU'], pre_eval=True)
checkpoint_config = dict(by_epoch=False, interval=1000)


# By default, models are trained on 8 GPUs with 2 images per GPU
# fp16 settings
optimizer_config = dict(type='Fp16OptimizerHook', loss_scale='dynamic')
# fp16 placeholder
fp16 = dict()

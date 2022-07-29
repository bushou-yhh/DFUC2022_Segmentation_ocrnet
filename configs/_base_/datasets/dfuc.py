# dataset settings
dataset_type = 'DFUCDataset'
data_root = 'data/dfuc2022'
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)

crop_size = (512, 512)

train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', reduce_zero_label=False),
    dict(type='Resize', img_scale=(480, 640), ratio_range=(0.5, 2.0)),
    # dict(type='Resize', img_scale=(640, 640), ratio_range=(0.5, 2.0)),
    dict(type='RandomCrop', crop_size=crop_size, cat_max_ratio=0.75),
    dict(type='RandomFlip', prob=0.5),
    # dict(type='RandomGamma'),
    dict(type='PhotoMetricDistortion'),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size=crop_size, pad_val=0, seg_pad_val=0),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_semantic_seg']),
]

test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='AdjustGamma',gamma=1.12),
    dict(
        type='MultiScaleFlipAug',
        # img_scale=(480, 640),
        img_scale=(576, 768),
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img']),
        ])
]
  
data = dict(
    samples_per_gpu=4,
    workers_per_gpu=2,
    train=dict(
        type=dataset_type,
        data_root=data_root,
        img_dir='images/images',
        ann_dir='annotations/n_labels',
        # img_dir='images/train_img_white',
        # ann_dir='annotations/n_labels',
        pipeline=train_pipeline),
    val=dict(
        type=dataset_type,
        data_root=data_root,
        img_dir='images/images',
        ann_dir='annotations/n_labels',
        # img_dir='images/train_img_white',
        # ann_dir='annotations/n_labels',
        pipeline=test_pipeline),
    test=dict(
        type=dataset_type,
        data_root=data_root,
        # img_dir='images/images',
        # ann_dir='annotations/n_labels',
        img_dir='images/images',
        ann_dir='annotations/n_labels',
        pipeline=test_pipeline))

_base_ = [
    './ocrnet_swin-t_512x1024_40k_dfuc2022.py'
]

checkpoint_file = 'https://download.openmmlab.com/mmsegmentation/v0.5/pretrain/swin/swin_small_patch4_window7_224_20220317-7ba6d6dd.pth'  # noqa
model = dict(
    backbone=dict(
        init_cfg=dict(type='Pretrained', checkpoint=checkpoint_file),
        depths=[2, 2, 18, 2]))

# By default, models are trained on 2 GPUs with 8 images per GPU
data = dict(
    samples_per_gpu=4,
    workers_per_gpu=4)

# runtime settings
runner = dict(type='IterBasedRunner', max_iters=80000)
evaluation = dict(interval=1000, metric=['mDice', 'mIoU'], pre_eval=True)
checkpoint_config = dict(by_epoch=False, interval=1000)

  
data = dict(
    samples_per_gpu=4,
    workers_per_gpu=2,
    train=dict(
        img_dir='images/train',
        ann_dir='annotations/n_train'),
    val=dict(
        img_dir='images/val',
        ann_dir='annotations/n_val'),
    test=dict(
         img_dir='images/val',
        ann_dir='annotations/n_val'))
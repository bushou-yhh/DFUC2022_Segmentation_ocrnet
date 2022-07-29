_base_ = './ocrnet_hr18_512x1024_40k_dfuc2022.py'
norm_cfg = dict(type='SyncBN', requires_grad=True)
model = dict(
    pretrained='open-mmlab://msra/hrnetv2_w48',
    backbone=dict(
        extra=dict(
            stage2=dict(num_channels=(48, 96)),
            stage3=dict(num_channels=(48, 96, 192)),
            stage4=dict(num_channels=(48, 96, 192, 384)))),
    decode_head=[
        dict(
            type='FCNHead',
            in_channels=[48, 96, 192, 384],
            channels=sum([48, 96, 192, 384]),
            input_transform='resize_concat',
            in_index=(0, 1, 2, 3),
            kernel_size=1,
            num_convs=1,
            norm_cfg=norm_cfg,
            concat_input=False,
            dropout_ratio=-1,
            num_classes=2,
            align_corners=False,
           loss_decode=[
                dict(type='CrossEntropyLoss',  loss_name='loss_ce',use_sigmoid=False, loss_weight=1.0, class_weight=[0.8373, 1.0180]),  
                dict(type='DiceLoss', loss_name='loss_dice', use_sigmoid=False, loss_weight=1.0, class_weight=[0.8373, 1.0180])]),
        dict(
            type='OCRHead',
            in_channels=[48, 96, 192, 384],
            channels=512,
            ocr_channels=256,
            input_transform='resize_concat',
            in_index=(0, 1, 2, 3),
            norm_cfg=norm_cfg,
            dropout_ratio=-1,
            num_classes=2,
            align_corners=False,
            loss_decode=[
                dict(type='CrossEntropyLoss',  loss_name='loss_ce',use_sigmoid=False, loss_weight=1.0,  class_weight=[0.8373, 1.0180]),  
                dict(type='DiceLoss', loss_name='loss_dice', use_sigmoid=False, loss_weight=1.0, class_weight=[0.8373, 1.0180])])
    ])
data = dict(
    samples_per_gpu=8,
    workers_per_gpu=4)
runner = dict(type='IterBasedRunner', max_iters=20000)
evaluation = dict(interval=500, metric='mDice', pre_eval=True)
checkpoint_config = dict(by_epoch=False, interval=500)
# load_from = "weight/ocrnet/ocrnet_hr48_512x512_160k_ade20k_20200615_184705-a073726d.pth"


optimizer=dict(
    paramwise_cfg = dict(
        custom_keys={
            'head': dict(lr_mult=10.)}))
#!/bin/bash

# CONFIG=configs/dfuc2022/deeplabv3plus_r50-d8_4x4_896x896_80k_dfuc2022.py
# # GPUS=2
# WORK_DIR=work_dirs/dfuc2022/deeplabv3plus_r50
# RESUMECKP=work_dirs/dfuc2022/upernet_swin_tiny_patch4_window7_512x512_20k_dfuc2022_pretrain_224x224_1K/iter_1600.pth
# CONFIG=configs/dfuc2022/upernet_swin_tiny_patch4_window7_512x512_20k_dfuc2022_pretrain_224x224_1K.py
# # GPUS=2
# WORK_DIR=work_dirs/dfuc2022/upernet_swin_tiny_patch4_window7_512x512_20k_dfuc2022_pretrain_224x224_1K
# # CUDA_LAUNCH_BLOCKING=1 
# bash dfuc2022/dist_train.sh $CONFIG  $GPUS --work-dir $WORK_DIR



# # RESUMECKP=work_dirs/dfuc2022/upernet_swin_tiny_patch4_window7_512x512_20k_dfuc2022_pretrain_224x224_1K/iter_1600.pth
# CONFIG=configs/dfuc2022/segformer_mit-b4_512x512_20k_dfuc2022.py
# GPUS=2
# WORK_DIR=work_dirs/dfuc2022/segformer_mit-b4
# # python dfuc2022/train.py $CONFIG   --work-dir $WORK_DIR  --gpu-id 0 
# bash dfuc2022/dist_train.sh $CONFIG  $GPUS --work-dir $WORK_DIR

# python dfuc2022/train.py $CONFIG   --work-dir $WORK_DIR  --gpu-id 0 --resume-from  $RESUMECKP


# # RESUMECKP=work_dirs/dfuc2022/upernet_swin_tiny_patch4_window7_512x512_20k_dfuc2022_pretrain_224x224_1K/iter_1600.pth
# CONFIG=configs/dfuc2022/ocrnet_hr48_512x512_20k_dfuc2022.py
# GPUS=2
# WORK_DIR=work_dirs/dfuc2022/ocrnet_hr48_diceloss
# # python dfuc2022/train.py $CONFIG   --work-dir $WORK_DIR  --gpu-id 0 
# bash dfuc2022/dist_train.sh $CONFIG  $GPUS --work-dir $WORK_DIR



CONFIG=configs/dfuc2022/edgeocrnet_convnext_xlarge_fp16_480x640_80k_dfuc.py
GPUS=2
WORK_DIR=work_dirs/dfuc2022/edgeocrnet_convnext_xlarge
# python dfuc2022/train.py $CONFIG   --work-dir $WORK_DIR  --gpu-id 0     
bash dfuc2022/dist_train.sh $CONFIG  $GPUS  --work-dir $WORK_DIR   \
     --auto-resume

#!/bin/bash
CONFIG=work_dirs/dfuc2022/ocrnet_swin-s_clip/ocrnet_swin-s_512x1024_40k_dfuc2022_clip.py
WEIGHT=work_dirs/dfuc2022/ocrnet_swin-s_clip/iter_60000.pth
IN_DIR=data/dfuc2022/images/DFUC2022_val_wound_clip_0.3
OUT_DIR=output/ocrnet_swin-s_clip/val_all_60000_480
# CUDA_LAUNCH_BLOCKING=1 \
python dfuc2022/infer.py $CONFIG  $WEIGHT   --img_dir   $IN_DIR   --out_dir  $OUT_DIR  --device cuda:1 
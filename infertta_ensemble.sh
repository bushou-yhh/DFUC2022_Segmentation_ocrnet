#!/bin/bash
# CONFIG=configs/dfuc2022/ocrnet_hr48_512x512_40k_dfuc2022.py
# WEIGHT=work_dirs/dfuc2022/ocrnet_hr48/iter_36000.pth
# IN_DIR=data/dfuc2022/images/test/DFUC2022_val_release
# OUT_DIR=output/ocrnet_hr48/val_all_3600_tta_12
# # CUDA_LAUNCH_BLOCKING=1 \
# python dfuc2022/infer.py $CONFIG  $WEIGHT   --img_dir   $IN_DIR   --out_dir  $OUT_DIR    --aug_test


# !/bin/bash
# CONFIG=configs/dfuc2022/ocrnet_swin-t_512x1024_40k_dfuc2022.py
# WEIGHT=work_dirs/dfuc2022/ocrnet_swin-t/iter_40000.pth
IN_DIR=data/dfuc2022/images/test/DFUC2022_test_release
OUT_DIR=Test/ocrnet/ensemble/convenext-xl
# CUDA_LAUNCH_BLOCKING=1 \
python dfuc2022/infer_ensemble.py $CONFIG  $WEIGHT   --img_dir   $IN_DIR   --out_dir  $OUT_DIR    --aug_test
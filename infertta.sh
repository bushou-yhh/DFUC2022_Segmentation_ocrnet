# !/bin/bash
# CONFIG=configs/dfuc2022/ocrnet_convnext_xlarge_fp16_480x640_80k_dfuc.py
# WEIGHT=work_dirs/dfuc2022/ocrnet_convnext_xlarge_fp16/iter_60000.pth
# IN_DIR=data/dfuc2022/images/test/DFUC2022_val_release
# OUT_DIR=output/enlight_gamma/ocrnet_convnext_xlarge_fp16/orign
# # CUDA_LAUNCH_BLOCKING=1 \
# python  dfuc2022/infer.py ${CONFIG}  ${WEIGHT}   --img_dir   $IN_DIR   --out_dir  $OUT_DIR    --aug_test  \
#      --device cuda:0  --thres 0.659



CONFIG=configs/dfuc2022/edgeocrnet_convnext_xlarge_fp16_480x640_80k_dfuc.py
WEIGHT=work_dirs/dfuc2022/edgeocrnet_convnext_xlarge/iter_60000.pth
IN_DIR=data/dfuc2022/images/test/DFUC2022_val_release
OUT_DIR=output/edgeocrnet_convnext_xlarge_fp16/gamma1.1_60000_0.650
# CUDA_LAUNCH_BLOCKING=1 \
python  dfuc2022/infer.py ${CONFIG}  ${WEIGHT}   --img_dir   $IN_DIR   --out_dir  $OUT_DIR    --aug_test  \
     --device cuda:1  --thres 0.650

# CONFIG=configs/dfuc2022/ocrnet_convnext_xlarge_fp16_480x640_80k_dfuc.py
# WEIGHT=work_dirs/dfuc2022/ocrnet_convnext_xlarge_fp16/iter_60000.pth
# IN_DIR=data/dfuc2022/images/test/DFUC2022_val_release
# OUT_DIR=output/edgeocrnet_convnext_xlarge_fp16/gamma1.1_60000_0.650
# # CUDA_LAUNCH_BLOCKING=1 \
# python  dfuc2022/infer.py ${CONFIG}  ${WEIGHT}   --img_dir   $IN_DIR   --out_dir  $OUT_DIR    --aug_test  \
#      --device cuda:1  --thres 0.650

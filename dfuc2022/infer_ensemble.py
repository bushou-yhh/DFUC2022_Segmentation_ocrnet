# Copyright (c) OpenMMLab. All rights reserved.
from argparse import ArgumentParser
import os
from unittest import result
from tqdm import tqdm
import mmcv
import cv2
import numpy as np

from mmseg.apis import inference_segmentor, init_segmentor, show_result_pyplot
from class_names import get_palette


def main():
    parser = ArgumentParser()
    # parser.add_argument('config', help='Config file')
    # parser.add_argument('checkpoint', help='Checkpoint file')
    parser.add_argument('--img_dir', help='the Image file path')
    parser.add_argument('--out_dir', default=None, help='Path to output file')
    parser.add_argument(
        '--device', default='cuda:0', help='Device used for inference')
    parser.add_argument(
        '--aug_test', action='store_true', help='Use Flip and Multi scale aug')
    parser.add_argument(
        '--palette',
        default='cityscapes',
        help='Color palette used for segmentation map')
    args = parser.parse_args()
    
    config1= "configs/dfuc2022/edgeocrnet_convnext_xlarge_fp16_480x640_80k_dfuc.py"
    config2= "configs/dfuc2022/ocrnet_convnext_xlarge_fp16_480x640_80k_dfuc.py"
    # config3= "configs/dfuc2022/ocrnet_hr48_512x512_40k_dfuc2022.py"

    checkpoint1= "work_dirs/dfuc2022/edgeocrnet_convnext_xlarge/iter_60000.pth"
    checkpoint2= "work_dirs/dfuc2022/ocrnet_convnext_xlarge_fp16/iter_60000.pth"
    # checkpoint3= "exp_log/ocrnet_hr48/iter_36000.pth"

    cfg1 = mmcv.Config.fromfile(config1)
    # import pdb;pdb.set_trace()
    # cfg.aug_test = True
    if args.aug_test:
        # hard code index
        # cfg.data.test.pipeline[1].img_ratios = [
        #     0.75, 0.875, 1.0, 1.125, 1.25
        # ]
        cfg1.data.test.pipeline[2].img_ratios = [
          0.75,  1.25,
        ]

        cfg1.data.test.pipeline[2].flip = True
        cfg1.data.test.pipeline[2].flip_direction=['horizontal', 'vertical']
    # build the model from a config file and a checkpoint file
    print(cfg1.data.test.pipeline)
    model1 = init_segmentor(cfg1, checkpoint1, device=args.device)


    cfg2 = mmcv.Config.fromfile(config2)
    
    if args.aug_test:
        cfg2.data.test.pipeline[2].img_ratios = [
             0.75,  1.25,
        ]
        cfg2.data.test.pipeline[2].flip = True
        cfg2.data.test.pipeline[2].flip_direction=['horizontal', 'vertical']
    # build the model from a config file and a checkpoint file
    print(cfg2.data.test.pipeline)
    model2 = init_segmentor(cfg2, checkpoint2, device=args.device)


    # cfg3 = mmcv.Config.fromfile(config3)
    # if args.aug_test:
    #     cfg3.data.test.pipeline[1].img_ratios = [
    #         0.5, 0.75, 1.0, 1.25, 1.5, 1.75
    #     ]
    #     cfg3.data.test.pipeline[1].flip = True
    # # build the model from a config file and a checkpoint file
    # model3 = init_segmentor(cfg3, checkpoint3, device=args.device)

    # test a single image
    # import  pdb;pdb.set_trace()
    for img in tqdm(os.listdir(args.img_dir)):
        if img.split(".")[-1] == 'jpg' or img.split(".")[-1] == 'png':
            _img = os.path.join(args.img_dir, img)
            img_ = os.path.join(args.out_dir, img).replace('.jpg', '.png')

# model1           
            seg_logit1 = 0
            seg_logits1, len1 = inference_segmentor(model1, _img)
            # import pdb;pdb.set_trace()
            for logit in seg_logits1:
                seg_logit1 += logit

            seg_logit1 = seg_logit1 / len1 

# model2
            seg_logit2 = 0
            seg_logits2, len2 = inference_segmentor(model2, _img)
            for logit in seg_logits2:
                seg_logit2 += logit

            seg_logit2 = seg_logit2 / len2 


# # model3
#             seg_logit3 = 0
#             seg_logits3, len3 = inference_segmentor(model3, _img)
#             for logit in seg_logits3:
#                 seg_logit3 += logit

#             seg_logit3 = seg_logit3 / len3 

            seg_logit = (seg_logit1 + seg_logit2 )/2
            # seg_pred = seg_logit.argmax(dim=1)
            seg_pred = seg_logit[:, 1].sigmoid() > 0.659
            seg_pred = seg_pred.long()
            seg_pred = seg_pred.cpu().numpy()
            # unravel batch dim
            result = list(seg_pred)



            palette = np.array(get_palette('dfuc'))
            seg = result[0]
            color_seg = np.zeros((seg.shape[0], seg.shape[1], 3), dtype=np.uint8)
            for label, color in enumerate(palette):
                # import pdb; pdb.set_trace()  
                color_seg[seg == label, :] = color
            color_seg = color_seg.astype(np.uint8)
            grayImage = cv2.cvtColor(color_seg,cv2.COLOR_BGR2GRAY)
            #小于阈值的像素点灰度值不变，大于阈值的像素点置为该阈值
            ret,thresh3 = cv2.threshold(grayImage,0,255, cv2.THRESH_BINARY)
            mmcv.imwrite(thresh3, img_)
            # show the results

            

if __name__ == '__main__':
    main()

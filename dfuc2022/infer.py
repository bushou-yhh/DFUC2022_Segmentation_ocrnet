# Copyright (c) OpenMMLab. All rights reserved.
from argparse import ArgumentParser
import os
from tqdm import tqdm
import mmcv
import cv2
import numpy as np

from mmseg.apis import inference_segmentor, init_segmentor, show_result_pyplot
from class_names import get_palette


def main():
    parser = ArgumentParser()
    parser.add_argument('config', help='Config file')
    parser.add_argument('checkpoint', help='Checkpoint file')
    parser.add_argument('--img_dir', help='the Image file path')
    parser.add_argument('--out_dir', default=None, help='Path to output file')
    parser.add_argument('--thres', default=0.618, type=float, help='threshold')
    parser.add_argument(
        '--device', default='cuda:0', help='Device used for inference')
    parser.add_argument(
        '--aug_test', action='store_true', help='Use Flip and Multi scale aug')
    
    
    
    args = parser.parse_args()


    cfg = mmcv.Config.fromfile(args.config)

    if args.aug_test:
        # hard code index
        # cfg.data.test.pipeline[1].img_ratios = [
        #     0.75, 0.875, 1.0, 1.125, 1.25
        # ]
        # cfg.data.test.pipeline[2].img_ratios = [
        #     0.5, 0.75, 1.0, 1.25, 1.5, 1.75
        # ]
        cfg.data.test.pipeline[2].img_ratios = [
            0.75, 1.25
        ]
        # cfg.data.test.pipeline[2].img_ratios = [
        #     0.75, 1.25
        # ]
        # cfg.data.test.pipeline[1].img_ratios = [
        #      1.0, 1.25, 1.5, 1.75
        # ]

        cfg.data.test.pipeline[2].flip = True
        cfg.data.test.pipeline[2].flip_direction=['horizontal', 'vertical']
        
        # cfg.model.test_cfg.logits=True 
        # cfg.model.test_cfg.binary_thres=0.5 
    # build the model from a config file and a checkpoint file
    print(cfg.data.test.pipeline)
    model = init_segmentor(cfg, args.checkpoint, device=args.device)

    # args.aug_test = True
    # if args.aug_test:
    #     # hard code index
    #     args.data.test.pipeline[1].img_ratios = [
    #         0.5, 0.75, 1.0, 1.25, 1.5, 1.75
    #     ]
    #     args.data.test.pipeline[1].flip = True

    # test a single image
    # import  pdb;pdb.set_trace()
    for img in tqdm(os.listdir(args.img_dir)):
        if img.split(".")[-1] == 'jpg' or img.split(".")[-1] == 'png':
            _img = os.path.join(args.img_dir, img)
            img_ = os.path.join(args.out_dir, img).replace('.jpg', '.png')
            # import pdb;pdb.set_trace()
            if args.aug_test:
                seg_logits, lenght = inference_segmentor(model, _img)
                seg_logit = 0
                for logit in seg_logits:
                    seg_logit += logit
                seg_logit /= lenght
               
                # import pdb;pdb.set_trace()
                # seg_pred = seg_logit.argmax(dim=1)
                seg_pred = seg_logit[:, 1].sigmoid() > args.thres
                seg_pred = seg_pred.long()
                # seg_pred = seg_logit.argmax(dim=1)
                seg_pred = seg_pred.cpu().numpy()
                result = list(seg_pred)
            else:
                result = inference_segmentor(model, _img)

            # unravel batch dim
            # import pdb;pdb.set_trace()
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

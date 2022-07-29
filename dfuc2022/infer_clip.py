# Copyright (c) OpenMMLab. All rights reserved.
from argparse import ArgumentParser
import os
from tqdm import tqdm
import mmcv
import cv2
import numpy as np
import csv
from collections import defaultdict

from mmseg.apis import inference_segmentor, init_segmentor, show_result_pyplot
from class_names import get_palette


def main():
    parser = ArgumentParser()
    parser.add_argument('config', help='Config file')
    parser.add_argument('checkpoint', help='Checkpoint file')
    parser.add_argument('--clip_img_dir', help='the cliped Image file path')
    parser.add_argument('--loc_csv', help='the csv file path')
    parser.add_argument('--orin_img_dir', help='the original Image file path')
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


    cfg = mmcv.Config.fromfile(args.config)
    # import pdb;pdb.set_trace()
    # cfg.aug_test = True
    if args.aug_test:
        # hard code index
        # cfg.data.test.pipeline[1].img_ratios = [
        #     0.75, 0.875, 1.0, 1.125, 1.25
        # ]
        cfg.data.test.pipeline[2].img_ratios = [
            0.5, 0.75, 1.0, 1.25, 1.5, 1.75
        ]

        # cfg.data.test.pipeline[1].img_ratios = [
        #      1.0, 1.25, 1.5, 1.75
        # ]

        cfg.data.test.pipeline[2].flip = True
        cfg.data.test.pipeline[2].flip_direction=['horizontal', 'vertical']
    print(cfg.data.test.pipeline, args.checkpoint)
    # build the model from a config file and a checkpoint file
    model = init_segmentor(cfg, args.checkpoint, device=args.device)

    locs =  defaultdict(list)
    with open(args.loc_csv, 'r') as f:
        fcsv = csv.DictReader(f)
        for row in fcsv:
            locs[row['filename']] .append(row['xmin'])
            locs[row['filename']] .append(row['ymin'])
            locs[row['filename']] .append(row['xmax'])
            locs[row['filename']] .append(row['ymax'])

    # import  pdb;pdb.set_trace()
    for img in tqdm(os.listdir(args.clip_img_dir)):
        if img.split(".")[-1] == 'jpg' or img.split(".")[-1] == 'png':
            _img = os.path.join(args.clip_img_dir, img)
            img_ = os.path.join(args.out_dir, img).replace('.jpg', '.png')
            # import pdb;pdb.set_trace()
            if args.aug_test:
                seg_logits, lenght = inference_segmentor(model, _img)
                seg_logit = 0
                for logit in seg_logits:
                    seg_logit += logit
                seg_logit /= lenght
                # seg_pred = seg_logit.argmax(dim=1)
                binary_thres = 0.64
                seg_pred = seg_logit[:, 1].sigmoid() > binary_thres
                seg_pred = seg_pred.long()
                seg_pred = seg_pred.cpu().numpy()
                result = list(seg_pred)
            else:
                result = inference_segmentor(model, _img)

            origin_img = cv2.imread(os.path.join(args.orin_img_dir, img))
            # unravel batch dim
            xmin,ymin, xmax, ymax = locs[img]
            palette = np.array(get_palette('dfuc'))
            seg = result[0]
            color_seg = np.zeros((seg.shape[0], seg.shape[1], 3), dtype=np.uint8)
            origin_seg = np.zeros((origin_img.shape[0], origin_img.shape[1],3), dtype=np.uint8)
            for label, color in enumerate(palette):
                # import pdb; pdb.set_trace()  
                color_seg[seg == label, :] = color
                # import pdb;pdb.set_trace()
                # print(img, locs[img], seg.shape)
                origin_seg[int(ymin):int(ymax),int(xmin):int(xmax), :] =color_seg
            origin_seg = origin_seg.astype(np.uint8)
            grayImage = cv2.cvtColor(origin_seg,cv2.COLOR_BGR2GRAY)
            #小于阈值的像素点灰度值不变，大于阈值的像素点置为该阈值
            ret,thresh = cv2.threshold(grayImage,0,255, cv2.THRESH_BINARY)
            mmcv.imwrite(thresh, img_)
            # show the results

            

if __name__ == '__main__':
    main()

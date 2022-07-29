import cv2
import os
from tqdm import tqdm
#相对路径下读取图片
in_path = "data/dfuc2022/annotations/labels"
out_path = "data/dfuc2022/annotations/n_labels"
for i in tqdm(os.listdir(in_path)):
    # import pdb;pdb.set_trace()
    o_img = os.path.join(in_path, i)
    o_img= cv2.imread(o_img)
    #灰度化处理
    # import pdb;pdb.set_trace()
    grayImage = cv2.cvtColor(o_img,cv2.COLOR_BGR2GRAY)
    #小于阈值的像素点灰度值不变，大于阈值的像素点置为该阈值
    ret,thresh3 = cv2.threshold(grayImage,0,1, cv2.THRESH_BINARY)
    n_img = os.path.join(out_path, i)
    cv2.imwrite(n_img, thresh3)
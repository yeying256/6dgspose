
import os,sys

from pathlib import Path
# 这个是需要将python的路径给到根目录，要不然找不到路径
dir_path = Path(os.path.dirname(os.path.realpath(__file__))).parents[0]
print(f"dir_path {dir_path}")
sys.path.append(dir_path.__str__())
import time
import random

import open3d as o3d
import cv2

import numpy as np

# 读取图像
imag_path = "/media/wangxiao/Newsmy/dataset_LINEMOD/datasets/train_data/0575-saltbottle-bottle/saltbottle-1/color"
gspath = "/media/wangxiao/Newsmy/dataset_LINEMOD/datasets/train_data_pgsr/0575-saltbottle-bottle/test/point_cloud/iteration_30000/point_cloud.ply"
boxpath = "/media/wangxiao/Newsmy/dataset_LINEMOD/datasets/train_data/0575-saltbottle-bottle/box3d_corners.txt"
img1 = cv2.imread(f'{imag_path}/0.png', cv2.IMREAD_GRAYSCALE)
img2 = cv2.imread(f'{imag_path}/10.png', cv2.IMREAD_GRAYSCALE)





# 初始化SIFT特征检测器
sift = cv2.SIFT_create()


# 显示结果
cv2.imshow('Feature Matches', img_matches)
cv2.waitKey(0)
cv2.destroyAllWindows()

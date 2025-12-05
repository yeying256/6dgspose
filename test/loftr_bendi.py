
import os,sys

import plotly.express as px
from pathlib import Path
from typing import List

dir_path = Path(os.path.dirname(os.path.realpath(__file__))).parents[0]
print(f"dir_path {dir_path}")
sys.path.append(dir_path.__str__())
import torch


import cv2
import numpy as np
import matplotlib.pyplot as plt


from submodules.LoFTR.src.loftr import LoFTR, default_cfg



def read_image(image_path):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    # img = cv2.imread(image_path)

    # img = cv2.resize(img, (640, 480))  # 调整图像大小以适应模型
    # img = cv2.resize(img, (640, 480))  # 调整图像大小以适应模型

    img_tensor = torch.from_numpy(img)[None][None] / 255.  # 归一化到[0, 1]
    return img, img_tensor.float().cuda()

def visualize_matches(img1, img2, mkpts1, mkpts2):
    plt.show(block=True)
    h1, w1 = img1.shape
    h2, w2 = img2.shape
    img_vis = np.zeros((max(h1, h2), w1 + w2), dtype=np.uint8)
    img_vis[:h1, :w1] = img1
    img_vis[:h2, w1:] = img2

    fig, ax = plt.subplots(figsize=(15, 5))
    ax.imshow(img_vis, cmap='gray')
    ax.scatter(mkpts1[:, 0], mkpts1[:, 1], s=10, c='red', marker='o')
    ax.scatter(mkpts2[:, 0] + w1, mkpts2[:, 1], s=10, c='blue', marker='o')
    for i in range(len(mkpts1)):
        ax.plot([mkpts1[i, 0], mkpts2[i, 0] + w1], [mkpts1[i, 1], mkpts2[i, 1]], 'g-', linewidth=1)
    plt.show()
    plt.waitforbuttonpress()  # 等待按键或鼠标点击
    plt.close()
# indoor_ds_new.ckpt
# Initialize LoFTR
    
    
matcher = LoFTR(config=default_cfg)
matcher.load_state_dict(torch.load("/home/wangxiao/6d_pose_learn/src/myproject/weight/indoor_ds_new.ckpt")['state_dict'])
matcher = matcher.eval().cuda()

image_path1 = "/home/wangxiao/6d_pose_learn/src/myproject/test/cup1.png"
image_path2 = "/home/wangxiao/6d_pose_learn/src/myproject/test/cup2.png"

img1, img1_tensor = read_image(image_path1)
img2, img2_tensor = read_image(image_path2)


with torch.no_grad():
    batch = {"image0": img1_tensor, "image1": img2_tensor}
    
# correspondences = model(input_dict)

# Inference

    matcher(batch)    # batch = {'image0': img0, 'image1': img1}
    mkpts0 = batch['mkpts0_f'].cpu().numpy()
    mkpts1 = batch['mkpts1_f'].cpu().numpy()


visualize_matches(img1, img2, mkpts0, mkpts1)



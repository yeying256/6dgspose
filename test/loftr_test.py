import torch
import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from kornia.feature import LoFTR

# 加载LoFTR模型
def load_loftr_model(ckpt_path):
    # 初始化LoFTR模型
    model = LoFTR(pretrained=None)
    # 加载预训练权重
    checkpoint = torch.load(ckpt_path, map_location='cpu')
    model.load_state_dict(checkpoint['state_dict'])
    model.eval()
    return model

# 读取图像并转换为灰度图
def read_image(image_path):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    # img = cv2.imread(image_path)

    # img = cv2.resize(img, (640, 480))  # 调整图像大小以适应模型
    # img = cv2.resize(img, (640, 480))  # 调整图像大小以适应模型

    img_tensor = torch.from_numpy(img)[None][None] / 255.  # 归一化到[0, 1]
    return img, img_tensor.float()

# 可视化匹配结果
def visualize_matches(img1, img2, mkpts1, mkpts2):
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

def main(image_path1, image_path2):

    #     Example:
    #     >>> img1 = torch.rand(1, 1, 320, 200)
    #     >>> img2 = torch.rand(1, 1, 128, 128)
    #     >>> input = {"image0": img1, "image1": img2}
    #     >>> loftr = LoFTR('outdoor')
    #     >>> out = loftr(input)
    # """
    # 加载模型
    model = LoFTR('outdoor')

    # 读取图像
    img1, img1_tensor = read_image(image_path1)
    img2, img2_tensor = read_image(image_path2)

    # 进行特征匹配
    with torch.no_grad():
        input_dict = {"image0": img1_tensor, "image1": img2_tensor}
        correspondences = model(input_dict)

    # 获取匹配点
    mkpts1 = correspondences['keypoints0'].cpu().numpy()
    mkpts2 = correspondences['keypoints1'].cpu().numpy()

    # 可视化匹配结果
    visualize_matches(img1, img2, mkpts1, mkpts2)

if __name__ == "__main__":
    # 图像路径
    image_path1 = "/media/wangxiao/Newsmy/linemod/raw/lm_test_all/test/000001/rgb/000000.png"
    image_path2 = "/media/wangxiao/Newsmy/linemod/raw/lm_train/train/000001/rgb/000454.png"

    
    # 预训练权重路径
    # ckpt_path = "weights/outdoor_ds.ckpt"

    main(image_path1, image_path2)
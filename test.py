from lightglue import LightGlue, SuperPoint, DISK, SIFT, ALIKED, DoGHardNet
from lightglue.utils import load_image, rbd

import cv2
import numpy as np
import torch


def tensor_to_cv2_image(tensor):
    """
    将 torch.Tensor 转换为 OpenCV 兼容的 NumPy 数组。
    
    参数:
    - tensor: 输入的 torch.Tensor，形状为 [C, H, W]
    
    返回:
    - img: 转换后的 NumPy 数组，形状为 [H, W, C]
    """
    # 确保张量在 CPU 上
    tensor = tensor.cpu()
    
    # 调整通道顺序：从 [C, H, W] 到 [H, W, C]
    img = tensor.permute(1, 2, 0).numpy()
    
    # 如果图像是 float 类型，缩放到 [0, 255] 并转换为 uint8
    if img.dtype == np.float32 or img.dtype == np.float64:
        img = (img * 255).astype(np.uint8)
    
    return img

def draw_matches(image0, image1, points0, points1, matches, output_path=None):
    """
    使用 OpenCV 绘制匹配的关键点和连线。
    
    参数:
    - image0: 第一张图像 (numpy array)
    - image1: 第二张图像 (numpy array)
    - points0: 第一张图像中的关键点坐标 (numpy array, shape: (K, 2))
    - points1: 第二张图像中的关键点坐标 (numpy array, shape: (K, 2))
    - matches: 匹配的索引 (numpy array, shape: (K, 2))
    - output_path: 保存结果图像的路径 (可选)
    """

 
    # 将 torch.Tensor 转换为 OpenCV 兼容的 NumPy 数组
    image0_np = tensor_to_cv2_image(image0)
    image1_np = tensor_to_cv2_image(image1)

    # 将关键点坐标从 torch.Tensor 转换为 NumPy 数组
    points0_np = points0.cpu().numpy()
    points1_np = points1.cpu().numpy()

    # 创建 DMatch 对象，用于表示匹配对
    dmatches = [cv2.DMatch(i, i, 0) for i in range(len(matches))]

    # 使用 OpenCV 的 drawMatches 函数绘制匹配结果
    img_matches = cv2.drawMatches(
        image0_np, [cv2.KeyPoint(float(pt[0]), float(pt[1]), 1) for pt in points0_np],
        image1_np, [cv2.KeyPoint(float(pt[0]), float(pt[1]), 1) for pt in points1_np],
        dmatches, None,
        flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS,
        matchesThickness=1  # 添加 matchesThickness 参数
    )

    # 显示结果图像

    cv2.namedWindow('Matches', cv2.WINDOW_NORMAL)  # 创建一个可调整大小的窗口
    cv2.resizeWindow('Matches', 800, 600)  # 设置窗口大小为 800x600 像素
    cv2.imshow('Matches', img_matches)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # 如果提供了输出路径，则保存结果图像
    if output_path:
        cv2.imwrite(output_path, img_matches)
        print(f"匹配结果已保存到 {output_path}")

# SuperPoint+LightGlue
extractor = SuperPoint(max_num_keypoints=2048).eval().cuda()  # load the extractor
matcher = LightGlue(features='superpoint').eval().cuda()  # load the matcher

# or DISK+LightGlue, ALIKED+LightGlue or SIFT+LightGlue
extractor = DISK(max_num_keypoints=2048).eval().cuda()  # load the extractor
matcher = LightGlue(features='disk').eval().cuda()  # load the matcher

# load each image as a torch.Tensor on GPU with shape (3,H,W), normalized in [0,1]
image0 = load_image('/home/wangxiao/6d_pose_learn/src/LightGlue/test/test3.jpeg').cuda()
image1 = load_image('/home/wangxiao/6d_pose_learn/src/LightGlue/test/test4.jpeg').cuda()

# extract local features
feats0 = extractor.extract(image0)  # auto-resize the image, disable with resize=None
feats1 = extractor.extract(image1)

# match the features
matches01 = matcher({'image0': feats0, 'image1': feats1})
feats0, feats1, matches01 = [rbd(x) for x in [feats0, feats1, matches01]]  # remove batch dimension
matches = matches01['matches']  # indices with shape (K,2)
points0 = feats0['keypoints'][matches[..., 0]]  # coordinates in image #0, shape (K,2)
points1 = feats1['keypoints'][matches[..., 1]]  # coordinates in image #1, shape (K,2)

draw_matches(image0, image1, points0, points1, matches)


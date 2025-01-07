
import os,sys

import plotly.express as px

from pathlib import Path
from typing import List
# 这个是需要将python的路径给到根目录，要不然找不到路径
dir_path = Path(os.path.dirname(os.path.realpath(__file__))).parents[0]
print(f"dir_path {dir_path}")
sys.path.append(dir_path.__str__())
import time
import random

import open3d as o3d

import numpy as np


import math

from gaussian_renderer import render as GS_Renderer
from gaussian_renderer.refine import GS_refine

from argparse import ArgumentParser
from arguments import ModelParams, PipelineParams, OptimizationParams
from scene import GaussianModel 
from scene.cameras import MiniCam,Camera

import torch
import torch.nn.functional as torch_F

import cv2
from scipy.spatial.transform import Rotation
from misc_utils.warmup_lr import CosineAnnealingWarmupRestarts
import string

from lightglue import LightGlue, SuperPoint, DISK, SIFT, ALIKED, DoGHardNet
from lightglue.utils import load_image, rbd

from misc_utils import match_utils


# # 读取图像
# imag_path = "/media/wangxiao/Newsmy/dataset_LINEMOD/datasets/train_data/0575-saltbottle-bottle/saltbottle-1/color"
# img1 = cv2.imread(f'{imag_path}/55.png', cv2.IMREAD_GRAYSCALE)
# intrin1path = "/media/wangxiao/Newsmy/dataset_LINEMOD/datasets/train_data/0575-saltbottle-bottle/saltbottle-1/intrin_ba/55.txt"
# # img2 = cv2.imread(f'{imag_path}/10.png', cv2.IMREAD_GRAYSCALE)
# boxpath = "/media/wangxiao/Newsmy/dataset_LINEMOD/datasets/train_data/0575-saltbottle-bottle/box3d_corners.txt"
# gs_path = "/media/wangxiao/Newsmy/dataset_LINEMOD/datasets/train_data_pgsr/0575-saltbottle-bottle/test/point_cloud/iteration_30000/point_cloud.ply"

imag_path = "/media/wangxiao/Newsmy/dataset_LINEMOD/datasets/train_data/0499-tiramisufranzzi-box/tiramisufranzzi-1/color"
img1 = cv2.imread(f'{imag_path}/106.png', cv2.IMREAD_GRAYSCALE)
intrin1path = "/media/wangxiao/Newsmy/dataset_LINEMOD/datasets/train_data/0499-tiramisufranzzi-box/tiramisufranzzi-1/intrin_ba/106.txt"
img2 = cv2.imread(f'{imag_path}/10.png', cv2.IMREAD_GRAYSCALE)
boxpath = "/media/wangxiao/Newsmy/dataset_LINEMOD/datasets/train_data/0499-tiramisufranzzi-box/box3d_corners.txt"
gs_path = "/media/wangxiao/Newsmy/dataset_LINEMOD/datasets/train_data_pgsr/0499-tiramisufranzzi-box/test/point_cloud/iteration_30000/point_cloud.ply"







parser = ArgumentParser()
gaussian_ModelP = ModelParams(parser)
gaussian_PipeP = PipelineParams(parser)
gaussian_OptimP = OptimizationParams(parser)
gaussian_BG = torch.zeros((3), device='cuda')

bbox_coords = match_utils.read_txt(boxpath)
initin1 = match_utils.read_txt(intrin1path)


height, width = img1.shape[:2]

# 一个camera队列
# cameras 中储存的R 都是从相机坐标系变换到世界坐标系 的转置 距离不变
cameras = match_utils.box2gscamera(box=bbox_coords,K=initin1,height=height,width=width,camera_num=7)


obj_gaussians = GaussianModel(3)
obj_gaussians.load_ply(gs_path)


obj_gaussians.clip_to_bounding_box_fromfile(boxpath)


img_xys =[]
gs_xyz_ws = []
for camera in cameras:
    render = GS_Renderer(camera, obj_gaussians, gaussian_PipeP, gaussian_BG)
    render_img = render['render']

    # 这里每个像素单位是m
    deepsimg = render['plane_depth']

    # depth_normalized = cv2.normalize(deepsimg.permute(1, 2, 0).detach().cpu().numpy(), None, 0, 255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    depth_normalized = deepsimg.permute(1, 2, 0).detach().cpu().numpy()
    # for y in range(depth_normalized.shape[0]):  # 遍历高度
    #     for x in range(depth_normalized.shape[1]):  # 遍历宽度
    #         value = depth_normalized[y, x]
    #         if not np.all(value == 0):  # 如果不是所有通道都为0（假设是多通道图像）
    #             print(f"Position ({x}, {y}): {value}")

    # depth_image_display = cv2.applyColorMap(cv2.convertScaleAbs(depth_normalized, alpha=255/10), cv2.COLORMAP_JET)
    
    # 应用伪彩色映射（例如使用热图）
    # colored_depth = cv2.applyColorMap(depth_normalized, cv2.COLORMAP_JET)

    


    # 沿第三个轴（通道轴）拼接三个通道
    cv2.imshow('Original Depth Image', depth_normalized)
    cv2.waitKey()  # 按任意键关闭窗口
    cv2.destroyAllWindows()
    # cv2.imshow('Colored Depth Image', colored_depth)


    render_img_np = render_img.permute(1, 2, 0).detach().cpu().numpy()
    render_img_np = (render_img_np * 255).astype(np.uint8)
    if render_img_np.shape[0] == 3:
        render_img_np = np.transpose(render_img_np, (1, 2, 0))

    # 更改rgb改为gbr
    img_cv = render_img_np[:, :, [2, 1, 0]]
    # 调整通道顺序：从 [H, W, C] 到 [C, H, W]
    img_cv = img_cv.transpose(2, 0, 1)

    torch_render = torch.from_numpy(img_cv).float().cuda()
    torch_render = torch_render / 255.0
    # torch_render = torch_render.unsqueeze(0)




    # 2. 将 BGR 图像转换为 RGB 格式
    img_rgb = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
    
    # 3. 将图像从 (H, W, C) 转换为 (C, H, W)
    img_rgb_transposed = img_rgb.transpose(2, 0, 1)
    
    # 4. 将像素值归一化到 [0, 1] 范围
    img_rgb_normalized = img_rgb_transposed / 255.0
    
    # 5. 将 NumPy 数组转换为 PyTorch 张量
    torch_targetimg = torch.from_numpy(img_rgb_normalized).float().cuda()
    
    # 6. 添加批次维度 (如果需要)，形状变为 (1, C, H, W)
    # torch_targetimg = img_tensor.unsqueeze(0).cuda()
    

    # SuperPoint+LightGlue
    extractor = SuperPoint(max_num_keypoints=2048).eval().cuda()  # load the extractor
    matcher = LightGlue(features='superpoint').eval().cuda()  # load the matcher

    # or DISK+LightGlue, ALIKED+LightGlue or SIFT+LightGlue
    # extractor = DISK(max_num_keypoints=2048).eval().cuda()  # load the extractor
    # matcher = LightGlue(features='disk').eval().cuda()  # load the matcher

    # extract local features
    feats0 = extractor.extract(torch_render)  # auto-resize the image, disable with resize=None
    feats1 = extractor.extract(torch_targetimg)

    # match the features
    matches01 = matcher({'image0': feats0, 'image1': feats1})
    feats0, feats1, matches01 = [rbd(x) for x in [feats0, feats1, matches01]]  # remove batch dimension
    matches = matches01['matches']  # indices with shape (K,2)
    # points0 = feats0['keypoints'][matches[..., 0]]  # coordinates in image #0, shape (K,2)
    # points1 = feats1['keypoints'][matches[..., 1]]  # coordinates in image #1, shape (K,2)

    scores = matches01['scores']

    confidence_threshold = 0.99  # 你可以根据需要调整这个阈值

    valid_indices = scores > confidence_threshold
    matches = matches[valid_indices]
    filtered_scores = scores[valid_indices]

    # 获取过滤后的匹配点坐标
    points0 = feats0['keypoints'][matches[..., 0]]  # coordinates in image #0, shape (K', 2)
    points1 = feats1['keypoints'][matches[..., 1]]  # coordinates in image #1, shape (K', 2)


    # 应用比例测试筛选匹配点
    # good_matches = []
    # good_matches.append(matches)

    # 返回img和物体坐标系上的xyz
    if not len(matches) < 3:
        img_xy,gs_xyz_w = match_utils.match2xy_xyz_LightGlue(kporg=points1,kpdeep=points0,img_deep=depth_normalized,camera=camera)
        img_xys.extend(img_xy)
        gs_xyz_ws.extend(gs_xyz_w)
        match_utils.draw_matches(torch_render, torch_targetimg, points0, points1, matches)

    #如果有匹配点
    # if not good_matches.count == 0:

    #     pass

    # 这里的distance是特征的相似度
    # 绘制匹配结果
    # img_matches = cv2.drawMatches(img1, points0, render_img_np, points1, matches, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

    # cv2.imshow('Rendered Image', img_matches)
    cv2.waitKey()  # 按任意键关闭窗口
    cv2.destroyAllWindows()
    
gs_xyz_ws = np.array(gs_xyz_ws, dtype=np.float32)
img_xys = np.array(img_xys, dtype=np.float32)



print(f"gs_xyz_ws.shape = {gs_xyz_ws.shape}")
print(f"img_xys.shape = {img_xys.shape}")

pcd = o3d.geometry.PointCloud()

# 将 NumPy 数组转换为 Open3D 的点云格式
pcd.points = o3d.utility.Vector3dVector(gs_xyz_ws)
coordinate_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1, origin=[0, 0, 0])
o3d.visualization.draw_geometries([pcd,coordinate_frame])

dist_coeffs = np.zeros(shape=[8, 1], dtype="float64")

# 计算基础矩阵
F, mask = cv2.findFundamentalMat(gs_xyz_ws, img_xys, cv2.FM_RANSAC, ransacReprojThreshold=8)

# 使用掩码过滤掉不符合几何约束的匹配点
# inliers = mask.ravel() == 1
# gs_xyz_ws = gs_xyz_ws[inliers]
# img_xys = img_xys[inliers]


# # 将 NumPy 数组转换为 Open3D 的点云格式
# pcd.points = o3d.utility.Vector3dVector(gs_xyz_ws)
# coordinate_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1, origin=[0, 0, 0])
# o3d.visualization.draw_geometries([pcd,coordinate_frame])


pnp_reprojection_error=1
_, rvec, tvec, inliers = cv2.solvePnPRansac(gs_xyz_ws, img_xys,cameras[0].get_k().cpu().numpy(),dist_coeffs,
                reprojectionError=pnp_reprojection_error,
                iterationsCount=10000,
                flags=cv2.SOLVEPNP_EPNP)


        # dist_coeffs是畸变系数
        # pnp_reprojection_error pnp重投影的误差阈值
        # inliers代表内点，标识哪些是有用的 形状：(N,1) N是输入点对的数量

# 1. 将旋转向量 rvec 转换为旋转矩阵 R
R_, _ = cv2.Rodrigues(rvec)

# 2. 构建齐次变换矩阵 T
# 初始化一个 4x4 的单位矩阵
T_ = np.eye(4)

# 将旋转矩阵 R 放入 T 的前 3x3 区域
T_[:3, :3] = R_

# 将平移向量 t 放入 T 的前 3 行的最后一列
T_[:3, 3] = tvec.flatten()

# 创建原始坐标系
original_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1.0, origin=[0, 0, 0])
# 创建变换后的坐标系
transformed_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1, origin=[0, 0, 0])
transformed_frame.transform(np.linalg.inv(T_))
# 可视化
o3d.visualization.draw_geometries([original_frame, transformed_frame])

# transformed_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1, origin=[0, 0, 0])
# transformed_frame.transform(T_)
# # 可视化
# o3d.visualization.draw_geometries([original_frame, transformed_frame])


# T_ = np.linalg.inv(T_)

K = match_utils.convert2K(initin1)
# item 是转化为标量
fx = K[0,0].item()
fy = K[1,1].item()
cx = K[0,2].item()
cy = K[1,2].item()
fovx = 2 * np.arctan(width / (2 * fx))
fovy = 2 * np.arctan(height / (2 * fy))
# cameras = box2gscamera(box=bbox_coords,K=initin1,height=height,width=width,camera_num=7)
camera = Camera(colmap_id=0,R=T_[:3,:3].T,T=T_[:3,3],
                FoVx=fovx,FoVy=fovy,Cx=cx,Cy=cy,image_height=height,image_width=width,
                image_name='',image_path='',uid=0,preload_img=False)

render_init = GS_Renderer(camera, obj_gaussians, gaussian_PipeP, gaussian_BG)

render_init_img = render_init['render']

render_img_np = render_init_img.permute(1, 2, 0).detach().cpu().numpy()
render_img_np = (render_img_np * 255).astype(np.uint8)
if render_img_np.shape[0] == 3:
    render_img_np = np.transpose(render_img_np, (1, 2, 0))

# 更改rgb改为gbr
img_cv = render_img_np[:, :, [2, 1, 0]] 

cv2.imshow('Rendered init imag', img_cv)
cv2.waitKey()  # 按任意键关闭窗口
cv2.destroyAllWindows()


class_gsrefine = GS_refine()


imag0_rgb = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
imag0_rgb_torch = torch.from_numpy(imag0_rgb).float().to('cuda')
output = class_gsrefine.GS_Refiner(image=imag0_rgb_torch,mask=None,init_camera=camera,gaussians=obj_gaussians,return_loss=True)




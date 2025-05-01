
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


def visualize_point_cloud_with_coordinate(points, camera_pose=None):
    # 创建Open3D点云对象
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    
    # 估计法线（可选，使可视化效果更好）
    pcd.estimate_normals()
    
    # 创建坐标系对象
    coordinate_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.01, origin=[0, 0, 0])
    
    # 创建可视化窗口
    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name='3D Point Cloud with Coordinate System', width=800, height=600)
    
    # 添加点云和坐标系到可视化
    vis.add_geometry(pcd)
    vis.add_geometry(coordinate_frame)
    
    # 如果提供了相机位姿，添加相机坐标系
    if camera_pose is not None:
        camera_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.5, origin=camera_pose[:3,3])
        camera_frame.rotate(camera_pose[:3,:3], center=camera_pose[:3,3])
        vis.add_geometry(camera_frame)
    
    # 设置渲染选项
    opt = vis.get_render_option()
    opt.background_color = np.asarray([0, 0, 0])  # 黑色背景
    opt.point_size = 1.5  # 点的大小
    
    # 设置视角
    ctr = vis.get_view_control()
    ctr.set_front([0, 0, -1])  # 设置相机朝向
    ctr.set_up([0, -1, 0])     # 设置相机上方向
    
    # 运行可视化
    vis.run()
    vis.destroy_window()




gs_path = "/media/wangxiao/Newsmy/linemod/linemod_pgsr/000001/test/point_cloud/iteration_30000/point_cloud.ply"

parser = ArgumentParser()
gaussian_ModelP = ModelParams(parser)
gaussian_PipeP = PipelineParams(parser)
gaussian_OptimP = OptimizationParams(parser)
gaussian_BG = torch.zeros((3), device='cuda')



# fx = 572.4114
# fy = 573.57043
# cx = 325.2611
# cy = 242.04899


fx = 2*572.4114
fy = 2*573.57043
cx = 325.2611
cy = 242.04899

K_org = np.array([[fx, 0, cx],
               [0, fy, cy],
               [0,  0,  1]])

# [{"cam_R_m2c": [-0.52573111, 0.85065081, 0.0, 0.84825128, 0.52424812, -0.07505775, -0.06384793, -0.03946019, -0.99717919], "cam_t_m2c": [-0.0, 0.0, 400.0],
# T_init = np.array([[-0.52573111, 0.85065081, 0.0,0],
#                     [0.84825128, 0.52424812, -0.07505775,0],
#                     [-0.06384793, -0.03946019, -0.99717919,0.4],
#                     [0,0,0,1]])

T_init = np.array([[-0.52573111, 0.85065081, 0.0,0],
                    [0.84825128, 0.52424812, -0.07505775,0],
                    [-0.06384793, -0.03946019, -0.99717919,0.4],
                    [0,0,0,1]])





# 一个camera队列
# cameras 中储存的R 都是从相机坐标系变换到世界坐标系 的转置 距离不变
# width = 640
# hight = 480

width = 640*2
hight = 480*2

fovx = 2*math.atan(width/(2*fx))
fovy = 2*math.atan(hight/(2*fy))

camera_base = Camera(T_init[:3,:3].T,T_init[:3,3],fovx,fovy,cx,cy,width,hight,preload_img=False)

obj_gaussians = GaussianModel(3)
obj_gaussians.load_ply(gs_path)

out = GS_Renderer(camera_base,obj_gaussians,gaussian_PipeP, gaussian_BG)
depth_map = out['plane_depth']

depth_map = np.squeeze(depth_map.detach().cpu().numpy())

y_indices, x_indices = np.nonzero(depth_map)
values = depth_map[y_indices, x_indices]


        # xy_org.append([y1, x1])


K_inv = camera_base.get_inv_k().cpu().numpy()
# 获取世界坐标系在相机坐标系中的表达，因为Camera中的R是以转置的形式储存的，所以取出来的时候要转置一下
R = camera_base.R.T
t = camera_base.T.reshape(3, 1)
T = np.eye(4)
T[:3, :3] = R
T[:3, 3] = t.flatten()

# 改成了从世界坐标系到camera坐标系
# T = np.linalg.inv(T)
T_inv = np.eye(4)
T_inv[:3, :3] = T[:3, :3].T
T_inv[:3, 3] = -T_inv[:3, :3] @ T[:3, 3]



# print(f"x={x1 + 512},y1 = {y1}... z = {z}")

xyz_gs = []
# 将所有像素坐标转换为齐次坐标
uv_homogeneous = np.column_stack([x_indices, y_indices, np.ones_like(x_indices)])

# 批量归一化
normalized_coords = (K_inv @ uv_homogeneous.T).T

# 获取所有深度值
z_values = depth_map[y_indices, x_indices]

# 计算相机坐标系下的3D坐标
camera_coords = normalized_coords * z_values[:, np.newaxis]

# 转换为齐次坐标
camera_coords_homogeneous = np.column_stack([camera_coords, np.ones_like(x_indices)])

# 批量转换到世界坐标系
world_coords = (T_inv @ camera_coords_homogeneous.T).T
world_coords = world_coords[:, :3]

xyz_gs = world_coords.tolist()


visualize_point_cloud_with_coordinate(xyz_gs)






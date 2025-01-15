import cv2
import math
import torch
import numpy as np



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


def rz_torch(theta):
    """
    创建一个绕 Z 轴旋转的 4x4 齐次矩阵。

    参数:
    theta (float): 旋转角度（以弧度为单位）

    返回:
    numpy.ndarray: 4x4 齐次矩阵
    """
    return torch.tensor([
        [np.cos(theta), -np.sin(theta), 0 ,0],
        [np.sin(theta),  np.cos(theta), 0, 0],
        [0,             0,              1 ,0],
        [0,0,0,1]
    ], dtype=torch.float, device='cuda')

def box2gscamera(width,height,K:np,box:np,camera_num:int = 4):
    '''
    此函数通过box和参数生成多个gs的camera
    K:相机内参矩阵,np数组格式
    box 边界框np数组格式
    '''

    cameras =[]


    boxponit = convert_to_3d_points(box)
    # 转换成内参矩阵
    K = convert2K(K)
    # item 是转化为标量
    fx = K[0,0].item()
    fy = K[1,1].item()
    cx = K[0,2].item()
    cy = K[1,2].item()


    # max_distance返回最远的两个点
    farthest_pair, max_line = find_farthest_points(boxponit)

    
    # 这里的

    T_init = poses_init(K=K,width=width,height=height,L=max_line)

    
    theta = 2 * math.pi / camera_num
    
    # 计算 cos(theta) 和 sin(theta)
    cos_theta = math.cos(theta)
    sin_theta = math.sin(theta)
    
    # 创建 4x4 的旋转矩阵
    R_y = torch.tensor([
        [cos_theta, 0, sin_theta, 0],
        [0, 1, 0, 0],
        [-sin_theta, 0, cos_theta, 0],
        [0, 0, 0, 1]
    ], dtype=torch.float, device='cuda')

    fovx = 2 * np.arctan(width / (2 * fx))
    fovy = 2 * np.arctan(height / (2 * fy))
    

    # 补充代码

    # 补充代码
    # for i in range(camera_num):
    #     test = torch.matrix_power(R_y,i)
    #     print(test)
    # 处理相机数据
    # @rz_torch(math.pi)
    for i in range(camera_num):
        if i == 0 :
            T_camera = rz_torch(math.pi) @ T_init@torch.eye(4,dtype=torch.float32,device='cuda')
        else:
            T_camera = rz_torch(math.pi) @ T_init@torch.matrix_power(R_y,i)
        T_camera = T_camera.cpu().numpy()

        # print(f"T_camera = {T_camera}")


        # 创建原始坐标系
        original_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1.0, origin=[0, 0, 0])
        # 创建变换后的坐标系
        transformed_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1, origin=[0, 0, 0])
        transformed_frame.transform(np.linalg.inv(T_camera))
        # 可视化
        # o3d.visualization.draw_geometries([original_frame, transformed_frame])
        
        camera = Camera(colmap_id=0,R=T_camera[:3,:3].T,T=T_camera[:3,3],
                        FoVx=fovx,FoVy=fovy,Cx=cx,Cy=cy,image_height=height,image_width=width,
                        image_name='',image_path='',uid=0,preload_img=False)
        cameras.append(camera)
        pass
    # 返回的是一个相机队列
    return cameras


def read_txt(file):
    with open(file, 'r') as f:
        bbox_coords = []
        # 每次迭代时会读取每一行的内容
        for line in f:
            coords = list(map(float, line.strip().split()))
            bbox_coords.extend(coords)
    return bbox_coords

def convert_to_3d_points(bbox_coords):
    """
    将一维列表转换为三维点的列表。
    
    :param bbox_coords: 一维列表，包含24个浮点数，表示8个三维点
    :return: 三维点的列表，每个点是一个长度为3的列表 [x, y, z]
    """
    if len(bbox_coords) != 24:
        raise ValueError("bbox_coords 应该包含24个元素")
    
    points = [bbox_coords[i:i+3] for i in range(0, len(bbox_coords), 3)]
    return points
def convert2K(K_array):
    '''
    返回torch类型的矩阵
    '''
    
    if len(K_array) != 9:
        raise ValueError("bbox_coords 应该包含9个元素")
    
    K = [K_array[i:i+3] for i in range(0, len(K_array), 3)]
    K_tensor = torch.tensor(K, dtype=torch.float, device='cuda')
    return K_tensor

def poses_init(K:torch.tensor,width,height,L):
    '''
    L:是最大长度
    K:是相机内参矩阵
    '''
    fx = K[0,0]
    fy = K[1,1]

    cx = K[0,2]
    cy = K[1,2]

    if width>=height:
        min_length = height
        f = K[1,1]
    else:
        min_length = width
        f = K[0,0]
    
    min_distance = (L * f) / min_length
    
    T_init = torch.eye(4,dtype=torch.float32, device='cuda')

    T_init[2,3] = min_distance

    # 定义旋转角度（-45度，即顺时针旋转45度）
    theta = -math.pi / 8  # -45度对应的弧度

    # 计算 cos(theta) 和 sin(theta)
    cos_theta = math.cos(theta)
    sin_theta = math.sin(theta)

    # 创建 4x4 的旋转矩阵
    R_x = torch.tensor([
        [1, 0, 0, 0],
        [0, cos_theta, -sin_theta, 0],
        [0, sin_theta, cos_theta, 0],
        [0, 0, 0, 1]
    ], dtype=torch.float32,device='cuda')

    initpose = T_init@R_x

    initpose[0,3] = -(min_distance *width/2 -cx*min_distance)/fx
    initpose[1,3] = -(min_distance *height/2 -cy*min_distance)/fy


    return initpose

def find_farthest_points(points):
    """
    找到距离最远的两个点。
    
    :param points: 三维点的列表，每个点是一个长度为3的列表 [x, y, z]
    :return: 距离最远的两个点及其距离
    """
    max_distance = 0
    farthest_pair = None
    
    for i in range(len(points)):
        for j in range(i + 1, len(points)):
            distance = np.linalg.norm(np.array(points[i]) - np.array(points[j]))
            if distance > max_distance:
                max_distance = distance
                farthest_pair = (points[i], points[j])
    
    return farthest_pair, max_distance


def match2xy_xyz(kporg,kpdeep,img_deep:np.ndarray,matchs:List[cv2.DMatch],camera:Camera):
    '''
    img_deep[w,h,1] 单位为m 深度图
    matchs 匹配信息
    camera 相机

    返回的是世界坐标系下的点的xyz坐标
    '''
    xy_org = []
    xyz_gs = []

    # cv2.imshow('chack deep', img_deep)
    # cv2.waitKey()  # 按任意键关闭窗口
    # cv2.destroyAllWindows()

    for match in matchs:
        # match.
        img_idx = match.queryIdx
        deep_idx = match.trainIdx

        # 获取关键点位置
        (x1, y1) = kporg[img_idx].pt
        (deepx, deepy) = kpdeep[deep_idx].pt

        xy_org.append([x1, y1])
        # xy_org.append([y1, x1])


        K_inv = camera.get_inv_k().cpu().numpy()
        R = camera.R.T
        t = camera.T.reshape(3, 1)
        T = np.eye(4)
        T[:3, :3] = R
        T[:3, 3] = t.flatten()

        # 改成了从世界坐标系到camera坐标系
        T = np.linalg.inv(T)



        # 取数值的时候xy要调换  
        z = img_deep[int(deepy),int(deepx)]

        uv_homogeneous = np.array([deepx, deepy, 1])

        normalized_coords = K_inv @ uv_homogeneous

        # 计算相机坐标系下的3D坐标
        X_c = z * normalized_coords[0]
        Y_c = z * normalized_coords[1]
        Z_c = z

        # 将相机坐标转换为世界坐标系下的3D坐标
        camera_coords = np.array([X_c, Y_c, Z_c,[1]]).reshape(4, 1)
        world_coords = T @ camera_coords

        world_coords = world_coords[:3,0]

        xyz_gs.append(world_coords)

    return xy_org,xyz_gs
    pass


def match2xy_xyz_LightGlue(kporg,kpdeep,img_deep:np.ndarray,camera:Camera):
    '''
    img_deep[w,h,1] 单位为m 深度图
    matchs 匹配信息
    camera 相机

    返回的是世界坐标系下的点的xyz坐标
    '''
    xy_org = []
    xyz_gs = []

    # cv2.imshow('chack deep', img_deep)
    # cv2.waitKey()  # 按任意键关闭窗口
    # cv2.destroyAllWindows()

    for i in range(kporg.shape[0]):
        # match.

        # 获取关键点位置
        (x1, y1) = kporg[i].cpu()                   
        # (deepy, deepx) = kpdeep[i].cpu()
        (deepx, deepy) = kpdeep[i].cpu()



        xy_org.append([x1, y1])
        # xy_org.append([y1, x1])


        K_inv = camera.get_inv_k().cpu().numpy()
        # 获取世界坐标系在相机坐标系中的表达，因为Camera中的R是以转置的形式储存的，所以取出来的时候要转置一下
        R = camera.R.T
        t = camera.T.reshape(3, 1)
        T = np.eye(4)
        T[:3, :3] = R
        T[:3, 3] = t.flatten()

        # 改成了从世界坐标系到camera坐标系
        # T = np.linalg.inv(T)
        T_inv = np.eye(4)
        T_inv[:3, :3] = T[:3, :3].T
        T_inv[:3, 3] = -T_inv[:3, :3] @ T[:3, 3]



        # 取数值的时候xy要调换  
        z = img_deep[int(deepy),int(deepx)]

        # print(f"x={x1 + 512},y1 = {y1}... z = {z}")


        uv_homogeneous = np.array([deepx, deepy, 1])

        normalized_coords = K_inv @ uv_homogeneous

        # 计算相机坐标系下的3D坐标
        X_c = z * normalized_coords[0]
        Y_c = z * normalized_coords[1]
        Z_c = z

        # 将相机坐标转换为世界坐标系下的3D坐标
        camera_coords = np.array([X_c, Y_c, Z_c,[1]]).reshape(4, 1)
        world_coords = T_inv @ camera_coords

        world_coords = world_coords[:3,0]

        xyz_gs.append(world_coords)


        # 可视化
        gs_xyz_ws = np.array(xyz_gs, dtype=np.float32)



        print(f"gs_xyz_ws.shape = {gs_xyz_ws.shape}")

        # pcd = o3d.geometry.PointCloud()

        # 将 NumPy 数组转换为 Open3D 的点云格式
        # pcd.points = o3d.utility.Vector3dVector(gs_xyz_ws)
        # coordinate_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1, origin=[0, 0, 0])
        # o3d.visualization.draw_geometries([pcd,coordinate_frame])



    return xy_org,xyz_gs
    pass
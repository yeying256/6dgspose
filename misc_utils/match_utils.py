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

import matplotlib.pyplot as plt


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


# def visualize_matches_loftr(img1: np.ndarray, img2: np.ndarray, mkpts1: np.ndarray, mkpts2: np.ndarray):
#         h1, w1 = img1.shape[:2]
#         h2, w2 = img2.shape[:2]
#         img_vis = np.zeros((max(h1, h2), w1 + w2, 3), dtype=np.uint8)
#         if len(img1.shape) == 2:
#             img_vis[:h1, :w1, :] = cv2.cvtColor(img1, cv2.COLOR_GRAY2BGR)
#         else:
#             img_vis[:h1, :w1, :] = img1
#         if len(img2.shape) == 2:
#             img_vis[:h2, w1:, :] = cv2.cvtColor(img2, cv2.COLOR_GRAY2BGR)
#         else:
#             img_vis[:h2, w1:, :] = img2

#         for pt1, pt2 in zip(mkpts1, mkpts2):
#             pt1_int = tuple(map(int, pt1))
#             pt2_int = tuple(map(int, pt2))
#             cv2.circle(img_vis, pt1_int, 2, (0, 255, 0), -1)
#             cv2.circle(img_vis, (pt2_int[0] + w1, pt2_int[1]), 2, (0, 0, 255), -1)
#             cv2.line(img_vis, pt1_int, (pt2_int[0] + w1, pt2_int[1]), (0, 255, 0), 1)

#         plt.figure(figsize=(15, 5))
#         plt.imshow(cv2.cvtColor(img_vis, cv2.COLOR_BGR2RGB))
#         plt.show()

def visualize_matches_loftr(img1: np.ndarray, img2: np.ndarray, mkpts1: np.ndarray, mkpts2: np.ndarray):
    # 获取图像的高度和宽度
    h1, w1 = img1.shape[:2]
    h2, w2 = img2.shape[:2]
    
    # 创建一个空白画布，用于拼接两张图像
    img_vis = np.zeros((max(h1, h2), w1 + w2, 3), dtype=np.uint8)
    
    # 将第一张图像放在画布的左侧
    if len(img1.shape) == 2:  # 如果是灰度图，转换为彩色图
        img_vis[:h1, :w1, :] = cv2.cvtColor(img1, cv2.COLOR_GRAY2BGR)
    else:
        img_vis[:h1, :w1, :] = img1
    
    # 将第二张图像放在画布的右侧
    if len(img2.shape) == 2:  # 如果是灰度图，转换为彩色图
        img_vis[:h2, w1:, :] = cv2.cvtColor(img2, cv2.COLOR_GRAY2BGR)
    else:
        img_vis[:h2, w1:, :] = img2
    
    # 绘制匹配点
    for pt1, pt2 in zip(mkpts1, mkpts2):
        pt1_int = tuple(map(int, pt1))  # 将点坐标转换为整数
        pt2_int = tuple(map(int, pt2))
        
        # 在第一张图像上绘制绿色点
        cv2.circle(img_vis, pt1_int, 2, (0, 255, 0), -1)
        
        # 在第二张图像上绘制红色点（注意 x 坐标需要加上第一张图像的宽度）
        cv2.circle(img_vis, (pt2_int[0] + w1, pt2_int[1]), 2, (0, 0, 255), -1)
        
        # 绘制连接线
        cv2.line(img_vis, pt1_int, (pt2_int[0] + w1, pt2_int[1]), (0, 255, 0), 1)
    
    # 使用 OpenCV 显示图像
    cv2.imshow('Matches', img_vis)
    cv2.waitKey(0)  # 等待用户按键
    cv2.destroyAllWindows()  # 关闭所有窗口

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

def box2gscamera(width,height,K:np,box:np,camera_num:int = 4,debug = False,camera_distance_scale = 1.0):
    '''
    此函数通过box和参数生成多个gs的camera
    K:相机内参矩阵,np数组格式
    box 边界框np数组格式
    '''

    cameras =[]

    # 假设 box 是一个输入变量
    if isinstance(box, np.ndarray):  # 检查 box 是否为 NumPy 数组
        if box.shape == (8, 3):  # 检查 shape 是否为 (8, 3)
            boxpoint = box
        else:
            # 调用 convert_to_3d_points 函数进行转换
            boxpoint = convert_to_3d_points(box)
    else:
        raise TypeError("The input 'box' must be a NumPy array.")
    # 转换成内参矩阵
    K = convert2K(K)
    # item 是转化为标量
    fx = K[0,0].item()
    fy = K[1,1].item()
    cx = K[0,2].item()
    cy = K[1,2].item()


    # max_distance返回最远的两个点
    farthest_pair, max_line = find_farthest_points(boxpoint)

    
    # 这里的

    T_init = poses_init(K=K,width=width,height=height,L=max_line*camera_distance_scale)

    
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
        if debug:
            o3d.visualization.draw_geometries([original_frame, transformed_frame])
        # o3d.visualization.draw_geometries([original_frame, transformed_frame])
        
        camera = Camera(colmap_id=0,R=T_camera[:3,:3].T,T=T_camera[:3,3],
                        FoVx=fovx,FoVy=fovy,Cx=cx,Cy=cy,image_height=height,image_width=width,
                        image_name='',image_path='',uid=0,preload_img=False)
        cameras.append(camera)
        pass
    # 返回的是一个相机队列
    return cameras


def box2gscamera_linemod(width,height,K:np,box:np,camera_num:int = 4):
    '''
    此函数通过box和参数生成多个gs的camera
    K:相机内参矩阵,np数组格式
    box 边界框np数组格式
    '''

    cameras =[]


    # boxponit = convert_to_3d_points(box)
    # 假设 box 是一个输入变量
    if isinstance(box, np.ndarray):  # 检查 box 是否为 NumPy 数组
        if box.shape == (8, 3):  # 检查 shape 是否为 (8, 3)
            boxpoint = box
        else:
            # 调用 convert_to_3d_points 函数进行转换
            boxpoint = convert_to_3d_points(box)
    else:
        raise TypeError("The input 'box' must be a NumPy array.")
    # 转换成内参矩阵
    K = convert2K(K)
    # item 是转化为标量
    fx = K[0,0].item()
    fy = K[1,1].item()
    cx = K[0,2].item()
    cy = K[1,2].item()


    # max_distance返回最远的两个点
    farthest_pair, max_line = find_farthest_points(boxpoint)

    
    # 这里的

    T_init = poses_init_lists(K=K,width=width,height=height,L=max_line,init_num=camera_num)

    
    

    fovx = 2 * np.arctan(width / (2 * fx))
    fovy = 2 * np.arctan(height / (2 * fy))

    for i in range(len(T_init)):

        T_camera = T_init[i].cpu().numpy()

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
    return np.array(points)
def convert2K(K_array):
    '''
    返回torch类型的矩阵
    '''
    
    # 如果 K_array 是一个 3x3 的 NumPy 数组
    if isinstance(K_array, np.ndarray) and K_array.shape == (3, 3):
        # 直接转换为 torch.Tensor 并返回
        return torch.tensor(K_array, dtype=torch.float, device='cuda')
    
    # 如果 K_array 是一个长度为 9 的一维数组或列表
    if len(K_array) != 9:
        raise ValueError("K_array 应该包含9个元素")
    
    # 将一维数组转换为 3x3 的矩阵
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

def T_rot_axis(theta , axis = 'z',return_torch=True,is_cuda = True):
    '''
    根据角度和旋转轴，返回一个描述旋转的齐次坐标变换的矩阵
    '''

        # 将角度转换为弧度制（如果输入是角度）
    theta = np.radians(theta) if theta > 2 * np.pi else theta

    # 初始化齐次坐标变换矩阵
    T_rot = np.eye(4)

    # 根据旋转轴计算旋转矩阵 R
    if axis.lower() == 'x':
        R = np.array([
            [1, 0, 0],
            [0, np.cos(theta), -np.sin(theta)],
            [0, np.sin(theta), np.cos(theta)]
        ])
    elif axis.lower() == 'y':
        R = np.array([
            [np.cos(theta), 0, np.sin(theta)],
            [0, 1, 0],
            [-np.sin(theta), 0, np.cos(theta)]
        ])
    elif axis.lower() == 'z':
        R = np.array([
            [np.cos(theta), -np.sin(theta), 0],
            [np.sin(theta), np.cos(theta), 0],
            [0, 0, 1]
        ])
    else:
        raise ValueError("无效的旋转轴！请指定 'x', 'y' 或 'z'。")

    # 将旋转矩阵嵌入齐次坐标变换矩阵
    T_rot[:3, :3] = R

    if return_torch == True:
        T_rot = torch.from_numpy(T_rot)

        if is_cuda == True:
            T_rot = T_rot.cuda()
        pass

    return T_rot
    pass

def poses_init_lists(K:torch.tensor,width,height,L,init_num=6):
    '''
    这个函数可以返回一系列的初始化相机位姿
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
    

    delta_theta = math.pi/ (init_num+1)

    # 生成很多T_init
    T_init_list = []
    for i in range(init_num):

        T_init = torch.eye(4,dtype=torch.float64, device='cuda')
        T_init[2,3] = min_distance
        T_init[0,3] = -(min_distance *width/2 -cx*min_distance)/fx
        T_init[1,3] = -(min_distance *height/2 -cy*min_distance)/fy

        rot_R1 = T_rot_axis(-math.pi/2 + delta_theta*(i+1),axis='x',return_torch=True,is_cuda=True)
        for i in range(init_num):
            rot_R2 = T_rot_axis(2 * delta_theta*(i+1),axis='z',return_torch=True,is_cuda=True)


            T_init_list.append(T_init @ rot_R1 @ rot_R2)

        pass

    return T_init_list

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


def match2xy_xyz_LightGlue(kporg,
                           kpdeep,
                           img_deep:np.ndarray,
                           camera:Camera):
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
        camera_coords = np.array([X_c, Y_c, Z_c,1]).reshape(4, 1)
        world_coords = T_inv @ camera_coords

        world_coords = world_coords[:3,0]

        xyz_gs.append(world_coords)


        # 可视化
        gs_xyz_ws = np.array(xyz_gs, dtype=np.float32)



        # print(f"gs_xyz_ws.shape = {gs_xyz_ws.shape}")

        # pcd = o3d.geometry.PointCloud()

        # 将 NumPy 数组转换为 Open3D 的点云格式
        # pcd.points = o3d.utility.Vector3dVector(gs_xyz_ws)
        # coordinate_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1, origin=[0, 0, 0])
        # o3d.visualization.draw_geometries([pcd,coordinate_frame])



    return xy_org,xyz_gs
    pass



def gradient_deep_mask(rander_deep:np.ndarray,
                       ksize = 3,
                       TG = 50,
                        debug=False):
    '''
    ksize: 梯度核的大小, 默认为3。
    TG: Thresholded Gradient: 梯度阈值化，用于提取深度图中的边缘。
    '''
        
    # 1. 计算深度图的梯度
    gradient_x = cv2.Sobel(rander_deep, cv2.CV_64F, 1, 0, ksize=ksize)
    gradient_y = cv2.Sobel(rander_deep, cv2.CV_64F, 0, 1, ksize=ksize)
    gradient_magnitude = np.sqrt(gradient_x**2 + gradient_y**2)

    # 归一化梯度幅值
    gradient_magnitude = cv2.normalize(gradient_magnitude, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)

    # 2. 阈值化梯度幅值，得到边缘掩码
    _, depth_edges = cv2.threshold(gradient_magnitude, TG, 255, cv2.THRESH_BINARY)

    # 显示深度边缘
    if debug:
        cv2.imshow('Depth Edges', depth_edges)
        # cv2.waitKey(0)

    return depth_edges

    pass


def crop_and_tile_mask(image, mask, K,debug = False):
    """
    将mask区域裁剪并平铺到整个图像中，并更新相机的内参矩阵 K。

    参数:
    - image: OpenCV读取的原始图像（BGR格式）
    - mask: OpenCV读取的mask图像（单通道灰度图）
    - K: 相机的内参矩阵 (3x3)

    返回:
    - result_image: 处理后的图像
    - new_K: 更新后的内参矩阵
    """
    # 检查输入是否有效
    if image is None or mask is None:
        raise ValueError("输入的图像或mask为空，请检查数据是否正确。")
    if K.shape != (3, 3):
        raise ValueError("K 矩阵必须是 3x3 的矩阵。")

    # 将mask转换为三通道，以便与图像进行按位与操作
    mask = cv2.merge([mask, mask, mask])

    # 应用mask
    masked_image = cv2.bitwise_and(image, mask)

    # 找到mask的边界框
    coords = cv2.findNonZero(mask[:, :, 0])  # 使用mask的第一个通道
    x, y, w, h = cv2.boundingRect(coords)

    # 裁剪图像
    cropped_image = masked_image[y:y+h, x:x+w]

    # 获取原始图像的大小
    original_height, original_width = image.shape[:2]

    # 将裁剪后的图像缩放到原始图像的大小
    resized_cropped_image = cv2.resize(cropped_image, (original_width, original_height))

    # 将缩放后的图像平铺到整个图像中
    result_image = resized_cropped_image

    # 更新 K 矩阵
    # 新的主点坐标需要根据裁剪区域调整
    new_K = K.copy()
    new_K[0, 2] -= x  # 更新 cx
    new_K[1, 2] -= y  # 更新 cy

    # 显示结果
    if debug:
        cv2.imshow('Result Image', result_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


    return result_image, new_K

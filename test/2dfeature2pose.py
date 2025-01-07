
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
        o3d.visualization.draw_geometries([original_frame, transformed_frame])
        
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

# /media/wangxiao/Newsmy/dataset_LINEMOD/datasets/train_data/0499-tiramisufranzzi-box/tiramisufranzzi-1/color
# 读取图像
# imag_path = "/media/wangxiao/Newsmy/dataset_LINEMOD/datasets/train_data/0575-saltbottle-bottle/saltbottle-1/color"
# img1 = cv2.imread(f'{imag_path}/0.png', cv2.IMREAD_GRAYSCALE)
# intrin1path = "/media/wangxiao/Newsmy/dataset_LINEMOD/datasets/train_data/0575-saltbottle-bottle/saltbottle-1/intrin_ba/0.txt"
# img2 = cv2.imread(f'{imag_path}/10.png', cv2.IMREAD_GRAYSCALE)
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

bbox_coords = read_txt(boxpath)
initin1 = read_txt(intrin1path)

bbox_coords


height, width = img1.shape[:2]

# 一个camera队列
# cameras 中储存的R 都是从相机坐标系变换到世界坐标系 的转置 距离不变
cameras = box2gscamera(box=bbox_coords,K=initin1,height=height,width=width,camera_num=7)


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
    gray_image = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)

    sift = cv2.SIFT_create()

    # 关键点
    # 
    # kp1[i].pt[0]：表示关键点的 x 坐标（水平方向），从左到右增加。
    # kp1[i].pt[1]：表示关键点的 y 坐标（垂直方向），从上到下增加。

    kp1, des1 = sift.detectAndCompute(img1, None)
    kp2, des2 = sift.detectAndCompute(gray_image, None)

    # 使用FLANN匹配器
    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)

    flann = cv2.FlannBasedMatcher(index_params, search_params)
    matches = flann.knnMatch(des1, des2, k=2)

    # 应用比例测试筛选匹配点
    good_matches = []
    for m, n in matches:
        if m.distance < 0.4 * n.distance:
            good_matches.append(m)

    # 返回img和物体坐标系上的xyz
    if not len(good_matches) < 2:
        img_xy,gs_xyz_w = match2xy_xyz(kporg=kp1,kpdeep=kp2,img_deep=depth_normalized,matchs=good_matches,camera=camera)
        img_xys.extend(img_xy)
        gs_xyz_ws.extend(gs_xyz_w)
    #如果有匹配点
    # if not good_matches.count == 0:

    #     pass

    # 这里的distance是特征的相似度
    # 绘制匹配结果
    img_matches = cv2.drawMatches(img1, kp1, gray_image, kp2, good_matches, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    

    cv2.imshow('Rendered Image', img_matches)
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
pnp_reprojection_error=5
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

K = convert2K(initin1)
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






# 初始化SIFT特征检测器
sift = cv2.SIFT_create()

# 检测特征点并计算描述符
kp1, des1 = sift.detectAndCompute(img1, None)
kp2, des2 = sift.detectAndCompute(img2, None)

# 使用FLANN匹配器
FLANN_INDEX_KDTREE = 1
index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
search_params = dict(checks=50)

flann = cv2.FlannBasedMatcher(index_params, search_params)
matches = flann.knnMatch(des1, des2, k=2)

# 应用比例测试筛选匹配点
good_matches = []
for m, n in matches:
    if m.distance < 0.7 * n.distance:
        good_matches.append(m)
# 这里的distance是特征的相似度
# 绘制匹配结果
img_matches = cv2.drawMatches(img1, kp1, img2, kp2, good_matches, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

# 显示结果
cv2.imshow('Feature Matches', img_matches)
cv2.waitKey(0)
cv2.destroyAllWindows()

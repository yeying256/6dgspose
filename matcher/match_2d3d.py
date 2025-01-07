
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

from datasets.datasets import datasets

from tqdm import tqdm


class match_2d3d():
    """
    这个类用于匹配2d和3d
    """
    def __init__(self,config) -> None:
        self.import_extractor(config)

        self.parser = ArgumentParser()
        self.gaussian_ModelP = ModelParams(self.parser)
        self.gaussian_PipeP = PipelineParams(self.parser)
        self.gaussian_OptimP = OptimizationParams(self.parser)
        self.gaussian_BG = torch.zeros((3), device='cuda')

        # 会绕gs模型生成一堆camera
        self.camera_num = config['match']['camera_num']
        self.pnp_reprojection_error = config['match']['pnp_reprojection_error']

        self.use_pose_refine = config['match']['use_pose_refine']
        
        pass

    def import_extractor(self,config):
        if config['match']['extractor']['name'] == "SuperPoint":
            max_num_keypoints = config['match']['extractor']['max_num_keypoints']
            self.extractor = SuperPoint(max_num_keypoints=max_num_keypoints).eval().cuda()  # load the extractor
        

        if config['match']['matcher']['name'] == "LightGlue":
            # 读取特征匹配类型
            features = config['match']['matcher']['features']
            self.matcher = LightGlue(features=features).eval().cuda()  # load the matcher

        self.match_confidence_threshold = config['match']['confidence_threshold']

        
        pass


    def match2d3d(self,img_rgb:np.ndarray,img_rander:torch.Tensor,rander_deep:np.ndarray,camera:Camera):
        '''
        img_rgb 是一个numpy数组，表示GBR图像,是opencv直接读取出来的
        '''
        # 如果img_rgb是numpy数组，则转换为torch.Tensor
        if isinstance(img_rgb, np.ndarray):
            # opencv直接读取出来的，所以要将GBR换成RGB
            img_rgb = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2RGB)
            # 换成torch格式，并且HWC 变成CHW
            img_rgb = torch.from_numpy(img_rgb).permute(2, 0, 1).float().cuda()
            img_rgb = img_rgb / 255.0

        depth_np = rander_deep.permute(1, 2, 0).detach().cpu().numpy()

        render_img_np = img_rander.permute(1, 2, 0).detach().cpu().numpy()
        render_img_np = (render_img_np * 255).astype(np.uint8)
        if render_img_np.shape[0] == 3:
            render_img_np = np.transpose(render_img_np, (1, 2, 0))

        # 更改rgb改为gbr
        img_cv = render_img_np[:, :, [2, 1, 0]]
        # 调整通道顺序：从 [H, W, C] 到 [C, H, W]
        img_cv = img_cv.transpose(2, 0, 1)

        torch_render = torch.from_numpy(img_cv).float().cuda()
        torch_render = torch_render / 255.0
        
        # extract local features
        feats0 = self.extractor.extract(torch_render)  # auto-resize the image, disable with resize=None 渲染图像
        feats1 = self.extractor.extract(img_rgb) #目标图像

        # match the features
        matches01 = self.matcher({'image0': feats0, 'image1': feats1})
        feats0, feats1, matches01 = [rbd(x) for x in [feats0, feats1, matches01]]  # remove batch dimension
        matches = matches01['matches']  # indices with shape (K,2)
        # points0 = feats0['keypoints'][matches[..., 0]]  # coordinates in image #0, shape (K,2)
        # points1 = feats1['keypoints'][matches[..., 1]]  # coordinates in image #1, shape (K,2)

        scores = matches01['scores']
 
        # 你可以根据需要在config中调整这个阈值
        valid_indices = scores > self.match_confidence_threshold
        matches = matches[valid_indices]
        filtered_scores = scores[valid_indices]

        # 获取过滤后的匹配点坐标
        points0 = feats0['keypoints'][matches[..., 0]]  # coordinates in image #0, shape (K', 2)
        points1 = feats1['keypoints'][matches[..., 1]]  # coordinates in image #1, shape (K', 2)
        img_xy = []
        gs_xyz_w = []
        if not len(matches) < 2:
            img_xy,gs_xyz_w = match_utils.match2xy_xyz_LightGlue(kporg=points1,kpdeep=points0,img_deep=depth_np,camera=camera)
        return img_xy,gs_xyz_w
    
    def camera_matchs(self,cameras:List[Camera],GS_model:GaussianModel,img_target:np.ndarray):
        '''
        匹配多张图片
        '''
        img_xys =[]
        gs_xyz_ws = []

        for camera in cameras:
            render = GS_Renderer(camera, GS_model, self.gaussian_PipeP, self.gaussian_BG)
            render_img = render['render']

            # 这里每个像素单位是m
            deepsimg = render['plane_depth']
            img_xy,gs_xyz_w = self.match2d3d(img_rander=render_img,img_rgb=img_target,rander_deep=deepsimg,camera=camera)

            img_xys.extend(img_xy)
            gs_xyz_ws.extend(gs_xyz_w)

        gs_xyz_ws = np.array(gs_xyz_ws, dtype=np.float32)
        img_xys = np.array(img_xys, dtype=np.float32)
        

        dist_coeffs = np.zeros(shape=[8, 1], dtype="float64")


        # 使用掩码过滤掉不符合几何约束的匹配点
        try:
            F, mask = cv2.findFundamentalMat(gs_xyz_ws, img_xys, cv2.FM_RANSAC, ransacReprojThreshold=8)
            inliers = mask.ravel() == 1
            gs_xyz_ws = gs_xyz_ws[inliers]
            img_xys = img_xys[inliers]
            return img_xys,gs_xyz_ws
        except:
            print("No inliers found.")
            return img_xys,gs_xyz_ws

            
        
        pass

    def pose_estimate(self,img_target:np.ndarray,GS_model:GaussianModel,initin1:List[float],bbox_coords:List[float]):

        # img1 = cv2.imread(f'{imag_path}/55.png', cv2.IMREAD_GRAYSCALE)
        height, width = img_target.shape[:2]



        cameras = match_utils.box2gscamera(box=bbox_coords,K=initin1,height=height,width=width,camera_num=self.camera_num)


        img_xys,gs_xyz_ws = self.camera_matchs(cameras= cameras,GS_model= GS_model,img_target=img_target)

        dist_coeffs = np.zeros(shape=[8, 1], dtype="float64")
        T_ = np.eye(4)
        try:
            _, rvec, tvec, inliers = cv2.solvePnPRansac(gs_xyz_ws, img_xys,cameras[0].get_k().cpu().numpy(),dist_coeffs,
                    reprojectionError=self.pnp_reprojection_error,
                    iterationsCount=10000,
                    flags=cv2.SOLVEPNP_EPNP)
            # 1. 将旋转向量 rvec 转换为旋转矩阵 R
            R_, _ = cv2.Rodrigues(rvec)

            # 2. 构建齐次变换矩阵 T
            # 初始化一个 4x4 的单位矩阵


            # 将旋转矩阵 R 放入 T 的前 3x3 区域
            T_[:3, :3] = R_

            # 将平移向量 t 放入 T 的前 3 行的最后一列
            T_[:3, 3] = tvec.flatten()
        except:
            print("No inliers found.")

        if self.use_pose_refine == True:
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

            render_init = GS_Renderer(camera, GS_model, self.gaussian_PipeP, self.gaussian_BG)

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


            imag0_rgb = cv2.cvtColor(img_target, cv2.COLOR_BGR2RGB)
            imag0_rgb_torch = torch.from_numpy(imag0_rgb).float().to('cuda')
            output = class_gsrefine.GS_Refiner(image=imag0_rgb_torch,mask=None,init_camera=camera,gaussians=GS_model,return_loss=True)
        
            # 返回的是4x4的np格式的位姿矩阵
            gs3d_delta_RT_np = output['gs3d_delta_RT']
            # gs3d_delta_RT_np[:3,:3] = 
            gs3d_pose_inv = np.linalg.inv(gs3d_delta_RT_np)
            T_ = (T_@gs3d_delta_RT_np)


        return T_
    

    def pose_estimate_batch(self,img_target_dir:List[str],GS_model_dir:str,initin1:List[str],bbox_coords:str):
        '''
        处理一个场景的位姿估计
        '''
        GS_model = GaussianModel(3)
        GS_model.load_ply(GS_model_dir)
        GS_model.clip_to_bounding_box_fromfile(bbox_coords)
        bbox_coords_np = match_utils.read_txt(bbox_coords)

        for i in tqdm(range(0,len(img_target_dir)),desc="Processing images"):
            img_target = cv2.imread(img_target_dir[i])
            initin_np = match_utils.read_txt(initin1[i])
            print(f"initin1[i] = {initin1[i]}")
            print(f"img_target_dir[i] = {img_target_dir[i]}")
            self.pose_estimate(img_target=img_target,GS_model=GS_model,initin1=initin_np,bbox_coords=bbox_coords_np)
            pass
    pass
    
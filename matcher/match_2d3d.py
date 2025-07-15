
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

from utils.graphics_utils import focal2fov,fov2focal


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
from misc_utils import gs_utils
from misc_utils import metric_utils
import string

from lightglue import LightGlue, SuperPoint, DISK, SIFT, ALIKED, DoGHardNet
from lightglue.utils import load_image, rbd

from kornia.feature import LoFTR

from misc_utils import match_utils
from misc_utils import misc
from torchvision.ops import roi_align


from datasets.datasets import datasets

from tqdm import tqdm

from configs import inference_cfg as CFG


from misc_utils import draw_bbox

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


def perform_segmentation_and_encoding(model_func, que_image, ref_database, device):
    with torch.no_grad():
        start_timer = time.time()
        
        if que_image.dim() == 3:
            que_image = que_image.unsqueeze(0)
        if que_image.shape[-1] == 3:
            que_image = que_image.permute(0, 3, 1, 2)
        que_feats = model_func.extract_DINOv2_feature(que_image.to(device))    
        pd_coarse_mask = model_func.query_cosegmentation(x_que=que_feats, 
                                                        x_ref=ref_database['obj_fps_feats'], 
                                                        ref_mask=ref_database['obj_fps_masks'],
                                                        ).sigmoid() # 1xCxHxW -> 1x1xHxW
        mask_threshold = CFG.coarse_threshold
        while True:
            que_binary_mask = (pd_coarse_mask.squeeze() >= mask_threshold).type(torch.uint8)
            if que_binary_mask.sum() < CFG.DINO_PATCH_SIZE**2:
                mask_threshold -= 0.01
                continue
            else:
                break
        _, pd_coarse_tight_scales, pd_coarse_centers = misc.torch_find_connected_component(
            que_binary_mask, include_supmask=CFG.CC_INCLUDE_SUPMASK, min_bbox_scale=CFG.DINO_PATCH_SIZE, return_bbox=True)

        pd_coarse_scales = pd_coarse_tight_scales * CFG.coarse_bbox_padding
        pd_coarse_bboxes = torch.stack([pd_coarse_centers[:, 0] - pd_coarse_scales / 2.0,
                                        pd_coarse_centers[:, 1] - pd_coarse_scales / 2.0,
                                        pd_coarse_centers[:, 0] + pd_coarse_scales / 2.0,
                                        pd_coarse_centers[:, 1] + pd_coarse_scales / 2.0], dim=-1)
        roi_RGB_crops = roi_align(que_image, boxes=[pd_coarse_bboxes],
                                  output_size=(CFG.zoom_image_scale, CFG.zoom_image_scale), 
                                  sampling_ratio=4) # 1x3xHxW -> Mx3xSxS
        
        if roi_RGB_crops.shape[0] == 1:
            rgb_img_crop = roi_RGB_crops  # 1x3xSxS 
            rgb_box_scale = pd_coarse_scales.squeeze(0)
            rgb_box_center = pd_coarse_centers.squeeze(0)
            rgb_box_tight_scale = pd_coarse_tight_scales.squeeze(0)
            rgb_img_feat = model_func.extract_DINOv2_feature(rgb_img_crop)
            rgb_crop_mask = model_func.query_cosegmentation(x_que=rgb_img_feat, 
                                                            x_ref=ref_database['obj_fps_feats'], 
                                                            ref_mask=ref_database['obj_fps_masks']).sigmoid()
        else:
            roi_img_feats, _, roi_dino_tokens = model_func.extract_DINOv2_feature(roi_RGB_crops, return_last_dino_feat=True)
            roi_img_masks = model_func.query_cosegmentation(x_que=roi_img_feats, 
                                                            x_ref=ref_database['obj_fps_feats'], 
                                                            ref_mask=ref_database['obj_fps_masks']).sigmoid() # KxCxSxS -> Kx1xSxS
            roi_obj_mask = torch_F.interpolate(roi_img_masks, 
                                                scale_factor=1.0/CFG.DINO_PATCH_SIZE, 
                                                mode='bilinear', align_corners=True, 
                                                recompute_scale_factor=True).flatten(2).permute(0, 2, 1).round() # Kx1xSxS -> KxLx1
            roi_dino_tokens = roi_obj_mask * roi_dino_tokens
            token_cosim = torch.einsum('klc,nc->kln', torch_F.normalize(roi_dino_tokens, dim=-1), 
                                                        torch_F.normalize(ref_database['obj_fps_dino_tokens'], dim=-1))
            if CFG.cosim_topk > 0:
                cosim_score = token_cosim.topk(dim=1, k=CFG.cosim_topk).values.mean(dim=-1).mean(dim=1)
            else:
                cosim_score = token_cosim.mean(dim=-1).sum(dim=1) / (1 + roi_obj_mask.squeeze(-1).sum(dim=1))  # KxLxN -> KxL -> K 
            
            optim_index = cosim_score.argmax()
            rgb_box_scale = pd_coarse_scales[optim_index]
            rgb_box_center = pd_coarse_centers[optim_index]
            rgb_box_tight_scale = pd_coarse_tight_scales[optim_index]
            
            rgb_img_crop = roi_RGB_crops[optim_index].unsqueeze(0)  # 1x3xSxS
            rgb_img_feat = roi_img_feats[optim_index].unsqueeze(0)  # 1xCxPxP
            rgb_crop_mask = roi_img_masks[optim_index].unsqueeze(0) # 1x1xSxS
        
        coarse_det_cost = time.time() - start_timer
        if CFG.enable_fine_detection:
            mask_threshold = CFG.finer_threshold
            while True:
                    fine_binary_mask = (rgb_crop_mask.squeeze() >= mask_threshold).type(torch.uint8)
                    if fine_binary_mask.sum() < CFG.DINO_PATCH_SIZE**2:
                        mask_threshold -= 0.1
                        continue
                    else:
                        break
            # 查找联通域
            # pd_fine_tight_scales：边界框（Bounding Box）的尺寸（宽高）。
            # pd_fine_centers：边界框的中心坐标
            _, pd_fine_tight_scales, pd_fine_centers = misc.torch_find_connected_component(
                fine_binary_mask, include_supmask=CFG.CC_INCLUDE_SUPMASK, min_bbox_scale=CFG.DINO_PATCH_SIZE, return_bbox=True)

            # 这个操作在归一化，然后减去0.5是为了转移到中心，这样的话就是计算了偏置多少个像素
            fine_offset_center = (pd_fine_centers / CFG.zoom_image_scale - 0.5) * rgb_box_scale[None]
            fine_bbox_centers = rgb_box_center[None, :] + fine_offset_center
            fine_bbox_tight_scales = rgb_box_scale[None] * pd_fine_tight_scales / CFG.zoom_image_scale
            
            fine_bbox_scales = fine_bbox_tight_scales * CFG.finer_bbox_padding 
            pd_fine_bboxes = torch.stack([fine_bbox_centers[:, 0] - fine_bbox_scales / 2.0,
                                            fine_bbox_centers[:, 1] - fine_bbox_scales / 2.0,
                                            fine_bbox_centers[:, 0] + fine_bbox_scales / 2.0,
                                            fine_bbox_centers[:, 1] + fine_bbox_scales / 2.0], dim=-1)
            roi_RGB_crops = roi_align(que_image, boxes=[pd_fine_bboxes],
                                        output_size=(CFG.zoom_image_scale, CFG.zoom_image_scale), 
                                        sampling_ratio=4) # 1x3xHxW -> Mx3xSxS
            
            if roi_RGB_crops.shape[0] == 1: 
                rgb_img_crop = roi_RGB_crops  # 1x3xSxS
                rgb_box_scale = fine_bbox_scales.squeeze(0)
                rgb_box_center = fine_bbox_centers.squeeze(0)
                rgb_box_tight_scale = fine_bbox_tight_scales.squeeze(0)
                rgb_img_feat = model_func.extract_DINOv2_feature(rgb_img_crop)
                rgb_crop_mask = model_func.query_cosegmentation(x_que=rgb_img_feat, 
                                                                x_ref=ref_database['obj_fps_feats'], 
                                                                ref_mask=ref_database['obj_fps_masks']).sigmoid() # 1xCxSxS -> 1x1xSxS
            else:
                roi_img_feats, _, roi_dino_tokens = model_func.extract_DINOv2_feature(roi_RGB_crops, return_last_dino_feat=True)
                roi_img_masks = model_func.query_cosegmentation(x_que=roi_img_feats, 
                                                                x_ref=ref_database['obj_fps_feats'], 
                                                                ref_mask=ref_database['obj_fps_masks']).sigmoid() # KxCxSxS -> Kx1xSxS
                roi_obj_mask = torch_F.interpolate(roi_img_masks, 
                                                    scale_factor=1.0/CFG.DINO_PATCH_SIZE, 
                                                    mode='bilinear', align_corners=True, 
                                                    recompute_scale_factor=True).flatten(2).permute(0, 2, 1).round() # KxLx1
                roi_dino_tokens = roi_obj_mask * roi_dino_tokens
                token_cosim = torch.einsum('klc,nc->kln', torch_F.normalize(roi_dino_tokens, dim=-1), 
                                                            torch_F.normalize(ref_database['obj_fps_dino_tokens'], dim=-1))
                if CFG.cosim_topk > 0:
                    cosim_score = token_cosim.topk(dim=1, k=CFG.cosim_topk).values.mean(dim=-1).mean(dim=1) # KxLxN -> KxTxN -> KxN -> K
                else:
                    cosim_score = token_cosim.mean(dim=-1).sum(dim=1) / (1 + roi_obj_mask.squeeze(-1).sum(dim=1))  # KxLxN -> KxL -> K 
                
                optim_index = cosim_score.argmax()
                rgb_box_scale = fine_bbox_scales[optim_index]
                rgb_box_center = fine_bbox_centers[optim_index]
                rgb_box_tight_scale = fine_bbox_tight_scales[optim_index]
                rgb_img_crop = roi_RGB_crops[optim_index].unsqueeze(0)  # 1x3xSxS
                rgb_img_feat = roi_img_feats[optim_index].unsqueeze(0)  # 1xCxPxP
                rgb_crop_mask = roi_img_masks[optim_index].unsqueeze(0) # 1x1xSxS                                
        
        fine_det_cost = time.time() - start_timer

        RAEncoder_timer = time.time()
        rgb_img_feat = model_func.extract_DINOv2_feature(rgb_img_crop)
        rgb_crop_mask = model_func.query_cosegmentation(x_que=rgb_img_feat, 
                                                        x_ref=ref_database['obj_fps_feats'], 
                                                        ref_mask=ref_database['obj_fps_masks']).sigmoid()
        obj_Remb_vec = model_func.generate_rotation_aware_embedding(rgb_img_feat, rgb_crop_mask)
        RAEncoder_cost = time.time() - RAEncoder_timer

    return {
        'bbox_scale': rgb_box_scale,
        'bbox_center': rgb_box_center,
        'bbox_tight_scale': rgb_box_tight_scale,
        'obj_Remb': obj_Remb_vec.squeeze(0),
        'rgb_image': rgb_img_crop.squeeze(0), # 3xSxS
        'rgb_mask': rgb_crop_mask.squeeze(0), # 1xSxS
        'coarse_det_cost': coarse_det_cost,
        'fine_det_cost': fine_det_cost,
        'RAEncoder_cost': RAEncoder_cost,
    }


def multiple_initial_pose_inference(obj_data, ref_database, device):
    # obj_data 目标
    # ref_database参考数据
    camK = obj_data['camK'].to(device).squeeze()
    # 旋转感知嵌入
    obj_Remb = obj_data['obj_Remb'].to(device).squeeze()
    obj_mask = obj_data['rgb_mask'].to(device).squeeze()
    # 边界框尺寸
    bbox_scale = obj_data['bbox_scale'].to(device).squeeze()
    bbox_center = obj_data['bbox_center'].to(device).squeeze()
    
    # round() 将掩码的值四舍五入为 0 或 1（二值化）。squeeze() 去除掩码中长度为 1 的维度，确保掩码是二维的。
    # torch.nonzero 返回输入张量中所有非零元素的索引。
    # as_tuple=True 将索引分别返回为两个张量：
    # que_msk_yy：非零元素的行坐标（y 坐标）。
    # que_msk_xx：非零元素的列坐标（x 坐标）。
    que_msk_yy, que_msk_xx = torch.nonzero(obj_mask.round().squeeze(), as_tuple=True)
    # 计算掩码的中心点
    que_msk_cx = (que_msk_xx.max() + que_msk_xx.min()) / 2
    que_msk_cy = (que_msk_yy.max() + que_msk_yy.min()) / 2
    # 计算掩码的面积，所有非零像素的总和
    que_bin_msk_area = obj_mask.round().sum()
    # 计算掩码的概率面积 不取整，直接算
    que_prob_msk_area = obj_mask.sum()

    # 点积运算 m是m个照片提取出来的数据，计算相似度
    Remb_cosim = torch.einsum('c, mc->m', obj_Remb, ref_database['refer_Remb_vectors'])

    # 选择最高的几个(CFG.ROT_TOPK个)参考索引
    max_inds = Remb_cosim.flatten().topk(dim=0, k=CFG.ROT_TOPK).indices

    # 从参考数据库中提取对应的旋转矩阵。有K个参考图像
    init_Rs = ref_database['refer_allo_Rs'][max_inds]           # Kx3x3

    # 根据选择的参考索引 max_inds，从参考数据库中提取对应的掩码信息。中心点、面积等
    selected_nnb_info = ref_database['refer_coseg_mask_info'][max_inds] # Kx4

    nnb_ref_Cx = selected_nnb_info[:, 0]   # K参考掩码的中心点坐标。
    nnb_ref_Cy = selected_nnb_info[:, 1]   # K参考掩码的中心点坐标。
    nnb_ref_Tz = selected_nnb_info[:, 2]   # K参考掩码对应的物体的深度值。
    nnb_ref_bin_area = selected_nnb_info[:, 3] # K参考掩码的二进制面积（非零像素的数量）。
    nnb_ref_prob_area = selected_nnb_info[:, 4] # K参考掩码的概率面积（所有像素值的总和）。

    # 二进制面积还是概率面积
    # 计算缩放比例 que_bin_msk_area目标的面积，nnb_ref_prob_area参考的面积
    if CFG.BINARIZE_MASK:
        delta_S = (que_bin_msk_area / nnb_ref_bin_area)**0.5
    else:
        delta_S = (que_prob_msk_area / nnb_ref_prob_area)**0.5

    # 计算偏移量 这是个比例
    delta_Px = (que_msk_cx - nnb_ref_Cx) / CFG.zoom_image_scale # K
    delta_Py = (que_msk_cy - nnb_ref_Cy) / CFG.zoom_image_scale # K
    delta_Pxy = torch.stack([delta_Px, delta_Py], dim=-1)   # Kx2
    
    # 深度估计
    que_Tz = nnb_ref_Tz / delta_S * CFG.zoom_image_scale / bbox_scale # K


    obj_Pxy = delta_Pxy * bbox_scale + bbox_center    # Kx2
    # 在最后一个维度填充1
    homo_pxpy = torch_F.pad(obj_Pxy, (0, 1), value=1) # Kx3
    # 初始平移向量 计算xyz
    init_Ts = torch.einsum('ij,kj->ki', torch.inverse(camK), homo_pxpy) * que_Tz.unsqueeze(1)
    
    # 这个是初始位姿
    init_RTs = torch.eye(4)[None, :, :].repeat(init_Rs.shape[0], 1, 1) # Kx4x4
    init_RTs[:, :3, :3] = init_Rs.detach().cpu()
    init_RTs[:, :3, 3] = init_Ts.detach().cpu()
    init_RTs = init_RTs.numpy()

    # Egocentric（以自我为中心） egocentric以场景为中心
    if CFG.USE_ALLOCENTRIC:
        for idx in range(init_RTs.shape[0]):
            init_RTs[idx, :3, :4] = gs_utils.allocentric_to_egocentric(init_RTs[idx, :3, :4])[:3, :4]

    return init_RTs

class match_2d3d():
    """
    这个类用于匹配2d和3d
    """
    def __init__(self,config) -> None:
        self.extractor = None
        self.matcher = None
        self.loftr = None
        self.match_name = config['match']['matcher']['name']

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

        if config['match']['extractor']['name'] == "loftr" and config['match']['matcher']['name'] == "loftr":
            
            self.loftr = LoFTR('outdoor').cuda()
            pass

        self.match_confidence_threshold = config['match']['confidence_threshold']

        
        pass

    def match2d3d(self,img_rgb:np.ndarray,
                  img_rander:torch.Tensor,
                  rander_deep:np.ndarray,
                  camera:Camera,
                  debug=False,
                  mask=None):
        '''
        img_rgb 是一个numpy数组，表示GBR图像,是opencv直接读取出来的
        img_rander：时0-1的 CHW 通道的图像
        '''
        # 如果img_rgb是numpy数组，则转换为torch.Tensor
        if isinstance(img_rgb, np.ndarray):
            # opencv直接读取出来的，所以要将BGR换成RGB
            img_rgb = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2RGB)
            # 换成torch格式，并且HWC 变成CHW
            img_rgb = torch.from_numpy(img_rgb).permute(2, 0, 1).float().cuda()
            img_rgb = img_rgb / 255.0

        # 从chw，转化为hwc
        depth_np = rander_deep.permute(1, 2, 0).detach().cpu().numpy()

        # 从chw，转化为hwc 
        render_img_np = img_rander.permute(1, 2, 0).detach().cpu().numpy()
        # 将0到1的改为0到255的rgb
        render_img_np = (render_img_np * 255).astype(np.uint8)
        if render_img_np.shape[0] == 3:
            render_img_np = np.transpose(render_img_np, (1, 2, 0))

        # 更改rgb改为bgr
        # img_cv = render_img_np[:, :, [2, 1, 0]]
        # 调整通道顺序：从 [H, W, C] 到 [C, H, W] rgb 
        img_cv = render_img_np.transpose(2, 0, 1)

        torch_render = torch.from_numpy(img_cv).float().cuda()
        torch_render = torch_render / 255.0
        # rgb 0到1 C H W
        
        # extract local features 
        # RGB格式
        feats0 = self.extractor.extract(img_rander)  # auto-resize the image, disable with resize=None 渲染图像
        # RGB格式
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


        # 如果提供了 mask，将其转换为 torch.Tensor 并移动到 GPU
        if mask is not None:
            mask_tensor = torch.from_numpy(mask).float().cuda()
        else:
            mask_tensor = None

        # 如果提供了 mask，进一步过滤匹配点（基于渲染图像）
        if mask_tensor is not None:
            # 检查匹配点是否位于有效区域
            valid_mask = mask_tensor[points0[:, 1].long(), points0[:, 0].long()] == 1
            points0 = points0[valid_mask]
            points1 = points1[valid_mask]
            matches = matches[valid_mask]  # 新增：同时过滤 matches
            filtered_scores = filtered_scores[valid_mask]


        if debug:
            match_utils.draw_matches(torch_render, img_rgb, points0, points1, matches)
            match_utils.draw_matches_with_depth(depth_np,torch_render, img_rgb, points0, points1, matches)
        # match_utils.draw_matches(torch_render, img_rgb, points0, points1, matches)

        img_xy = []
        gs_xyz_w = []
        if not len(matches) < 2:
            img_xy,gs_xyz_w = match_utils.match2xy_xyz_LightGlue(kporg=points1,
                                                                 kpdeep=points0,
                                                                 img_deep=depth_np,
                                                                 camera=camera)
        return img_xy,gs_xyz_w
    


    
    
    def match2d3d_loftr(self,img_rgb:np.ndarray,
                img_rander:torch.Tensor,
                rander_deep:np.ndarray,
                camera:Camera,
                debug=False,
                mask=None):
        '''
        img_rgb 是一个numpy数组，表示GBR图像,是opencv直接读取出来的
        '''
        # 如果img_rgb是numpy数组，则转换为torch.Tensor
        
         # 处理渲染图像 CHW -> HWC
        render_img_np = img_rander.permute(1, 2, 0).detach().cpu().numpy()
        render_img_np = (render_img_np * 255).astype(np.uint8)
        # 渲染图改为灰度图 RGB->gay 
        if render_img_np.shape[2] == 3:
            render_img_np = cv2.cvtColor(render_img_np, cv2.COLOR_RGB2GRAY)  # 转换为灰度图

        # 处理目标图像
        img_target_gray = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2GRAY)  # 转换为灰度图


        img_target_tensor = torch.from_numpy(img_target_gray)[None][None].float().cuda() / 255.0

        img_render_tensor = torch.from_numpy(render_img_np)[None][None].float().cuda() / 255.0 


        # 0是target 1是render
        with torch.no_grad():
            input_dict = {"image0": img_target_tensor, "image1": img_render_tensor}
            correspondences = self.loftr(input_dict)
        
        # 获取匹配点
        mkpts0 = correspondences['keypoints0']
        mkpts1 = correspondences['keypoints1']
        scores = correspondences['confidence']

        # 根据置信度过滤匹配点
        valid_indices = scores > self.match_confidence_threshold
        mkpts0 = mkpts0[valid_indices]
        mkpts1 = mkpts1[valid_indices]
        filtered_scores = scores[valid_indices]

        # 如果提供了mask，进一步过滤匹配点（基于渲染图像）
        if mask is not None:
            mask_tensor = torch.from_numpy(mask).float().cuda()
            valid_mask = mask_tensor[mkpts0[:, 1].long(), mkpts0[:, 0].long()] == 1
            mkpts0 = mkpts0[valid_mask]
            mkpts1 = mkpts1[valid_mask]
            filtered_scores = filtered_scores[valid_mask]

        # 0是target 1是render
            
        depth_np = rander_deep.permute(1, 2, 0).detach().cpu().numpy().squeeze()
        if debug:

            # match_utils.draw_matches(img_render_tensor, img_rgb, points0, points1, matches)
            # match_utils.draw_matches_with_depth(depth_np,torch_render, img_rgb, points0, points1, matches)
            match_utils.visualize_matches_loftr(img_rgb, render_img_np, mkpts0, mkpts1,depth_img=depth_np)


                           
        if debug == True:
            y_indices, x_indices = np.nonzero(depth_np)

            K_inv = camera.get_inv_k().cpu().numpy()
            # 获取世界坐标系在相机坐标系中的表达，因为Camera中的R是以转置的形式储存的，所以取出来的时候要转置一下
            R_temp = camera.R.T
            t_temp = camera.T.reshape(3, 1)
            T_temp = np.eye(4)
            T_temp[:3, :3] = R_temp
            T_temp[:3, 3] = t_temp.flatten()

            # 改成了从世界坐标系到camera坐标系
            # T = np.linalg.inv(T)
            T_inv = np.eye(4)
            T_inv[:3, :3] = T_temp[:3, :3].T
            T_inv[:3, 3] = -T_inv[:3, :3] @ T_temp[:3, 3]



            # print(f"x={x1 + 512},y1 = {y1}... z = {z}")

            xyz_gs = []
            # 将所有像素坐标转换为齐次坐标
            uv_homogeneous = np.column_stack([x_indices, y_indices, np.ones_like(x_indices)])

            # 批量归一化
            normalized_coords = (K_inv @ uv_homogeneous.T).T

            # 获取所有深度值
            z_values = depth_np[y_indices, x_indices]

            # 计算相机坐标系下的3D坐标
            camera_coords = normalized_coords * z_values[:, np.newaxis]

            # 转换为齐次坐标
            camera_coords_homogeneous = np.column_stack([camera_coords, np.ones_like(x_indices)])

            # 批量转换到世界坐标系
            world_coords = (T_inv @ camera_coords_homogeneous.T).T
            world_coords = world_coords[:, :3]

            xyz_gs = world_coords.tolist()


            visualize_point_cloud_with_coordinate(xyz_gs)

            pass

        img_xy = []
        gs_xyz_w = []
        if not len(mkpts0) < 2:
            img_xy,gs_xyz_w = match_utils.match2xy_xyz_LightGlue(kporg=mkpts0,
                                                                 kpdeep=mkpts1,
                                                                 img_deep=depth_np,
                                                                 camera=camera)
        return img_xy,gs_xyz_w
     
    def camera_matchs(self,cameras:List[Camera],
                      GS_model:GaussianModel,
                      img_target:np.ndarray,
                      debug=False,
                      mask=None,
                      match = 'LightGlue'):
        '''
        匹配多张图片,只负责匹配，不负责计算任何的位姿估计
        '''
        img_xys =[]
        gs_xyz_ws = []

        for camera in cameras:
            render = GS_Renderer(camera, GS_model, self.gaussian_PipeP, self.gaussian_BG)
            render_img = render['render']

            # 这里每个像素单位是m
            deepsimg = render['plane_depth']

            deepsimg_np = deepsimg.permute(1, 2, 0).detach().cpu().numpy()

            dg_mask = match_utils.gradient_deep_mask(deepsimg_np,TG=10,debug=debug)

            render_img_np = render_img.permute(1, 2, 0).detach().cpu().numpy()
            # 生成 mask
            mask_black = np.any(render_img_np != 0, axis=-1).astype(np.uint8)


            inverted_dg_mask = ~dg_mask 

            final_mask = np.logical_and(inverted_dg_mask, mask_black).astype(np.uint8)

            if match == 'LightGlue':
                img_xy,gs_xyz_w = self.match2d3d(img_rander=render_img,
                                                img_rgb=img_target,
                                                rander_deep=deepsimg,
                                                camera=camera,
                                                debug=debug,
                                                mask=final_mask)
            elif match == 'loftr':
                img_xy,gs_xyz_w = self.match2d3d_loftr(img_rgb=img_target,
                                                        img_rander=render_img,
                                                        rander_deep=deepsimg,
                                                        camera=camera,
                                                        debug=debug,
                                                        mask=final_mask)

            img_xys.extend(img_xy)
            gs_xyz_ws.extend(gs_xyz_w)

        gs_xyz_ws = np.array(gs_xyz_ws, dtype=np.float32)
        img_xys = np.array(img_xys, dtype=np.float32)
        

        dist_coeffs = np.zeros(shape=[8, 1], dtype="float64")


        if debug:
            pcd = o3d.geometry.PointCloud()
            # 将 NumPy 数组转换为 Open3D 的点云格式
            pcd.points = o3d.utility.Vector3dVector(gs_xyz_ws)
            coordinate_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1, origin=[0, 0, 0])
            o3d.visualization.draw_geometries([pcd,coordinate_frame])


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

    def pose_estimate(self,img_target:np.ndarray,GS_model:GaussianModel,initin1:List[float],
                      bbox_coords:List[float],
                      camera_ref = None,
                      debug = False,
                      mask = None,
                      camera_distance_scale = 1.0,
                      refine_cfg = None,
                      mask_cut = False):
        '''
        args:
            camera_ref:是参考的camera
        '''
        # img1 = cv2.imread(f'{imag_path}/55.png', cv2.IMREAD_GRAYSCALE)
        height, width = img_target.shape[:2]



        cameras = match_utils.box2gscamera(box=bbox_coords,
                                           K=initin1,
                                           height=height,
                                           width=width,
                                           camera_num=self.camera_num,
                                           debug=debug,
                                           camera_distance_scale = camera_distance_scale)
        
        # cameras = match_utils.box2gscamera_linemod(box=bbox_coords,K=initin1,height=height,width=width,camera_num=self.camera_num)
        K_img_np = match_utils.convert2K(initin1).cpu().numpy()

        # 如果linemod需要裁剪，那么这里就是根据mask裁剪rgb图片
        if mask_cut:
            img_target_cut,K_cut = match_utils.crop_and_tile_mask(img_target,mask,K_img_np,debug=debug)
            img_target = img_target_cut
            K_img_np = K_cut
        
        if camera_ref is not None:
            cameras = cameras + camera_ref

        img_xys,gs_xyz_ws = self.camera_matchs(cameras= cameras,
                                               GS_model= GS_model,
                                               img_target=img_target,
                                               debug=debug,
                                               match=self.match_name)

        dist_coeffs = np.zeros(shape=[8, 1], dtype="float64")
        T_ = np.eye(4)
        try:
            # 这里一定是camera[0]因为这里的camera0是根据数据集生成的。跟targetimg相同。
            _, rvec, tvec, inliers = cv2.solvePnPRansac(gs_xyz_ws, img_xys,K_img_np,dist_coeffs,
                    reprojectionError=self.pnp_reprojection_error,
                    iterationsCount=10000,
                    flags=cv2.SOLVEPNP_EPNP)
            
            if np.all(tvec > 100):
            # 如果是，将 tvec 设置为零向量
                tvec = np.zeros_like(tvec)
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

            if debug == True:
                cv2.imshow('Rendered init imag', img_cv)
                cv2.waitKey()  # 按任意键关闭窗口
                cv2.destroyAllWindows()
            # cv2.imshow('Rendered init imag', img_cv)
            # cv2.waitKey()  # 按任意键关闭窗口
            # cv2.destroyAllWindows()


            class_gsrefine = GS_refine()


            imag0_rgb = cv2.cvtColor(img_target, cv2.COLOR_BGR2RGB)
            imag0_rgb_torch = torch.from_numpy(imag0_rgb).float().to('cuda')
            output = class_gsrefine.GS_Refiner(image=imag0_rgb_torch,
                                               mask=None,
                                               init_camera=camera,
                                               gaussians=GS_model,
                                               return_loss=True,
                                               debug=debug)
        
            # 返回的是4x4的np格式的位姿矩阵
            gs3d_delta_RT_np = output['gs3d_delta_RT']
            # gs3d_delta_RT_np[:3,:3] = 
            gs3d_pose_inv = np.linalg.inv(gs3d_delta_RT_np)
            init_T = T_
            T_ = (T_@gs3d_delta_RT_np)


        return T_,init_T
    



    def pose_estimate_batch(self,img_target_dir:List[str],GS_model_dir:str,initin1:List[str],bbox_coords:str,object_name:str,dataset:datasets):
        '''
        处理一个场景的位姿估计
        '''
        GS_model = GaussianModel(3)
        GS_model.load_ply(GS_model_dir)
        GS_model.clip_to_bounding_box_fromfile(bbox_coords)
        bbox_coords_np = match_utils.read_txt(bbox_coords)
        bbox_points_np = match_utils.convert_to_3d_points(bbox_coords_np)
        with open(dataset.dataset.metric_out_dir, 'a') as f :
            f.write(f"{object_name}\n")

        # 计算直径
        dataset.dataset.diameter[object_name] = gs_utils.torch_compute_diameter_from_pointcloud(GS_model._xyz)

        metrics = {"R_errs":[],
                   "t_errs":[]}
        init_metrics = {"R_errs":[],
                   "t_errs":[]}
        for i in tqdm(range(0,len(img_target_dir)),desc="Processing images"):
            img_target = cv2.imread(img_target_dir[i])
            initin_np = match_utils.read_txt(initin1[i])
            print(f"initin1[i] = {initin1[i]}")
            print(f"img_target_dir[i] = {img_target_dir[i]}")
            T_p,init_T = self.pose_estimate(img_target=img_target,GS_model=GS_model,initin1=initin_np,bbox_coords=bbox_points_np)

            camK = match_utils.convert2K(initin_np)
            # 读取数据集pose
            pose_data = match_utils.read_txt(dataset.dataset.pose_dir_lists[object_name][i])
            pose_data_np = np.array(pose_data).reshape(4, 4)
            
            # 初始的位姿
            init_add = metric_utils.calc_add_metric(GS_model._xyz.detach().cpu().numpy(), dataset.dataset.diameter[object_name],init_T,pose_data_np)
            # 优化的位姿
            refine_add = metric_utils.calc_add_metric(GS_model._xyz.detach().cpu().numpy(), dataset.dataset.diameter[object_name],T_p,pose_data_np)
            
            # refine的数据
            if 'ADD_metric' not in metrics.keys():
                metrics['ADD_metric'] = list()
            metrics['ADD_metric'].append(refine_add)

            refine_proj_err = metric_utils.calc_projection_2d_error(bbox_points_np, T_p, pose_data_np, camK.cpu().numpy())
            if 'Proj2D' not in metrics.keys():
                metrics['Proj2D'] = list()
            metrics['Proj2D'].append(refine_proj_err)


            R_err,t_err= metric_utils.calc_pose_error(T_p,pose_data_np)
            metrics["R_errs"].append(R_err)
            metrics["t_errs"].append(t_err)


            # init的数据

            if 'ADD_metric' not in init_metrics.keys():
                init_metrics['ADD_metric'] = list()
            init_metrics['ADD_metric'].append(init_add)

            refine_proj_err = metric_utils.calc_projection_2d_error(bbox_points_np, init_T, pose_data_np, camK.cpu().numpy())
            if 'Proj2D' not in init_metrics.keys():
                init_metrics['Proj2D'] = list()
            init_metrics['Proj2D'].append(refine_proj_err)


            R_err_init,t_err_init= metric_utils.calc_pose_error(init_T,pose_data_np)
            init_metrics["R_errs"].append(R_err_init)
            init_metrics["t_errs"].append(t_err_init)


            pass


        agg_metric = metric_utils.aggregate_metrics(metrics)

        init_agg_metric = metric_utils.aggregate_metrics(init_metrics)

        with open(dataset.dataset.metric_out_dir, 'a') as f :
            f.write(f"name = {object_name} : \n")

            f.write("init_metric\n")
            for key, value in init_agg_metric.items():
                f.write(f"{key}: {value}\n")
            f.write("refine_metric\n")
            for key, value in agg_metric.items():
                f.write(f"{key}: {value}\n")


        print(agg_metric)
        return metrics,init_metrics
    pass


    def pose_estimate_batch2(self,img_target_dir:List[str],
                             GS_model_dir:str,
                             initin1:List[str],
                             bbox_coords:str,
                             object_name:str,
                             dataset:datasets):
        '''
        处理一个场景的位姿估计
        '''
        GS_model = GaussianModel(3)
        GS_model.load_ply(GS_model_dir)
        GS_model.clip_to_bounding_box_fromfile(bbox_coords)
        bbox_coords_np = match_utils.read_txt(bbox_coords)
        bbox_points_np = match_utils.convert_to_3d_points(bbox_coords_np)
        with open(dataset.dataset.metric_out_dir, 'a') as f :
            f.write(f"{object_name}\n")

        # 计算直径
        dataset.dataset.diameter[object_name] = gs_utils.torch_compute_diameter_from_pointcloud(GS_model._xyz)

        metrics = {"R_errs":[],
                   "t_errs":[]}
        init_metrics = {"R_errs":[],
                   "t_errs":[]}
        
        camera_ref =  self.camera_ref_gen(dataset,object_name)


        # 记录pose
        # 创建这个文件夹
        pose_out_dir_temp = f"{dataset.dataset.pose_out_dir}/{object_name}"
        os.makedirs(pose_out_dir_temp, exist_ok=True)

        for i in tqdm(range(0,len(img_target_dir)),desc="Processing images"):
            img_target = cv2.imread(img_target_dir[i])
            initin_np = match_utils.read_txt(initin1[i])
            print(f"initin1[i] = {initin1[i]}")
            print(f"img_target_dir[i] = {img_target_dir[i]}")
            T_p,init_T = self.pose_estimate(img_target=img_target,
                                            GS_model=GS_model,
                                            initin1=initin_np,
                                            bbox_coords=bbox_points_np,
                                            camera_ref=camera_ref,
                                            refine_cfg=dataset.dataset.refine_CFG,
                                            debug=dataset.dataset.debug)
            
            GS_model.initialize_pose()

            camK = match_utils.convert2K(initin_np)
            # 读取数据集pose
            pose_data = match_utils.read_txt(dataset.dataset.pose_dir_lists[object_name][i])
            pose_data_np = np.array(pose_data).reshape(4, 4)

            # pose_out_dir_temp
            name = os.path.splitext(os.path.basename(img_target_dir[i]))[0]

            # 这个路径是到每张照片的输出文件夹
            pose_dir = f"{pose_out_dir_temp}/{name}"

            os.makedirs(pose_dir, exist_ok=True)

            pose_init_dir = f"{pose_dir}/pose_init.txt"
            pose_refine_dir = f"{pose_dir}/pose_refine.txt"


            # 所有文件路径
            pose_files = [pose_init_dir, pose_refine_dir]

            # 删除已存在的文件
            for file_path in pose_files:
                if os.path.exists(file_path):
                    os.remove(file_path)

            # 创建新的空文件
            for file_path in pose_files:
                with open(file_path, 'w') as f:
                    pass  # 创建空文件即可


            with open(pose_init_dir, 'w') as f:
                    # init_T
                    np.savetxt(f, init_T, fmt='%.6f', delimiter=' ')
                    pass  # 创建空文件即可
            
            with open(pose_refine_dir, 'w') as f:
                    # init_T
                    np.savetxt(f, T_p, fmt='%.6f', delimiter=' ')
                    pass  # 创建空文件即可
            
            
            # traget image 的np表示 w,h,c
            rgb_np = cv2.cvtColor(img_target, cv2.COLOR_RGB2BGR)

            if rgb_np.dtype != np.uint8:
                if rgb_np.max() <= 1.0:
                    rgb_np = (rgb_np * 255).astype(np.uint8)
                else:
                    rgb_np = rgb_np.astype(np.uint8)


            rgb_np = cv2.cvtColor(rgb_np, cv2.COLOR_BGR2RGB)



            img_init_dir = f"{pose_dir}/pose_init.png"
            img_refine_dir = f"{pose_dir}/pose_refine.png"

            camK_np = camK.cpu().numpy().copy()
            initpose_rgb_show = draw_bbox.render_bboxes(pose_data_np,init_T,bbox_points_np,camK_np,rgb_np)
            refinpose_rgb_show = draw_bbox.render_bboxes(pose_data_np,T_p,bbox_points_np,camK_np,rgb_np)

            # 保存图像，若文件已存在则覆盖
            cv2.imwrite(img_init_dir, initpose_rgb_show)
            cv2.imwrite(img_refine_dir, refinpose_rgb_show)



            # cv2.imshow('Rendered Bboxes', initpose_rgb_show)
            # 等待任意键按下后关闭窗口
            # cv2.waitKey(0)
            # cv2.destroyAllWindows()
            






            # 每次优化完参数之后就要清空一下，要不然就会叠加上一次的数据

            
            # 初始的位姿
            init_add = metric_utils.calc_add_metric(GS_model._xyz.detach().cpu().numpy(), dataset.dataset.diameter[object_name],init_T,pose_data_np)
            # 优化的位姿
            refine_add = metric_utils.calc_add_metric(GS_model._xyz.detach().cpu().numpy(), dataset.dataset.diameter[object_name],T_p,pose_data_np)
            
            # refine的数据
            if 'ADD_metric' not in metrics.keys():
                metrics['ADD_metric'] = list()
            metrics['ADD_metric'].append(refine_add)

            refine_proj_err = metric_utils.calc_projection_2d_error(bbox_points_np, T_p, pose_data_np, camK.cpu().numpy())
            if 'Proj2D' not in metrics.keys():
                metrics['Proj2D'] = list()
            metrics['Proj2D'].append(refine_proj_err)


            R_err,t_err= metric_utils.calc_pose_error(T_p,pose_data_np)
            metrics["R_errs"].append(R_err)
            metrics["t_errs"].append(t_err)


            # init的数据

            if 'ADD_metric' not in init_metrics.keys():
                init_metrics['ADD_metric'] = list()
            init_metrics['ADD_metric'].append(init_add)

            refine_proj_err = metric_utils.calc_projection_2d_error(bbox_points_np, init_T, pose_data_np, camK.cpu().numpy())
            if 'Proj2D' not in init_metrics.keys():
                init_metrics['Proj2D'] = list()
            init_metrics['Proj2D'].append(refine_proj_err)


            R_err_init,t_err_init= metric_utils.calc_pose_error(init_T,pose_data_np)
            init_metrics["R_errs"].append(R_err_init)
            init_metrics["t_errs"].append(t_err_init)


            pass


        agg_metric = metric_utils.aggregate_metrics(metrics)

        init_agg_metric = metric_utils.aggregate_metrics(init_metrics)

        with open(dataset.dataset.metric_out_dir, 'a') as f :
            f.write(f"name = {object_name} : \n")

            f.write("init_metric\n")
            for key, value in init_agg_metric.items():
                f.write(f"{key}: {value}\n")
            f.write("refine_metric\n")
            for key, value in agg_metric.items():
                f.write(f"{key}: {value}\n")


        print(agg_metric)
        return metrics,init_metrics
    pass
    
    def camera_ref_gen(self,detaset:datasets,id:str,gap=10):
        '''
        抽取一部分相机
        '''
        cameras:list[Camera] = []
        for i in range(len(detaset.dataset.cam_intr_dir_lists[id])):
            if i%gap==0:

                pose_data = match_utils.read_txt(detaset.dataset.pose_dir_lists[id][i])
                pose_data_np = np.array(pose_data).reshape(4, 4)
                R_ = pose_data_np[:3, :3]
                t_ = pose_data_np[:3, 3]

                K_ref = match_utils.convert2K(match_utils.read_txt( detaset.dataset.cam_intr_dir_lists[id][i]) )
                img_ref = cv2.imread(detaset.dataset.color_dir_lists[id][i])
                height,width = img_ref.shape[:2]
                fx = K_ref[0,0]
                fy = K_ref[1,1]

                fovx =  2*math.atan(width/(2*fx))
                fovy =  2*math.atan(height/(2*fy))

                camera = Camera(R_.T,t_,fovx,fovy,K_ref[0,2],K_ref[1,2],width,height,image_name=img_ref,image_path='',uid=0,preload_img=False)
                cameras.append(camera)



            pass
            # camera.append(Camera(detaset.pose_lists[detaset.sub_name_lists[i]][detaset.ID_lists[i]],detaset.cam_K_lists[detaset.sub_name_lists[i]][detaset.ID_lists[i]]))

        return cameras
        pass


    def camera_ref_gen_linemod(self,detaset:datasets,id:str):
        '''
        抽取一部分相机
        '''
        id = str(int(id))
        cameras:list[Camera] = []
        for i in range(len(detaset.dataset.ref_T[id])):
            if i%10==0:

                pose_data = detaset.dataset.ref_T[id][i]
                pose_data_np = np.array(pose_data).reshape(4, 4)


                R_ = pose_data_np[:3, :3]
                t_ = pose_data_np[:3, 3]

                K_ref = detaset.dataset.cam_K_lists[id][0]
                img_ref = cv2.imread(detaset.dataset.color_dir_lists[id][0]).shape[:2]
                pixx,pixy = img_ref
                fx = K_ref[0,0]
                fy = K_ref[1,1]

                fovx =  2*math.atan(pixx/(2*fx))
                fovy =  2*math.atan(pixy/(2*fy))

                camera = Camera(R_.T,t_,fovx,fovy,K_ref[0,2],K_ref[1,2],pixy,pixx,image_name='',image_path='',uid=0,preload_img=False)
                cameras.append(camera)



            pass
            # camera.append(Camera(detaset.pose_lists[detaset.sub_name_lists[i]][detaset.ID_lists[i]],detaset.cam_K_lists[detaset.sub_name_lists[i]][detaset.ID_lists[i]]))

        return cameras
        pass


    def pose_estimate_batch_linemod(self,img_target_dir:List[str],
                             GS_model_dir:str,
                             initin1:List[np.ndarray],
                             bbox_coords:str,
                             object_name:str,
                             dataset:datasets,
                             reference_database,
                             model_func,
                             device = "cuda",
                             save_pred_mask = True):
        '''
        处理一个场景的位姿估计
        model_func:是gspose的预训练模型
        ref_data:是基于预训练模型生成的一个参考数据集
        '''
        GS_model = GaussianModel(3)
        GS_model.load_ply(GS_model_dir)
        # 这里的box已经是8x3的矩阵了
        box = dataset.dataset.box[object_name]

        # 改为从数据集中获取
        GS_model.clip_to_bounding_box(box)

        # GS_model.clip_to_bounding_box_fromfile(bbox_coords)
        # bbox_coords_np = match_utils.read_txt(bbox_coords)
        # bbox_points_np = match_utils.convert_to_3d_points(box)
        with open(dataset.dataset.metric_out_dir, 'a') as f :
            f.write(f"{object_name}\n")

        # 计算直径 不需要计算，本来就有
        # dataset.dataset.diameter[object_name] = gs_utils.torch_compute_diameter_from_pointcloud(GS_model._xyz)

        metrics = {"R_errs":[],
                   "t_errs":[]}
        init_metrics = {"R_errs":[],
                   "t_errs":[]}
        gspose_metric = {"R_errs":[],
                   "t_errs":[]}
        

        # 记录pose
        # 创建这个文件夹
        pose_out_dir_temp = f"{dataset.dataset.pose_out_dir}/{object_name}"
        os.makedirs(pose_out_dir_temp, exist_ok=True)

        
        # camera_ref =  self.camera_ref_gen_linemod(dataset,object_name)
        # 进入每个照片
        for i in tqdm(range(0,len(img_target_dir)),desc="Processing images"):
            # 这个是bgr
            img_target = cv2.imread(img_target_dir[i])
            GS_model.initialize_pose()
            # GS_model没有问题
            # GS_model.save_ply(f"/media/wangxiao/Newsmy/linemod/testply/test_{i}.ply")

            print(f"initin1[i] = {initin1[i]}")
            print(f"img_target_dir[i] = {img_target_dir[i]}")

            mask_dir =  dataset.dataset.mask_dir_lists[object_name][i]

            mask_image = cv2.imread(mask_dir, cv2.IMREAD_GRAYSCALE)  # mask图像（灰度模式读取）

            # 检查mask是否为二值图像，如果不是则进行二值化

            # crop_target, new_K = match_utils.crop_and_tile_mask(img_target,mask_image,initin1[i],debug=dataset.dataset.debug)
            # …………………………………………………………………………gspose


            # 图像的原始尺寸
            que_hei, que_wid = img_target.shape[:2]
            raw_hei, raw_wid = img_target.shape[:2]
            # 长边
            raw_long_size = max(raw_hei, raw_wid)
            # 短边
            raw_short_size = min(raw_hei, raw_wid)
            # 原始照片短边长边的比例 宽高比
            raw_aspect_ratio = raw_short_size / raw_long_size
            # 调整图像大小
            if raw_hei < raw_wid:
                # 672 原始图像哪个长宽像素比较长，那么就把哪个的像素变成 672 另外一边按照比例
                new_wid = CFG.query_longside_scale
                new_hei = int(new_wid * raw_aspect_ratio)
            else:
                new_hei = CFG.query_longside_scale
                new_wid = int(new_hei * raw_aspect_ratio)
            # 计算像素变换的比例！！！！！！！！！！！
            query_rescaling_factor = CFG.query_longside_scale / raw_long_size
            

            # 这个更改为rgb图像
            image_tensor = torch.from_numpy(cv2.cvtColor(img_target, cv2.COLOR_BGR2RGB)).float()/255.0  # 转为 float32
            # que_image = image_tensor.permute(2, 0, 1) 
            # 这个更改为 (B,H,W,C) 转为 (B,C,H,W)。
            que_image = image_tensor[None, ...].permute(0, 3, 1, 2).to(device)

            # 将原始的像素变换成这个大概是长边672，另外一边成比例
            que_image = torch_F.interpolate(que_image, size=(new_hei, new_wid), mode='bilinear', align_corners=True)
            
            # obj_data = naive_perform_segmentation_and_encoding(model_func, 
                                                                # device=device,
                                                                # que_image=que_image,
                                                                # ref_database=reference_database)
            
            # 对查询图像进行分割和编码。
            obj_data = perform_segmentation_and_encoding(model_func, 
                                                        device=device,
                                                        que_image=que_image,
                                                        ref_database=reference_database)
            
            # 'rgb_image': rgb_img_crop.squeeze(0), # 3xSxS
            # 'rgb_mask': rgb_crop_mask.squeeze(0), # 1xSxS
            
            # 这个原来是因为神经网络输入的像素是确定的，那么需要将原始图片更改大小，
            # 然后除以这个缩放比例，就是改变之前的参数,也就是说，在裁剪之前，已经变换了一次像素
            obj_data['bbox_scale'] /= query_rescaling_factor  # back to the original image scale
            obj_data['bbox_center'] /= query_rescaling_factor # back to the original image scale
            # 
            bbox_center = obj_data['bbox_center']
            bbox_scale = obj_data['bbox_scale']
            # 裁剪计算

            target_size = CFG.GS_RENDER_SIZE




            # 裁剪过的target照片的K
            
            if save_pred_mask:
                # mask的path
                coseg_mask_path = os.path.join(dataset.dataset.ref_output_dir,
                                object_name,
                                'pred_coseg_mask', 
                                '{:06d}.png'.format(i))
                # coseg_mask_path = que_data['coseg_mask_path']
                # 根据参数还原mask
                orig_mask = gs_utils.zoom_out_and_uncrop_image(obj_data['rgb_mask'].squeeze(),
                                                                        bbox_center=obj_data['bbox_center'],
                                                                        bbox_scale=obj_data['bbox_scale'],
                                                                        orig_hei=que_hei,
                                                                        orig_wid=que_wid) # 1xHxWx1
                # 原始尺寸的mask
                orig_mask = (orig_mask.detach().cpu().squeeze() * 255).numpy().astype(np.uint8) # HxW
                
                if not os.path.exists(os.path.dirname(coseg_mask_path)):
                    os.makedirs(os.path.dirname(coseg_mask_path))
                # 保存了mask到coseg_mask_path这个路径
                cv2.imwrite(coseg_mask_path, orig_mask)

                rgb_crop = (obj_data['rgb_image'].detach().cpu().squeeze().permute(1, 2, 0) * 255).numpy().astype(np.uint8)[:, :, ::-1]
                # 把rgb_crop保存到coseg_mask_path这个路径 .png 替换为_rgb.png
                cv2.imwrite(coseg_mask_path.replace('.png', '_rgb_crop.png'), rgb_crop)

            # 这里的camK是裁剪之前的相机内参
                            # 换成tensor
            # obj_data['rgb_mask']
            
            camK = match_utils.convert2K(initin1[i])
            obj_data['camK'] = camK  # object sequence-wise camera intrinsics, e.g., a fixed camera intrinsics for all frames within a sequence

            # obj_data['camK'] = camK  # object sequence-wise camera intrinsics, e.g., a fixed camera intrinsics for all frames within a sequence
            obj_data['img_scale'] = max(que_hei, que_wid)
            
            initilizer_timer = time.time()
            # initpose计算 mask继承在obj data了
            init_RTs = multiple_initial_pose_inference(obj_data=obj_data,
                                                        ref_database=reference_database,
                                                        device=device)
            
            cameras:list[Camera] = []

            pose = init_RTs[0]  # 获取第 i 个位姿 (4x4)

            pose_data = pose
            pose_data_np = np.array(pose_data).reshape(4, 4)


            R_ = pose_data_np[:3, :3].copy()
            t_ = pose_data_np[:3, 3].copy()

            height, width, channels = rgb_crop.shape


            if CFG.APPLY_ZOOM_AND_CROP:
                print(f"bbox_scale = {bbox_scale}\n")
                # que_zoom_rescaling_factor = target_size / bbox_scale
                # zoom_cam_fx = camK[0, 0] * que_zoom_rescaling_factor
                # zoom_cam_fy = camK[1, 1] * que_zoom_rescaling_factor
                # que_zoom_offsetX = -2 * (bbox_center[0] - camK[0, 2]) / bbox_scale
                # que_zoom_offsetY = -2 * (bbox_center[1] - camK[1, 2]) / bbox_scale
                
                # 这里有个bug：query_rescaling_factor
                # 这里是 /边界框在原                                        图像中的像素 target_size = 224 可能是裁剪后的像素
                que_zoom_rescaling_factor = target_size / bbox_scale
                zoom_cam_fx = camK[0, 0] * que_zoom_rescaling_factor
                zoom_cam_fy = camK[1, 1] * que_zoom_rescaling_factor
                que_zoom_FovX = focal2fov(zoom_cam_fx, target_size)
                que_zoom_FovY = focal2fov(zoom_cam_fy, target_size)
                # que_zoom_offsetX = -2 * (bbox_center[0] - camK[0, 2]) / bbox_scale
                # que_zoom_offsetY = -2 * (bbox_center[1] - camK[1, 2]) / bbox_scale
                cx = (camK[0, 2]-bbox_center[0]+bbox_scale/2)*que_zoom_rescaling_factor
                cy = (camK[1, 2]-bbox_center[1]+bbox_scale/2)*que_zoom_rescaling_factor

                    # cx = (camK[0, 2]-bbox_center[0])*que_zoom_rescaling_factor
                    # cy = (camK[1, 2]-bbox_center[1])*que_zoom_rescaling_factor

                # que_zoom_FovX = focal2fov(zoom_cam_fx, target_size)
                # que_zoom_FovY = focal2fov(zoom_cam_fy, target_size)
                # fovx =  2*math.atan(target_size/(2*zoom_cam_fx))
                # fovy =  2*math.atan(target_size/(2*zoom_cam_fy))

            K_crop = np.array([[zoom_cam_fx.cpu(), 0, cx.cpu()],
                                [0, zoom_cam_fy.cpu(), cy.cpu()],
                                [0, 0, 1]])

            camera = Camera(R_.T,t_,que_zoom_FovX,que_zoom_FovY,cx.cpu(),cy.cpu(),width,height,image_name='',image_path='',uid=0,preload_img=False)
            
            cameras.append(camera)


            if dataset.dataset.debug:
                print(f"裁剪之前的K = {camK}")
                print(f"裁剪之后的K = {K_crop}")
                print(f"目标照片的像素(长宽等长) ={target_size} ")
                print(f"裁剪中心 ={bbox_center} ")
                print(f"box像素大小 ={bbox_scale} ")
                print(f"que_zoom_rescaling_factor = {que_zoom_rescaling_factor}")
                print(f"裁剪后的cameraK = {cameras[0].get_k()}")
                pass



            # 添加一个没裁剪过的相机试一试raw_hei, raw_wid
                
            # 'rgb_mask': rgb_crop_mask.squeeze(0), # 1xSxS         
                
            # 获取 RGB 图像 (3, s, s) 和 mask (1, s, s)
            rgb_image = obj_data['rgb_image'].detach().cpu()  # shape: (3, s, s)
            rgb_mask = obj_data['rgb_mask'].detach().cpu()    # shape: (1, s, s)

            # 将 mask 扩展成 (3, s, s) 以匹配 RGB 图像
            rgb_mask_expanded = rgb_mask.expand_as(rgb_image)  # shape: (3, s, s)

            # 应用 mask：mask=1 的部分保留原值，mask=0 的部分置 0
            masked_rgb = rgb_image * rgb_mask_expanded  # shape: (3, s, s)

            # 转换为 numpy 并调整通道顺序 (H, W, C)
            masked_rgb_np = masked_rgb.permute(1, 2, 0).numpy()  # (s, s, 3)
            masked_rgb_np = (masked_rgb_np * 255).astype(np.uint8)[:, :, ::-1]   # 0-255 范围，保持 RGB

                

            

            # …………………………………………………………………………gspose
            T_p,init_T,linemod_init = self.pose_estimate_linemod(img_target=masked_rgb_np,
                                            GS_model=GS_model,
                                            initin1=K_crop,
                                            bbox_coords=box,
                                            camera_ref=cameras,
                                            debug=dataset.dataset.debug,
                                            camera_distance_scale = dataset.dataset.camera_distance_scale,
                                            mask_cut=True)
            if dataset.dataset.debug:
                print("")
            # 读取数据集pose
            pose_data_np = dataset.dataset.pose_lists[object_name][i]

            print(f"i = {i}")
            print(f"pose_data_np = {pose_data_np}")

            print(f"T linmode = {linemod_init}")
            print(f"T init = {init_T}")
            print(f"T refin = {T_p}")

            print(f"file = {img_target_dir[i]}")


            # pose_out_dir_temp
            name = os.path.splitext(os.path.basename(img_target_dir[i]))[0]

            # 这个路径是到每张照片的输出文件夹
            pose_dir = f"{pose_out_dir_temp}/{name}"

            os.makedirs(pose_dir, exist_ok=True)

            pose_init_dir = f"{pose_dir}/pose_init.txt"
            pose_refine_dir = f"{pose_dir}/pose_refine.txt"
            pose_gspose_dir = f"{pose_dir}/pose_gspose.txt"


            # 所有文件路径
            pose_files = [pose_init_dir, pose_refine_dir, pose_gspose_dir]

            # 删除已存在的文件
            for file_path in pose_files:
                if os.path.exists(file_path):
                    os.remove(file_path)

            # 创建新的空文件
            for file_path in pose_files:
                with open(file_path, 'w') as f:
                    pass  # 创建空文件即可


            with open(pose_init_dir, 'w') as f:
                    # init_T
                    np.savetxt(f, init_T, fmt='%.6f', delimiter=' ')
                    pass  # 创建空文件即可
            
            with open(pose_refine_dir, 'w') as f:
                    # init_T
                    np.savetxt(f, T_p, fmt='%.6f', delimiter=' ')
                    pass  # 创建空文件即可
            
            with open(pose_gspose_dir, 'w') as f:
                    # init_T
                    np.savetxt(f, linemod_init, fmt='%.6f', delimiter=' ')
                    pass  # 创建空文件即可
            
            # traget image 的np表示 w,h,c
            rgb_np = rgb_image.permute(1, 2, 0).numpy()

            if rgb_np.dtype != np.uint8:
                if rgb_np.max() <= 1.0:
                    rgb_np = (rgb_np * 255).astype(np.uint8)
                else:
                    rgb_np = rgb_np.astype(np.uint8)


            rgb_np = cv2.cvtColor(rgb_np, cv2.COLOR_BGR2RGB)



            img_init_dir = f"{pose_dir}/pose_init.png"
            img_refine_dir = f"{pose_dir}/pose_refine.png"
            img_gspose_dir = f"{pose_dir}/pose_gspose.png"

            initpose_rgb_show = draw_bbox.render_bboxes(pose_data_np,init_T,box,K_crop,rgb_np)
            gspose_rgb_show = draw_bbox.render_bboxes(pose_data_np,linemod_init,box,K_crop,rgb_np)
            refinpose_rgb_show = draw_bbox.render_bboxes(pose_data_np,T_p,box,K_crop,rgb_np)

            # 保存图像，若文件已存在则覆盖
            cv2.imwrite(img_init_dir, initpose_rgb_show)
            cv2.imwrite(img_refine_dir, refinpose_rgb_show)
            cv2.imwrite(img_gspose_dir, gspose_rgb_show)



            # cv2.imshow('Rendered Bboxes', initpose_rgb_show)
            # 等待任意键按下后关闭窗口
            # cv2.waitKey(0)
            # cv2.destroyAllWindows()
            




            
            # 初始的位姿
            init_add = metric_utils.calc_add_metric(GS_model._xyz.detach().cpu().numpy(),
                                                    dataset.dataset.diameter[object_name],
                                                    init_T,
                                                    pose_data_np,
                                                    syn=dataset.dataset.is_sym[object_name])
            
            # 优化的位姿
            refine_add = metric_utils.calc_add_metric(GS_model._xyz.detach().cpu().numpy(),
                                                      dataset.dataset.diameter[object_name],
                                                      T_p,
                                                      pose_data_np,
                                                      syn=dataset.dataset.is_sym[object_name])
            
            gspose_add = metric_utils.calc_add_metric(GS_model._xyz.detach().cpu().numpy(),
                                                    dataset.dataset.diameter[object_name],
                                                    linemod_init,
                                                    pose_data_np,
                                                    syn=dataset.dataset.is_sym[object_name])
            
            

            if 'ADD_metric' not in gspose_metric.keys():
                gspose_metric['ADD_metric'] = list()
            gspose_metric['ADD_metric'].append(gspose_add)

            refine_proj_err = metric_utils.calc_projection_2d_error(box, linemod_init, pose_data_np, camK.cpu().numpy())
            if 'Proj2D' not in gspose_metric.keys():
                gspose_metric['Proj2D'] = list()
            gspose_metric['Proj2D'].append(refine_proj_err)


            R_err,t_err= metric_utils.calc_pose_error(linemod_init,pose_data_np)
            gspose_metric["R_errs"].append(R_err)
            gspose_metric["t_errs"].append(t_err)


            # refine的数据
            if 'ADD_metric' not in metrics.keys():
                metrics['ADD_metric'] = list()
            metrics['ADD_metric'].append(refine_add)

            refine_proj_err = metric_utils.calc_projection_2d_error(box, T_p, pose_data_np, camK.cpu().numpy())
            if 'Proj2D' not in metrics.keys():
                metrics['Proj2D'] = list()
            metrics['Proj2D'].append(refine_proj_err)


            R_err,t_err= metric_utils.calc_pose_error(T_p,pose_data_np)
            metrics["R_errs"].append(R_err)
            metrics["t_errs"].append(t_err)


            # init的数据

            if 'ADD_metric' not in init_metrics.keys():
                init_metrics['ADD_metric'] = list()
            init_metrics['ADD_metric'].append(init_add)

            refine_proj_err = metric_utils.calc_projection_2d_error(box, init_T, pose_data_np, camK.cpu().numpy())
            if 'Proj2D' not in init_metrics.keys():
                init_metrics['Proj2D'] = list()
            init_metrics['Proj2D'].append(refine_proj_err)


            R_err_init,t_err_init= metric_utils.calc_pose_error(init_T,pose_data_np)
            init_metrics["R_errs"].append(R_err_init)
            init_metrics["t_errs"].append(t_err_init)






        add_gspose_metric= metric_utils.aggregate_metrics(gspose_metric)

        agg_metric = metric_utils.aggregate_metrics(metrics)

        init_agg_metric = metric_utils.aggregate_metrics(init_metrics)

        with open(dataset.dataset.metric_out_dir, 'a') as f :
            f.write(f"name = {object_name} : \n")
            f.write("gspose_metric\n")
            for key, value in add_gspose_metric.items():
                f.write(f"{key}: {value}\n")

            f.write("init_metric\n")
            for key, value in init_agg_metric.items():
                f.write(f"{key}: {value}\n")

            
            f.write("refine_metric\n")
            for key, value in agg_metric.items():
                f.write(f"{key}: {value}\n")
            


        print(agg_metric)
        return metrics,init_metrics,gspose_metric
    pass


    # def pose_estimate(self,img_target:np.ndarray,GS_model:GaussianModel,initin1:List[float],
    #                   bbox_coords:List[float],
    #                   camera_ref = None,
    #                   debug = False,
    #                   mask = None,
    #                   camera_distance_scale = 1.0,
    #                   refine_cfg = None,
    #                   mask_cut = False):
    def pose_estimate_linemod(self,img_target:np.ndarray,
                              GS_model:GaussianModel,
                              initin1:List[float],
                              bbox_coords:List[float], 
                              camera_ref = None,
                            debug = False,
                            mask_img = None,
                            camera_distance_scale = 1.0,
                            refine_cfg = None,
                            mask_cut = False):

        # img1 = cv2.imread(f'{imag_path}/55.png', cv2.IMREAD_GRAYSCALE)
        height, width = img_target.shape[:2]


        

        # cameras = match_utils.box2gscamera(box=bbox_coords,
        #                             K=initin1,
        #                             height=height,
        #                             width=width,
        #                             camera_num=self.camera_num,
        #                             debug=debug,
        #                             camera_distance_scale = camera_distance_scale)

        # cameras = match_utils.box2gscamera_linemod(box=bbox_coords,K=initin1,height=height,width=width,camera_num=self.camera_num)


        # target图片的内参
        K_img_np = match_utils.convert2K(initin1).cpu().numpy()

        # 如果linemod需要裁剪，那么这里就是根据mask裁剪rgb图片
        # if mask_cut:
        #     img_target_cut,K_cut = match_utils.crop_and_tile_mask(img_target,mask,K_img_np,debug=debug)
        #     img_target = img_target_cut
        #     K_img_np = K_cut
        
        # if camera_ref is not None:
        cameras = camera_ref

        

        img_xys,gs_xyz_ws = self.camera_matchs(cameras= cameras,
                                               GS_model= GS_model,
                                               img_target=img_target,
                                               match=self.match_name,
                                               debug=debug)

        dist_coeffs = np.zeros(shape=[8, 1], dtype="float64")
        T_init_ref = np.eye(4)
        T_init_ref[:3,:3] = cameras[0].R.T
        T_init_ref[:3,3] = cameras[0].T
        T_ = T_init_ref.copy()
        try:
            _, rvec, tvec, inliers = cv2.solvePnPRansac(gs_xyz_ws, 
                                                        img_xys,cameras[0].get_k().cpu().numpy(),
                                                        dist_coeffs,
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

            d_trans = np.linalg.norm(T_[:3, 3]-T_init_ref[:3,3])
            # 如果ref估计的位置跟差的太远
            # ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^


            # if d_trans>0.08:
            #     T_ = T_init_ref.copy()
            #     camera = cameras[0]
            # else:
            #     camera = Camera(colmap_id=0,R=T_[:3,:3].T,T=T_[:3,3],
            #                     FoVx=fovx,FoVy=fovy,Cx=cx,Cy=cy,image_height=height,image_width=width,
            #                     image_name='',image_path='',uid=0,preload_img=False)
            
            # if d_trans>0.08:
            T_ = T_init_ref.copy()
            camera = cameras[0]
            # else:
            #     camera = Camera(colmap_id=0,R=T_[:3,:3].T,T=T_[:3,3],
            #                     FoVx=fovx,FoVy=fovy,Cx=cx,Cy=cy,image_height=height,image_width=width,
            #                     image_name='',image_path='',uid=0,preload_img=False)

            # ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

            render_init = GS_Renderer(camera, GS_model, self.gaussian_PipeP, self.gaussian_BG)
            # render_init = GS_Renderer(cameras[0], GS_model, self.gaussian_PipeP, self.gaussian_BG)


            render_init_img = render_init['render']

            render_img_np = render_init_img.permute(1, 2, 0).detach().cpu().numpy()
            render_img_np = (render_img_np * 255).astype(np.uint8)
            if render_img_np.shape[0] == 3:
                render_img_np = np.transpose(render_img_np, (1, 2, 0))

            # 更改rgb改为gbr
            img_cv = render_img_np[:, :, [2, 1, 0]] 

            # cv2.imshow('Rendered init imag', img_cv)
            # cv2.waitKey()  # 按任意键关闭窗口
            # cv2.destroyAllWindows()


            class_gsrefine = GS_refine()


            imag0_rgb = cv2.cvtColor(img_target, cv2.COLOR_BGR2RGB)
            imag0_rgb_torch = torch.from_numpy(imag0_rgb).float().to('cuda')
            output = class_gsrefine.GS_Refiner(image=imag0_rgb_torch,
                                               mask=None,
                                               init_camera=camera,
                                               gaussians=GS_model,
                                               return_loss=True,
                                               debug=debug)
        
            # 返回的是4x4的np格式的位姿矩阵
            gs3d_delta_RT_np = output['gs3d_delta_RT']
            # gs3d_delta_RT_np[:3,:3] = 
            gs3d_pose_inv = np.linalg.inv(gs3d_delta_RT_np)
            init_T = T_
            T_ = (T_@gs3d_delta_RT_np)


        return T_,init_T,T_init_ref
    
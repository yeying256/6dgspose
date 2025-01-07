import os,sys

from pathlib import Path
# 这个是需要将python的路径给到根目录，要不然找不到路径
dir_path = Path(os.path.dirname(os.path.realpath(__file__))).parents[0]
print(f"dir_path {dir_path}")
sys.path.append(dir_path.__str__())
import time
import random


from gaussian_renderer import render as GS_Renderer
from argparse import ArgumentParser
from arguments import ModelParams, PipelineParams, OptimizationParams
from scene import GaussianModel 
from scene.cameras import MiniCam,Camera
import torch
import torch.nn.functional as torch_F
import numpy as np
import cv2
from scipy.spatial.transform import Rotation
from misc_utils.warmup_lr import CosineAnnealingWarmupRestarts
import string

# 将字典转化为类
from types import SimpleNamespace


import os

from pytorch_msssim import SSIM, MS_SSIM

from scene.colmap_loader import read_extrinsics_text, read_intrinsics_text, qvec2rotmat, \
    read_extrinsics_binary, read_intrinsics_binary, read_points3D_binary, read_points3D_text

SSIM_METRIC = SSIM(data_range=1, size_average=True, channel=3) # channel=1 for grayscale images
MS_SSIM_METRIC = MS_SSIM(data_range=1, size_average=True, channel=3)


def show_gs_img(target_img:torch.Tensor):
    image_refine_torch = target_img
    image_refine_torch = image_refine_torch.cpu()
    image_refine = image_refine_torch.detach().numpy()
    
    # 转换一下格式
    image_refine = image_refine.transpose(1, 2, 0)
    image_refine = ((image_refine - image_refine.min()) / (image_refine.max() - image_refine.min()) * 255).astype(np.uint8)

    image_refine_BGR = cv2.cvtColor(image_refine,cv2.COLOR_RGB2BGR)

    cv2.imshow('Rendered Image', image_refine_BGR)
    cv2.waitKey()  # 按任意键关闭窗口
    return image_refine_BGR

class GS_refine:
    def __init__(self,CFG = None) -> None:

        if CFG ==None:
            # CFG = {
            #     'START_LR' : 5e-3,
            #     'MAX_STEPS' : 1000,
            #     'END_LR' : 1e-6,
            #     'WARMUP' : 10,
            #     'USE_SSIM' : True,
            #     'USE_MS_SSIM' : True,
            #     'EARLY_STOP_MIN_STEPS' : 5,
            #     'EARLY_STOP_LOSS_GRAD_NORM' : 1e-4
            # }
            CFG = SimpleNamespace(
                # 初始学习率
                START_LR=5e-4,
                # 最大步数
                MAX_STEPS=1000,
                END_LR=1e-6,
                WARMUP=10,
                USE_SSIM=True,
                USE_MS_SSIM=True,
                EARLY_STOP_MIN_STEPS=10,
                EARLY_STOP_LOSS_GRAD_NORM=5e-5
                )
        self.CFG = CFG
        self.SSIM_METRIC = SSIM(data_range=1, size_average=True, channel=3) # channel=1 for grayscale images
        self.MS_SSIM_METRIC = MS_SSIM(data_range=1, size_average=True, channel=3)


        self.device = torch.device('cuda')
        self.parser = ArgumentParser()
        self.gaussian_ModelP = ModelParams(self.parser)
        self.gaussian_PipeP = PipelineParams(self.parser)
        self.gaussian_OptimP = OptimizationParams(self.parser)
        self.gaussian_BG = torch.zeros((3), device=self.device)

    def GS_Refiner(self,image:torch.Tensor , mask:torch.Tensor , init_camera:Camera, gaussians:GaussianModel, return_loss=False):
        if image.dim() == 4:
            image = image.squeeze(0)
        if image.shape[2] == 3:
            image = image.permute(2, 0, 1) # 3xSxS
        if mask ==None:
            mask = torch.ones((1,) + image.shape[1:], dtype=torch.bool)
        else:
            if mask.dim() == 2:
                mask = mask[None, :, :]
            if mask.dim() == 4:
                mask = mask.squeeze(0)
            if mask.shape[2] == 1:
                mask = mask.permute(2, 0, 1) # 1xSxS

        
        assert(image.dim() == 3 and image.shape[0] == 3), image.shape


        gaussians.initialize_pose()
        params = [gaussians._delta_R, gaussians._delta_T]
        optimizer = torch.optim.AdamW(params, lr=self.CFG.START_LR)

        lr_scheduler = CosineAnnealingWarmupRestarts(optimizer, self.CFG.MAX_STEPS,
                                                    warmup_steps=self.CFG.WARMUP, 
                                                    max_lr=self.CFG.START_LR, 
                                                    min_lr=self.CFG.END_LR)
        trunc_mask = (image.sum(dim=0, keepdim=True) > 0).type(torch.float32) # 1xSxS        
        target_img = (image * mask.float().to(self.device)).to(self.device)
        target_img = target_img/255

        # 测试代码
        # image_refine_torch = target_img
        # image_refine_torch = image_refine_torch.cpu()
        # image_refine = image_refine_torch.detach().numpy()
        
        # # 转换一下格式
        # image_refine = image_refine.transpose(1, 2, 0)
        # image_refine = ((image_refine - image_refine.min()) / (image_refine.max() - image_refine.min()) * 255).astype(np.uint8)

        # image_refine_BGR = cv2.cvtColor(image_refine,cv2.COLOR_RGB2BGR)

        # cv2.imshow('Rendered Image', image_refine_BGR)
        # cv2.waitKey()  # 按任意键关闭窗口
        
        # show_gs_img(target_img/2 + target_img/2)
        # show_gs_img(target_img)

        # 测试代码

        iter_losses = list()
        for iter_step in range(self.CFG.MAX_STEPS):
            # GS_Renderer是渲染器
            render_img = GS_Renderer(init_camera, gaussians, self.gaussian_PipeP, self.gaussian_BG)['render'] * trunc_mask
            # show_gs_img(render_img)
            show_gs_img(render_img/2+target_img/2)
            # show_gs_img(target_img)
            loss = 0.0

            if self.CFG.USE_SSIM:
                loss += (1 - self.SSIM_METRIC(render_img[None, ...], target_img[None, ...]))
                print(f"step ={iter_step} loss = {loss}")
            if self.CFG.USE_MS_SSIM:
                loss += (1 - self.MS_SSIM_METRIC(render_img[None, ...], target_img[None, ...]))

            loss.backward()
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()

            iter_losses.append(loss.item())
            if iter_step >= self.CFG.EARLY_STOP_MIN_STEPS:
                loss_grads = (torch.as_tensor(iter_losses)[1:] - torch.as_tensor(iter_losses)[:-1]).abs()
                if loss_grads[-self.CFG.EARLY_STOP_MIN_STEPS:].mean() < self.CFG.EARLY_STOP_LOSS_GRAD_NORM: # early stop the refinement
                    break
        
        gs3d_delta_RT = gaussians.get_delta_pose.squeeze(0).detach().cpu().numpy()


        outp = {
            'gs3d_delta_RT': gs3d_delta_RT,
            'iter_step': iter_step,
            'render_img': render_img,
        }
        
        if return_loss:
            sobel_x = torch.tensor([[1, 0, -1], [2, 0, -2], [1, 0, -1]], dtype=torch.float32).view(1, 1, 3, 3).repeat(1, 3, 1, 1)
            sobel_x = sobel_x.to(image.device)
            sobel_y = sobel_x.transpose(-2, -1)
            # Apply Sobel filter to the images
            query_sobel_h = torch_F.conv2d(image[None], sobel_x, padding=0)
            query_sobel_v = torch_F.conv2d(image[None], sobel_y, padding=0)
            rend_sobel_h = torch_F.conv2d(render_img[None], sobel_x, padding=0)
            rend_sobel_v = torch_F.conv2d(render_img[None], sobel_y, padding=0)
            edge_err = (query_sobel_h - rend_sobel_h).abs().mean() + (query_sobel_v - rend_sobel_v).abs().mean()
            outp['edge_err'] = edge_err

        return outp
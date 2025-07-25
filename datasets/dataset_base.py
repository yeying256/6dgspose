import yaml
import argparse
import logging
from pathlib import Path
import os,sys
import numpy as np
import logging

from typing import Dict, List
import cv2

from types import SimpleNamespace

# 这个是需要将python的路径给到根目录，要不然找不到路径
dir_path = Path(os.path.dirname(os.path.realpath(__file__))).parents[0]
print(f"dir_path {dir_path}")
sys.path.append(dir_path.__str__())

class dataset_base:
    def __init__(self) -> None:
        #这里包含了 这个数据集的子目标的所有名字 如果是linemod，就是标号，没有前面的0
        self.sub_name_lists = []
        self.ID_lists = []
        # 这里面包含了数据集的所有照片路径 这里的id是去掉前面的0
        self.color_dir_lists:Dict[str, List[str]]  = {}
        # 所有相机内参路径
        self.cam_intr_dir_lists:Dict[str, List[str]]  = {}
        self.cam_K_lists:Dict[str,List[np.ndarray]] = {}

        # 所有的位姿路径
        self.pose_dir_lists:Dict[str, List[str]]  = {}
        self.pose_lists:Dict[str,List[np.ndarray]] = {}
        self.allo_pose_lists:Dict[str,List[np.ndarray]] = {}
        # 3dgs的
        self.gs_model_dir_lists:Dict[str, List[str]] = {}
        # box边界
        self.box_dir_lists:Dict[str,str] = {}
        self.box:Dict[str,np.ndarray] = {}

        # 参考的内参和外参,这个是用来生成参考的
        self.ref_K:Dict[str,List[np.ndarray]] = {}
        self.ref_T:Dict[str,List[np.ndarray]] = {}

        self.ref_dir:str = '' 

        # 3dgs的直径
        self.diameter:Dict[str,float] ={}

        self.is_sym:Dict[str,bool] ={}


        self.metric_out_dir:str = ""

        self.colmap_dir:str = ""

        self.color_forder_name = ""

        # 每张照片mask的路径
        self.mask_dir_lists:Dict[str, List[str]]  = {}
        self.maskforder_dir = ""
        
        self.debug = False
        self.camera_distance_scale = 1.0

        self.refine_CFG = SimpleNamespace(
                # 初始学习率
                START_LR=8e-4,
                # 最大步数
                MAX_STEPS=1000,
                END_LR=1e-6,
                WARMUP=10,
                USE_SSIM=True,
                USE_MS_SSIM=True,
                EARLY_STOP_MIN_STEPS=10,
                EARLY_STOP_LOSS_GRAD_NORM=5e-6
                )
        
        self.gspose_model_path = ""
        
        self.camera_gap = 10

        self.debug_target = None

        self.pose_out_dir = None
        pass

    def get_imag(self,imag_dir:str):
        '''
        imag_dir:单一图像的路径
        return: 
        '''
        # 读取图像
        image = cv2.imread(imag_dir)
        # 将BGR格式转换为RGB格式
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return image_rgb
    

    def get_intrinsics(self,intrinsics_dir:str):
        intrinsics_data = np.loadtxt(intrinsics_dir)
        # 验证文件内容是否为 3x3 矩阵
        if intrinsics_data.shape != (3, 3):
            logging.error(f"Invalid matrix shape: expected 3x3, got {intrinsics_data.shape}")
            raise ValueError(f"Invalid matrix shape: expected 3x3, got {intrinsics_data.shape}")
        return intrinsics_data
    
    def get_pose(self,pose_dir:str):
        pose_data = np.loadtxt(pose_dir)
        # 验证文件内容是否为 3x3 矩阵
        if pose_data.shape != (4, 4):
            logging.error(f"Invalid matrix shape: expected 4x4, got {pose_data.shape}")
            raise ValueError(f"Invalid matrix shape: expected 4x4, got {pose_data.shape}")
        return pose_data
        pass

    pass
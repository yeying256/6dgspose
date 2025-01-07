import yaml
import argparse
import logging
from pathlib import Path
import os,sys
import numpy as np
import logging

from typing import Dict, List
import cv2

# 这个是需要将python的路径给到根目录，要不然找不到路径
dir_path = Path(os.path.dirname(os.path.realpath(__file__))).parents[0]
print(f"dir_path {dir_path}")
sys.path.append(dir_path.__str__())

class dataset_base:
    def __init__(self) -> None:
        #这里包含了 这个数据集的子目标的所有名字 
        self.sub_name_lists = []
        # 这里面包含了数据集的所有照片路径
        self.color_dir_lists:Dict[str, List[str]]  = {}
        # 所有相机内参路径
        self.cam_intr_dir_lists:Dict[str, List[str]]  = {}
        # 所有的位姿路径
        self.pose_dir_lists:Dict[str, List[str]]  = {}
        # 3dgs的
        self.gs_model_dir_lists:Dict[str, List[str]] = {}
        # box边界
        self.box_dir_lists:Dict[str,str] = {}
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
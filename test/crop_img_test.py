import yaml
import argparse
import logging
from pathlib import Path
import os,sys
import numpy as np
import cv2
import torch




# 这个是需要将python的路径给到根目录，要不然找不到路径
dir_path = Path(os.path.dirname(os.path.realpath(__file__))).parents[0]
print(f"dir_path {dir_path}")
sys.path.append(dir_path.__str__())

from datasets.datasets import datasets
from matcher.match_2d3d import match_2d3d

from tqdm import tqdm

from scene import GaussianModel 
from misc_utils import metric_utils
from misc_utils import match_utils




def load_yaml_config(config_file):
    """加载 YAML 配置文件"""
    with open(config_file, 'r', encoding='utf-8') as file:
        try:
            config = yaml.safe_load(file)
            return config
        except yaml.YAMLError as exc:
            logging.error(f"Error loading YAML file: {exc}")
            raise

def main():

    # 加载数据集
    imag_path = "/media/wangxiao/Newsmy/linemod/raw/lm_test_all/test/000001/rgb/000000.png"
    mask_path = "/media/wangxiao/Newsmy/linemod/raw/lm_test_all/test/000001/mask/000000_000000.png"

    image = cv2.imread(imag_path)
    mask = cv2.imread(mask_path,cv2.IMREAD_GRAYSCALE)

    K = np.array([[572.4114, 0, 325.2611],
                    [0, 573.57043, 242.04899],
                    [0, 0, 1]])


    result, new_K = match_utils.crop_and_tile_mask(image,mask,K,True)

    # 打印更新后的 K 矩阵
    print("更新后的 K 矩阵:")
    print(new_K)
    
    
    pass

if __name__ == '__main__':
    # 使用 argparse 解析命令行参数


    # 
    # 调用 main 函数
    main()
    # 运行时运行这个指令 python main.py config.yaml
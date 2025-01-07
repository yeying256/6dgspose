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




def load_yaml_config(config_file):
    """加载 YAML 配置文件"""
    with open(config_file, 'r', encoding='utf-8') as file:
        try:
            config = yaml.safe_load(file)
            return config
        except yaml.YAMLError as exc:
            logging.error(f"Error loading YAML file: {exc}")
            raise

def main(config_file:str):
    config = load_yaml_config(config_file)
    # 加载数据集
    dataset = datasets(config)
    pose_match2d3d = match_2d3d(config)
    
    

    for object in tqdm(dataset.dataset.object_lists, desc="Processing objects"):
        # 进入到了object
        
        pose_match2d3d.pose_estimate_batch(GS_model_dir = dataset.dataset.gs_model_dir_lists[object],
                                            bbox_coords = dataset.dataset.box_dir_lists[object],
                                            initin1 = dataset.dataset.cam_intr_dir_lists[object],
                                            img_target_dir = dataset.dataset.color_dir_lists[object])
        pass

        


    print(config['data'])
    pass

if __name__ == '__main__':
    # 使用 argparse 解析命令行参数
    parser = argparse.ArgumentParser(description="Load and process YAML configuration file.")
    parser.add_argument('config_file', type=str, help="Path to the YAML configuration file")
    args = parser.parse_args()

    # 
    # 调用 main 函数
    main(args.config_file)
    # 运行时运行这个指令 python main.py config.yaml
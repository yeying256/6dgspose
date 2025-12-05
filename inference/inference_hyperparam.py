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
    # pose_match2d3d.camera_num :最小相机数量，
    # pose_match2d3d.camera_num_gap : 每次循环结束后相机添加的数量
    # pose_match2d3d.camera_num_uplimit :最大相机数量不能超过这个数
    start = pose_match2d3d.camera_num
    init_camera_num = pose_match2d3d.camera_num
    gap = pose_match2d3d.camera_num_gap
    uplimit = pose_match2d3d.camera_num_uplimit


    for current_camera_num in range(start, uplimit + 1, gap):

        print(f"Processing with {current_camera_num} cameras...")
        pose_match2d3d.camera_num = current_camera_num
    
    
        refine_metrics = {'R_errs': [], 't_errs': [],'ADD_metric':[],'Proj2D':[]}
        init_metrics = {'R_errs': [], 't_errs': [],'ADD_metric':[],'Proj2D':[]}

        for object in tqdm(dataset.dataset.object_lists, desc="Processing objects"):
            # 进入到了object

            if dataset.dataset.debug_target is not None:
                if object not in dataset.dataset.debug_target:
                    continue


            # metrics,init_metric = pose_match2d3d.pose_estimate_batch(GS_model_dir = dataset.dataset.gs_model_dir_lists[object],
            #                                     bbox_coords = dataset.dataset.box_dir_lists[object],
            #                                     initin1 = dataset.dataset.cam_intr_dir_lists[object],
            #                                     img_target_dir = dataset.dataset.color_dir_lists[object],
            #                                     object_name=object,dataset=dataset)
                

            with open(dataset.dataset.metric_out_dir, 'a') as f :
                f.write(f"current_camera_num = {pose_match2d3d.camera_num} \n")
                f.write(f"match_confidence_threshold = {pose_match2d3d.match_confidence_threshold} \n")


            metrics,init_metric = pose_match2d3d.pose_estimate_batch2(GS_model_dir = dataset.dataset.gs_model_dir_lists[object],
                                                    bbox_coords = dataset.dataset.box_dir_lists[object],
                                                    initin1 = dataset.dataset.cam_intr_dir_lists[object],
                                                    img_target_dir = dataset.dataset.color_dir_lists[object],
                                                    object_name=object,dataset=dataset)

            # 拼接字典
            refine_metrics = {key: refine_metrics.get(key, []) + metrics.get(key, []) for key in set(refine_metrics) | set(metrics)}
            init_metrics = {key: init_metrics.get(key, []) + init_metric.get(key, []) for key in set(init_metrics) | set(init_metric)}

        # 测试下代码
            # break

            pass
        agg_refine_metric = metric_utils.aggregate_metrics(refine_metrics)
        agg_init_metric = metric_utils.aggregate_metrics(init_metrics)


        with open(dataset.dataset.metric_out_dir, 'a') as f :
            f.write(f"current_camera_num = {current_camera_num} \n")
            
            f.write(f"name = ALL : \n")
            for key, value in agg_refine_metric.items():
                f.write(f"{key}: {value}\n")
            for key, value in agg_init_metric.items():
                f.write(f"{key}: {value}\n")
        
        print(config['data'])
        pass

    # pose_match2d3d.gap :  每次循环结束后相机添加的置信度参数
    # pose_match2d3d.up_limit : 置信度的上限
    # pose_match2d3d.match_confidence_threshold :初始的置信度
    start = pose_match2d3d.match_confidence_threshold
    stop = pose_match2d3d.up_limit + pose_match2d3d.gap  # 因为 arange 不包含 stop
    step = pose_match2d3d.gap

    pose_match2d3d.camera_num = init_camera_num
    for confidence in np.arange(start, stop, step):

        if confidence > pose_match2d3d.up_limit:

            break
        pose_match2d3d.match_confidence_threshold = confidence
        print(f"Processing with confidence threshold: {confidence:.2f}")
        refine_metrics = {'R_errs': [], 't_errs': [],'ADD_metric':[],'Proj2D':[]}
        init_metrics = {'R_errs': [], 't_errs': [],'ADD_metric':[],'Proj2D':[]}

        for object in tqdm(dataset.dataset.object_lists, desc="Processing objects"):
            # 进入到了object

            if dataset.dataset.debug_target is not None:
                if object not in dataset.dataset.debug_target:
                    continue


            # metrics,init_metric = pose_match2d3d.pose_estimate_batch(GS_model_dir = dataset.dataset.gs_model_dir_lists[object],
            #                                     bbox_coords = dataset.dataset.box_dir_lists[object],
            #                                     initin1 = dataset.dataset.cam_intr_dir_lists[object],
            #                                     img_target_dir = dataset.dataset.color_dir_lists[object],
            #                                     object_name=object,dataset=dataset)
                


            with open(dataset.dataset.metric_out_dir, 'a') as f :
                f.write(f"current_camera_num = {pose_match2d3d.camera_num} \n")
                f.write(f"match_confidence_threshold = {pose_match2d3d.match_confidence_threshold} \n")

            metrics,init_metric = pose_match2d3d.pose_estimate_batch2(GS_model_dir = dataset.dataset.gs_model_dir_lists[object],
                                                    bbox_coords = dataset.dataset.box_dir_lists[object],
                                                    initin1 = dataset.dataset.cam_intr_dir_lists[object],
                                                    img_target_dir = dataset.dataset.color_dir_lists[object],
                                                    object_name=object,dataset=dataset)

            # 拼接字典
            refine_metrics = {key: refine_metrics.get(key, []) + metrics.get(key, []) for key in set(refine_metrics) | set(metrics)}
            init_metrics = {key: init_metrics.get(key, []) + init_metric.get(key, []) for key in set(init_metrics) | set(init_metric)}

        # 测试下代码
            # break

            pass
        agg_refine_metric = metric_utils.aggregate_metrics(refine_metrics)
        agg_init_metric = metric_utils.aggregate_metrics(init_metrics)


        with open(dataset.dataset.metric_out_dir, 'a') as f :
            
            
            f.write(f"name = ALL : \n")
            for key, value in agg_refine_metric.items():
                f.write(f"{key}: {value}\n")
            for key, value in agg_init_metric.items():
                f.write(f"{key}: {value}\n")
        


        print(config['data'])
        pass
        # 执行你的逻辑

if __name__ == '__main__':
    # 使用 argparse 解析命令行参数
    parser = argparse.ArgumentParser(description="Load and process YAML configuration file.")
    parser.add_argument('config_file', type=str, help="Path to the YAML configuration file")
    args = parser.parse_args()

    # 
    # 调用 main 函数
    main(args.config_file)
    # 运行时运行这个指令 python main.py config.yaml
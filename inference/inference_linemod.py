import yaml
import argparse
import logging
from pathlib import Path
import os,sys
import numpy as np
import cv2
import torch

import pickle





# 这个是需要将python的路径给到根目录，要不然找不到路径
dir_path = Path(os.path.dirname(os.path.realpath(__file__))).parents[0]
print(f"dir_path {dir_path}")
sys.path.append(dir_path.__str__())

from datasets.datasets import datasets
from datasets import linemod_dataset as lm_data
from matcher.match_2d3d import match_2d3d

from tqdm import tqdm

from scene import GaussianModel 
from misc_utils import metric_utils

from model.network import model_arch as ModelNet

from configs import inference_cfg as CFG

from datasets.inference_datasets import datasetCallbacks

from arguments import ModelParams, PipelineParams, OptimizationParams
# from gaussian_object.build_3DGaussianObject import create_3D_Gaussian_object


def load_yaml_config(config_file):
    """加载 YAML 配置文件"""
    with open(config_file, 'r', encoding='utf-8') as file:
        try:
            config = yaml.safe_load(file)
            return config
        except yaml.YAMLError as exc:
            logging.error(f"Error loading YAML file: {exc}")
            raise

def main(config_file:str,train_data_file:str):
    # 测试数据集config
    config = load_yaml_config(config_file)
    # 参考数据集config
    train_data_config= load_yaml_config(train_data_file)
    # 加载数据集
    dataset = datasets(config)
    train_data = datasets(train_data_config)
    
    pose_match2d3d = match_2d3d(config)


    device = torch.device('cuda:0')
    # 这是那个 网络
    model_net = ModelNet().to(device)

    gspose_path = dataset.dataset.gspose_model_path


    ckpt_weight = torch.load(gspose_path, map_location=device)
    model_net.load_state_dict(ckpt_weight)
    print('Pretrained weights are loaded from ', gspose_path.split('/')[-1])
    model_net.eval()

    dataset_name = "LINEMOD"
    # 设置gspose的数据集
    datasetLoader = datasetCallbacks[dataset_name]['DATASETLOADER']

    gaussian_ModelP = ModelParams(parser)
    gaussian_PipeP  = PipelineParams(parser)
    gaussian_OptimP = OptimizationParams(parser)
    gaussian_BG = torch.zeros((3), device=device)



    
    
    refine_metrics = {'R_errs': [], 't_errs': [],'ADD_metric':[],'Proj2D':[]}
    init_metrics = {'R_errs': [], 't_errs': [],'ADD_metric':[],'Proj2D':[]}

    gspose_metrics = {'R_errs': [], 't_errs': [],'ADD_metric':[],'Proj2D':[]}

    for object in tqdm(dataset.dataset.ID_lists, desc="Processing objects"):


        
        gs_model_dir = dataset.dataset.gs_model_dir_lists[object]
        gs_model_dir

                # 新的文件名
        new_filename = "database.pkl"

        # 找到最后一个斜杠的位置
        last_slash_index = gs_model_dir.rfind('/')

        # 如果找到了斜杠，则构建新的路径；否则，假设整个字符串就是一个文件名（这种情况比较少见）
        obj_ref_database_path = os.path.join(dataset.dataset.ref_output_dir,
                                object,
                                'database.pkl', 
                                )



        if not os.path.exists(obj_ref_database_path): 
            ref_database = lm_data.create_reference_database_from_RGB_images(model_func=model_net,
                                                    obj_dataset=dataset.dataset,
                                                    device=device,
                                                    save_pred_mask=True,
                                                    id=object)
        else:
            print('load database from ', obj_ref_database_path)
            with open(obj_ref_database_path, 'rb') as df:
                ref_database = pickle.load(df)
    
        ref_database['obj_bbox3D'] = torch.as_tensor(train_data.dataset.box[object], dtype=torch.float32)


        ref_database['bbox3d_diameter'] = torch.as_tensor(train_data.dataset.diameter[object], dtype=torch.float32)

        for _key, _val in ref_database.items():
            if isinstance(_val, np.ndarray):
                # 换成torch
                ref_database[_key] = torch.as_tensor(_val, dtype=torch.float32).to(device)

        for _key, _val in ref_database.items():
            if isinstance(_val, torch.Tensor):
                ref_database[_key] = _val.cuda() 

        # 进入到了object
        
        # obj_refer_database_dir = os.path.join(dataset.dataset.ref_output_dir, object)
        # obj_ref_database_path = os.path.join(obj_refer_database_dir, f'{object}_database.pkl') 

        # if not os.path.exists(obj_ref_database_path):
        #     print(f'preprocess reference data for {object}')
        #     datapath = Path.joinpath(dataset.dataset.dataset_dir,'{:06d}.png'.format(object))
        #     obj_refer_dataset = datasetLoader(datapath, object, 
        #                                     subset_mode='train', 
        #                                     num_refer_views=-1,
        #                                     use_binarized_mask=CFG.BINARIZE_MASK,
        #                                     obj_database_dir=obj_refer_database_dir)
            # 利用哪个网络生成参考数据集
            # ref_database = create_reference_database_from_RGB_images(model_net, obj_refer_dataset, device=device, save_pred_mask=True)
            # ref_database['obj_bbox3D'] = torch.as_tensor(obj_refer_dataset.obj_bbox3d, dtype=torch.float32)
            # ref_database['bbox3d_diameter'] = torch.as_tensor(obj_refer_dataset.bbox3d_diameter, dtype=torch.float32)




        metrics,init_metric,gspose_metric = pose_match2d3d.pose_estimate_batch_linemod(GS_model_dir = dataset.dataset.gs_model_dir_lists[object],
                                            bbox_coords = dataset.dataset.box[object],
                                            initin1 = dataset.dataset.cam_K_lists[object],
                                            img_target_dir = dataset.dataset.color_dir_lists[object],
                                            object_name=object,
                                            dataset=dataset,
                                            reference_database=ref_database,
                                            model_func=model_net,
                                            device = device,
                                            save_pred_mask = True)

        # 拼接字典
        refine_metrics = {key: refine_metrics.get(key, []) + metrics.get(key, []) for key in set(refine_metrics) | set(metrics)}
        init_metrics = {key: init_metrics.get(key, []) + init_metric.get(key, []) for key in set(init_metrics) | set(init_metric)}
        gspose_metrics = {key: gspose_metrics.get(key, []) + gspose_metric.get(key, []) for key in set(gspose_metrics) | set(gspose_metric)}


# 测试下代码
        # break

        pass
    agg_refine_metric = metric_utils.aggregate_metrics(refine_metrics)
    agg_init_metric = metric_utils.aggregate_metrics(init_metrics)
    agg_gspose_metric = metric_utils.aggregate_metrics(gspose_metrics)


    with open(dataset.dataset.metric_out_dir, 'a') as f :
        f.write(f"name = ALL : \n")
        f.write(f"gspose_metric : \n")
        for key, value in agg_gspose_metric.items():
            f.write(f"{key}: {value}\n")
        f.write(f"init : \n")
        for key, value in agg_init_metric.items():
            f.write(f"{key}: {value}\n")
        f.write(f"refine : \n")
        for key, value in agg_refine_metric.items():
            f.write(f"{key}: {value}\n")

        


    print(config['data'])
    pass

if __name__ == '__main__':
    # 使用 argparse 解析命令行参数
    parser = argparse.ArgumentParser(description="Load and process YAML configuration file.")
    parser.add_argument('config_file', type=str, help="Path to the YAML configuration file")
    parser.add_argument('ref_data', type=str, help="Path to the YAML configuration file")


        # Set up command line argument parser
    parser.add_argument('--ip', type=str, default="127.0.0.1")
    parser.add_argument('--port', type=int, default=6009)
    parser.add_argument('--debug_from', type=int, default=-1)
    parser.add_argument('--detect_anomaly', action='store_true', default=False)
    parser.add_argument("--test_iterations", nargs="+", type=int, default=[30_000])
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[30_000])
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[])
    parser.add_argument("--start_checkpoint", type=str, default = None)
    
    ###### arguments for CoSegPose ########
    parser.add_argument('--num_refer_views', type=int, default=-1)
    parser.add_argument('--dataset_name', default='LINEMOD', type=str, help='dataset name')
    parser.add_argument('--outpose_dir', default='output_pose', type=str, help='output pose directory')
    parser.add_argument('--database_dir', default='reference_database', type=str, help='reference database directory')

    parser.add_argument('--build_GS_model', action='store_true', help='enable fine detection')
    parser.add_argument('--enable_GS_Refiner', action='store_true', help='enable 3D Gaussian Splatting Refiner')

    args = parser.parse_args()
    
    # 
    # 调用 main 函数
    main(args.config_file,args.ref_data)
    # 运行时运行这个指令 python main.py config.yaml
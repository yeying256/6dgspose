# 这两个是加入配置参数
from omegaconf.dictconfig import DictConfig
import hydra
# 进度条
from tqdm import tqdm
# 系统指令
import os
import os.path as osp
# 路径指令
from pathlib import Path
import numpy as np
import torch
import math

# 导入onepose网络
from src.inference.inference_OnePosePlus import inference_onepose_plus


from argparse import ArgumentParser
from arguments import ModelParams, PipelineParams, OptimizationParams
from scene import GaussianModel 
parser = ArgumentParser()
gaussian_ModelP = ModelParams(parser)
gaussian_PipeP = PipelineParams(parser)
gaussian_OptimP = OptimizationParams(parser)
gaussian_BG = torch.zeros((3), device='cuda')



# 这个就是推断处理数据的入口
def inference_worker(data_dirs, cfg, pba=None, worker_id=0):
    # # data_dirs：
    # 0:'data/datasets/test_data/0408-colorbox-box colorbox-4'
    # 1:'data/datasets/test_data/0409-aptamil-box aptamil-3'
    print(
        f"Worker {worker_id} will process: {[(data_dir.split(' ')[0]).split('/')[-1][:4] for data_dir in data_dirs]}, total: {len(data_dirs)} objects"
    )
    data_dirs = tqdm(data_dirs) if pba is None else data_dirs

    obj_name2metrics = {}
    # 遍历每一个数据路径
    for data_dir in data_dirs:
        print(f"Processing {data_dir}.")

        # Load obj name and inference sequences 把目录分开
        root_dir, sub_dirs = data_dir.split(" ")[0], data_dir.split(" ")[1:]
        sfm_mapping_sub_dir = '-'.join([sub_dirs[0].split("-")[0], '1'])
        num_img_in_mapping_seq = len(os.listdir(osp.join(root_dir, sfm_mapping_sub_dir, 'color')))
        obj_name = root_dir.split("/")[-1]
        # sfm输出的路径
        sfm_base_path = cfg.sfm_base_dir
        # 3dgs的输出路径
        gs_dir = cfg.gs_dir

        # GT：Ground Truth 已经标注好的真实数据作为对象检测方法
        if "object_detector_method" in cfg:
            object_detector_method = cfg.object_detector_method
        else:
            object_detector_method = 'GT'

        # Get all inference image path
        all_image_paths = []
        for sub_dir in sub_dirs:

            if object_detector_method == 'GT':
                color_dir = osp.join(root_dir, sub_dir, "color")
            else:
                raise NotImplementedError

            img_paths = list(Path(color_dir).glob("*.png"))
            if len(img_paths) == num_img_in_mapping_seq:
                print(f"Same num of images in test sequence:{sub_dir}")
            # 把里面每个路径改成字符串
            image_paths = [str(img_path) for img_path in img_paths]
            all_image_paths += image_paths

        if len(all_image_paths) == 0:
            print(f"No png image in {root_dir}")
            if pba is not None:
                pba.update.remote(1)
            continue
        
        # 找到sfm输出的工作文件夹
        sfm_results_dir = osp.join(
            sfm_base_path,
            "outputs_"
            + cfg.match_type
            + "_"
            + cfg.network.detection
            + "_"
            + cfg.network.matching,
            obj_name,
        )
        gs_result_dir = osp.join(gs_dir,obj_name,"test","point_cloud","iteration_30000","point_cloud.ply")
        # 边界框sub_dirs[0]
        boxdir = osp.join(root_dir,"box3d_corners.txt")

        obj_gaussians = GaussianModel(gaussian_ModelP.sh_degree)
        obj_gaussians.load_ply(gs_result_dir)
        # 裁剪gs模型
        obj_gaussians.clip_to_bounding_box_fromfile(boxdir)

        # 把数据三维模型加进去

        metrics = inference_onepose_plus(sfm_results_dir =sfm_results_dir, all_image_paths = all_image_paths, 
                                         gaussion_model=obj_gaussians ,
                                         cfg = cfg, use_ray=cfg.use_local_ray, verbose=cfg.verbose)
        obj_name2metrics[obj_name] = metrics
        if pba is not None:
            pba.update.remote(1)
    # 结果就是精度评价
    return obj_name2metrics


def inference(cfg):
    # Load all test objects
    # data_dir 为data_base_dir
    data_dirs = cfg.data_dir
    # data_base_dir 是用作测试的照片，用来测试数据集中的照片的位姿
    # data_base_dir: "data/datasets/lowtexture_test_data"
    # sfm_output 是用作保存三维空间结构的位置，保存了三维点云和特征。
    # sfm_base_dir: "data/datasets/sfm_output"

    if isinstance(data_dirs, str):
        # 从哪一个开始选择
        print(
            f"Process all objects in directory:{data_dirs} sequences"
        )

        # data/datasets/lowtexture_test_data
        object_names = os.listdir(data_dirs)
        data_dirs_list = []
        want_seq_id = 1

        if cfg.ids is not None:
            # Use data ids
            # 生成一个键值对列表，键值对索引是前四个字符，对应数据集的数字标号，name是每个模型文件夹的全名
            id2full_name = {name[:4]: name for name in object_names if "-" in name}
            # 将object_names 更新为新的列表中，排除那些没用的文件夹名字
            object_names = [id2full_name[id] for id in cfg.ids if id in id2full_name]


        # 从这里开始处理数据
        for object_name in object_names:
            if "-" not in object_name:
                continue
            # 前两个if都是用来判断是是否为有效数据集，和排除数据集

            # 查找这个数据集(储存照片的数据集)路径下的所有目录，因为这个数据集每个物体都在4个场景下做了相应的数据集
            sequence_names = sorted(os.listdir(osp.join(data_dirs, object_name)))

            # 分割两次
            obj_short_name = object_name.split('-', 2)[1]

            # 分割字符串
            if want_seq_id is not None:
                sequence_names = ['-'.join([obj_short_name, str(want_seq_id)])]

            print(sequence_names)
            # " ".join([...])：将列表中的元素用空格分隔符拼接成一个字符串。data_dirs_list = ['data/object1 seq1 seq2']
            # # data_dirs_list：
            # 0:'data/datasets/test_data/0408-colorbox-box colorbox-4'
            # 1:'data/datasets/test_data/0409-aptamil-box aptamil-3'
            data_dirs_list.append(
                " ".join([osp.join(data_dirs, object_name)] + sequence_names)
            )
    else:
        raise NotImplementedError

    data_dirs = data_dirs_list  # [obj_name]

    name2metrics = inference_worker(data_dirs, cfg)

    
    # Parse metrics:
    gathered_metrics = {}
    for name, metrics in name2metrics.items():
        for metric_name, metric in metrics.items():
            if metric_name not in gathered_metrics:
                gathered_metrics[metric_name] = [metric]
            else:
                gathered_metrics[metric_name].append(metric)
        
    # Dump metrics:
    os.makedirs(cfg.output.txt_dir, exist_ok=True)
    with open(osp.join(cfg.output.txt_dir, 'metrics.txt'), 'w') as f:
        for name, metrics in name2metrics.items():
            f.write(f'{name}: \n')
            for metric_name, metric in metrics.items():
                f.write(f"{metric_name}: {metric}  ")
            f.write('\n ---------------- \n')
    
    with open(osp.join(cfg.output.txt_dir, 'metrics.txt'), 'a') as f:
        for metric_name, metric in gathered_metrics.items():
            print(f'{metric_name}:')
            metric_np = np.array(metric)
            metric_mean = np.mean(metric)
            print(metric_mean)
            print('---------------------')

            f.write('Summary: \n')
            f.writelines(str(metric_mean))


@hydra.main(config_path="configs/", config_name="config.yaml")
def main(cfg: DictConfig):
    globals()[cfg.type](cfg)
    # type: inference

if __name__ == "__main__":
    main()

# python inference.py +experiment=inference_onepose.yaml verbose=True
from itertools import chain
# import ray
import os
import math
import numpy as np
# from loguru import logger
import torch

from src.datasets.OnePosePlus_inference_dataset import OnePosePlusInferenceDataset
# from src.utils.ray_utils import ProgressBar, chunks, chunk_index, split_dict
from src.models.OnePosePlus.OnePosePlusModel import OnePosePlus_model
from src.utils.metric_utils import aggregate_metrics

from .inference_OnePosePlus_worker import (
    inference_onepose_plus_worker
)

# args = {
#     "ray": {
#         "slurm": False,
#         "n_workers": 2,
#         "n_cpus_per_worker": 1,
#         "n_gpus_per_worker": 0.5,
#         "local_mode": False,
#     },
# }

def build_model(model_configs, ckpt_path):
    match_model = OnePosePlus_model(model_configs)
    # load checkpoints
    print(f"Load ckpt:{ckpt_path}")
    state_dict = torch.load(ckpt_path, map_location="cpu")["state_dict"]
    for k in list(state_dict.keys()):
        state_dict[k.replace("matcher.", "")] = state_dict.pop(k)

    match_model.load_state_dict(state_dict, strict=True)
    match_model.eval()
    return match_model

# 这个函数是用来加载模型进行匹配的
def inference_onepose_plus(
    sfm_results_dir, all_image_paths, cfg,gaussion_model, use_ray=True, verbose=True
):
    """
    Inference for one object
    """
    # Build dataset:
    dataset = OnePosePlusInferenceDataset(
        sfm_results_dir,
        all_image_paths,
        load_3d_coarse=cfg.datamodule.load_3d_coarse,
        shape3d=cfg.datamodule.shape3d_val,
        img_pad=cfg.datamodule.img_pad,
        img_resize=cfg.datamodule.img_resize,
        df=cfg.datamodule.df,
        pad=cfg.datamodule.pad3D,
        load_pose_gt=True,
        n_images=None
    )
    # 这个是加载onepose++的模型
    match_model = build_model(cfg['model']["OnePosePlus"], cfg['model']['pretrained_ckpt'])

    # Run matching
    all_ids = np.arange(0, len(dataset))


    results = inference_onepose_plus_worker(dataset=dataset, match_model = match_model, subset_ids =all_ids,
                                             gaussian_model=gaussion_model,cfgs =cfg['model'], verbose=verbose)
    # logger.info("Match and compute pose error finish!")
    
    # Parse results:
    R_errs = []
    t_errs = []
    if 'ADD_metric' in results[0]:
        add_metric = []
        proj2d_metric = []
    else:
        add_metric = None
        proj2d_metric = None
    
    # Gather results metrics:
    for result in results:
        R_errs.append(result['R_errs'])
        t_errs.append(result['t_errs'])
        if add_metric is not None:
            add_metric.append(result['ADD_metric'])
            proj2d_metric.append(result['proj2D_metric'])
    
    # Aggregate metrics: 
    pose_errs = {'R_errs': R_errs, "t_errs": t_errs}
    if add_metric is not None:
        pose_errs.update({'ADD_metric': add_metric, "proj2D_metric": proj2d_metric})
    metrics = aggregate_metrics(pose_errs, cfg['model']['eval_metrics']['pose_thresholds'])

    return metrics
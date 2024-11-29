import os,sys
from pathlib import Path
# 这个是需要将python的路径给到根目录，要不然找不到路径
dir_path = Path(os.path.dirname(os.path.realpath(__file__))).parents[0]
print(f"dir_path {dir_path}")
sys.path.append(dir_path.__str__())
from argparse import ArgumentParser
from arguments import ModelParams, PipelineParams, OptimizationParams
from scene import GaussianModel 

import torch
import torch.nn as nn
import numpy as np
import cv2

from tqdm import tqdm

parser = ArgumentParser()
gaussian_ModelP = ModelParams(parser)
# gaussian_PipeP = PipelineParams(parser)

gs_pwd = "/media/wangxiao/Newsmy/dataset_LINEMOD/datasets/train_data_pgsr/0575-saltbottle-bottle/test/point_cloud/iteration_30000/point_cloud.ply"
npz_patrh = "/media/wangxiao/Newsmy/dataset_LINEMOD/datasets/sfm_output/outputs_softmax_loftr_loftr/0575-saltbottle-bottle/anno/anno_3d_average_coarse.npz"
output_path = "/media/wangxiao/Newsmy/dataset_LINEMOD/datasets/sfm_output/outputs_softmax_loftr_loftr/0575-saltbottle-bottle/anno/anno_3dgs_coarse.npz"
# anno_3d_average_coarse

# 单位为m
max_distance = 0.001

# mode = 'avg'
mode = 'max'


def main():
    print("GS2npz test")
    obj_gaussians = GaussianModel(gaussian_ModelP.sh_degree)
    obj_gaussians.load_ply(gs_pwd)

    # gs模型的xyz坐标
    gs_xyzs = obj_gaussians.get_xyz.data
    # gs_cov = obj_gaussians.get_scaling

    onepose_npz = np.load(npz_patrh)
    print(f"数组名称{onepose_npz.files}")
    # 数组名称['keypoints3d', 'descriptors3d', 'scores3d']
    onepose_xyzs = onepose_npz['keypoints3d'] #(n,3)
    descriptors3d =torch.tensor( onepose_npz['descriptors3d'], device='cuda').float() #(128,n)
    # 转置一下，因为原来的数据是
    descriptors3d.transpose_(0,1)
    scores3d = torch.tensor( onepose_npz['scores3d'], device='cuda').float() #(n,1)
    
    # 每个gs的xyz点

    # 将 one_pose_xyzs 转换为 torch 张量并移动到 cuda 设备
    onepose_xyzs_tensor = torch.tensor(onepose_xyzs, device='cuda').float()

    # keypoint3d_new = torch.empty(size=(0,0),device='cuda')
    # descriptors3d_new = torch.empty(size=(0,0),device='cuda')
    # scores3d_new = torch.empty(size=(0,0),device='cuda')
    keypoint3d_new_list = [] 
    descriptors3d_new_list = []
    scores3d_new_list = []

    # for gs_id in range(gs_xyzs.size(0)):
    for gs_id in tqdm(range(gs_xyzs.size(0)), desc="Processing keypoints"):
        gs_xyz = gs_xyzs[gs_id]

        # 计算所有 one_pose_xyzs 与 gs_xyz 之间的距离
        distances = torch.norm(onepose_xyzs_tensor - gs_xyz, dim=1)

        # 获取排序后的索引
        sorted_distances,sorted_indices = torch.sort(distances)
        
        maxid=-1
        for id in range(sorted_indices.size(0)):
            if sorted_distances[id]>max_distance:
                maxid = id-1
                break
        
        if maxid == -1:
            continue
        else:
            if mode == 'avg':
                weight_torch = 1.0 - sorted_distances[:maxid+1] / torch.sum(sorted_distances[:maxid+1])
                weight_torch = weight_torch / torch.sum(weight_torch,dim=0)
                weight_torch = weight_torch.unsqueeze(0)
                weight_torch.transpose_(0,1)
                this_descriptors3d = torch.sum(weight_torch * descriptors3d[sorted_indices[:maxid+1],:] ,dim=0)
                this_scores3d = torch.sum(scores3d[sorted_indices[:maxid+1],:],dim = 0)
                # 新的关键点

            elif mode == 'max':
                this_descriptors3d = descriptors3d[sorted_indices[0],:]
                this_scores3d = scores3d[sorted_indices[0],:]
            
            keypoint3d_new_list.append(gs_xyz)
            descriptors3d_new_list.append(this_descriptors3d)
            scores3d_new_list.append(this_scores3d)


            # keypoint3d_new = torch.cat((keypoint3d_new,gs_xyz),dim=0)
            # descriptors3d_new = torch.cat((descriptors3d_new,this_descriptors3d),dim=0)
            # scores3d_new = torch.cat((scores3d_new,this_scores3d),dim=0)
    
    # 将列表转换为张量 出循环了
    keypoint3d_new = torch.stack(keypoint3d_new_list, dim=0)
    descriptors3d_new = torch.stack(descriptors3d_new_list, dim=0)
    scores3d_new = torch.stack(scores3d_new_list, dim=0)
    # 再转置回来
    descriptors3d_new.transpose_(0,1)

     # 将张量转换为 numpy 数组
    keypoint3d_new_np = keypoint3d_new.cpu().numpy()
    descriptors3d_new_np = descriptors3d_new.cpu().numpy()
    scores3d_new_np = scores3d_new.cpu().numpy()

    # 保存到 .npz 文件
    np.savez_compressed(output_path, keypoints3d=keypoint3d_new_np, descriptors3d=descriptors3d_new_np, scores3d=scores3d_new_np)           




if __name__=="__main__":
    main()
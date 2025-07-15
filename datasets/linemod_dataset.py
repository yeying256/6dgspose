import yaml
import argparse
import logging
from pathlib import Path
import os,sys
import numpy as np
import json

# 这个是需要将python的路径给到根目录，要不然找不到路径
dir_path = Path(os.path.dirname(os.path.realpath(__file__))).parents[0]
print(f"dir_path {dir_path}")
sys.path.append(dir_path.__str__())

from configs import inference_cfg as CFG
from misc_utils import gs_utils
import torch
import torch.nn.functional as torch_F
from torchvision.ops import roi_align
from pytorch3d import ops as py3d_ops
from PIL import Image

import cv2
import time


# 调用基类
from datasets.dataset_base import dataset_base

from types import SimpleNamespace



def extract_model_info(json_file):
    """
    从 models_info.json 文件中提取模型信息。

    参数:
        json_file (str): 包含模型信息的 JSON 文件路径。

    返回:
        tuple: 包含以下三个字典的元组：
            - diameter (dict): 模型直径，格式为 {'标号': float}
            - box (dict): 模型边界框，格式为 {'标号': np.array(8, 3)}
            - is_sym (dict): 模型是否对称，格式为 {'标号': bool}
    """
    # 初始化输出变量
    diameter = {}
    box = {}
    is_sym = {}

    # 加载 JSON 数据
    with open(json_file, 'r') as f:
        models_info = json.load(f)

    # 遍历每个模型的信息
    for model_id, info in models_info.items():
        # 提取直径
        diameter[model_id] = float(info['diameter'])/1000.0

        # 计算 bounding box 的 8 个顶点坐标
        min_x, max_x = info['min_x'], info['min_x'] + info['size_x']
        min_y, max_y = info['min_y'], info['min_y'] + info['size_y']
        min_z, max_z = info['min_z'], info['min_z'] + info['size_z']
        vertices = np.array([
            [min_x, min_y, min_z],  # 左下后
            [max_x, min_y, min_z],  # 右下后
            [max_x, max_y, min_z],  # 右上后
            [min_x, max_y, min_z],  # 左上后
            [min_x, min_y, max_z],  # 左下前
            [max_x, min_y, max_z],  # 右下前
            [max_x, max_y, max_z],  # 右上前
            [min_x, max_y, max_z]   # 左上前
        ])
        box[model_id] = vertices/1000.0

        # 判断是否有对称性
        if 'symmetries_continuous' in info or 'symmetries_discrete' in info:
            is_sym[model_id] = True
        else:
            is_sym[model_id] = False

    return diameter, box, is_sym


# 定义函数：从 JSON 文件中提取内参矩阵
def extract_cam_K_from_file(json_file):
    """
    从 scene_camera.json 文件中提取每个标号的内参矩阵 cam_K。

    参数:
        json_file (str): JSON 文件路径。

    返回:
        dict: 包含每个标号的内参矩阵 K 的字典。
    """
    # 初始化输出字典
    K_matrices = {}

    # 加载 JSON 数据
    with open(json_file, 'r') as f:
        data = json.load(f)

    # 遍历数据，提取 cam_K 并转换为 3x3 矩阵
    for key, value in data.items():
        cam_K = value.get("cam_K", None)  # 获取 cam_K 字段
        if cam_K and len(cam_K) == 9:  # 确保 cam_K 存在且长度为 9
            K_matrices[key] = np.array(cam_K).reshape(3, 3)  # 转换为 3x3 矩阵

    return K_matrices

# 定义函数：从 JSON 文件中提取齐次变换矩阵
def extract_homogeneous_matrices_from_file(json_file):
    """
    从 scene_gt.json 文件中提取每个标号的齐次变换矩阵。

    参数:
        json_file (str): JSON 文件路径。

    返回:
        dict: 包含每个标号的齐次变换矩阵的字典。
    """
    # 初始化输出字典
    homogeneous_matrices = {}

    # 加载 JSON 数据
    with open(json_file, 'r') as f:
        data = json.load(f)

    # 遍历数据，提取 cam_R_m2c 和 cam_t_m2c 并拼接为齐次变换矩阵
    for key, value in data.items():
        if isinstance(value, list) and len(value) > 0:  # 确保值是一个非空列表
            entry = value[0]  # 假设每个标号只有一个对象
            cam_R_m2c = entry.get("cam_R_m2c", None)  # 获取旋转矩阵
            cam_t_m2c = entry.get("cam_t_m2c", None)  # 获取平移向量

            # 检查数据完整性
            if cam_R_m2c and len(cam_R_m2c) == 9 and cam_t_m2c and len(cam_t_m2c) == 3:
                # 转换为 NumPy 数组
                R = np.array(cam_R_m2c).reshape(3, 3)  # 旋转矩阵
                t = np.array(cam_t_m2c).reshape(3, 1)/1000.0  # 平移向量

                # 构造齐次变换矩阵
                T = np.hstack((R, t))  # 水平拼接 R 和 t
                T = np.vstack((T, np.array([0, 0, 0, 1])))  # 添加最后一行 [0, 0, 0, 1]

                # 存储到字典中
                homogeneous_matrices[key] = T

    return homogeneous_matrices


def search_mask_dir(path):
    """
    根据路径下的文件生成索引字典。

    参数:
        path (str): 文件所在的目录路径。

    返回:
        dict: 以文件名中提取的数字为键，文件完整路径为值的字典。
    """
    index = {}
    # 确保路径存在且是一个目录
    if not os.path.isdir(path):
        raise ValueError(f"指定的路径 {path} 不是一个有效的目录")

    # 遍历目录中的所有文件
    for filename in os.listdir(path):
        # 检查文件名是否符合模式 "XXXXXX_YYYYYY.png"
        if filename.endswith(".png") and "_" in filename:
            # 提取数字部分（第一个下划线前的部分）
            key = str(int(filename.split("_")[0]))
            # 构造文件的完整路径
            full_path = os.path.join(path, filename)
            # 添加到索引字典中
            index[key] = full_path

    return index

class linemod_dataset(dataset_base):
    def __init__(self,config) -> None:
        super().__init__()
        # 3dgs文件夹
        self.cwd_3dgs = config['data']['gsdir']
        self.gs_subdir = config['data']['gs_subdir']

        # sfm路径
        self.colmap_dir = config['data']['colmap_path']

        # 图像路径
        self.dataset_dir = config['data']['dataset_dir']

        # 这里是取需要的子文件夹
        subdir = config['data']['subdir']
        # rgb图像 文件夹
        self.color_forder_name = config['data']['color']
        # 内参文件夹
        intrinsics_forder_name = config['data']['intrin']
        # 位姿文件夹
        self.pose_forder_name = config['data']['pose']
        # 3dbox文件夹
        self.box_dir = config['data']['box_dir']

        self.metric_out_dir = config['metric_out_dir']

        self.debug = config['debug']

        # 获取每个对象的直径，边界框，是否是对称的
        self.diameter,self.box,self.is_sym = extract_model_info(self.box_dir)

        self.maskforder_dir = config['data']['maskforder_dir']

        self.no_obj_lists = config['data']['no_target_name_lists']


        ref_dir = config['data']['data_ref_dir']

        self.camera_distance_scale = config['camera_distance_scale']
        # 查找所有的子文件夹
        objs = os.listdir(self.dataset_dir)

        try:
            self.ref_output_dir = config['ref_output_dir']
        except:
            self.ref_output_dir = None

        try:
            self.pose_out_dir = config['pose_out_dir']
        except:
            self.pose_out_dir = None

        # refin参数
        self.refine_CFG = SimpleNamespace(
                # 初始学习率
                START_LR=config['refine']['START_LR'],
                # 最大步数
                MAX_STEPS=config['refine']['MAX_STEPS'],
                END_LR=config['refine']['END_LR'],
                WARMUP=config['refine']['WARMUP'],
                USE_SSIM=config['refine']['USE_SSIM'],
                USE_MS_SSIM=config['refine']['USE_MS_SSIM'],
                EARLY_STOP_MIN_STEPS=config['refine']['EARLY_STOP_MIN_STEPS'],
                EARLY_STOP_LOSS_GRAD_NORM=config['refine']['EARLY_STOP_LOSS_GRAD_NORM']
                )
        
        self.gspose_model_path = config['gspose_model_path']

        self.log_dir = config['log_dir']

        # 遍历所有的目标，比如说瓶子等
        for object_name in objs:
            # 这里的文件夹前面带着好多0

            # 如果在排除的目标列表中，则跳过
            if object_name in self.no_obj_lists:
                continue
            else:
                self.sub_name_lists.append(object_name)

            
            if object_name.isdigit():  # 确保文件名是纯数字
                number = str(int(object_name))  # 转换为整数以去掉前导零
            else:
                continue

            id = number
            # 这里已经是去掉前面的0了
            self.ID_lists.append(id)

            # 到了每一个数据集的文件夹，在里面查找所有的object文件夹
            temp_dir = os.path.join(self.dataset_dir,object_name)

            # 查找所有的图片
            color_forder_dir = os.path.join(temp_dir,self.color_forder_name)
            imags= os.listdir(color_forder_dir)
            # 查找这个目标的所有的内参 
            cam_intr_dir = os.path.join(temp_dir,intrinsics_forder_name)
            camera_K = extract_cam_K_from_file(cam_intr_dir)
            # 查找这个目标的所有的外参
            pose_dir = os.path.join(temp_dir,self.pose_forder_name)
            homogeneous_matrices = extract_homogeneous_matrices_from_file(pose_dir)

            ref_pose_id = extract_homogeneous_matrices_from_file(f"{ref_dir}/{object_name}/{self.pose_forder_name}")
            
            for id_ in ref_pose_id:
                self.ref_T.setdefault(id,[]).append(ref_pose_id[id_])
                pass
            
            camera_K_ref = extract_cam_K_from_file(f"{ref_dir}/{object_name}/{intrinsics_forder_name}")
            for id_ in camera_K_ref:
                # self.ref_T.setdefault(id,[]).append(ref_pose_id[id_])
                self.ref_K.setdefault(id,[]).append(camera_K_ref[id_])
                pass

            # = extract_homogeneous_matrices_from_file(f"{ref_dir}/{object_name}/{self.pose_forder_name}")
            # 查找mask
            masks_dirs = search_mask_dir(f'{temp_dir}/{self.maskforder_dir}')

            for imag in imags:
                # 遍历所有的图片
                if not os.path.splitext(imag)[1] == '.png' and not os.path.splitext(imag)[1] == '.jpg':
                    continue
                img_id = str(int(os.path.splitext(imag)[0]))

                self.color_dir_lists.setdefault(id,[]).append(os.path.join(color_forder_dir,imag))
                self.cam_K_lists.setdefault(id,[]).append(camera_K[img_id])
                self.pose_lists.setdefault(id,[]).append(homogeneous_matrices[img_id])
                self.mask_dir_lists.setdefault(id,[]).append(masks_dirs[img_id])

                allo_pose = homogeneous_matrices[img_id].copy()
                allo_pose[:3,:3] = gs_utils.egocentric_to_allocentric(homogeneous_matrices[img_id])[:3,:3]

                self.allo_pose_lists.setdefault(id,[]).append(allo_pose)

                # txt_name = os.path.splitext(imag)[0] + '.txt'
                # self.cam_intr_dir_lists.setdefault(id,[]).append(os.path.join(cam_intr_dir,txt_name))
                # self.pose_dir_lists.setdefault(id,[]).append(os.path.join(pose_dir,txt_name))


            pass
            temp = os.path.join(self.cwd_3dgs,object_name,self.gs_subdir)
            self.gs_model_dir_lists.setdefault(id,temp)
            



        pass
    pass


def create_reference_database_from_RGB_images(model_func, 
                                            obj_dataset:linemod_dataset,
                                            device="cuda",
                                            save_pred_mask=False,
                                            id = ""):    
    '''
    model_func:预训练的gspose模型
    obj_dataset:数据集
    device:cuda是否用显卡
    id:目标id 如 '1','2'…… 去掉前面0的字符串类型
    '''
    # if CFG.USE_ALLOCENTRIC:
    #     # true
    #     # 大概就是以目标为坐标系中心
    #     obj_poses = np.stack(obj_dataset.allo_poses, axis=0)
    # else:
    #     # 这就是大概以相机位姿为中心
    #     obj_poses = np.stack(obj_dataset.poses, axis=0)
                
    obj_poses = obj_dataset.allo_pose_lists[id]

    obj_poses = torch.as_tensor(obj_poses, dtype=torch.float32).to(device)
    obj_matRs = obj_poses[:, :3, :3]
    obj_vecRs = obj_matRs[:, 2, :3]
    # 选出来一系列的图像 采样出来CFG.refer_view_num个点
    fps_inds = py3d_ops.sample_farthest_points(
        obj_vecRs[None, :, :], K=CFG.refer_view_num, random_start_point=False)[1].squeeze(0)  # obtain the FPS indices
    ref_fps_images = list()
    ref_fps_poses = list()
    ref_fps_camKs = list()
    for ref_idx in fps_inds:
        # 获取第ref_idx个图像的信息，是个序号

        view_idx = ref_idx.item()
        # datum = obj_dataset[view_idx]
        camK = torch.as_tensor(obj_dataset.cam_K_lists[id][ref_idx], dtype=torch.float32)
        # camK = datum['camK']      # 3x3
        # 出来的是rgb通道的，然后归一化处理了
        image_np = np.array(Image.open(obj_dataset.color_dir_lists[id][ref_idx]), dtype=np.uint8) / 255.0
        image = torch.as_tensor(image_np, dtype=torch.float32)   
        # image = 

        # image = datum['image']    # HxWx3
        # 如果 'allo_pose' 不存在，则返回默认值 datum['pose']。
        # pose = datum.get('allo_pose', datum['pose']) # 4x4
        pose = torch.as_tensor(obj_dataset.allo_pose_lists[id][ref_idx], dtype=torch.float32)

        ref_fps_images.append(image)
        ref_fps_poses.append(pose)
        ref_fps_camKs.append(camK)
    ref_fps_poses = torch.stack(ref_fps_poses, dim=0)
    ref_fps_camKs = torch.stack(ref_fps_camKs, dim=0)
    ref_fps_images = torch.stack(ref_fps_images, dim=0)
    # 把这些图像裁剪一下
    zoom_fps_images = gs_utils.zoom_in_and_crop_with_offset(image=ref_fps_images, # KxHxWx3 -> KxSxSx3
                                                                K=ref_fps_camKs, 
                                                                t=ref_fps_poses[:, :3, 3], 
                                                                radius=obj_dataset.diameter[id]/2,
                                                                target_size=CFG.zoom_image_scale, 
                                                                margin=CFG.zoom_image_margin)['zoom_image']
    with torch.no_grad():
        # 适配 PyTorch 输入：将 (B,H,W,C) 转为 (B,C,H,W)。
        if zoom_fps_images.shape[-1] == 3:
            zoom_fps_images = zoom_fps_images.permute(0, 3, 1, 2)
        # 使用DINOv2提取特征
        obj_fps_feats, _, obj_fps_dino_tokens = model_func.extract_DINOv2_feature(zoom_fps_images.to(device), return_last_dino_feat=True) # Kx768x16x16
        # obj_fps_feats：用于后续分割任务的特征。
        # obj_fps_dino_tokens：DINOv2 的特征向量。
        obj_fps_masks = model_func.refer_cosegmentation(obj_fps_feats).sigmoid() # Kx1xSxS

        obj_token_masks = torch_F.interpolate(obj_fps_masks,
                                             scale_factor=1.0/model_func.dino_patch_size, 
                                             mode='bilinear', align_corners=True, recompute_scale_factor=True) # Kx1xS/14xS/14
        obj_fps_dino_tokens = obj_fps_dino_tokens.flatten(0, 1)[obj_token_masks.view(-1).round().type(torch.bool), :] # KxLxC -> KLxC -> MxC

    refer_allo_Rs = list()
    refer_pred_masks = list()
    refer_Remb_vectors = list()
    refer_coseg_mask_info = list()
    num_instances = len(obj_dataset.cam_K_lists[id])

    # 这里是每一张图片
    for idx in range(num_instances):
        # ref_data = obj_dataset[idx]

        # camK = ref_data['camK']
        camK = torch.as_tensor(obj_dataset.cam_K_lists[id][idx],dtype=torch.float32)

        # image = ref_data['image']
        image_np = np.array(Image.open(obj_dataset.color_dir_lists[id][idx]), dtype=np.uint8) / 255.0
        image = torch.as_tensor(image_np, dtype=torch.float32)   

        pose = torch.as_tensor(obj_dataset.allo_pose_lists[id][idx],dtype=torch.float32)


        # pose = ref_data.get('allo_pose', ref_data['pose']) # 4x4

        refer_allo_Rs.append(pose[:3, :3])
        ref_tz = (1 + CFG.zoom_image_margin) * camK[:2, :2].max() * obj_dataset.diameter[id] / CFG.zoom_image_scale
        zoom_outp = gs_utils.zoom_in_and_crop_with_offset(image=image, # HxWx3 -> SxSx3
                                                            K=camK, 
                                                            t=pose[:3, 3], 
                                                            radius=obj_dataset.diameter[id]/2,
                                                            target_size=CFG.zoom_image_scale, 
                                                            margin=CFG.zoom_image_margin) # SxSx3
        with torch.no_grad():
            zoom_image = zoom_outp['zoom_image'].unsqueeze(0)
            if zoom_image.shape[-1] == 3:
                zoom_image = zoom_image.permute(0, 3, 1, 2)
            zoom_feat = model_func.extract_DINOv2_feature(zoom_image.to(device))
            zoom_mask = model_func.query_cosegmentation(zoom_feat, 
                                                        x_ref=obj_fps_feats, 
                                                        ref_mask=obj_fps_masks).sigmoid()
            y_Remb = model_func.generate_rotation_aware_embedding(zoom_feat, zoom_mask)
            refer_Remb_vectors.append(y_Remb.squeeze(0)) # 64
            try:
                msk_yy, msk_xx = torch.nonzero(zoom_mask.detach().cpu().squeeze().round().type(torch.uint8), as_tuple=True)
                msk_cx = (msk_xx.max() + msk_xx.min()) / 2
                msk_cy = (msk_yy.max() + msk_yy.min()) / 2
            except: # no mask is found
                msk_cx = CFG.zoom_image_scale / 2
                msk_cy = CFG.zoom_image_scale / 2

            prob_mask_area = zoom_mask.detach().cpu().sum()
            bin_mask_area = zoom_mask.round().detach().cpu().sum()            
            refer_coseg_mask_info.append(torch.tensor([msk_cx, msk_cy, ref_tz, bin_mask_area, prob_mask_area]))

        if save_pred_mask:
            orig_mask = gs_utils.zoom_out_and_uncrop_image(zoom_mask.squeeze(), # SxS
                                                            bbox_center=zoom_outp['bbox_center'],
                                                            bbox_scale=zoom_outp['bbox_scale'],
                                                            orig_hei=image.shape[0],
                                                            orig_wid=image.shape[1],
                                                            )# 1xHxWx1
            # coseg_mask_path = ref_data['coseg_mask_path']
            # coseg_mask_path = f"{obj_dataset.ref_output_dir}/{id}" 

            coseg_mask_path = os.path.join(obj_dataset.ref_output_dir,
                                           id,
                                           'pred_coseg_mask', 
                                           '{:06d}.png'.format(idx))


            orig_mask = (orig_mask.detach().cpu().squeeze() * 255).numpy().astype(np.uint8) # HxW
            if not os.path.exists(os.path.dirname(coseg_mask_path)):
                os.makedirs(os.path.dirname(coseg_mask_path))
            cv2.imwrite(coseg_mask_path, orig_mask)
            # 储存一下剪切过的图片看一下
            if zoom_image.shape[1] == 3:  # 检查是否为 [1, 3, H, W] 格式
                img_np = zoom_image.squeeze(0).permute(1, 2, 0).cpu().numpy()  # [H, W, 3]
            else:
                img_np = zoom_image.squeeze(0).cpu().numpy()  # 单通道 [H, W]

            # 归一化到 [0, 255] 并转为uint8
            img_np = (img_np * 255).clip(0, 255).astype(np.uint8)

            # 保存为文件（确保路径存在）
            coseg_rgb_path = os.path.join(obj_dataset.ref_output_dir,
                                id,
                                'pred_coseg_mask', 
                                '{:06d}rgb_.png'.format(idx))

            cv2.imwrite(coseg_rgb_path, cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR))  # OpenCV需要BGR格式



        else:
            refer_pred_masks.append(zoom_mask.detach().cpu().squeeze()) # SxS

        if (idx + 1) % 100 == 0:
            time_stamp = time.strftime('%d-%H:%M:%S', time.localtime())
            print('[{}/{}], {}'.format(idx+1, num_instances, time_stamp))

    refer_allo_Rs = torch.stack(refer_allo_Rs, dim=0).squeeze() # Nx3x3
    refer_Remb_vectors = torch.stack(refer_Remb_vectors, dim=0).squeeze()      # Nx64
    refer_coseg_mask_info = torch.stack(refer_coseg_mask_info, dim=0).squeeze() # Nx3
    
    ref_database = dict()
    if not save_pred_mask:
        refer_pred_masks = torch.stack(refer_pred_masks, dim=0).squeeze() # NxSxS
        ref_database['refer_pred_masks'] = refer_pred_masks

    ref_database['obj_fps_inds'] = fps_inds
    # ref_database['obj_fps_images'] = zoom_fps_images

    ref_database['obj_fps_feats'] = obj_fps_feats
    ref_database['obj_fps_masks'] = obj_fps_masks
    ref_database['obj_fps_dino_tokens'] = obj_fps_dino_tokens 

    ref_database['refer_allo_Rs'] = refer_allo_Rs
    ref_database['refer_Remb_vectors'] = refer_Remb_vectors
    ref_database['refer_coseg_mask_info'] = refer_coseg_mask_info
    
    return ref_database
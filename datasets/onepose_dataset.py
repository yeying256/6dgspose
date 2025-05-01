import yaml
import argparse
import logging
from pathlib import Path
import os,sys

# 这个是需要将python的路径给到根目录，要不然找不到路径
dir_path = Path(os.path.dirname(os.path.realpath(__file__))).parents[0]
print(f"dir_path {dir_path}")
sys.path.append(dir_path.__str__())

# 调用基类
from datasets.dataset_base import dataset_base
from types import SimpleNamespace


class onepose_dataset(dataset_base):
    def __init__(self,config) -> None:
        super().__init__()
        # 3dgs文件夹
        self.cwd_3dgs = config['data']['gsdir']
        self.gs_subdir = config['data']['gs_subdir']

        # 图像路径
        self.dataset_dir = config['data']['dataset_dir']

        # 这里是取需要的子文件夹
        subdir = config['data']['subdir']
        # rgb图像 文件夹
        color_forder_name = config['data']['color']
        # 内参文件夹
        intrinsics_forder_name = config['data']['intrin']
        # 位姿文件夹
        self.pose_forder_name = config['data']['pose']
        # 3dbox文件夹
        self.box_dir = config['data']['box_dir']

        self.metric_out_dir = config['metric_out_dir']

        self.debug = config['debug']

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
        
        try:
            self.debug_target = config['data']['debug_target']
        except:
            self.debug_target = None
        

        self.camera_gap = config['camera_gap']

        # 查找所有的子文件夹
        self.object_lists = os.listdir(self.dataset_dir)
        # 遍历所有的目标，比如说瓶子等
        for object_name in self.object_lists:
            self.sub_name_lists.append(object_name)
            # 3dbox文件夹
            self.box_dir_lists[object_name] = os.path.join(self.dataset_dir,object_name,self.box_dir)
            # 到了每一个数据集的文件夹，在里面查找所有的子文件夹
            temp_dir = os.path.join(self.dataset_dir,object_name)

            sub_lists = os.listdir(temp_dir)
            # 选出带有横杠-的文件夹名字
            subdirs_with_minus = [subdir for subdir in sub_lists if '-' in subdir]
            if subdir == 'all':
                for sub_name in subdirs_with_minus:
                    # 遍历所有的子文件夹 -1,-2,-3……
                    temp_subdir = os.path.join(temp_dir,sub_name)
                    # 查找所有的图片
                    color_forder_dir = os.path.join(temp_subdir,color_forder_name)
                    imags= os.listdir(color_forder_dir)
                    # 查找所有的内参
                    cam_intr_dir = os.path.join(temp_subdir,intrinsics_forder_name)

                    pose_dir = os.path.join(temp_subdir,self.pose_forder_name)
                    

                    for imag in imags:
                        # 遍历所有的图片
                        self.color_dir_lists[object_name].append(os.path.join(color_forder_dir,imag))
                        txt_name = os.path.splitext(imag)[0] + '.txt'
                        self.cam_intr_dir_lists[object_name].append(os.path.join(cam_intr_dir,txt_name))
                        self.pose_dir_lists[object_name].append(os.path.join(pose_dir,txt_name))

                pass
            else:
                subdir
                for sub_name in subdirs_with_minus:
                    if subdir not in sub_name:
                        continue
                    # 遍历所有的子文件夹 -1,-2,-3……
                    temp_subdir = os.path.join(temp_dir,sub_name)
                    # 查找所有的图片
                    color_forder_dir = os.path.join(temp_subdir,color_forder_name)
                    imags= os.listdir(color_forder_dir)
                    # 查找所有的内参
                    cam_intr_dir = os.path.join(temp_subdir,intrinsics_forder_name)

                    pose_dir = os.path.join(temp_subdir,self.pose_forder_name)
                    

                    for imag in imags:
                        # 遍历所有的图片
                        if not os.path.splitext(imag)[1] == '.png' and not os.path.splitext(imag)[1] == '.jpg':
                            continue
                        self.color_dir_lists.setdefault(object_name,[]).append(os.path.join(color_forder_dir,imag))
                        txt_name = os.path.splitext(imag)[0] + '.txt'
                        self.cam_intr_dir_lists.setdefault(object_name,[]).append(os.path.join(cam_intr_dir,txt_name))
                        self.pose_dir_lists.setdefault(object_name,[]).append(os.path.join(pose_dir,txt_name))

                pass
            pass
            temp = os.path.join(self.cwd_3dgs,object_name,self.gs_subdir)
            self.gs_model_dir_lists.setdefault(object_name,temp)
            



        pass
    pass
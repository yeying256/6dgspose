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

# 调用基类
from datasets.dataset_base import dataset_base



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
        diameter[model_id] = float(info['diameter'])

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
        box[model_id] = vertices

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

        self.diameter,self.box,self.is_sym = extract_model_info(self.box_dir)

        self.maskforder_dir = config['data']['maskforder_dir']


        
        # 查找所有的子文件夹
        self.sub_name_lists = os.listdir(self.dataset_dir)
        # 遍历所有的目标，比如说瓶子等
        for object_name in self.sub_name_lists:
            # 这里的文件夹前面带着好多0

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

                # txt_name = os.path.splitext(imag)[0] + '.txt'
                # self.cam_intr_dir_lists.setdefault(id,[]).append(os.path.join(cam_intr_dir,txt_name))
                # self.pose_dir_lists.setdefault(id,[]).append(os.path.join(pose_dir,txt_name))


            pass
            temp = os.path.join(self.cwd_3dgs,object_name,self.gs_subdir)
            self.gs_model_dir_lists.setdefault(object_name,temp)
            



        pass
    pass
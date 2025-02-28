import numpy as np
import json
from argparse import ArgumentParser
import os
import cv2
from PIL import Image, ImageFile
from glob import glob
import math
import sys
from pathlib import Path

dir_path = Path(os.path.dirname(os.path.realpath(__file__))).parents[0]
print(f"dir_path {dir_path}")
sys.path.append(dir_path.__str__())

import argparse
import yaml
import logging

from datasets.datasets import datasets


import open3d as o3d






# dir_path = Path(os.path.dirname(os.path.realpath(__file__))).parents[2]
# sys.path.append(dir_path.__str__())

from preprocess.database import COLMAPDatabase  # NOQA
from preprocess.read_write_model import read_model, rotmat2qvec  # NOQA

ImageFile.LOAD_TRUNCATED_IMAGES = True


def create_folder(folder_path):
    """
    创建一个文件夹，如果文件夹已经存在则不执行任何操作。

    :param folder_path: 要创建的文件夹路径
    """
    try:
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
            print(f"文件夹 '{folder_path}' 已创建。")
        else:
            print(f"文件夹 '{folder_path}' 已经存在。")
    except Exception as e:
        print(f"创建文件夹时发生错误: {e}")

def read_txt(path):
    """
    读取 poseba 文件并将其内容转换为一个 4x4 的矩阵。
    参数:
    - path: poseba 文件的路径
    返回:
    - matrix: 4x4 的 numpy 数组
    """
    # 读取文件内容
    with open(path, 'r') as file:
        lines = file.readlines()
    # 解析文件内容为矩阵
    matrix = []
    for line in lines:
        # 去除每行的空白字符并分割为浮点数列表
        row = [float(num) for num in line.strip().split()]
        matrix.append(row)
    # 将列表转换为 numpy 数组
    matrix = np.array(matrix)
    return matrix


def create_init_files(pinhole_dict_file, db_file, out_dir,color_dir):
    # Partially adapted from https://github.com/Kai-46/nerfplusplus/blob/master/colmap_runner/run_colmap_posed.py

    if not os.path.exists(out_dir):
        os.mkdir(out_dir)

    # create template
    with open(pinhole_dict_file) as fp:
        pinhole_dict = json.load(fp)

    template = {}
    cameras_line_template = '{camera_id} PINHOLE {width} {height} {fx} {fy} {cx} {cy}\n'
    images_line_template = '{image_id} {qw} {qx} {qy} {qz} {tx} {ty} {tz} {camera_id} {image_name}\n\n'
    
    for img_name in pinhole_dict:
        # w, h, fx, fy, cx, cy, qvec, t
        params = pinhole_dict[img_name]
        w = params[0]
        h = params[1]
        fx = params[2]
        fy = params[3]
        # fx = str(0.6 * float(w))
        # fy = str(0.6 * float(w))
        # cx = str(float(w) / 2.0)
        # cy = str(float(h) / 2.0)
        cx = params[4]
        cy = params[5]
        qvec = params[6:10]
        tvec = params[10:13]

        imag_path = os.path.join(color_dir, img_name)

        cam_line = cameras_line_template.format(
            camera_id="{camera_id}", width=w, height=h, fx=fx, fy=fy, cx=cx, cy=cy)
        img_line = images_line_template.format(image_id="{image_id}", qw=qvec[0], qx=qvec[1], qy=qvec[2], qz=qvec[3],
                                               tx=tvec[0], ty=tvec[1], tz=tvec[2], camera_id="{camera_id}",
                                               image_name=img_name)
        template[img_name] = (cam_line, img_line)

    # read database
    db = COLMAPDatabase.connect(db_file)
    table_images = db.execute("SELECT * FROM images")
    img_name2id_dict = {}
    for row in table_images:
        img_name2id_dict[row[1]] = row[0]

    # cameras_txt_lines = [template[img_name][0].format(camera_id=1)]
    cameras_txt_lines = []
    images_txt_lines = []
    for img_name, img_id in img_name2id_dict.items():
        image_line = template[img_name][1].format(image_id=img_id, camera_id=img_id)

        cameras_line = template[img_name][0].format(camera_id=img_id)
        cameras_txt_lines.append(cameras_line)
        images_txt_lines.append(image_line)

    with open(os.path.join(out_dir, 'cameras.txt'), 'w') as fp:
        fp.writelines(cameras_txt_lines)
        fp.write('\n')

    with open(os.path.join(out_dir, 'images.txt'), 'w') as fp:
        fp.writelines(images_txt_lines)
        fp.write('\n')

    # create an empty points3D.txt
    fp = open(os.path.join(out_dir, 'points3D.txt'), 'w')
    fp.close()


def compare_filenames(file1, file2, file3):
    """
    判断三个文件的文件名（除了扩展名）是否相同。

    参数:
    - file1: 第一个文件的路径
    - file2: 第二个文件的路径
    - file3: 第三个文件的路径

    返回:
    - bool: 如果三个文件的文件名（除了扩展名）相同，返回 True；否则返回 False
    """
    # 提取文件名（不包括扩展名）
    base_name1 = os.path.splitext(os.path.basename(file1))[0]
    base_name2 = os.path.splitext(os.path.basename(file2))[0]
    base_name3 = os.path.splitext(os.path.basename(file3))[0]

    # 比较文件名
    return base_name1 == base_name2 == base_name3

def create_camera_frame(T, scale=1.0):
    """
    创建一个表示相机位姿的坐标系框架。

    参数:
    - T: 4x4 的变换矩阵，表示相机的位姿
    - scale: 坐标轴的缩放因子

    返回:
    - frame: Open3D 的 TriangleMesh 对象
    """
    frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=scale)
    frame.transform(T)
    return frame

def visualize_camera_poses(poses, scale=1.0):
    """
    可视化多个相机位姿。

    参数:
    - poses: 一个列表，包含多个 4x4 的变换矩阵
    - scale: 坐标轴的缩放因子
    """
    frames = []
    for pose in poses:
        frame = create_camera_frame(pose, scale)
        frames.append(frame)
    identity_matrix = np.eye(4, dtype=int)
    frame = create_camera_frame(identity_matrix, 0.1)
    frames.append(frame)

    # 可视化相机位姿
    o3d.visualization.draw_geometries(frames)


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
    linemod_dataset = datasets(config)

    linemod_path = linemod_dataset.dataset.dataset_dir

    color_forder_name = linemod_dataset.dataset.color_forder_name


    # 储存了数据集所有的路径
    # scene_list = os.listdir()
    # 进入到每一个场景下
    colmap_dir = linemod_dataset.dataset.colmap_dir
    for scene in linemod_dataset.dataset.sub_name_lists:

        object_id = str(int(scene))

        db_file = os.path.join(f"{colmap_dir}/{scene}", 'database.db')
        sfm_dir = os.path.join(f"{colmap_dir}/{scene}", 'sparse')
        imags_dir = os.path.join(f"{colmap_dir}/{scene}", 'images')

        os.system(f"rm -rf {colmap_dir}/{scene}/images/*")
        os.system(f"rm -rf {colmap_dir}/{scene}/sparse/*")
        os.system(f"rm {colmap_dir}/{scene}/database.db")
        # 创建工作目录和一些文件夹
        create_folder(f"{colmap_dir}/{scene}/images")
        create_folder(f"{colmap_dir}/{scene}/sparse")

        color_dir = os.path.join(f"{linemod_path}/{scene}/{color_forder_name}",)
        
        # extract features
        os.system(f"colmap feature_extractor --database_path {db_file} \
                --image_path {color_dir} \
                --ImageReader.single_camera 0 \
                --ImageReader.camera_model=PINHOLE \
                --SiftExtraction.use_gpu=true \
                --SiftExtraction.num_threads=32"
                  )


        # 获取列表


        # 获取文件列表并排序
        # images_lis = sorted(glob(os.path.join(color_dir, '*.png')))


        # images_lis = glob(os.path.join(color_dir, '*.png'))
        # poses_ba_lis = glob(os.path.join(poses_ba_dir, '*.txt'))
        # intrin_ba_lis = glob(os.path.join(intrin_ba_dir, '*.txt'))


        w, h = Image.open(linemod_dataset.dataset.color_dir_lists[object_id][0]).size
        pinhole_dict = {}
        # enumerate 是一个内置函数，用于将一个可迭代对象（如列表）转换为一个枚举对象，该对象包含索引和值的元组。
        
        poses = []
        for idx, image in enumerate(linemod_dataset.dataset.color_dir_lists[object_id]):
            # 取出这一个图像
            image = os.path.basename(image)
            
            # 源程序p是投影矩阵 我们可以从数据集中直接取出相机位姿矩阵和相机内参矩阵
            T_w2c = linemod_dataset.dataset.pose_lists[object_id][idx]
            # print(T_w2c)

            # T_w2c = np.linalg.inv(T_w2c)
            # T_w2c[:3, :3] = np.transpose(T_w2c[:3, :3])
            # T_w2c = np.transpose(T_w2c)
            
            
            intrinsic_param = linemod_dataset.dataset.cam_K_lists[object_id][idx]
            # 旋转矩阵改为四元数表示
            qvec = rotmat2qvec(T_w2c[:3, :3]).reshape(4, )
            tvec = T_w2c[:3, 3].reshape(3, )

            # 打印结果

            # tvec = pose[:3, 3].reshape(3, )
            # qvec = rotmat2qvec(pose[:3, :3]).reshape(4, )

            poses.append(T_w2c)

            fx = intrinsic_param[0][0]
            fy = intrinsic_param[1][1]
            cx = intrinsic_param[0][2]
            cy = intrinsic_param[1][2]
# 1485.0338812206555,1485.0338812206555,714.9350380907669,935.2746092331771
            # fx = 1485.0338812206555
            # fy = 1485.0338812206555
            # cx = 714.9350380907669
            # cy = 935.2746092331771

            params = [str(w), str(h), str(fx), str(fy), str(cx), str(cy),
                    str(qvec[0]), str(qvec[1]), str(qvec[2]), str(qvec[3]),
                    str(tvec[0]), str(tvec[1]), str(tvec[2])]
            # 生成一对索引 image 就是用字符串生成的索引头
            pinhole_dict[image] = params

        # 可视化相机位姿
        # visualize_camera_poses(poses, scale=0.01)
        # convert to colmap files 
        pinhole_dict_file = os.path.join(f"{colmap_dir}/{scene}", 'pinhole_dict.json')

        with open(pinhole_dict_file, 'w') as fp:
            json.dump(pinhole_dict, fp, indent=2, sort_keys=True)



        create_init_files(pinhole_dict_file, db_file, sfm_dir,color_dir)


        
        # match features
        # 最基础的匹配
        os.system(f"colmap exhaustive_matcher \
                --database_path {db_file} \
                --SiftMatching.use_gpu=true"
                  )

        # os.system(f"colmap sequential_matcher \
        # --database_path {db_file} \
        # --SiftMatching.use_gpu=true"
        #     )
        
    

        # # bundle adjustment
        os.system(f"colmap point_triangulator \
                --database_path {db_file} \
                --image_path {color_dir} \
                --input_path {sfm_dir} \
                --output_path {sfm_dir} \
                --Mapper.tri_ignore_two_view_tracks=true"
                  )
        # os.system(f"colmap bundle_adjuster \
        #         --input_path {sfm_dir} \
        #         --output_path {sfm_dir} \
        #         --BundleAdjustment.refine_extrinsics=true"
        #           )

        # os.system(f"colmap mapper \
        # --database_path {db_file} \
        # --image_path {color_dir} \
        # --output_path {sfm_dir} "
        # )
        # undistortion 消除畸变
        os.system(f"colmap image_undistorter \
            --image_path {color_dir} \
            --input_path {sfm_dir} \
            --output_path {colmap_dir}/{scene} \
            --output_type COLMAP"
                  )
        
        

        pass

    

    print(linemod_dataset.dataset.sub_name_lists)



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Load and process YAML configuration file.")
    parser.add_argument('config_file', type=str, help="Path to the YAML configuration file")
    args = parser.parse_args()

    # 
    # 调用 main 函数
    main(args.config_file)

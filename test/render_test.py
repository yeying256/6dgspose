

import os,sys

from pathlib import Path
# 这个是需要将python的路径给到根目录，要不然找不到路径
dir_path = Path(os.path.dirname(os.path.realpath(__file__))).parents[0]
print(f"dir_path {dir_path}")
sys.path.append(dir_path.__str__())
import time
import random

import open3d as o3d


from gaussian_renderer import render as GS_Renderer
from gaussian_renderer.refine import GS_refine

from argparse import ArgumentParser
from arguments import ModelParams, PipelineParams, OptimizationParams
from scene import GaussianModel 
from scene.cameras import MiniCam,Camera
import torch
import torch.nn.functional as torch_F
import numpy as np
import cv2
from scipy.spatial.transform import Rotation
from misc_utils.warmup_lr import CosineAnnealingWarmupRestarts
import string




# 将字典转化为类
from types import SimpleNamespace


import collections
# Camera = collections.namedtuple(
#     "Camera", ["id", "model", "width", "height", "params"])

import os

from pytorch_msssim import SSIM, MS_SSIM

# L1Loss = torch.nn.L1Loss(reduction='mean')
SSIM_METRIC = SSIM(data_range=1, size_average=True, channel=3) # channel=1 for grayscale images
MS_SSIM_METRIC = MS_SSIM(data_range=1, size_average=True, channel=3)

from scene.colmap_loader import read_extrinsics_text, read_intrinsics_text, qvec2rotmat, \
    read_extrinsics_binary, read_intrinsics_binary, read_points3D_binary, read_points3D_text


# from gaussian_object.arguments import ModelParams, PipelineParams, OptimizationParams


device = torch.device('cuda')
parser = ArgumentParser()
gaussian_ModelP = ModelParams(parser)
gaussian_PipeP = PipelineParams(parser)
gaussian_OptimP = OptimizationParams(parser)
gaussian_BG = torch.zeros((3), device='cuda')


point_pwd = "/media/wangxiao/Newsmy/dataset_LINEMOD/datasets/train_data_pgsr/0575-saltbottle-bottle/test/point_cloud/iteration_30000/point_cloud.ply"
image_pwd = "/media/wangxiao/Newsmy/dataset_LINEMOD/datasets/train_data/0575-saltbottle-bottle/saltbottle-1/color"

# 定义配置参数
CFG = SimpleNamespace(
    # 初始学习率
    START_LR=5e-4,
    # 最大步数
    MAX_STEPS=1000,
    END_LR=1e-6,
    WARMUP=10,
    USE_SSIM=True,
    USE_MS_SSIM=True,
    EARLY_STOP_MIN_STEPS=10,
    EARLY_STOP_LOSS_GRAD_NORM=5e-6
)
# CFG = {
#     'START_LR' : 5e-3,
#     'MAX_STEPS' : 1000,
#     'END_LR' : 1e-6,
#     'WARMUP' : 10,
#     'USE_SSIM' : True,
#     'USE_MS_SSIM' : True,
#     'EARLY_STOP_MIN_STEPS' : 5,
#     'EARLY_STOP_LOSS_GRAD_NORM' : 1e-4
# }


# 配置视频储存路径等
base_video_path = '/media/wangxiao/Newsmy/dataset_LINEMOD/datasets/train_data/0575-saltbottle-bottle/video'

def generate_random_string(length=6):
    """生成指定长度的随机字符串"""
    letters = string.ascii_letters + string.digits
    return ''.join(random.choice(letters) for _ in range(length))

def show_gs_img(target_img:torch.Tensor):
    image_refine_torch = target_img
    image_refine_torch = image_refine_torch.cpu()
    image_refine = image_refine_torch.detach().numpy()
    
    # 转换一下格式
    image_refine = image_refine.transpose(1, 2, 0)
    image_refine = ((image_refine - image_refine.min()) / (image_refine.max() - image_refine.min()) * 255).astype(np.uint8)

    image_refine_BGR = cv2.cvtColor(image_refine,cv2.COLOR_RGB2BGR)

    cv2.imshow('Rendered Image', image_refine_BGR)
    cv2.waitKey()  # 按任意键关闭窗口
    return image_refine_BGR

def add_noise_to_quaternion(qvec, noise_level=0.01):
    # 生成一个小的随机旋转四元数
    np.random.seed(int(time.time()))
    rotation_qvec = Rotation.from_quat(qvec)
    angle = np.random.uniform(-noise_level, noise_level)
    axis = np.random.randn(3)
    axis /= np.linalg.norm(axis)  # 归一化
    sin_half_angle = np.sin(angle / 2)
    cos_half_angle = np.cos(angle / 2)
    noise_quaternion = np.array([sin_half_angle * axis[0], sin_half_angle * axis[1], sin_half_angle * axis[2], cos_half_angle])
    
    # 将噪声四元数转换为 Rotation 对象
    rotation_noise = Rotation.from_quat(noise_quaternion)
    
    # 将噪声四元数与原始四元数相乘
    rotation_qvec_noisy = rotation_qvec * rotation_noise
    
    # 将结果转换回四元数
    qvec_noisy = rotation_qvec_noisy.as_quat()
    return qvec_noisy

def add_noise_to_position(tvec, noise_level=0.01):
    noise = np.random.normal(0, noise_level, size=tvec.shape)
    tvec_noisy = tvec + noise
    return tvec_noisy

def GS_Refiner(image:torch.Tensor , mask:torch.Tensor , init_camera:Camera, gaussians:GaussianModel, return_loss=False):
        if image.dim() == 4:
            image = image.squeeze(0)
        if image.shape[2] == 3:
            image = image.permute(2, 0, 1) # 3xSxS
        if mask ==None:
            mask = torch.ones((1,) + image.shape[1:], dtype=torch.bool)
        else:
            if mask.dim() == 2:
                mask = mask[None, :, :]
            if mask.dim() == 4:
                mask = mask.squeeze(0)
            if mask.shape[2] == 1:
                mask = mask.permute(2, 0, 1) # 1xSxS

        
        assert(image.dim() == 3 and image.shape[0] == 3), image.shape

        # 制作视频
        video_images = {}
        video_extension = '.mp4'
        random_suffix = generate_random_string()
        video_path = f"{base_video_path}/{random_suffix}{video_extension}"
        fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
        video = cv2.VideoWriter(video_path, fourcc, 15, (image.shape[1], image.shape[2]))
        # 制作视频

        gaussians.initialize_pose()
        optimizer = torch.optim.AdamW([gaussians._delta_R, gaussians._delta_T], lr=CFG.START_LR)
        print(optimizer.state_dict())
        lr_scheduler = CosineAnnealingWarmupRestarts(optimizer, CFG.MAX_STEPS,
                                                    warmup_steps=CFG.WARMUP, 
                                                    max_lr=CFG.START_LR, 
                                                    min_lr=CFG.END_LR)
        trunc_mask = (image.sum(dim=0, keepdim=True) > 0).type(torch.float32) # 1xSxS        
        target_img = (image * mask.float().to(device)).to(device)
        target_img = target_img/255

        # 测试代码
        # image_refine_torch = target_img
        # image_refine_torch = image_refine_torch.cpu()
        # image_refine = image_refine_torch.detach().numpy()
        
        # # 转换一下格式
        # image_refine = image_refine.transpose(1, 2, 0)
        # image_refine = ((image_refine - image_refine.min()) / (image_refine.max() - image_refine.min()) * 255).astype(np.uint8)

        # image_refine_BGR = cv2.cvtColor(image_refine,cv2.COLOR_RGB2BGR)

        # cv2.imshow('Rendered Image', image_refine_BGR)
        # cv2.waitKey()  # 按任意键关闭窗口
        
        # show_gs_img(target_img/2 + target_img/2)
        show_gs_img(target_img)

        # 测试代码

        iter_losses = list()
        for iter_step in range(CFG.MAX_STEPS):
            # GS_Renderer是渲染器
            render_img = GS_Renderer(init_camera, gaussians, gaussian_PipeP, gaussian_BG)['render'] * trunc_mask

            video_img = show_gs_img(render_img/2+target_img/2)
            video.write(video_img)
            loss = 0.0

            if CFG.USE_SSIM:
                loss += (1 - SSIM_METRIC(render_img[None, ...], target_img[None, ...]))
                print(f"step ={iter_step} loss = {loss}")
            if CFG.USE_MS_SSIM:
                loss += (1 - MS_SSIM_METRIC(render_img[None, ...], target_img[None, ...]))
                
            loss.backward()
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()

            iter_losses.append(loss.item())
            if iter_step >= CFG.EARLY_STOP_MIN_STEPS:
                loss_grads = (torch.as_tensor(iter_losses)[1:] - torch.as_tensor(iter_losses)[:-1]).abs()
                if loss_grads[-CFG.EARLY_STOP_MIN_STEPS:].mean() < CFG.EARLY_STOP_LOSS_GRAD_NORM: # early stop the refinement
                    break
        
        gs3d_delta_RT = gaussians.get_delta_pose.squeeze(0).detach().cpu().numpy()




        # 释放视频写入器
        video.release()

        outp = {
            'gs3d_delta_RT': gs3d_delta_RT,
            'iter_step': iter_step,
            'render_img': render_img,
        }
        
        if return_loss:
            sobel_x = torch.tensor([[1, 0, -1], [2, 0, -2], [1, 0, -1]], dtype=torch.float32).view(1, 1, 3, 3).repeat(1, 3, 1, 1)
            sobel_x = sobel_x.to(image.device)
            sobel_y = sobel_x.transpose(-2, -1)
            # Apply Sobel filter to the images
            query_sobel_h = torch_F.conv2d(image[None], sobel_x, padding=0)
            query_sobel_v = torch_F.conv2d(image[None], sobel_y, padding=0)
            rend_sobel_h = torch_F.conv2d(render_img[None], sobel_x, padding=0)
            rend_sobel_v = torch_F.conv2d(render_img[None], sobel_y, padding=0)
            edge_err = (query_sobel_h - rend_sobel_h).abs().mean() + (query_sobel_v - rend_sobel_v).abs().mean()
            outp['edge_err'] = edge_err

        return outp

def main():
    print("GS_render test")

    path = "/media/wangxiao/Newsmy/dataset_LINEMOD/datasets/sfm_output/outputs_softmax_loftr_loftr/0575-saltbottle-bottle/sfm_ws"

    cameras_extrinsic_file = os.path.join(path, "sparse", "images.bin")
    cameras_intrinsic_file = os.path.join(path, "sparse", "cameras.bin")

    cam_extrinsics = read_extrinsics_binary(cameras_extrinsic_file)
    cam_intrinsics = read_intrinsics_binary(cameras_intrinsic_file)
    # Camera = collections.namedtuple(
    # "Camera", ["id", "model", "width", "height", "params"])

    # images[image_id] = Image(
    # id=image_id, qvec=qvec, tvec=tvec,
    # camera_id=camera_id, name=image_name,
    # xys=xys, point3D_ids=point3D_ids)

    id=1
    Camera_one = cam_intrinsics[id]
    w = Camera_one.width
    h = Camera_one.height
    fx = Camera_one.params[0]
    fy = Camera_one.params[1]
    cx = Camera_one.params[2]
    cy = Camera_one.params[3]

    fovx = 2 * np.arctan(w / (2 * fx))
    fovy = 2 * np.arctan(h / (2 * fy))

    intrinsic_matrix = torch.zeros(3,3)
    intrinsic_matrix[0,0]=fx
    intrinsic_matrix[1,1]=fy
    intrinsic_matrix[0,2]=cx
    intrinsic_matrix[1,2]=cy
    intrinsic_matrix[2,2]=1.0
    intrinsic_matrix.float().cuda()

    qvec_gt = cam_extrinsics[id].qvec
    tvec_gt = cam_extrinsics[id].tvec

    tvec = tvec_gt
    # w x y z
    qvec = qvec_gt[[1, 2, 3, 0]]
    qvec = add_noise_to_quaternion(qvec,0.01)
    tvec = add_noise_to_position(tvec_gt,0.005)

    R = Rotation.from_quat(qvec)

    # 测试可视化坐标系
    # 从colmap中读取的位姿
    T_colmap = np.eye(4)
    T_colmap[:3, :3] = R.as_matrix()
    T_colmap[:3, 3] = tvec


    R = R.as_matrix().T


    # 这是直接读取到的位姿
    T_gt = [-6.983123996165305769e-01, 1.403005627748399042e-01, 7.019085016060609972e-01, 9.476508797357724792e-03,
            -3.653195919952658688e-01, -9.131302507420553383e-01, -1.809274464644956837e-01, -2.332960280871109598e-02,
            6.155496635091396440e-01, -3.827648067218480943e-01, 6.889047209077401313e-01 ,4.347392819213768411e-01,
            0.000000000000000000e+00 ,0.000000000000000000e+00, 0.000000000000000000e+00, 1.000000000000000000e+00]
    T_gt_matrix = np.array(T_gt).reshape(4, 4)
    # 创建原始坐标系
    original_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.3, origin=[0, 0, 0])
    transformed_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1, origin=[0, 0, 0])
    transformed_frame.transform(T_gt_matrix)

    colmap_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.20, origin=[0, 0, 0])
    colmap_frame.transform(T_colmap)
    
    T_colmap_inv = np.linalg.inv(T_colmap)
    frame_inv = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.20, origin=[0, 0, 0])
    frame_inv.transform(T_colmap_inv)

    


    o3d.visualization.draw_geometries([original_frame,colmap_frame,frame_inv])


    camerapath = Path(cam_extrinsics[id].name)
    cameraname = camerapath.name



    gs_camera =  Camera(colmap_id=id,R=R,T=tvec,Cx=cx,Cy=cy,image_height=h,image_width=w,FoVx=fovx,FoVy=fovy,
                        image_name=cameraname,image_path=camerapath,uid=id)

    # gs_camera =  MiniCam(width=w,height=h,fovx=fovx,fovy=fovy,znear=0.1,zfar=10000,world_view_transform=Tmatix,full_proj_transform=intrinsic_matrix)


    obj_gaussians = GaussianModel(gaussian_ModelP.sh_degree)
    obj_gaussians.load_ply(point_pwd)

    # 这里需要裁剪模型
    # bbox_coords = [
    #     -0.411161676964370060e-01, -7.079713634457378280e-02, -0.453013546280710155e-01,
    #     -0.390649987936478371e-01, -7.425646133443529473e-02, 0.412349492191197547e-01,
    #     -0.383780033148180599e-01, 6.105837375882193496e-02, 0.42950554310113622e-01,
    #     -0.404291722176072288e-01, 6.451769874868344690e-02, -0.432412484161794219e-01,
    #     0.535089001252602692e-01, -7.220261973354145935e-02, -0.478999496518403722e-01,
    #     0.555600690280494380e-01, -7.566194472340297128e-02, 0.463635419535041193e-01,
    #     0.562470645068792152e-01, 5.965289036985425841e-02, 0.406964604072420194e-01,
    #     0.541958956040900464e-01, 6.311221535971577035e-02, -0.458398434399487509e-01
    # ]

    # 源数据集的
    bbox_coords =[-3.796371205057989806e-02,-7.839022570733904238e-02 ,-3.724212625509088631e-02,
                -3.588311474391941980e-02,-7.920277835394752552e-02 ,4.183483401333373819e-02,
                -3.598206953301359701e-02 ,7.300152765654545350e-02 ,4.340140784623270492e-02,
                -3.806266683967407527e-02 ,7.381408030315393665e-02 ,-3.567555242219191958e-02,
                3.846474454893995809e-02 ,-7.831984615020971818e-02 ,-3.925231548490171890e-02,
                4.054534185560043635e-02 ,-7.913239879681820133e-02 ,3.982464478352290560e-02,
                4.044638706650625914e-02 ,7.307190721367477770e-02 ,4.139121861642187233e-02,
                3.836578975984578088e-02 ,7.388445986028326085e-02 ,-3.768574165200275217e-02]
    obj_gaussians.clip_to_bounding_box(bbox_coords)
    
    obj_gaussians.save_ply("/media/wangxiao/Newsmy/dataset_LINEMOD/datasets/train_data_pgsr/0575-saltbottle-bottle/test/point_cloud/iteration_30000/clip.ply")


    # 


    render_img = GS_Renderer(gs_camera, obj_gaussians, gaussian_PipeP, gaussian_BG)['render']
    # 将 render_img 的维度从 (C, H, W) 调整为 (H, W, C)。
    render_img_np = render_img.permute(1, 2, 0).detach().cpu().numpy()
    # 将 NumPy 数组的值从浮点数（通常是 [0, 1] 范围内的值）转换为 [0, 255] 范围内的无符号 8 位整数
    render_img_np = (render_img_np * 255).astype(np.uint8)

    # 这个是更换一下 像素在前，通道在后
    if render_img_np.shape[0] == 3:
        render_img_np = np.transpose(render_img_np, (1, 2, 0))
    
    # 如果axis=-1也行，代表最后一个轴 
    mask_path = os.path.join(image_pwd,'0_mask.png')
    # mask_from_path = cv2.imread(mask_path)
    # mask = np.ones(mask_from_path.shape[:2], dtype=bool)
    # mask = np.all(mask_from_path == 255, axis=2)

    # mask = np.all(render_img_np == 0, axis=2)
    # mask = 1 - mask.astype(np.uint8)
    # mask_visualize = (mask * 255).astype(np.uint8)
    # print("Mask shape:", mask_visualize.shape)
    # print("Mask data type:", mask_visualize.dtype)


    # cv2.imshow('Mask Visualization', mask_visualize)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    # 取出不是黑色的部分

    

    imag0_path = os.path.join(image_pwd,'0.png')
    imag0 = cv2.imread(imag0_path)
    # 换一下颜色
    imag0_rgb = cv2.cvtColor(imag0, cv2.COLOR_BGR2RGB)
    imag0_rgb_torch = torch.from_numpy(imag0_rgb).float().to('cuda')

    render_img_torch = torch.from_numpy(render_img_np).float().to('cuda')
    # mask_torch = torch.from_numpy(mask).to('cuda')


    class_gsrefine = GS_refine()
    class_gsrefine.GS_Refiner(image=imag0_rgb_torch,init_camera=gs_camera,gaussians=obj_gaussians,return_loss=True,mask=None)

    ret_outp = GS_Refiner(image=imag0_rgb_torch, mask=None, init_camera=gs_camera, gaussians=obj_gaussians, return_loss=True)

    image_refine_torch = ret_outp['render_img']
    image_refine_torch = image_refine_torch.cpu()
    image_refine = image_refine_torch.detach().numpy()
    
    # 转换一下格式
    image_refine = image_refine.transpose(1, 2, 0)
    image_refine = ((image_refine - image_refine.min()) / (image_refine.max() - image_refine.min()) * 255).astype(np.uint8)

    image_refine_BGR = cv2.cvtColor(image_refine,cv2.COLOR_RGB2BGR)

    # rgb_image[:, :, [2, 1, 0]] 把rgb，改成bgr显示
    img_cv = render_img_np[:, :, [2, 1, 0]] 
    # 使用 OpenCV 显示图像

            # 'gs3d_delta_RT': gs3d_delta_RT,
            # 'iter_step': iter_step,
            # 'render_img': render_img,
    print(f"'gs3d_delta_RT' = {ret_outp['gs3d_delta_RT']}")
    print(f"'iter_step' = {ret_outp['iter_step']}")


    combined_image = cv2.vconcat([img_cv, image_refine_BGR, imag0])
    cv2.imshow('Rendered Image', combined_image)
    cv2.waitKey()  # 按任意键关闭窗口
    # cv2.destroyAllWindows()


if __name__=="__main__":
    main()
    








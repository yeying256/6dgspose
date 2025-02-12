import os

scenes = ['0575-saltbottle-bottle']
data_base_path='/media/wangxiao/Newsmy/dataset_LINEMOD/datasets/sfm_output/outputs_softmax_loftr_loftr'
out_base_path='/media/wangxiao/Newsmy/dataset_LINEMOD/datasets/train_data_pgsr'
imagpath = '/media/wangxiao/Newsmy/dataset_LINEMOD/datasets/train_data'
eval_path='dtu_eval'
out_name='test'
gpu_id=0

for scene in scenes:
    cmd = f'rm -rf {out_base_path}/{scene}/{out_name}/*'
    print(cmd)
    os.system(cmd)

    scene_sfm_dir = f"{data_base_path}/{scene}/sfm_ws/model_filted_bbox"

    # 将图像复制过来
    source_dir = f'{imagpath}/{scene}/images'
    destination_dir = f'{data_base_path}/{scene}/images'

    cmd = f'rsync -av {source_dir}/* {destination_dir}/'
    # 使用 cp 命令复制文件
    # cmd = f'cp -r {data_base_path}/{scene}/sfm_ws/model_filted_bbox/* {destination_dir}/'
    print(cmd)
    os.system(cmd)

    # 使用 rsync 命令复制文件或文件夹
    cmd = f'rsync -av {scene_sfm_dir}/* {data_base_path}/{scene}/sparse/'
    # /media/wangxiao/Newsmy/dataset_LINEMOD/datasets/sfm_output/outputs_softmax_loftr_loftr/0575-saltbottle-bottle/sfm_ws/model_filted_bbox


    # cmd = f'cp -rf {data_base_path}/{scene}/sparse/3/* {data_base_path}/{scene}/sparse/'
    print(cmd)
    os.system(cmd)

    # cmd = f'cp -rf {data_base_path}/scan{scene}/sparse/0/* {data_base_path}/{scene}/sparse/'
    # print(cmd)
    # os.system(cmd)


    common_args = "--quiet -r2 --ncc_scale 0.5"
    cmd = f'CUDA_VISIBLE_DEVICES={gpu_id} python train.py -s {data_base_path}/{scene} -m {out_base_path}/{scene}/{out_name}'
    print(cmd)
    os.system(cmd)

    # common_args = "--quiet --num_cluster 1 --voxel_size 0.002 --max_depth 5.0"
    # cmd = f'CUDA_VISIBLE_DEVICES={gpu_id} python render.py -m {out_base_path}/{scene}/{out_name} {common_args}'
    # print(cmd)
    # os.system(cmd)

    # cmd = f"CUDA_VISIBLE_DEVICES={gpu_id} python scripts/eval_dtu/evaluate_single_scene.py " + \
    #     f"--input_mesh {out_base_path}/{scene}/{out_name}/mesh/tsdf_fusion_post.ply " + \
    #     f"--scan_id {scene} --output_dir {out_base_path}/{scene}/{out_name}/mesh " + \
    #     f"--mask_dir {data_base_path} " + \
    #     f"--DTU {eval_path}"
import os


data_base_path='/media/wangxiao/Newsmy/dataset_LINEMOD/datasets/lowtexture_test_data_sfm'
out_base_path='/media/wangxiao/Newsmy/dataset_LINEMOD/datasets/lowtexture_test_data_pgsr'
imagpath = '/media/wangxiao/Newsmy/dataset_LINEMOD/datasets/lowtexture_test_data'
# eval_path='dtu_eval'
out_name='test'
gpu_id=0

# scenes = os.listdir(data_base_path)
scenes = ['0724-vitamin-others']

cmd = 'conda activate pgsr '
os.system(cmd)
print(cmd)


for scene in scenes:
    cmd = f'rm -rf {out_base_path}/{scene}/{out_name}/*'
    print(cmd)
    os.system(cmd)


    cmd = f'CUDA_VISIBLE_DEVICES={gpu_id} python train.py -s {data_base_path}/{scene} -m {out_base_path}/{scene}/{out_name} -i {data_base_path}/{scene}/images --max_abs_split_points 0 --opacity_cull_threshold 0.05'

    print(cmd)
    os.system(cmd)
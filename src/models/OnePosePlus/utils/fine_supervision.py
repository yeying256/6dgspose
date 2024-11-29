import torch

@torch.no_grad()
def fine_supervision(data, config):
    """
    这个函数的主要目的是计算细粒度匹配的监督信号，并将其存储在 data 字典中，以便后续在模型训练时使用。
    通过这种方式，模型可以在粗匹配的基础上进一步调整其预测，以更准确地对齐图像特征点。
    Update:
        data (dict): {
            "expec_f_gt": [M, 2]
        }
    """
    # 从config中读取具体参数
    coarse_scale, fine_scale = list(config['OnePosePlus']['loftr_backbone']['resolution'])
    radius = config['OnePosePlus']['loftr_fine']['window_size'] // 2

    # 获取索引信息
    # 2. get coarse prediction
    b_ids, i_ids, j_ids = data['b_ids'], data['i_ids'], data['j_ids']

    # 3. compute gt
    coarse_scale = coarse_scale * data['query_image_scale'][b_ids][:, [1, 0]] if 'query_image_scale' in data else fine_scale 
    fine_scale = fine_scale * data['query_image_scale'][b_ids][:, [1, 0]] if 'query_image_scale' in data else fine_scale

    mkpts_query = (
        torch.stack([j_ids % data["q_hw_c"][1], j_ids // data["q_hw_c"][1]], dim=1)
        * coarse_scale
    ) # [M, 2]

    fine_gt_location = data['fine_location_matrix_gt'][b_ids, i_ids, j_ids]
    gt_offset = fine_gt_location - mkpts_query
    expec_f_gt = (gt_offset) / fine_scale / radius  # [M, 2]

    
    data.update({"expec_f_gt": expec_f_gt})
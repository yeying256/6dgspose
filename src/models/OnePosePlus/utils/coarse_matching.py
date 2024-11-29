# from loguru import logger

import torch
import torch.nn as nn
from torch.nn import functional as F
from einops.einops import rearrange
# from src.utils.profiler import PassThroughProfiler


def mask_border(m, b: int, v):
    """ Mask borders with value
    Args:
        m (torch.Tensor): [N, n_pointcloud, H1, W1]
        b (int)
        v (m.dtype)
    """
    m[:, :, :b] = v
    m[:, :, :, :b] = v
    m[:, :, -b:0] = v
    m[:, :, :, -b:0] = v


def mask_border_with_padding(m, bd, v, p_m0, p_m1):
    m[:, :bd] = v
    m[:, :, :bd] = v
    m[:, :, :, :bd] = v
    m[:, :, :, :, :bd] = v

    h0s, w0s = p_m0.sum(1).max(-1)[0].int(), p_m0.sum(-1).max(-1)[0].int()
    h1s, w1s = p_m1.sum(1).max(-1)[0].int(), p_m1.sum(-1).max(-1)[0].int()
    for b_idx, (h0, w0, h1, w1) in enumerate(zip(h0s, w0s, h1s, w1s)):
        m[b_idx, h0 - bd :] = v
        m[b_idx, :, w0 - bd :] = v
        m[b_idx, :, :, h1 - bd :] = v
        m[b_idx, :, :, :, w1 - bd :] = v


def calc_max_candidates(p_m0, p_m1):
    """Calculate the max candidates of all pairs within a batch"""
    h0s, w0s = p_m0.sum(1).max(-1)[0], p_m0.sum(-1).max(-1)[0]
    h1s, w1s = p_m1.sum(1).max(-1)[0], p_m1.sum(-1).max(-1)[0]
    max_cand = torch.sum(torch.min(torch.stack([h0s * w0s, h1s * w1s], -1), -1)[0])
    return max_cand


def build_feat_normalizer(method, **kwargs):
    if method == "sqrt_feat_dim":
        # 一种特别的归一化方式
        return lambda feat: feat / feat.shape[-1] ** 0.5
    elif method == "none" or method is None:
        return lambda feat: feat
    elif method == "temparature":
        return lambda feat: feat / kwargs["temparature"]
    else:
        raise ValueError

class CoarseMatching(nn.Module):
    # def __init__(self, config, profiler=None):
    def __init__(self, config):
        super().__init__()
        self.config = config
        # 正则化处理
        self.feat_normalizer = build_feat_normalizer(config["feat_norm_method"])

        self.type = config["type"]
        if self.type == "dual-softmax":
            self.temperature = config['dual_softmax']['temperature']
        else:
            raise NotImplementedError()

        # from conf_matrix to prediction
        self.thr = config["thr"]
        # border_rm边界移出宽度
        self.border_rm = config["border_rm"]
        self.train_coarse_percent = config["train"]["train_coarse_percent"]

        # 被用来确定需要添加多少个真实匹配点作为填充，以使总的匹配数量达到 max_num_matches_train。
        self.train_pad_num_gt_min = config["train"]["train_pad_num_gt_min"]

        # self.profiler = profiler or PassThroughProfiler

    def forward(self, feat_db_3d, feat_query, data, mask_query=None):
        """
        输入参数：
            feat_db_3d 表示数据库中的3D特征 N代表批次大小，L代表3D特征点的数量，C是特征的纬度
            feat_query 查询图像的2D特征 N表示批次大小，S表示2D特征点的数量 C是特征的纬度
            data 字典，包含图像尺寸等数据
            mask_query图像的掩码，可选
        Args:

            feat_db_3d (torch.Tensor): [N, L, C]
            feat_query (torch.Tensor): [N, S, C]
            data (dict)
            mask_query (torch.Tensor): [N, S] (optional)
        
        匹配结果：
        mkpts_3d_db：形状为 [M, 3]，表示匹配的3D关键点的位置坐标。
        mkpts_query_c: 形状为 [M, 2]，表示匹配的2D查询图像上的特征点位置坐标。
        mconf: 形状为 [M]，表示匹配的置信度分数。
        b_ids, i_ids, j_ids: 分别表示匹配结果的批次索引、3D特征索引和2D特征索引。
        
        Update:    
            data (dict): {
                'b_ids' (torch.Tensor): [M'],
                'i_ids' (torch.Tensor): [M'],
                'j_ids' (torch.Tensor): [M'],
                'gt_mask' (torch.Tensor): [M'],
                'mkpts_3d_db' (torch.Tensor): [M, 3],
                'mkpts_query_c' (torch.Tensor): [M, 2],
                'mconf' (torch.Tensor): [M]}
            NOTE: M' != M during training.
        """
        N, L, S, C = (
            feat_db_3d.size(0),
            feat_db_3d.size(1),
            feat_query.size(1),
            feat_query.size(2),
        )

        # normalize  正则化，在init里面 map() 函数是一个内置函数，用于将一个函数依次作用于一个或多个可迭代对象的每个元素上，并返回一个迭代器，
        # 该迭代器包含了每个元素经过函数处理后的结果。feat_normalizer返回一个lambda函数，目的是归一化
        feat_db_3d, feat_query = map(self.feat_normalizer, [feat_db_3d, feat_query])

        if self.type == "dual-softmax":
            # 相似度矩阵 
            # torch.einsum 是一个非常灵活的函数，用于计算多维数组的各种线性代数运算。
            # 这里使用的子句 "nlc,nsc->nls" 描述了如何将两个张量 feat_db_3d 和 feat_query 相结合。
            # 对于每个 l（3D特征点），计算它与所有 s（2D特征点）的 点积。
            sim_matrix = (
                torch.einsum("nlc,nsc->nls", feat_db_3d, feat_query) / (self.temperature + 1e-4)
            )

            # 这里的掩码就是二维图像的掩码
            if mask_query is not None:
                fake_mask3D = torch.ones((N, L), dtype=torch.bool, device=mask_query.device)
                valid_sim_mask = fake_mask3D[..., None] * mask_query[:, None]
                _inf = torch.zeros_like(sim_matrix)
                _inf[~valid_sim_mask.bool()] = -1e9
                del valid_sim_mask
                sim_matrix += _inf
            conf_matrix = F.softmax(sim_matrix, 1) * F.softmax(sim_matrix, 2)
        else:
            raise NotImplementedError

        data.update({"conf_matrix": conf_matrix})

        data.update(**self.get_coarse_match(conf_matrix, data))

        # predict coarse matches from conf_matrix
        # with self.profiler.record_function("LoFTR/coarse-matching/get_coarse_match"):
        #     data.update(**self.get_coarse_match(conf_matrix, data))

    @torch.no_grad()
    def get_coarse_match(self, conf_matrix, data):
        """
        输入：
        Args:
            conf_matrix (torch.Tensor): [N, L, S]

            # 查询图像的原始图像的分辨率和特征图图像的分辨率
            data (dict): with keys ['hw1_i', 'hw1_c']
        Returns:
            coarse_matches (dict): {
                'b_ids' (torch.Tensor): [M'],
                'i_ids' (torch.Tensor): [M'],
                'j_ids' (torch.Tensor): [M'],
                'gt_mask' (torch.Tensor): [M'],
                'm_bids' (torch.Tensor): [M],
                'mkpts_3d_db' (torch.Tensor): [M, 3],
                'mkpts_query_c' (torch.Tensor): [M, 2],
                'mconf' (torch.Tensor): [M]}
        """
        # 储存了特征图的高度和宽度
        axes_lengths = {"h1c": data["q_hw_c"][0], "w1c": data["q_hw_c"][1]}
        device = conf_matrix.device
        # confidence thresholding
        # 对置信度矩阵 conf_matrix 进行阈值筛选，得到一个布尔掩码 mask。
        # 重排 mask 的维度，使其与特征图的尺寸对齐。 得到一个布尔掩码，这个掩码就是识别到特征的地方
        # 这里是为了排除某一个点没有匹配点的情况
        mask = conf_matrix > self.thr
        mask = rearrange(
            mask, "b n_point_cloud (h1c w1c) -> b n_point_cloud h1c w1c", **axes_lengths
            # 最后那一串向量转化为一个二维的图
        )
        if "mask0" not in data:
            # border_rm边界移除的宽度。这是个是要把边缘的特征点移除
            mask_border(mask, self.border_rm, False)
        else:
            # 这程序还没写，不知道要干嘛，也不知道mask0是干嘛的
            raise NotImplementedError
        
        # 处理完边界之后再处理回来向量的形式
        mask = rearrange(
            mask, "b n_point_cloud h1c w1c -> b n_point_cloud (h1c w1c)", **axes_lengths
        )

        # mutual nearest  “互最近邻”（mutual nearest neighbor, MNN）
        # 验证特征点是否近邻 conf_matrix == conf_matrix.max(dim=2, keepdim=True)[0] 这里是一个true的矩阵
        # 提取出每一个三维点匹配的最大的二维点
        # 再提取出每一个二维点匹配最大的三维点
        # 
        mask = (
            mask
            * (conf_matrix == conf_matrix.max(dim=2, keepdim=True)[0])
            * (conf_matrix == conf_matrix.max(dim=1, keepdim=True)[0])
        )

        # 3. find all valid coarse matches
        # this only works when at most one `True` in each row
        # 每一行的最大值及其索引 all_j_ids就是索引 mask_v就是一个 1Xn的张量。all_j_ids也是一个1Xn的张量
        mask_v, all_j_ids = mask.max(dim=2)
        # with self.profiler.record_function(
        #     "LoFTR/coarse-matching/get_coarse_match/argmax-conf"
        # ):
            # b_ids, i_ids = torch.where(mask_v)
        # 找到所有 True 元素的位置，并返回这些位置的索引
        b_ids, i_ids = torch.where(mask_v)
        j_ids = all_j_ids[b_ids, i_ids]
        # 把所有匹配路径点索引列出来
        mconf = conf_matrix[b_ids, i_ids, j_ids]

        # when TRAINING
        # select only part of coarse matches for fine-level training
        # pad with gt coarses matches   检查是否是训练模式 并且是否开启了训练时填充的训练
        # 通过填充真实的匹配点，可以确保每次迭代都有足够数量的正样本参与训练，从而避免模型偏向于负样本。
        if self.training and self.config['train']['train_padding']:
            if "mask0" not in data:
                num_candidates_max = mask.size(0) * min(mask.size(1), mask.size(2))
            else:
                raise not NotImplementedError
            # train_coarse_percent 配置参数中的百分比，也就是你最大可以匹配多少个点
            max_num_matches_train = int(num_candidates_max * self.train_coarse_percent) # Max train number
            # 计算最大匹配数

            # 计算预测匹配的数量，找一列计算就行了
            num_matches_pred = len(b_ids)

            # 验证最小填充数肯定比最大比配数大
            assert (
                self.train_pad_num_gt_min < max_num_matches_train
            ), "min-num-gt-pad should be less than num-train-matches"

            # pred_indices is to select from prediction
            # 选择索引预测，如果预测匹配的数量小于等于最大匹配数量减去最小填充数量，则选择所有预测索引，否则随机选择一部分索引以达到所需数量
            if num_matches_pred <= max_num_matches_train - self.train_pad_num_gt_min:
                pred_indices = torch.arange(num_matches_pred, device=device)
            else:
                # 随机生成了一个一维张量，其中的内容从0到num_matches_pred（不包括这个值）随机生成，其长度为：(max_num_matches_train - self.train_pad_num_gt_min,)
                pred_indices = torch.randint(
                    num_matches_pred,
                    (max_num_matches_train - self.train_pad_num_gt_min,),
                    device=device,
                )

            # 这里使用 torch.where 函数来获取 conf_matrix_gt 中非零元素的位置，这些位置对应于真实匹配点的索引。然后，确保至少有一个真实匹配点。
            # 这是数据集给出的数据ground truth
            spv_b_ids, spv_i_ids, spv_j_ids = torch.where(data['conf_matrix_gt'])
            assert len(spv_b_ids) != 0
            # 这里是真实匹配点。
            gt_pad_indices = torch.randint(
                len(spv_b_ids),
                (max(max_num_matches_train - num_matches_pred, self.train_pad_num_gt_min),),
                device=device,
            )
            # 将真实的匹配点数据的置信度设置为0
            mconf_gt = torch.zeros(
                len(spv_b_ids), device=device
            )  # set conf of gt paddings to all zero

            # 合并预测点和真实的匹配点
            b_ids, i_ids, j_ids, mconf = map(
                lambda x, y: torch.cat([x[pred_indices], y[gt_pad_indices]], dim=0),
                *zip(
                    [b_ids, spv_b_ids],
                    [i_ids, spv_i_ids],
                    [j_ids, spv_j_ids],
                    [mconf, mconf_gt],
                ),
            )
            # 在训练过程中，通过结合预测匹配点和真实匹配点，可以确保每次训练都有相同数量的匹配样本，从而帮助模型更好地学习和优化。

        # These matches select patches that feed into fine-level network
        coarse_matches = {"b_ids": b_ids, "i_ids": i_ids, "j_ids": j_ids}

        # 4. Update with matches in original image resolution
        # 更新匹配信息到原图像的分辨率中
        # scale缩放因子就是因为分辨率不同导致的缩放问题。
        scale = data["q_hw_i"][0] / data["q_hw_c"][0]
        scale_total = scale * data["query_image_scale"][b_ids][:, [1, 0]] if "query_image_scale" in data else scale
        mkpts_query = (
            torch.stack([j_ids % data["q_hw_c"][1], j_ids // data["q_hw_c"][1]], dim=1)
            * scale_total
        )
        mkpts_3d_db = data["keypoints3d"][b_ids, i_ids]

        # These matches is the current prediction (for visualization)
        coarse_matches.update(
            {
                # 真实匹配掩码
                "gt_mask": mconf == 0,
                # 非零置信度匹配的批量索引
                "m_bids": b_ids[mconf != 0],  # mconf == 0 => gt matches
                # 非零置信度3d点的坐标
                "mkpts_3d_db": mkpts_3d_db[mconf != 0],
                # 非零置信度匹配的查询图像特征点坐标： 是一个二维张量，第一维度代表匹配点的数量，第二维度代表坐标
                "mkpts_query_c": mkpts_query[mconf != 0],
                # 非零置信度匹配的置信度：
                "mconf": mconf[mconf != 0],
            }
        )

        return coarse_matches

    @property
    def n_rand_samples(self):
        return self._n_rand_samples

    @n_rand_samples.setter
    def n_rand_samples(self, value):
        print(f"Setting {type(self).__name__}.n_rand_samples to {value}.")
        self._n_rand_samples = value

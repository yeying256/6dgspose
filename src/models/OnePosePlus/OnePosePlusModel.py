
# from loguru import logger
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops.einops import rearrange, repeat

# from src.utils.profiler import PassThroughProfiler

from .backbone import (
    build_backbone,
    _extract_backbone_feats,
    _get_feat_dims,
)
from .utils.normalize import normalize_3d_keypoints
from .loftr_module import LocalFeatureTransformer, FinePreprocess
from .utils.position_encoding import (
    PositionEncodingSine,
    KeypointEncoding_linear,
)
from .utils.coarse_matching import CoarseMatching
from .utils.fine_matching import FineMatching


class OnePosePlus_model(nn.Module):
    def __init__(self, config, profiler=None, debug=False):
        super().__init__()
        # Misc
        self.config = config
        # self.profiler = profiler or PassThroughProfiler()
        self.debug = debug

        # Modules
        # Used to extract 2D query image feature 2D特征提取，提取了特征，提取完了之后并进行了位置编码 
        self.backbone = build_backbone(self.config["loftr_backbone"])

        # For query image and 3D points 位置编码
        if self.config["positional_encoding"]["enable"]:
            self.dense_pos_encoding = PositionEncodingSine(
                self.config["loftr_coarse"]["d_model"],
                max_shape=self.config["positional_encoding"]["pos_emb_shape"],
            )
        else:
            self.dense_pos_encoding = None
        
        # 关键点编码 
        if self.config["keypoints_encoding"]["enable"]:
            # NOTE: from Gats part
            if config["keypoints_encoding"]["type"] == "mlp_linear":
                encoding_func = KeypointEncoding_linear
            else:
                raise NotImplementedError

            self.kpt_3d_pos_encoding = encoding_func(
                inp_dim=3,
                feature_dim=self.config["keypoints_encoding"]["descriptor_dim"],
                layers=self.config["keypoints_encoding"]["keypoints_encoder"],
                norm_method=self.config["keypoints_encoding"]["norm_method"],
            )
        else:
            self.kpt_3d_pos_encoding = None
        # 进行粗粒度特征变换
        self.loftr_coarse = LocalFeatureTransformer(self.config["loftr_coarse"])

        # 匹配模块 CoarseMatching特征粗匹配
        self.coarse_matching = CoarseMatching(
            self.config["coarse_matching"]
        )

        # 特征细匹配前处理模块
        self.fine_preprocess = FinePreprocess(
            self.config["loftr_fine"],
            cf_res=self.config["loftr_backbone"]["resolution"],
            feat_ids=self.config["loftr_backbone"]["resnetfpn"]["output_layers"],
            feat_dims=_get_feat_dims(self.config["loftr_backbone"]),
        )

        # 细粒度特征transformer
        self.loftr_fine = LocalFeatureTransformer(self.config["loftr_fine"])
        # 细粒度匹配模块
        self.fine_matching = FineMatching(self.config["fine_matching"])

        # 看看配置文件里面是不是有预训练模型，如果有的话就载入，如果没有的话就是训练模式
        self.loftr_backbone_pretrained = self.config["loftr_backbone"]["pretrained"]
        if self.loftr_backbone_pretrained is not None:
            print(
                f"Load pretrained backbone from {self.loftr_backbone_pretrained}"
            )
            ckpt = torch.load(self.loftr_backbone_pretrained, "cpu")["state_dict"]
            for k in list(ckpt.keys()):
                if "backbone" in k:
                    newk = k[k.find("backbone") + len("backbone") + 1 :]
                    ckpt[newk] = ckpt[k]
                ckpt.pop(k)
            self.backbone.load_state_dict(ckpt)

            if self.config["loftr_backbone"]["pretrained_fix"]:
                for param in self.backbone.parameters():
                    param.requires_grad = False

    def forward(self, data):
        """
        Update:
        keypoints3d: [N, n1, 3]
        N代表批次的大小，n1代表关键点的数量，keypoints3d是代表关键点的位置三维坐标，query_image_scale代表缩放比例，长宽，mask是可选的

        descriptors3d_db: [N, dim, n1] 含义: 表示的是3D关键点的描述符。
        N: 批次大小。
        dim: 描述符的维度，即用来表示每个3D关键点特征的向量长度。
        n1: 每个样本中3D关键点的数量。

        scores3d_db: [N, n1, 1]

        形状: [N, n1, 1]
        含义: 表示的是每个3D关键点的置信度或评分。
        N: 批次大小。
        n1: 每个样本中3D关键点的数量。
        1: 每个关键点有一个评分值，表示该关键点的有效性或可信度。

        4. query_image: (N, 1, H, W)

        形状: (N, 1, H, W)
        含义: 表示的是查询图像的数据。
            N: 批次大小。
            1: 表示图像通道数，这里为1，说明是灰度图（单通道）。
            H: 图像的高度。
            W: 图像的宽度。

        5. query_image_scale: (N, 2)
        形状: (N, 2)
        含义: 表示的是查询图像的缩放因子。
            N: 批次大小。
            2: 分别表示图像的宽度和高度的缩放因子。

            data (dict): {
                keypoints3d: [N, n1, 3]
                descriptors3d_db: [N, dim, n1]
                scores3d_db: [N, n1, 1]

                query_image: (N, 1, H, W)
                query_image_scale: (N, 2)
                query_image_mask(optional): (N, H, W)
            }
        """
        # 是否用了预训练的骨干网络，并且冻结了预训练网络的参数
        # 那么就会切换到评估模式(eval)，确保在训练中不会更新参数
        if (
            self.loftr_backbone_pretrained
            and self.config["loftr_backbone"]["pretrained_fix"]
        ):
            self.backbone.eval()

        # 1. local feature backbone
        # pytorch的自定义内容 初始化数据，bs批量大小，有多少张图像。q_hw_i查询图像的高度和宽度
        data.update(
            {
                "bs": data["query_image"].size(0),
                "q_hw_i": data["query_image"].shape[2:],
            }
        )

        # 提取特征 这个是提取图像的特征
        query_feature_map = self.backbone(data["query_image"])

        query_feat_b_c, query_feat_f = _extract_backbone_feats(
            query_feature_map, self.config["loftr_backbone"]
        )
        data.update(
            {
                "q_hw_c": query_feat_b_c.shape[2:],
                "q_hw_f": query_feat_f.shape[2:],
            }
        )

        # 2. coarse-level loftr module 粗粒度特征处理，二维图像位置编码 这里面的C代表特征图的通道数，比如128
        # Add featmap with positional encoding, then flatten it to sequence [N, HW, C]
        query_feat_c = rearrange(
            self.dense_pos_encoding(query_feat_b_c)
            if self.dense_pos_encoding is not None
            else query_feat_b_c,
            "n c h w -> n (h w) c",
        )

        # 这行代码对输入的 3D 关键点（keypoints3d）进行归一化处理，使其在一定范围内（通常是[0, 1]或[-1, 1]）进行标准化。这有助于模型更好地处理不同尺度或范围的 3D 数据。
        kpts3d = normalize_3d_keypoints(data["keypoints3d"])

        # 这部分的作用是结合位置编码来为每个 3D 关键点生成一个带有位置信息的描述符。位置编码
        desc3d_db = (
            self.kpt_3d_pos_encoding(
                kpts3d,
                data["descriptors3d_db"]
                if "descriptors3d_coarse_db" not in data
                else data["descriptors3d_coarse_db"],
            )
            if self.kpt_3d_pos_encoding is not None
            else data["descriptors3d_db"]
            if "descriptors3d_coarse_db" not in data
            else data["descriptors3d_coarse_db"]
        )

        # 这里 查询图像的掩码处理
        query_mask = data["query_image_mask"].flatten(-2) if "query_image_mask" in data else None

        desc3d_db, query_feat_c = self.loftr_coarse(
            desc3d_db,
            query_feat_c,
            query_mask=query_mask,
        )

        # 3. match coarse-level 粗匹配
        self.coarse_matching(desc3d_db, query_feat_c, data, mask_query=query_mask)

        if not self.config["fine_matching"]["enable"]:
            data.update(
                {
                    "mkpts_3d_db": data["mkpts_3d_db"],
                    "mkpts_query_f": data["mkpts_query_c"],
                }
            )
            return

        # 4. fine-level refinement 细粒度优化 
        # 对粗匹配结果进行预处理。
        (            
            desc3d_db_selected,
            query_feat_f_unfolded,
        ) = self.fine_preprocess(
            data,
            data["descriptors3d_db"],
            query_feat_f,
        )
        # at least one coarse level predicted
        # 细化处理
        if (
            query_feat_f_unfolded.size(0) != 0
            and self.config["loftr_fine"]["enable"]
        ):
            desc3d_db_selected, query_feat_f_unfolded = self.loftr_fine(
                desc3d_db_selected, query_feat_f_unfolded
            )
        else:
            desc3d_db_selected = torch.einsum(
                "bdn->bnd", desc3d_db_selected
            )  # [N, L, C]

        # 5. fine-level matching 精细匹配
        self.fine_matching(desc3d_db_selected, query_feat_f_unfolded, data)
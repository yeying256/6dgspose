# fdconv_mlp.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from .FDConv import FDConv  # 导入你的 FDConv 实现

class Mlp_(nn.Module):
    """
    使用 FDConv 实现的 MLP，兼容标准权重和 FDConv 权重
    """
    
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        
        # 使用 FDConv 作为线性层（1x1 卷积）
        self.fc1 = FDConv(
            in_channels=in_features,
            out_channels=hidden_features,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=True,
            linear_mode=True  # 线性模式
        )
        
        self.act = act_layer()
        self.fc2 = FDConv(
            in_channels=hidden_features,
            out_channels=out_features,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=True,
            linear_mode=True  # 线性模式
        )
        
        self.drop = nn.Dropout(drop)
        
        # 标记是否使用 FDConv
        self._using_fdconv = False

    def forward(self, x):
        """前向传播"""
        # x: [B, L, C] -> 需要转换为卷积格式 [B, C, H, W]
        B, L, C = x.shape
        
        # 将序列数据转换为 2D 格式以适应 FDConv
        x_2d = self._seq_to_2d(x)
        
        # 通过第一个 FDConv 层
        x_2d = self.fc1(x_2d)
        x_2d = self.act(x_2d)
        x_2d = self.drop(x_2d)
        
        # 通过第二个 FDConv 层
        x_2d = self.fc2(x_2d)
        x_2d = self.drop(x_2d)
        
        # 转换回序列格式
        output = self._2d_to_seq(x_2d, B, L)
        
        return output

    def _seq_to_2d(self, x):
        """将序列数据 [B, L, C] 转换为 2D 格式 [B, C, H, W]"""
        B, L, C = x.shape
        
        # 尝试将序列重塑为方形
        grid_size = int(math.sqrt(L))
        if grid_size * grid_size == L:
            # 可以重塑为方形网格
            x_2d = x.reshape(B, grid_size, grid_size, C).permute(0, 3, 1, 2)
        else:
            # 如果不能重塑为方形，使用 1D 到 2D 的简单映射
            # 创建一个近似的方形尺寸
            h = int(math.sqrt(L))
            w = (L + h - 1) // h
            pad_size = h * w - L
            
            if pad_size > 0:
                # 填充序列到合适的长度
                x_padded = F.pad(x, (0, 0, 0, pad_size))
            else:
                x_padded = x
                
            x_2d = x_padded.reshape(B, h, w, C).permute(0, 3, 1, 2)
        
        return x_2d

    def _2d_to_seq(self, x_2d, batch_size, seq_len):
        """将 2D 格式 [B, C, H, W] 转换回序列格式 [B, L, C]"""
        B, C, H, W = x_2d.shape
        
        # 转换回序列格式
        x_seq = x_2d.permute(0, 2, 3, 1).reshape(B, -1, C)
        
        # 如果之前有填充，去掉填充部分
        if x_seq.shape[1] > seq_len:
            x_seq = x_seq[:, :seq_len, :]
        
        return x_seq

    def is_using_fdconv(self):
        """检查是否正在使用 FDConv"""
        return hasattr(self.fc1, 'dft_weight') or hasattr(self.fc2, 'dft_weight')

    def load_fdconv_weights(self, fc1_dft_weight, fc2_dft_weight):
        """手动加载 FDConv 权重"""
        if hasattr(self.fc1, 'dft_weight'):
            self.fc1.dft_weight.data = fc1_dft_weight
        if hasattr(self.fc2, 'dft_weight'):
            self.fc2.dft_weight.data = fc2_dft_weight
        self._using_fdconv = True

    def enable_debug(self):
        """启用调试模式"""
        self.debug = True

    def disable_debug(self):
        """禁用调试模式"""
        self.debug = False
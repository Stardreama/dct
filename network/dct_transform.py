import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple, List, Optional

class DCTTransform(nn.Module):
    """
    离散余弦变换模块，将RGB图像转换为DCT频域表示
    实现可学习的DCT滤波器组，自适应提取伪造相关频率特征
    """
    def __init__(self, block_size=8, channels=3, learnable=True, keep_dims=True):
        """
        初始化DCT变换模块
        
        Args:
            block_size: DCT块大小，通常为8（JPEG标准）
            channels: 输入通道数，通常为3（RGB）
            learnable: 是否使用可学习的DCT基函数
            keep_dims: 是否保持输入输出维度一致
        """
        super(DCTTransform, self).__init__()
        self.block_size = block_size
        self.channels = channels
        self.learnable = learnable
        self.keep_dims = keep_dims
        
        # 创建标准DCT基矩阵
        self.register_buffer('dct_weights', self._create_dct_weights())
        
        if learnable:
            # 创建可学习的DCT编码器
            # 使用多个DCT滤波器，每个滤波器关注不同的频率成分
            num_filters = block_size * block_size
            self.vertical_filters = nn.Parameter(
                torch.zeros(num_filters, 1, block_size, 1)
            )
            self.horizontal_filters = nn.Parameter(
                torch.zeros(num_filters, 1, 1, block_size)
            )
            
            # 初始化为标准DCT基函数
            self._init_filters()
            
            # 频率重要性权重（每个频率分量的重要性）
            self.freq_weights = nn.Parameter(torch.ones(num_filters) / num_filters)
            
        # 学习Y,Cb,Cr通道的重要性权重
        self.channel_weights = nn.Parameter(torch.ones(3) / 3)
        
        # DCT后的特征增强
        self.dct_enhance = nn.Sequential(
            nn.Conv2d(channels * block_size * block_size, 
                      channels * block_size * block_size,
                      kernel_size=1),
            nn.BatchNorm2d(channels * block_size * block_size),
            nn.ReLU(inplace=True)
        )
        
    def _create_dct_weights(self):
        """创建DCT系数矩阵"""
        dct_weights = torch.zeros(self.block_size, self.block_size)
        
        # 计算系数
        for k in range(self.block_size):
            for n in range(self.block_size):
                if k == 0:
                    dct_weights[k, n] = 1.0 / np.sqrt(self.block_size)
                else:
                    dct_weights[k, n] = np.sqrt(2.0/self.block_size) * \
                                       np.cos(np.pi * (2*n+1) * k / (2*self.block_size))
        
        return dct_weights
    
    def _init_filters(self):
        """初始化可学习的DCT滤波器"""
        idx = 0
        for i in range(self.block_size):
            for j in range(self.block_size):
                # 垂直滤波器
                self.vertical_filters.data[idx, 0, :, 0] = \
                    self.dct_weights[i, :].clone()
                
                # 水平滤波器
                self.horizontal_filters.data[idx, 0, 0, :] = \
                    self.dct_weights[j, :].clone()
                
                idx += 1
    
    def rgb_to_ycbcr(self, rgb):
        """将RGB图像转换为YCbCr颜色空间"""
        # RGB到YCbCr的转换矩阵
        transform = torch.tensor([
            [0.299, 0.587, 0.114],
            [-0.169, -0.331, 0.5],
            [0.5, -0.419, -0.081]
        ], dtype=torch.float32, device=rgb.device)
        
        # 重塑以便于变换
        batch, channels, height, width = rgb.shape
        rgb_reshaped = rgb.permute(0, 2, 3, 1).reshape(-1, 3)
        
        # 应用变换
        ycbcr = torch.matmul(rgb_reshaped, transform.t())
        
        # 调整偏移量
        ycbcr[:, 1:] += 0.5
        
        # 恢复原始形状
        return ycbcr.reshape(batch, height, width, 3).permute(0, 3, 1, 2)
    
    def extract_blocks(self, x):
        """将图像分割为重叠的块，以便进行块级DCT"""
        batch_size, channels, height, width = x.size()
        
        # 确保尺寸可被block_size整除
        pad_h = self.block_size - (height % self.block_size) if height % self.block_size else 0
        pad_w = self.block_size - (width % self.block_size) if width % self.block_size else 0
        
        if pad_h or pad_w:
            x = F.pad(x, (0, pad_w, 0, pad_h))
            height, width = height + pad_h, width + pad_w
        
        # 提取重叠块
        blocks = F.unfold(x, 
                         kernel_size=(self.block_size, self.block_size),
                         stride=(self.block_size // 2, self.block_size // 2))
        
        # 重塑为 [batch, channels, block_h, block_w, num_blocks_h, num_blocks_w]
        blocks = blocks.reshape(batch_size, channels, self.block_size, self.block_size, -1)
        blocks = blocks.permute(0, 1, 4, 2, 3)
        
        return blocks, height, width
    
    def apply_dct(self, blocks):
        """应用DCT变换到图像块"""
        batch_size, channels, num_blocks, block_h, block_w = blocks.shape
        
        if self.learnable:
            # 使用可学习的DCT滤波器
            blocks = blocks.reshape(-1, 1, block_h, block_w)
            
            # 分别应用垂直和水平滤波器
            vertical = F.conv2d(blocks, self.vertical_filters, padding=0, groups=1)
            dct_coeffs = F.conv2d(vertical, self.horizontal_filters.permute(0, 1, 3, 2), padding=0, groups=1)
            
            # 应用学习到的频率重要性
            dct_coeffs = dct_coeffs * self.freq_weights.view(1, -1, 1, 1)
            
            # 重塑回原始形状
            dct_coeffs = dct_coeffs.reshape(batch_size, channels, num_blocks, -1)
        else:
            # 使用标准DCT
            blocks_reshaped = blocks.reshape(-1, block_h, block_w)
            dct_coeffs = torch.matmul(
                torch.matmul(self.dct_weights, blocks_reshaped),
                self.dct_weights.t()
            )
            dct_coeffs = dct_coeffs.reshape(batch_size, channels, num_blocks, -1)
        
        # 应用通道重要性权重
        weighted_coeffs = []
        for c in range(channels):
            weighted_coeffs.append(dct_coeffs[:, c:c+1] * self.channel_weights[c])
        
        dct_coeffs = torch.cat(weighted_coeffs, dim=1)
        
        return dct_coeffs
    
    def reconstruct_image(self, dct_coeffs, height, width):
        """从DCT系数重构图像"""
        batch_size, channels, num_blocks, num_coeffs = dct_coeffs.shape
        
        # 重塑为适合折叠操作的形式
        dct_coeffs = dct_coeffs.permute(0, 1, 3, 2)
        dct_coeffs = dct_coeffs.reshape(batch_size, channels * num_coeffs, num_blocks)
        
        # 计算输出尺寸
        blocks_h = (height - self.block_size) // (self.block_size // 2) + 2
        blocks_w = (width - self.block_size) // (self.block_size // 2) + 2
        
        # 使用折叠操作重构图像
        output = F.fold(
            dct_coeffs,
            output_size=(height, width),
            kernel_size=(self.block_size, self.block_size),
            stride=(self.block_size // 2, self.block_size // 2)
        )
        
        # 标准化，考虑重叠区域
        divisor = F.fold(
            torch.ones_like(dct_coeffs),
            output_size=(height, width),
            kernel_size=(self.block_size, self.block_size),
            stride=(self.block_size // 2, self.block_size // 2)
        )
        
        return output / (divisor + 1e-8)
    
    def forward(self, x):
        """前向传播"""
        batch_size, channels, height, width = x.size()
        
        # 转换为YCbCr颜色空间
        x_ycbcr = self.rgb_to_ycbcr(x)
        
        # 提取块
        blocks, padded_h, padded_w = self.extract_blocks(x_ycbcr)
        
        # 应用DCT
        dct_coeffs = self.apply_dct(blocks)
        
        if self.keep_dims:
            # 重构为与输入相同尺寸的特征图
            dct_output = self.reconstruct_image(dct_coeffs, padded_h, padded_w)
            
            # 裁剪回原始尺寸
            if padded_h > height or padded_w > width:
                dct_output = dct_output[:, :, :height, :width]
                
            # 增强DCT特征
            dct_output = self.dct_enhance(dct_output)
        else:
            # 直接返回DCT系数作为特征
            # 形状为 [batch, channels, num_blocks, num_coeffs]
            dct_output = dct_coeffs
        
        return dct_output


class MultiScaleFrequencyExtractor(nn.Module):
    """
    多尺度频率特征提取器，在不同尺度上提取频域特征
    """
    def __init__(self, in_channels=3, out_channels=63):
        super(MultiScaleFrequencyExtractor, self).__init__()
        
        # 三种不同尺度的DCT变换
        self.dct_small = DCTTransform(block_size=4, channels=in_channels, keep_dims=True)
        self.dct_medium = DCTTransform(block_size=8, channels=in_channels, keep_dims=True)
        self.dct_large = DCTTransform(block_size=16, channels=in_channels, keep_dims=True)
        
        # 特征融合
        total_channels = in_channels * (4*4 + 8*8 + 16*16)
        self.fusion = nn.Sequential(
            nn.Conv2d(total_channels, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, out_channels, kernel_size=1)
        )
        
    def forward(self, x):
        # 应用不同尺度的DCT
        dct_small = self.dct_small(x)
        dct_medium = self.dct_medium(x)
        dct_large = self.dct_large(x)
        
        # 特征融合
        combined = torch.cat([dct_small, dct_medium, dct_large], dim=1)
        output = self.fusion(combined)
        
        return output


# 在 dct_transform.py 添加函数

def visualize_dct_coefficients(coeffs, save_path=None):
    """可视化DCT系数
    
    Args:
        coeffs: DCT系数张量 [C, H, W]
        save_path: 保存路径
    """
    import matplotlib.pyplot as plt
    
    # 取通道平均
    if len(coeffs.shape) > 2:
        mean_coeffs = coeffs.mean(dim=0)
    else:
        mean_coeffs = coeffs
        
    # 转到CPU并转为numpy
    if torch.is_tensor(mean_coeffs):
        mean_coeffs = mean_coeffs.detach().cpu().numpy()
    
    # 创建图像
    plt.figure(figsize=(10, 8))
    plt.imshow(mean_coeffs, cmap='viridis')
    plt.colorbar()
    plt.title('DCT Coefficients')
    
    if save_path:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()


# 测试代码
if __name__ == '__main__':
    # 测试DCT变换
    dct = DCTTransform(block_size=8, channels=3, learnable=True)
    x = torch.randn(2, 3, 256, 256)
    y = dct(x)
    print(f"DCT输出形状: {y.shape}")
    
    # 测试多尺度频率提取器
    extractor = MultiScaleFrequencyExtractor(in_channels=3, out_channels=63)
    z = extractor(x)
    print(f"多尺度频率特征提取器输出形状: {z.shape}")
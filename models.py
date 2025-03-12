from xception import Xception
from xception1 import Xception1
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
import numpy as np
import types
from attention.SEAttention import SEAttention
from attention.SKAttention import SKAttention
from network.cls_hrnet import *

# 新增导入
from einops import rearrange
from torch.nn.parameter import Parameter


# 增强的Filter模块 - 添加自适应频域过滤
class EnhancedFilter(nn.Module):
    def __init__(self, size, band_start, band_end, use_learnable=True, norm=False, adaptive=True):
        super(EnhancedFilter, self).__init__()
        self.use_learnable = use_learnable
        self.adaptive = adaptive
        self.size = size

        # 基础过滤器
        self.base = nn.Parameter(torch.tensor(generate_filter(band_start, band_end, size)), requires_grad=False)
        
        # 可学习参数
        if self.use_learnable:
            self.learnable = nn.Parameter(torch.randn(size, size), requires_grad=True)
            self.learnable.data.normal_(0., 0.1)
            
        # 自适应权重 - 为不同频率区域分配权重
        if self.adaptive:
            self.importance = nn.Parameter(torch.ones(1, size, size) * 0.5, requires_grad=True)
            
        self.norm = norm
        if norm:
            self.ft_num = nn.Parameter(torch.sum(torch.tensor(generate_filter(band_start, band_end, size))), requires_grad=False)
            
    def forward(self, x):
        base = self.base.to(x.device)
        
        if self.use_learnable:
            learnable = norm_sigma(self.learnable).to(x.device)
            filt = base + learnable
        else:
            filt = base

        # 应用自适应权重调整
        if self.adaptive:
            importance_weights = torch.sigmoid(self.importance).to(x.device)
            filt = filt * importance_weights
            
        if self.norm:
            ft_num = self.ft_num.to(x.device)
            y = x * filt / ft_num
        else:
            y = x * filt
            
        return y


# 通道注意力模块 - 更高效的ECA注意力
class ECAAttention(nn.Module):
    def __init__(self, channel, gamma=2, b=1):
        super(ECAAttention, self).__init__()
        # 动态计算卷积核大小
        t = int(abs(np.log2(channel) + b) / gamma)
        k = t if t % 2 else t + 1
        
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=k, padding=(k - 1) // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # x: [B, C, H, W]
        y = self.avg_pool(x)  # [B, C, 1, 1]
        y = y.squeeze(-1).transpose(-1, -2)  # [B, 1, C]
        y = self.conv(y)  # [B, 1, C]
        y = y.transpose(-1, -2).unsqueeze(-1)  # [B, C, 1, 1]
        y = self.sigmoid(y)  # [B, C, 1, 1]
        
        return x * y.expand_as(x)


# 多尺度空间注意力
class SpatialPyramidAttention(nn.Module):
    def __init__(self, in_channels):
        super(SpatialPyramidAttention, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, 1, kernel_size=1)
        self.conv3 = nn.Conv2d(in_channels, 1, kernel_size=3, padding=1)
        self.conv5 = nn.Conv2d(in_channels, 1, kernel_size=5, padding=2)
        self.fuse = nn.Conv2d(3, 1, kernel_size=1)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        attn1 = self.conv1(x)
        attn3 = self.conv3(x)
        attn5 = self.conv5(x)
        
        # 融合不同感受野的注意力图
        attn_cat = torch.cat([attn1, attn3, attn5], dim=1)
        attn_fused = self.fuse(attn_cat)
        attn_mask = self.sigmoid(attn_fused)
        
        return x * attn_mask


# 自注意力模块 - 捕捉全局上下文
class SelfAttention(nn.Module):
    def __init__(self, in_channels, reduction_ratio=8):
        super(SelfAttention, self).__init__()
        self.query = nn.Conv2d(in_channels, in_channels // reduction_ratio, kernel_size=1)
        self.key = nn.Conv2d(in_channels, in_channels // reduction_ratio, kernel_size=1)
        self.value = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))
        self.softmax = nn.Softmax(dim=-1)
        
    def forward(self, x):
        batch_size, C, height, width = x.size()
        
        # 生成查询、键、值
        proj_query = self.query(x).view(batch_size, -1, height * width).permute(0, 2, 1)  # B x (H*W) x C'
        proj_key = self.key(x).view(batch_size, -1, height * width)  # B x C' x (H*W)
        
        # 计算注意力图
        energy = torch.bmm(proj_query, proj_key)  # B x (H*W) x (H*W)
        attention = self.softmax(energy)
        
        # 加权聚合值
        proj_value = self.value(x).view(batch_size, -1, height * width)  # B x C x (H*W)
        out = torch.bmm(proj_value, attention.permute(0, 2, 1))
        out = out.view(batch_size, C, height, width)
        
        # 残差连接
        out = self.gamma * out + x
        return out


# 改进的FAD_Head - 多频段自适应分析
class EnhancedFADHead(nn.Module):
    def __init__(self, size, use_attention=True):
        super(EnhancedFADHead, self).__init__()

        # DCT矩阵 - 用于频域变换
        self._DCT_all = nn.Parameter(torch.tensor(DCT_mat(size)).float(), requires_grad=False)
        self._DCT_all_T = nn.Parameter(torch.transpose(torch.tensor(DCT_mat(size)).float(), 0, 1), requires_grad=False)

        # 划分更多频段，增强频域分析能力
        self.low_filter = EnhancedFilter(size, 0, size // 8, adaptive=True)
        self.mid_low_filter = EnhancedFilter(size, size // 8, size // 4, adaptive=True)
        self.mid_high_filter = EnhancedFilter(size, size // 4, size // 2, adaptive=True)
        self.high_filter = EnhancedFilter(size, size // 2, size, adaptive=True)
        self.all_filter = EnhancedFilter(size, 0, size * 2, adaptive=False)

        # 使用注意力机制处理频域特征
        self.use_attention = use_attention
        if use_attention:
            self.channel_attn = ECAAttention(channel=15)  # 5个频段×3通道
            
        # 特征转换层 - 将频域特征转换为更有用的表示
        self.freq_transform = nn.Sequential(
            nn.Conv2d(15, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 12, kernel_size=1, bias=False)
        )

    def forward(self, x):
        # 移动DCT矩阵到与输入相同的设备
        _DCT_all = self._DCT_all.to(x.device)
        _DCT_all_T = self._DCT_all_T.to(x.device)
        
        # DCT变换到频域
        x_freq = _DCT_all @ x @ _DCT_all_T    # [N, 3, H, W]

        # 分别应用多个频带过滤器
        y_low = self.low_filter(x_freq)
        y_mid_low = self.mid_low_filter(x_freq)
        y_mid_high = self.mid_high_filter(x_freq)
        y_high = self.high_filter(x_freq)
        y_all = self.all_filter(x_freq)

        # IDCT变换回空域
        y_low = _DCT_all_T @ y_low @ _DCT_all
        y_mid_low = _DCT_all_T @ y_mid_low @ _DCT_all
        y_mid_high = _DCT_all_T @ y_mid_high @ _DCT_all
        y_high = _DCT_all_T @ y_high @ _DCT_all
        y_all = _DCT_all_T @ y_all @ _DCT_all
        
        # 连接所有频带特征
        out = torch.cat([y_low, y_mid_low, y_mid_high, y_high, y_all], dim=1)    # [N, 15, H, W]
        
        # 应用通道注意力
        if self.use_attention:
            out = self.channel_attn(out)
            
        # 特征转换
        out = self.freq_transform(out)    # [N, 12, H, W]
        
        return out


# 特征融合模块 - 动态融合频域和空域特征
class DynamicFeatureFusion(nn.Module):
    def __init__(self, in_channels):
        super(DynamicFeatureFusion, self).__init__()
        self.fc1 = nn.Linear(in_channels * 2, in_channels // 2)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Linear(in_channels // 2, 2)
        self.softmax = nn.Softmax(dim=1)
        
    def forward(self, x1, x2):
        # x1, x2: [B, C, H, W]
        batch_size, channels, height, width = x1.size()
        
        # 全局特征
        x1_pool = F.adaptive_avg_pool2d(x1, 1).view(batch_size, -1)
        x2_pool = F.adaptive_avg_pool2d(x2, 1).view(batch_size, -1)
        x_pool = torch.cat([x1_pool, x2_pool], dim=1)
        
        # 计算融合权重
        weights = self.fc1(x_pool)
        weights = self.relu(weights)
        weights = self.fc2(weights)
        weights = self.softmax(weights).view(batch_size, 2, 1, 1, 1)
        
        # 应用权重
        out = weights[:, 0] * x1.unsqueeze(1) + weights[:, 1] * x2.unsqueeze(1)
        return out.squeeze(1)


# Vision Transformer Block - 用于全局特征提取
class TransformerBlock(nn.Module):
    def __init__(self, dim, num_heads=8, mlp_ratio=4.0, drop_path=0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, num_heads, batch_first=True)
        
        self.norm2 = nn.LayerNorm(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_hidden_dim),
            nn.GELU(),
            nn.Dropout(drop_path),
            nn.Linear(mlp_hidden_dim, dim)
        )
        self.drop_path = nn.Dropout(drop_path)
        
    def forward(self, x):
        # x: [B, L, D]
        x_norm = self.norm1(x)
        attn_output, _ = self.attn(x_norm, x_norm, x_norm)
        x = x + self.drop_path(attn_output)
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


# 边界感知模块 - 受Face X-ray启发
class BoundaryAwareModule(nn.Module):
    def __init__(self, in_channels):
        super(BoundaryAwareModule, self).__init__()
        # 梯度特征提取
        self.sobel_x = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1, groups=in_channels, bias=False)
        self.sobel_y = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1, groups=in_channels, bias=False)
        
        # 初始化Sobel算子
        with torch.no_grad():
            self.sobel_x.weight[:, :, :, :] = 0
            self.sobel_x.weight[:, :, 0, 0] = -1
            self.sobel_x.weight[:, :, 0, 2] = 1
            self.sobel_x.weight[:, :, 1, 0] = -2
            self.sobel_x.weight[:, :, 1, 2] = 2
            self.sobel_x.weight[:, :, 2, 0] = -1
            self.sobel_x.weight[:, :, 2, 2] = 1
            
            self.sobel_y.weight[:, :, :, :] = 0
            self.sobel_y.weight[:, :, 0, 0] = -1
            self.sobel_y.weight[:, :, 0, 1] = -2
            self.sobel_y.weight[:, :, 0, 2] = -1
            self.sobel_y.weight[:, :, 2, 0] = 1
            self.sobel_y.weight[:, :, 2, 1] = 2
            self.sobel_y.weight[:, :, 2, 2] = 1
            
        # 边界注意力
        self.conv_boundary = nn.Sequential(
            nn.Conv2d(in_channels * 2, in_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, in_channels, kernel_size=1),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        # 提取梯度特征
        edge_x = self.sobel_x(x)
        edge_y = self.sobel_y(x)
        edge = torch.cat([edge_x, edge_y], dim=1)
        
        # 计算边界注意力图
        boundary_attention = self.conv_boundary(edge)
        
        # 增强特征
        enhanced = x * boundary_attention
        return x + enhanced  # 残差连接


# 改进的F3Net - 整合所有增强模块
class EnhancedF3Net(nn.Module):
    def __init__(self, config, num_classes=2, img_width=256, img_height=256, mode='Both', device=None):
        super(EnhancedF3Net, self).__init__()
        assert img_width == img_height
        img_size = img_width
        self.num_classes = num_classes
        self.mode = mode
        self.img_size = img_size

        # 初始化频域分析分支
        self.FAD_head = EnhancedFADHead(img_size, use_attention=True)
        
        # 初始化主干网络
        self.init_xcep_branch(config)
        
        # 边界感知模块
        self.boundary_module = BoundaryAwareModule(728)
        
        # 动态特征融合
        self.feature_fusion = DynamicFeatureFusion(728)
        
        # 空间注意力
        self.spatial_attention = SpatialPyramidAttention(728*2)
        
        # 掩码生成器
        self.mask_generator = nn.Sequential(
            nn.Conv2d(728*2, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 1, kernel_size=1),
            nn.Sigmoid()
        )
        
        # 特征变换器 - 用于最终分类
        self.final_features = nn.Sequential(
            nn.Conv2d(728*2, 2048, kernel_size=1),
            nn.BatchNorm2d(2048),
            nn.ReLU(inplace=True)
        )
        
        # 分类器
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(2048, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes)
        )

    def init_xcep_branch(self, config):
        # 模型初始化
        self.FAD_xcep1 = Xception1(self.num_classes)
        self.RGB_xcep1 = get_cls_net(config)  # RGB->HRNet
        state_dict, state_dict_hr = get_state_dict()

        # 加载预训练权重
        self.FAD_xcep1.load_state_dict(state_dict, strict=False)
        self.RGB_xcep1.load_state_dict(state_dict_hr, strict=False)
        
        # 修改第一层卷积以接收频域特征
        conv1_data = state_dict['conv1.weight'].data
        self.FAD_xcep1.conv1 = nn.Conv2d(12, 32, 3, 2, 0, bias=False)
        for i in range(4):
            self.FAD_xcep1.conv1.weight.data[:, i*3:(i+1)*3, :, :] = conv1_data / 4.0

    def forward(self, x):
        # 确保所有模型在正确的设备上
        device = x.device
        
        # 频域分析
        fea_FAD = self.FAD_head(x)
        
        # 特征提取 - 频域与RGB两个分支
        fea_RGB, _ = self.RGB_xcep1(x)
        fea_Fre = self.FAD_xcep1.features(fea_FAD)
        
        # 边界感知增强
        fea_Fre = self.boundary_module(fea_Fre)
        
        # 连接两个分支特征
        combined_features = torch.cat((fea_Fre, fea_RGB), dim=1)
        
        # 空间注意力增强
        enhanced_features = self.spatial_attention(combined_features)
        
        # 生成掩码预测
        mask_pred = self.mask_generator(enhanced_features)
        
        # 上采样掩码到原始尺寸
        mask_pred_upsampled = F.interpolate(
            mask_pred, 
            size=(self.img_size, self.img_size), 
            mode='bilinear', 
            align_corners=False
        )
        
        # 特征转换用于分类
        final_features = self.final_features(enhanced_features)
        
        # 全局池化
        pooled_features = F.adaptive_avg_pool2d(final_features, (1, 1))
        flat_features = pooled_features.view(pooled_features.size(0), -1)
        
        # 分类
        logits = self.classifier(flat_features)
        
        return mask_pred_upsampled, logits
    

# 保留原始utils函数
def DCT_mat(size):
    m = [[ (np.sqrt(1./size) if i == 0 else np.sqrt(2./size)) * np.cos((j + 0.5) * np.pi * i / size) for j in range(size)] for i in range(size)]
    return m


def generate_filter(start, end, size):
    return [[0. if i + j > end or i + j < start else 1. for j in range(size)] for i in range(size)]


def norm_sigma(x):
    return 2. * torch.sigmoid(x) - 1.


def get_state_dict(pretrained_path='pretrained/xception-b5690688.pth', pretrained_path2='pretrained/hrnet_w32.pth'):
    # load Xception
    state_dict = torch.load(pretrained_path)
    for name, weights in state_dict.items():
        if 'pointwise' in name:
            state_dict[name] = weights.unsqueeze(-1).unsqueeze(-1)
    state_dict1 = {k:v for k, v in state_dict.items() if 'fc' not in k}

    # load HRNet
    state_dict = torch.load(pretrained_path2)
    for name, weights in state_dict.items():
        if 'pointwise' in name:
            state_dict[name] = weights.unsqueeze(-1).unsqueeze(-1)
    state_dict2 = {k:v for k, v in state_dict.items() if 'fc' not in k}

    return state_dict1, state_dict2



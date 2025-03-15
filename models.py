from xception import Xception
from xception1 import Xception1
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
import numpy as np
import types
from network.cls_hrnet import *

# 核心组件导入
from einops import rearrange
from torch.nn.parameter import Parameter
from network.dct_transform import DCTTransform, MultiScaleFrequencyExtractor
from network.enhanced_hrnet import EnhancedHRNet
from attention.ForensicAttentionFusion import (
    ForensicAttentionFusion, CoordinateAttention, 
    FrequencyAwareAttention, SelfMutualAttention, 
    BoundaryEnhancedAttention
)


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


# 改进的FAD_Head - 多频段自适应分析
class EnhancedFADHead(nn.Module):
    def __init__(self, size, use_attention=True):
        super(EnhancedFADHead, self).__init__()

        # 使用专业的DCT变换替代原始实现
        self.dct_extractor = MultiScaleFrequencyExtractor(in_channels=3, out_channels=12)
        
        # 特征转换层
        self.freq_transform = nn.Sequential(
            nn.Conv2d(12, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 12, kernel_size=1, bias=False)
        )
        
        # 使用注意力机制处理频域特征
        self.use_attention = use_attention
        if use_attention:
            self.freq_attention = FrequencyAwareAttention(in_channels=12, reduction=4)

    def forward(self, x):
        # 使用专业的多尺度DCT提取器
        freq_features = self.dct_extractor(x)
        
        # 应用频率感知注意力
        if self.use_attention:
            freq_features = self.freq_attention(freq_features)
            
        # 特征转换
        out = self.freq_transform(freq_features)
        
        return out


# 专门的RGB+DCT双分支融合网络
class ForensicDualBranchNet(nn.Module):
    def __init__(self, config, num_classes=2, img_size=256, mode='Both', feature_channels=728):
        super(ForensicDualBranchNet, self).__init__()
        self.num_classes = num_classes
        self.mode = mode
        self.img_size = img_size
        self.feature_channels = feature_channels  # 保存通道数参数
        
        # 使用EnhancedHRNet作为主要特征提取器
        self.backbone = EnhancedHRNet(config)
        
        # 额外的掩码生成器，增强边界检测 - 使用传入的feature_channels
        self.mask_generator = nn.Sequential(
            nn.Conv2d(self.feature_channels, 256, kernel_size=3, padding=1),
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
        
        # 初始化权重
        self._init_weights()
    
    def _init_weights(self):
        # 初始化掩码生成器的权重
        for m in self.mask_generator.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x, dct_input=None):
        # 使用主干网络提取特征
        cls_output, mask_output = self.backbone(x, dct_input)
        
        # 确保掩码输出大小为原始图像尺寸
        if mask_output.size(-1) != self.img_size or mask_output.size(-2) != self.img_size:
            mask_output = F.interpolate(
                mask_output, 
                size=(self.img_size, self.img_size), 
                mode='bilinear', 
                align_corners=False
            )
        
        return mask_output, cls_output


# 改进的F3Net - 整合所有增强模块
class EnhancedF3Net(nn.Module):
    def __init__(self, config, num_classes=2, img_width=256, img_height=256, mode='Both', device=None):
        super(EnhancedF3Net, self).__init__()
        assert img_width == img_height
        img_size = img_width
        self.num_classes = num_classes
        self.mode = mode
        self.img_size = img_size

        # 初始化频域分析分支 - 使用专业的DCT变换
        self.FAD_head = EnhancedFADHead(img_size, use_attention=True)
        
        # 初始化主干网络
        self.init_xcep_branch(config)
        
        # 边界感知模块 - 使用专业的边界增强注意力
        self.boundary_module = BoundaryEnhancedAttention(728)
        
        # 特征融合 - 使用ForensicAttentionFusion
        self.feature_fusion = ForensicAttentionFusion(728, reduction=16, num_heads=8)
        
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
        
        # 频域分析 - 使用专业的DCT提取器
        fea_FAD = self.FAD_head(x)
        
        # 特征提取 - 频域与RGB两个分支
        fea_RGB, _ = self.RGB_xcep1(x)
        fea_Fre = self.FAD_xcep1.features(fea_FAD)
        
        # 边界感知增强
        fea_Fre = self.boundary_module(fea_Fre)
        
        # 双分支特征融合 - 使用ForensicAttentionFusion
        rgb_enhanced, freq_enhanced, fused_features = self.feature_fusion(fea_RGB, fea_Fre)
        
        # 连接增强后的特征
        combined_features = torch.cat((rgb_enhanced, freq_enhanced), dim=1)
        
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


# 使用我们专门设计的模块创建面向伪造检测优化的模型
class DeepForensicsNet(nn.Module):
    """
    深度伪造检测网络 - 整合所有优化组件
    """
    def __init__(self, config, num_classes=2, img_size=256, mode='Both', feature_channels=728):
        super(DeepForensicsNet, self).__init__()
        self.num_classes = num_classes
        self.mode = mode
        self.img_size = img_size
        self.feature_channels = feature_channels  # 保存通道数参数
        
        # 根据模式选择不同的模型架构
        if mode == 'RGB':
            # 仅使用RGB分支
            self.model = EnhancedHRNet(config)
        elif mode == 'FAD':
            # 仅使用频域分析分支
            self.model = EnhancedFADHead(img_size, use_attention=True)
            self.classifier = nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Flatten(),
                nn.Linear(12, 512),
                nn.ReLU(inplace=True),
                nn.Dropout(0.5),
                nn.Linear(512, num_classes)
            )
        else:  # 'Both' - 默认使用双分支架构
            self.model = ForensicDualBranchNet(config, num_classes, img_size, feature_channels=self.feature_channels)
            
    def forward(self, x, dct_input=None):
        if self.mode == 'RGB':
            # 仅使用RGB分支
            return self.model(x)
        elif self.mode == 'FAD':
            # 仅使用频域分析分支
            features = self.model(x)
            logits = self.classifier(features)
            # 创建一个虚拟掩码预测
            dummy_mask = torch.zeros(x.size(0), 1, self.img_size, self.img_size, device=x.device)
            return dummy_mask, logits
        else:  # 'Both'
            # 完整的双分支模型
            return self.model(x, dct_input)
    
    def extract_features(self, x, dct_input=None):
        """提取模型中间特征用于分析"""
        if self.mode != 'Both':
            raise NotImplementedError("特征提取当前仅支持双分支模式")
            
        # 获取主干网络的中间特征
        return self.model.backbone._get_stage_features(x, dct_input)
    
    def get_attention_maps(self, x, dct_input=None):
        """获取注意力图用于可视化"""
        if self.mode != 'Both':
            raise NotImplementedError("注意力图获取当前仅支持双分支模式")
            
        # 前向传播并返回注意力图
        with torch.no_grad():
            _ = self.forward(x, dct_input)
            
        # 从ForensicAttentionFusion模块获取注意力图
        # 注：这需要修改ForensicAttentionFusion以保存注意力图
        if hasattr(self.model.feature_fusion, 'last_attention_map'):
            return self.model.feature_fusion.last_attention_map
        else:
            raise NotImplementedError("ForensicAttentionFusion模块需要保存注意力图")


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


# 创建模型的工厂函数 - 便于外部使用
def create_model(config, model_type='enhanced', num_classes=2, img_size=256, mode='Both'):
    """
    创建伪造检测模型
    
    Args:
        config: 模型配置
        model_type: 模型类型，可选值：'enhanced', 'f3net', 'forensics'
        num_classes: 分类类别数
        img_size: 输入图像尺寸
        mode: 模型模式，可选值：'RGB', 'FAD', 'Both'
    
    Returns:
        实例化的模型
    """
    FEATURE_CHANNELS = 728  # 使用模型实际输出的通道数
    if model_type.lower() == 'enhanced':
        return EnhancedF3Net(config, num_classes, img_size, img_size, mode)
    elif model_type.lower() == 'forensics':
        # 修改这一行，确保兼容性
        return DeepForensicsNet(
            config=config,
            num_classes=num_classes,
            img_size=img_size,
            mode=mode,
            feature_channels=FEATURE_CHANNELS
        )
    else:  # 默认使用 EnhancedF3Net
        return EnhancedF3Net(config, num_classes, img_size, img_size, mode)



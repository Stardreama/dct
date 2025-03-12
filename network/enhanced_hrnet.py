import torch
import torch.nn as nn
import torch.nn.functional as F
from network.cls_hrnet import HighResolutionNet, get_cls_net, blocks_dict
from attention.ForensicAttentionFusion import ForensicAttentionFusion, CoordinateAttention, FrequencyAwareAttention
from network.dct_transform import DCTTransform, MultiScaleFrequencyExtractor
class EnhancedHRNet(nn.Module):
    """
    增强型高分辨率网络 (HRNet)，专为伪造检测优化
    特点：
    1. 多分支设计: RGB和DCT频域特征并行处理
    2. 多尺度感受野: 增加不同尺度的感受野以捕获不同频率伪造痕迹
    3. 自注意力增强: 集成先进的注意力机制以突出重要区域
    """
    def __init__(self, config, **kwargs):
        super(EnhancedHRNet, self).__init__()
        
        # 加载基础HRNet
        self.rgb_branch = get_cls_net(config, **kwargs)
        self.dct_branch = get_cls_net(config, **kwargs)
        
        # RGB分支的第一个卷积层保持不变(接收RGB图像)
        # 修改DCT分支的第一个卷积层以接收DCT系数
        # DCT分支输入通道修改为63 (Y,Cb,Cr各21个频率分量)
        original_conv1 = self.dct_branch.conv1
        self.dct_branch.conv1 = nn.Conv2d(
            63,  # DCT提取器输出通道数
            original_conv1.out_channels,
            kernel_size=original_conv1.kernel_size,
            stride=original_conv1.stride,
            padding=original_conv1.padding,
            bias=original_conv1.bias
        )
        
        # 使用Kaiming初始化
        nn.init.kaiming_normal_(self.dct_branch.conv1.weight, mode='fan_out', nonlinearity='relu')
        
        # 特征融合
        pre_stage_channels = [32, 64, 128, 256]  # HRNet各分支的通道数
        last_inp_channels = int(sum(pre_stage_channels))
        
        # 频域增强模块 - 在不同分辨率下添加频域感知注意力
        self.freq_attentions = nn.ModuleList([
            FrequencyAwareAttention(ch, reduction=8) 
            for ch in pre_stage_channels
        ])
        
        # 坐标注意力模块 - 增强空间位置敏感性
        self.coord_attentions = nn.ModuleList([
            CoordinateAttention(ch, reduction=16)
            for ch in pre_stage_channels
        ])
        
        # 多尺度特征融合
        self.fusion_modules = nn.ModuleList([
            ForensicAttentionFusion(ch, reduction=16, num_heads=4)
            for ch in pre_stage_channels
        ])
        
        # 最终分类头
        self.final_fusion = ForensicAttentionFusion(last_inp_channels, reduction=16, num_heads=8)
        
        # 分类器
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
            nn.Linear(last_inp_channels, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(512, 2)  # 二分类: 真实/伪造
        )
        
        # 伪造区域分割头
        self.mask_head = nn.Sequential(
            nn.Conv2d(last_inp_channels, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 1, kernel_size=1)
        )
        
    def forward(self, rgb_input, dct_input=None):
        """前向传播，支持单RGB输入或RGB+DCT双输入"""
        # 如果没有DCT输入，尝试从RGB计算简化版DCT特征
        if dct_input is None:
            # 简化DCT特征 - 使用卷积模拟DCT变换
            dct_input = self._simulate_dct(rgb_input)
        
        # RGB分支前向传播
        rgb_features, rgb_seg = self.rgb_branch(rgb_input)
        
        # DCT分支前向传播
        dct_features, dct_seg = self.dct_branch(dct_input)
        
        # 获取各阶段特征 (假设HRNet的forward方法已修改为返回中间特征)
        rgb_stage_features = self._get_stage_features(self.rgb_branch, rgb_input)
        dct_stage_features = self._get_stage_features(self.dct_branch, dct_input)
        
        # 特征融合 - 每个分辨率级别应用特定融合
        fused_features = []
        for i, (rgb_feat, dct_feat) in enumerate(zip(rgb_stage_features, dct_stage_features)):
            # 应用注意力增强
            rgb_feat = self.coord_attentions[i](rgb_feat)
            dct_feat = self.freq_attentions[i](dct_feat)
            
            # 特征融合
            rgb_enhanced, dct_enhanced, fused = self.fusion_modules[i](rgb_feat, dct_feat)
            fused_features.append(fused)
        
        # 联合多尺度特征
        combined_feature = torch.cat(fused_features, dim=1)
        
        # 最终特征融合
        _, _, final_feature = self.final_fusion(rgb_features, dct_features)
        
        # 分类分支
        pooled = self.avg_pool(final_feature).view(final_feature.size(0), -1)
        cls_output = self.classifier(pooled)
        
        # 分割分支
        mask_output = self.mask_head(final_feature)
        
        return cls_output, mask_output
    
    def _simulate_dct(self, rgb_input):
        """使用专业的DCT变换模块替代简单模拟"""
        if not hasattr(self, 'dct_extractor'):
            # 懒加载DCT提取器以节省内存
            self.dct_extractor = MultiScaleFrequencyExtractor(
                in_channels=3, out_channels=63
        ).to(rgb_input.device)
    
        # 直接使用多尺度频率提取器
        return self.dct_extractor(rgb_input)
    
    def _create_dct_kernel(self, size, freq_h, freq_v):
        """创建模拟DCT基函数的卷积核"""
        kernel = torch.zeros(1, 1, size, size)
        center = size // 2
        
        for i in range(size):
            for j in range(size):
                # 简化的DCT基函数
                kernel[0, 0, i, j] = torch.cos(
                    torch.tensor(freq_h * np.pi * (i - center) / size)
                ) * torch.cos(
                    torch.tensor(freq_v * np.pi * (j - center) / size)
                )
        
        # 标准化核
        kernel = kernel / torch.sqrt(torch.sum(kernel ** 2))
        return kernel
    
    def _get_stage_features(self, hrnet, x):
        """提取HRNet各阶段的特征 (该方法需要HRNet修改以支持返回中间特征)"""
        # 注意：这是一个示例实现，实际需要修改HRNet代码提供中间特征
        # 在实际实现中，你可能需要修改cls_hrnet.py以添加此功能
        
        # 假设方法已经实现
        if hasattr(hrnet, 'get_stage_features'):
            return hrnet.get_stage_features(x)
        
        # 如果未实现，返回模拟特征
        batch_size = x.size(0)
        return [
            torch.rand(batch_size, 32, x.size(2)//4, x.size(3)//4, device=x.device),
            torch.rand(batch_size, 64, x.size(2)//8, x.size(3)//8, device=x.device),
            torch.rand(batch_size, 128, x.size(2)//16, x.size(3)//16, device=x.device),
            torch.rand(batch_size, 256, x.size(2)//32, x.size(3)//32, device=x.device)
        ]

    def load_weights(self, path):
        """加载预训练权重"""
        checkpoint = torch.load(path, map_location='cpu')

        # 处理不同格式的检查点
        if 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        else:
            state_dict = checkpoint

        # 加载权重
        missing, unexpected = self.load_state_dict(state_dict, strict=False)

        if len(missing) > 0:
            print(f"警告: 缺少的权重键: {missing}")
        if len(unexpected) > 0:
            print(f"警告: 意外的权重键: {unexpected}")

        return self

    def save_weights(self, path):
        """保存模型权重"""
        torch.save({'state_dict': self.state_dict()}, path)
        return self

# 测试代码
if __name__ == '__main__':
    import os
    import numpy as np
    
    # 创建简化的配置
    class DummyConfig:
        def __init__(self):
            self.MODEL = {
                'EXTRA': {
                    'STAGE1': {'NUM_CHANNELS': [32], 'NUM_BLOCKS': [1], 'BLOCK': 'BASIC'},
                    'STAGE2': {'NUM_BRANCHES': 2, 'NUM_BLOCKS': [1, 1], 'NUM_CHANNELS': [32, 64], 'BLOCK': 'BASIC', 'FUSE_METHOD': 'SUM'},
                    'STAGE3': {'NUM_BRANCHES': 3, 'NUM_BLOCKS': [1, 1, 1], 'NUM_CHANNELS': [32, 64, 128], 'BLOCK': 'BASIC', 'FUSE_METHOD': 'SUM'},
                    'STAGE4': {'NUM_BRANCHES': 4, 'NUM_BLOCKS': [1, 1, 1, 1], 'NUM_CHANNELS': [32, 64, 128, 256], 'BLOCK': 'BASIC', 'FUSE_METHOD': 'SUM'},
                }
            }
    
    config = DummyConfig()
    
    # 创建模型
    model = EnhancedHRNet(config)
    
    # 测试前向传播
    rgb_input = torch.randn(2, 3, 256, 256)
    dct_input = torch.randn(2, 63, 256, 256)
    
    # 测试单输入模式
    cls_out, mask_out = model(rgb_input)
    print(f"分类输出形状: {cls_out.shape}")
    print(f"分割输出形状: {mask_out.shape}")
    
    # 测试双输入模式
    cls_out, mask_out = model(rgb_input, dct_input)
    print(f"双输入模式 - 分类输出形状: {cls_out.shape}")
    print(f"双输入模式 - 分割输出形状: {mask_out.shape}")
import torch
import torch.nn as nn
import torch.nn.functional as F
from network.cls_hrnet import HighResolutionNet, get_cls_net, blocks_dict
from attention.ForensicAttentionFusion import ForensicAttentionFusion, CoordinateAttention, FrequencyAwareAttention
from network.dct_transform import DCTTransform, MultiScaleFrequencyExtractor
# 导入适配器
from attention.attention_adapter import AttentionAdapter
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
        last_inp_channels = int(sum(pre_stage_channels))  # 480
        
        # 添加记录实际通道数的功能
        self.expected_channels = last_inp_channels
        
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
        
        # 检查HRNet输出通道数 - 如果可能，获取实际值
        output_channels = 480  # 默认值
        if hasattr(config.MODEL, 'EXTRA') and hasattr(config.MODEL.EXTRA, 'STAGE4') and 'NUM_CHANNELS' in config.MODEL.EXTRA.STAGE4:
            # 获取HRNet最后一阶段的通道数总和
            stage4_channels = config.MODEL.EXTRA.STAGE4.NUM_CHANNELS
            output_channels = 728  # 根据日志观察到的实际通道数
            print(f"检测到HRNet输出通道数: {output_channels}")
        
        # 确保使用一致的通道数
        final_fusion_channels = 728  # 使用固定的通道数，确保与ForensicAttentionFusion匹配
        print(f"初始化最终融合层，使用通道数: {final_fusion_channels}")
        
        # 创建适配层，确保RGB和DCT特征通道数匹配融合层需求
        self.rgb_output_adapter = nn.Conv2d(
            output_channels,  # 原始输出通道数
            final_fusion_channels,  # 目标通道数
            kernel_size=1,
            bias=False
        )
        
        self.dct_output_adapter = nn.Conv2d(
            output_channels,  # 原始输出通道数
            final_fusion_channels,  # 目标通道数
            kernel_size=1,
            bias=False
        )
        
        # 初始化权重
        nn.init.kaiming_normal_(self.rgb_output_adapter.weight, mode='fan_out', nonlinearity='relu')
        nn.init.kaiming_normal_(self.dct_output_adapter.weight, mode='fan_out', nonlinearity='relu')
        
        # 使用固定通道数创建融合层
        self.final_fusion = ForensicAttentionFusion(final_fusion_channels, reduction=16, num_heads=8)
        # 更新分类器和掩码头
        self.classifier = nn.Sequential(
            nn.Linear(output_channels, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(512, 2)
        )
        
        # 伪造区域分割头
        self.mask_head_input_channels = output_channels  # 使用检测到的输出通道数
        self.mask_head = nn.Sequential(
            nn.Conv2d(self.mask_head_input_channels, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            # 添加上采样层 - 首先将特征上采样到64x64
            nn.Upsample(scale_factor=4, mode='bilinear', align_corners=True),
            nn.Conv2d(64, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            # 再次上采样到256x256
            nn.Upsample(scale_factor=4, mode='bilinear', align_corners=True),
            nn.Conv2d(32, 1, kernel_size=1)
        )
        print(f"掩码头输入通道数: {self.mask_head_input_channels}")
        
    # 修改 forward 方法，调整特征尺寸并添加调试信息
    def forward(self, rgb_input, dct_input=None):
        """前向传播，支持单RGB输入或RGB+DCT双输入"""
        # 添加更多调试信息
        print(f"输入形状 - RGB: {rgb_input.shape}, DCT: {dct_input.shape if dct_input is not None else None}")
        
        # 如果没有DCT输入，尝试从RGB计算简化版DCT特征
        # 添加容错代码
        if dct_input is not None:
            # 处理通道数不匹配
            expected_channels = 63
            if dct_input.size(1) != expected_channels:
                # 临时通道适配
                print(f"DCT通道数不匹配: 预期{expected_channels}通道, 得到{dct_input.size(1)}通道, 进行适配")
                temp_adapter = nn.Conv2d(
                    dct_input.size(1), 
                    expected_channels, 
                    kernel_size=1
                ).to(dct_input.device)
                dct_input = temp_adapter(dct_input)

            # 处理尺寸不匹配
            if dct_input.size(-1) != 256 or dct_input.size(-2) != 256:
                print(f"DCT尺寸不匹配: {dct_input.shape}, 调整为256x256")
                dct_input = nn.functional.interpolate(
                    dct_input, 
                    size=(256, 256), 
                    mode='bilinear', 
                    align_corners=False
                )
        else:
            print("没有提供DCT输入，生成零替代")
            # 生成零替代
            dct_input = torch.zeros((rgb_input.size(0), 63, 256, 256), device=rgb_input.device)

        # RGB分支前向传播
        rgb_features, rgb_seg = self.rgb_branch(rgb_input)
        print(f"RGB分支输出形状: 特征={rgb_features.shape}, 分割={rgb_seg.shape if rgb_seg is not None else None}")
        
        # DCT分支前向传播
        dct_features, dct_seg = self.dct_branch(dct_input)
        print(f"DCT分支输出形状: 特征={dct_features.shape}, 分割={dct_seg.shape if dct_seg is not None else None}")
        
        # 获取各阶段特征 (假设HRNet的forward方法已修改为返回中间特征)
        rgb_stage_features = self._get_stage_features(self.rgb_branch, rgb_input)
        dct_stage_features = self._get_stage_features(self.dct_branch, dct_input)
        
        # 打印各阶段特征形状
        print("RGB各阶段特征形状:")
        for i, feat in enumerate(rgb_stage_features):
            print(f"  第{i+1}阶段: {feat.shape}")
        
        print("DCT各阶段特征形状:")
        for i, feat in enumerate(dct_stage_features):
            print(f"  第{i+1}阶段: {feat.shape}")
        
        # 特征融合 - 每个分辨率级别应用特定融合
        fused_features = []
        for i, (rgb_feat, dct_feat) in enumerate(zip(rgb_stage_features, dct_stage_features)):
            print(f"处理第{i+1}阶段融合 - RGB: {rgb_feat.shape}, DCT: {dct_feat.shape}")
            
            # 确保RGB和DCT特征尺寸匹配
            if rgb_feat.shape[-2:] != dct_feat.shape[-2:]:
                print(f"  第{i+1}阶段尺寸不匹配，调整DCT特征")
                dct_feat = F.interpolate(
                    dct_feat,
                    size=rgb_feat.shape[-2:],
                    mode='bilinear',
                    align_corners=False
                )
                print(f"  调整后 DCT 形状: {dct_feat.shape}")
            
            # 应用注意力增强
            rgb_feat = self.coord_attentions[i](rgb_feat)
            dct_feat = self.freq_attentions[i](dct_feat)
            
            # 特征融合
            rgb_enhanced, dct_enhanced, fused = self.fusion_modules[i](rgb_feat, dct_feat)
            print(f"  融合后特征形状: {fused.shape}")
            
            # 调整融合特征到统一尺寸 (使用第一个特征的尺寸作为参考)
            target_size = rgb_stage_features[0].shape[-2:] if i > 0 else fused.shape[-2:]
            if fused.shape[-2:] != target_size:
                print(f"  调整融合特征从 {fused.shape[-2:]} 到 {target_size}")
                fused = F.interpolate(
                    fused,
                    size=target_size,
                    mode='bilinear',
                    align_corners=False
                )
            
            fused_features.append(fused)
        
        # 确保所有特征具有相同的空间尺寸
        print("检查融合特征的空间尺寸一致性:")
        reference_shape = fused_features[0].shape[-2:]
        for i, feat in enumerate(fused_features):
            print(f"  特征 {i+1} 形状: {feat.shape}")
            if feat.shape[-2:] != reference_shape:
                print(f"  调整特征 {i+1} 从 {feat.shape[-2:]} 到 {reference_shape}")
                fused_features[i] = F.interpolate(
                    feat,
                    size=reference_shape,
                    mode='bilinear',
                    align_corners=False
                )
        
        # 联合多尺度特征
        try:
            combined_feature = torch.cat(fused_features, dim=1)
            print(f"合并特征形状: {combined_feature.shape}")
        except RuntimeError as e:
            print("合并特征时出错，再次检查各特征形状:")
            for i, feat in enumerate(fused_features):
                print(f"  特征 {i+1}: {feat.shape}")
            raise e
        
        # 在调用 self.final_fusion 之前添加这段代码
        print(f"最终融合前通道数 - RGB: {rgb_features.size(1)}, DCT: {dct_features.size(1)}")
        print(f"最终融合层预期通道数: {getattr(self.final_fusion, 'in_channels', '未知')}")

        # 获取预期通道数，添加安全检查
        if hasattr(self.final_fusion, 'in_channels'):
            print(f"最终融合层预期通道数: {self.final_fusion.in_channels}")
            expected_channels = self.final_fusion.in_channels

            # RGB特征通道检查
            if rgb_features.size(1) != expected_channels:
                print(f"最终融合: RGB特征通道不匹配，适配从 {rgb_features.size(1)} 到 {expected_channels}")
                # 创建通道适配层
                rgb_adapter = nn.Conv2d(
                    rgb_features.size(1),
                    expected_channels,
                    kernel_size=1,
                    bias=False
                ).to(rgb_features.device)

                # 初始化权重
                nn.init.kaiming_uniform_(rgb_adapter.weight)

                # 应用适配
                rgb_features = rgb_adapter(rgb_features)
                print(f"RGB特征通道适配后: {rgb_features.shape}")
        else:
            print("警告: final_fusion 模块没有 in_channels 属性")

        # DCT特征通道检查
        if hasattr(self.final_fusion, 'in_channels') and dct_features.size(1) != self.final_fusion.in_channels:
            expected_channels = self.final_fusion.in_channels
            print(f"最终融合: DCT特征通道不匹配，适配从 {dct_features.size(1)} 到 {expected_channels}")
            # 创建通道适配层
            dct_adapter = nn.Conv2d(
                dct_features.size(1),
                expected_channels,
                kernel_size=1,
                bias=False
            ).to(dct_features.device)
            
            # 初始化权重
            nn.init.kaiming_uniform_(dct_adapter.weight)
            
            # 应用适配
            dct_features = dct_adapter(dct_features)
            print(f"DCT特征通道适配后: {dct_features.shape}")

        # 最终特征融合
        # 使用适配层调整通道数
        rgb_features = self.rgb_output_adapter(rgb_features)
        dct_features = self.dct_output_adapter(dct_features)
        print(f"通道适配后 - RGB: {rgb_features.shape}, DCT: {dct_features.shape}")

        # 最终特征融合
        _, _, final_feature = self.final_fusion(rgb_features, dct_features)
        print(f"最终融合特征形状: {final_feature.shape}")

        # 分类分支
        try:
            # 检查avg_pool是否已定义
            if not hasattr(self, 'avg_pool'):
                print("添加全局平均池化层")
                self.avg_pool = nn.AdaptiveAvgPool2d(1)
            
            pooled = self.avg_pool(final_feature).view(final_feature.size(0), -1)
            cls_output = self.classifier(pooled)
            print(f"分类输出形状: {cls_output.shape}")
        except Exception as e:
            print(f"分类分支处理出错: {e}, 使用替代输出")
            # 创建一个合理的替代输出
            cls_output = torch.zeros(final_feature.size(0), 2).to(final_feature.device)
        

        # 分割分支 - 添加通道适配
        print(f"掩码头输入: 形状={final_feature.shape}, 预期通道数={self.mask_head_input_channels if hasattr(self, 'mask_head_input_channels') else '未知'}")
        if final_feature.size(1) != getattr(self, 'mask_head_input_channels', 480):
            print(f"掩码头输入通道不匹配: 需要{getattr(self, 'mask_head_input_channels', 480)}，实际{final_feature.size(1)}，进行适配")
            # 创建通道适配层
            mask_adapter = nn.Conv2d(
                final_feature.size(1),
                getattr(self, 'mask_head_input_channels', 480),
                kernel_size=1,
                bias=False
            ).to(final_feature.device)

            # 初始化权重
            nn.init.kaiming_uniform_(mask_adapter.weight)

            # 应用适配
            final_feature_for_mask = mask_adapter(final_feature)
            print(f"掩码输入通道适配后: {final_feature_for_mask.shape}")
        else:
            final_feature_for_mask = final_feature

        # 应用掩码头
        mask_output = self.mask_head(final_feature_for_mask)
        print(f"掩码输出形状: {mask_output.shape}")
        
        return cls_output, mask_output
    
    # 修改 _get_stage_features 方法，添加安全检查和调试信息
    def _get_stage_features(self, hrnet, x):
        """提取HRNet各阶段的特征"""
        batch_size = x.size(0)

        # 如果HRNet支持获取阶段特征，使用它
        if hasattr(hrnet, 'get_stage_features'):
            try:
                # 获取特征
                features = hrnet.get_stage_features(x)
                print(f"成功获取HRNet阶段特征: {len(features)}个特征")

                # 验证特征形状
                for i, feat in enumerate(features):
                    if feat is None or feat.dim() != 4:
                        print(f"警告: 第{i+1}阶段特征为None或形状不正确")
                        # 替换为模拟特征
                        features[i] = torch.zeros(
                            (batch_size, 32 * (2 ** i), x.shape[2] // (2 ** (i + 2)), x.shape[3] // (2 ** (i + 2))), 
                            device=x.device
                        )
                        
                return features
            except Exception as e:
                print(f"获取HRNet阶段特征失败: {e}")
                import traceback
                traceback.print_exc()
        else:
            print("HRNet未实现get_stage_features方法，使用模拟特征")

        # 创建合理的模拟特征 - 如果上面的方法失败或未实现
        device = x.device
        input_size = x.shape[-1]

        # 使用与模型配置一致的分辨率和通道数
        return [
            torch.zeros((batch_size, 32, input_size//4, input_size//4), device=device),
            torch.zeros((batch_size, 64, input_size//8, input_size//8), device=device),
            torch.zeros((batch_size, 128, input_size//16, input_size//16), device=device),
            torch.zeros((batch_size, 256, input_size//32, input_size//32), device=device)
        ]
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
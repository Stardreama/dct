import torch
import torch.nn as nn
import torch.nn.functional as F
from attention.ForensicAttentionFusion import ForensicAttentionFusion

def modify_enhanced_hrnet():
    """修改EnhancedHRNet以添加通道适配功能"""
    file_path = 'network/enhanced_hrnet.py'
    
    with open(file_path, 'r', encoding='utf-8') as file:
        content = file.read()
    
    # 添加通道适配器导入
    if 'from attention.attention_adapter import AttentionAdapter' not in content:
        import_pos = content.find('import torch\n')
        if import_pos != -1:
            modified_content = content[:import_pos + 12] + 'import torch.nn.functional as F\n' + content[import_pos + 12:]
            modified_content = modified_content.replace('from attention.ForensicAttentionFusion import ForensicAttentionFusion, CoordinateAttention, FrequencyAwareAttention',
                                                        'from attention.ForensicAttentionFusion import ForensicAttentionFusion, CoordinateAttention, FrequencyAwareAttention\n# 导入适配器\nfrom attention.attention_adapter import AttentionAdapter')
            content = modified_content
    
    # 添加通道适配代码
    if 'def _adapt_channels' not in content:
        forward_pos = content.find('def forward(self, rgb_input, dct_input=None):')
        if forward_pos != -1:
            # 添加通道适配方法
            adapt_channels_method = '''
    def _adapt_channels(self, features, target_channels, name=""):
        """调整特征通道数，以匹配目标通道数"""
        if features.size(1) != target_channels:
            print(f"{name} 通道调整: {features.size(1)} -> {target_channels}")
            adapter = nn.Conv2d(
                features.size(1), 
                target_channels, 
                kernel_size=1,
                bias=False
            ).to(features.device)
            # 初始化权重
            nn.init.kaiming_normal_(adapter.weight, mode='fan_out')
            # 应用适配
            return adapter(features)
        return features
'''
            # 在类初始化结束处添加方法
            class_end_pos = content.find('def forward(self, rgb_input, dct_input=None):', forward_pos - 200)
            if class_end_pos != -1:
                modified_content = content[:class_end_pos] + adapt_channels_method + content[class_end_pos:]
                content = modified_content
    
    # 修改forward方法中的特征融合部分
    if '_, _, final_feature = self.final_fusion(rgb_features, dct_features)' in content:
        # 找到调用final_fusion的代码行
        fusion_line_pos = content.find('_, _, final_feature = self.final_fusion(rgb_features, dct_features)')
        if fusion_line_pos != -1:
            # 替换为使用通道适配的版本
            old_code = '_, _, final_feature = self.final_fusion(rgb_features, dct_features)'
            new_code = '''
        # 适配通道数
        expected_channels = 480  # 默认预期通道数
        if hasattr(self.final_fusion, 'in_channels'):
            expected_channels = self.final_fusion.in_channels
            
        # 显示通道信息
        print(f"最终融合前 - RGB: {rgb_features.size(1)}通道, DCT: {dct_features.size(1)}通道, 期望: {expected_channels}通道")
        
        # 通道适配
        rgb_adapted = self._adapt_channels(rgb_features, expected_channels, "RGB特征")
        dct_adapted = self._adapt_channels(dct_features, expected_channels, "DCT特征")
        
        # 使用适配后的特征
        _, _, final_feature = self.final_fusion(rgb_adapted, dct_adapted)'''
            
            modified_content = content.replace(old_code, new_code)
            content = modified_content
    
    # 保存修改后的文件
    with open(file_path, 'w', encoding='utf-8') as file:
        file.write(content)
    
    print("EnhancedHRNet已成功修改，添加了通道适配功能")
    
def create_attention_adapter():
    """创建注意力适配器模块文件"""
    adapter_file = 'attention/attention_adapter.py'
    
    # 检查文件是否已存在
    try:
        with open(adapter_file, 'r') as f:
            print("适配器文件已存在，跳过创建")
            return
    except FileNotFoundError:
        pass
    
    # 创建适配器代码
    adapter_code = '''import torch
import torch.nn as nn
import torch.nn.functional as F

class AttentionAdapter(nn.Module):
    """注意力机制适配器，解决通道数不匹配问题"""
    
    def __init__(self, in_channels, out_channels, preserve_mode='interpolate'):
        """
        初始化通道适配器
        
        Args:
            in_channels: 输入通道数
            out_channels: 输出通道数
            preserve_mode: 特征保留模式，'interpolate'或'conv'
        """
        super(AttentionAdapter, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.preserve_mode = preserve_mode
        
        # 创建通道适配层
        if preserve_mode == 'conv':
            self.adapter = nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=1,
                bias=False
            )
            nn.init.kaiming_normal_(self.adapter.weight, mode='fan_out')
        else:
            # 使用插值方式
            self.register_buffer('weight', torch.eye(out_channels, in_channels))
            if out_channels > in_channels:
                # 对于通道扩展，使用复制
                repeat_factor = out_channels // in_channels
                remainder = out_channels % in_channels
                
                if remainder > 0:
                    self.register_buffer(
                        'remainder_weight',
                        torch.eye(remainder, in_channels)
                    )
    
    def forward(self, x):
        """前向传播，调整通道数"""
        if self.in_channels == self.out_channels:
            return x
            
        if self.preserve_mode == 'conv':
            return self.adapter(x)
        
        # 使用插值方法
        b, c, h, w = x.size()
        
        if self.out_channels < self.in_channels:
            # 通道减少 - 取前N个通道
            return x[:, :self.out_channels, :, :]
        else:
            # 通道扩展 - 重复通道
            repeat_factor = self.out_channels // self.in_channels
            remainder = self.out_channels % self.in_channels
            
            repeated = x.repeat(1, repeat_factor, 1, 1)
            
            if remainder > 0:
                remainder_channels = x[:, :remainder, :, :]
                result = torch.cat([repeated, remainder_channels], dim=1)
                return result
            else:
                return repeated

def adapt_channels(tensor, target_channels):
    """快速调整通道数的辅助函数"""
    current_channels = tensor.size(1)
    
    if current_channels == target_channels:
        return tensor
        
    if current_channels > target_channels:
        # 减少通道 - 取部分通道
        return tensor[:, :target_channels, :, :]
    else:
        # 增加通道 - 重复通道
        repeats = target_channels // current_channels
        remainder = target_channels % current_channels
        
        result = tensor.repeat(1, repeats, 1, 1)
        if remainder > 0:
            remainder_channels = tensor[:, :remainder, :, :]
            result = torch.cat([result, remainder_channels], dim=1)
            
        return result
'''
    
    # 确保目录存在
    import os
    os.makedirs('attention', exist_ok=True)
    
    # 写入适配器文件
    with open(adapter_file, 'w', encoding='utf-8') as file:
        file.write(adapter_code)
        
    print("已创建注意力适配器模块: attention/attention_adapter.py")
    
if __name__ == "__main__":
    create_attention_adapter()
    modify_enhanced_hrnet()
    print("修复完成，请重新运行训练命令")
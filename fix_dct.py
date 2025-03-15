# 创建文件 fix_dct.py

import os
import torch.nn as nn

# 1. 首先检查DCT变换模块的接口
def fix_dct_transform():
    dct_file = "network/dct_transform.py"
    with open(dct_file, "r", encoding="utf-8") as f:
        content = f.read()
        
    print("修复DCT变换模块...")
    
    # 检查DCTTransform类的参数
    if "def __init__(self, out_channels=" in content:
        print("DCTTransform类已有out_channels参数，无需修改")
    else:
        # 创建正确的DCTTransform实现
        with open(dct_file, "w", encoding="utf-8") as f:
            # 添加一个兼容的实现
            new_content = """import torch
import torch.nn as nn
import torch.nn.functional as F

class DCTTransform(nn.Module):
    def __init__(self, out_channels=12, multi_scale=True):
        super(DCTTransform, self).__init__()
        self.out_channels = out_channels
        self.multi_scale = multi_scale
        
        # 初始化DCT基函数
        self.register_buffer('dct_weights', self._get_dct_filter(out_channels))
        
    def _get_dct_filter(self, out_channels):
        # 简化的DCT过滤器生成
        filters = []
        for u in range(8):
            for v in range(8):
                if u * 8 + v < out_channels:
                    filter = torch.zeros(8, 8)
                    for i in range(8):
                        for j in range(8):
                            filter[i, j] = self._dct_basis(u, v, i, j)
                    filters.append(filter.unsqueeze(0))
        return torch.cat(filters, dim=0).unsqueeze(1)
    
    def _dct_basis(self, u, v, i, j):
        # DCT基函数计算
        cu = 1.0 / (8 ** 0.5) if u == 0 else (2.0 / 8) ** 0.5
        cv = 1.0 / (8 ** 0.5) if v == 0 else (2.0 / 8) ** 0.5
        return cu * cv * torch.cos(torch.tensor((2*i+1)*u*3.14159/(2*8))) * torch.cos(torch.tensor((2*j+1)*v*3.14159/(2*8)))
    
    def forward(self, x):
        # 转换输入为灰度
        if x.size(1) == 3:  # RGB
            # 转为灰度
            x_gray = 0.299 * x[:, 0:1] + 0.587 * x[:, 1:2] + 0.114 * x[:, 2:3]
        else:
            x_gray = x
            
        # 提取DCT特征
        dct_features = F.conv2d(x_gray, self.dct_weights)
        
        if self.multi_scale:
            # 添加多尺度特征
            x_down = F.avg_pool2d(x_gray, 2)
            dct_features_down = F.conv2d(x_down, self.dct_weights)
            dct_features_up = F.interpolate(dct_features_down, scale_factor=2, mode='bilinear')
            dct_features = torch.cat([dct_features, dct_features_up], dim=1)
            
        return dct_features

class MultiScaleFrequencyExtractor(nn.Module):
    def __init__(self, in_channels=3):
        super(MultiScaleFrequencyExtractor, self).__init__()
        self.dct = DCTTransform(out_channels=12, multi_scale=True)
        
    def forward(self, x):
        return self.dct(x)
"""
            f.write(new_content)
            print("已创建兼容的DCTTransform实现")

# 2. 调整模型架构，处理通道不匹配问题
def fix_model_architecture():
    models_file = "models.py"
    
    with open(models_file, "r", encoding="utf-8") as f:
        content = f.readlines()
    
    modified = False
    new_content = []
    
    for i, line in enumerate(content):
        new_content.append(line)
        # 查找需要修改的位置
        if "self.dct_branch =" in line and "HRNet" in line:
            # 查找下一行，检查是否有设置初始通道
            if i+1 < len(content) and "init_channel" not in content[i+1]:
                # 在卷积层之前添加通道适配层
                new_content.append("        # 添加通道适配层\n")
                new_content.append("        self.dct_adapter = nn.Conv2d(12, 63, kernel_size=1)\n")
                modified = True
        
        # 修改forward方法
        if "dct_features, dct_seg = self.dct_branch" in line and modified:
            # 在传给DCT分支前，先适配通道数
            indent = line.find("dct_features")
            new_content[-1] = f"{' ' * indent}dct_input = self.dct_adapter(dct_input)\n{line}"
    
    if modified:
        with open(models_file, "w", encoding="utf-8") as f:
            f.writelines(new_content)
        print("已修改模型架构，添加通道适配层")
    else:
        print("未找到需要修改的模型代码，请检查代码结构")

# 执行修复
if __name__ == "__main__":
    print("开始修复DCT相关问题...")
    fix_dct_transform()
    fix_model_architecture()
    print("修复完成，请使用以下命令尝试训练：")
    print("python [train.py](http://_vscodecontentref_/1) --config [config.yaml](http://_vscodecontentref_/2) --model_type forensics --mode Both")
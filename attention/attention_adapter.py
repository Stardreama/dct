# 创建一个新文件: D:\project\DCT_RGB_HRNet\attention\attention_adapter.py

import torch
import torch.nn as nn
import torch.nn.functional as F

class AttentionAdapter(nn.Module):
    """包装注意力模块以确保输入尺寸兼容"""
    
    def __init__(self, attention_module):
        super().__init__()
        self.attention_module = attention_module
    
    def forward(self, *args, **kwargs):
        try:
            return self.attention_module(*args, **kwargs)
        except RuntimeError as e:
            # 如果错误与维度有关
            if "size of tensor" in str(e) and "must match" in str(e):
                # 获取参数
                q, k, v = args[:3]
                height, width = None, None
                
                if len(args) > 3:
                    height = args[3]
                if len(args) > 4:
                    width = args[4]
                    
                # 从kwargs获取
                if height is None and 'height' in kwargs:
                    height = kwargs['height']
                if width is None and 'width' in kwargs:
                    width = kwargs['width']
                
                # 如果维度已指定
                if height is not None and width is not None:
                    # 确保q、k、v都有合适的形状
                    target_shape = (height, width)
                    q = self._adapt_tensor(q, target_shape)
                    k = self._adapt_tensor(k, target_shape)
                    v = self._adapt_tensor(v, target_shape)
                    
                    # 使用调整后的参数重新调用
                    args_list = list(args)
                    args_list[0:3] = [q, k, v]
                    return self.attention_module(*args_list, **kwargs)
                
            # 如果无法解决，重新引发错误
            raise e
    
    def _adapt_tensor(self, tensor, target_shape):
        """调整张量形状以匹配目标"""
        if tensor.dim() < 3:
            return tensor
            
        # 获取当前形状的最后两个维度
        *batch_dims, h, w, d = tensor.shape
        
        # 如果形状已经匹配
        if h == target_shape[0] and w == target_shape[1]:
            return tensor
            
        # 重塑为适合插值的形式
        tensor_4d = tensor.reshape(-1, d, h, w).permute(0, 1, 2, 3)
        
        # 进行插值
        resized = F.interpolate(
            tensor_4d, 
            size=target_shape,
            mode='bilinear', 
            align_corners=False
        )
        
        # 恢复原始维度顺序
        return resized.permute(0, 1, 2, 3).reshape(*batch_dims, target_shape[0], target_shape[1], d)
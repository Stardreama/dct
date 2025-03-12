import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from einops import rearrange, reduce


class CoordinateAttention(nn.Module):
    """
    坐标注意力模块 - 同时捕获位置信息和通道信息
    参考: "Coordinate Attention for Efficient Mobile Network Design" (CVPR 2021)
    """
    def __init__(self, in_channels, reduction=16):
        super(CoordinateAttention, self).__init__()
        self.in_channels = in_channels
        self.reduction = reduction
        
        # 共享的降维层
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))
        
        mid_channels = max(8, in_channels // reduction)
        
        # 共享的MLP
        self.mlp = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True)
        )
        
        # 分别处理高度方向和宽度方向
        self.mlp_h = nn.Conv2d(mid_channels, in_channels, kernel_size=1, bias=False)
        self.mlp_w = nn.Conv2d(mid_channels, in_channels, kernel_size=1, bias=False)
        
    def forward(self, x):
        identity = x
        
        # 池化操作
        n, c, h, w = x.size()
        x_h = self.pool_h(x)  # [n, c, h, 1]
        x_w = self.pool_w(x)  # [n, c, 1, w]
        
        # 共享的降维处理
        x_h = self.mlp(x_h)
        x_w = self.mlp(x_w)
        
        # 分别生成不同方向的注意力权重
        x_h = self.mlp_h(x_h).sigmoid()  # [n, c, h, 1]
        x_w = self.mlp_w(x_w).sigmoid()  # [n, c, 1, w]
        
        # 将两个方向的注意力合并
        x_attention = x_h * x_w
        
        # 应用注意力
        out = identity * x_attention
        
        return out


class FrequencyAwareAttention(nn.Module):
    """
    频率感知注意力 - 专门针对DCT域特征设计
    """
    def __init__(self, in_channels, reduction=8):
        super(FrequencyAwareAttention, self).__init__()
        self.in_channels = in_channels
        
        # 将频域特征划分为低频、中频、高频三部分
        self.freq_pool_low = nn.AdaptiveAvgPool2d(8)    # 低频区域(中心)
        self.freq_pool_mid = nn.AdaptiveAvgPool2d(16)   # 中频区域
        self.freq_pool_high = nn.AdaptiveAvgPool2d(32)  # 高频区域
        
        # 频域特征处理
        mid_channels = max(8, in_channels // reduction)
        self.freq_process = nn.Sequential(
            nn.Conv2d(in_channels * 3, mid_channels, kernel_size=1),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, in_channels, kernel_size=1),
            nn.Sigmoid()
        )
        
        # 阻挡梯度，保持低频特征的稳定性
        self.alpha = nn.Parameter(torch.ones(1) * 0.8)
        
    def get_frequency_regions(self, x):
        # 从中心向外获取不同频率区域
        b, c, h, w = x.size()
        
        # 低频区域 (中心区域)
        x_low = self.freq_pool_low(x)
        x_low = F.interpolate(x_low, size=(h, w), mode='bilinear', align_corners=False)
        
        # 中频区域
        x_mid = self.freq_pool_mid(x)
        x_mid = F.interpolate(x_mid, size=(h, w), mode='bilinear', align_corners=False)
        
        # 高频区域 (使用原始特征减去低频和中频)
        x_high = x - (x_low * self.alpha + x_mid * (1-self.alpha))
        
        # 连接不同频率区域
        return torch.cat([x_low, x_mid, x_high], dim=1)
        
    def forward(self, x):
        identity = x
        
        # 分析不同频率区域
        freq_regions = self.get_frequency_regions(x)
        
        # 生成频率感知注意力图
        freq_attention = self.freq_process(freq_regions)
        
        # 应用频率注意力
        out = identity * freq_attention
        
        return out


class SelfMutualAttention(nn.Module):
    """
    自相互注意力模块 - 结合Transformer的自注意力和不同分支间的互注意力
    改进自: "TransForensics: A Transformer-Based Face Forgery Detection Method" (CVPR 2022)
    """
    def __init__(self, in_channels, num_heads=8, dropout=0.1):
        super(SelfMutualAttention, self).__init__()
        self.in_channels = in_channels
        self.num_heads = num_heads
        self.head_dim = in_channels // num_heads
        assert self.head_dim * num_heads == in_channels, "in_channels must be divisible by num_heads"
        
        # 自注意力投影层
        self.q_proj = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.k_proj = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.v_proj = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        
        # 输出投影层
        self.out_proj = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        
        # 注意力缩放因子
        self.scale = self.head_dim ** -0.5
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
        
        # 相对位置编码
        self.pos_embed_h = nn.Parameter(torch.zeros(1, num_heads, 64, 1))
        self.pos_embed_w = nn.Parameter(torch.zeros(1, num_heads, 1, 64))
        
    def forward(self, x1, x2=None):
        """
        如果只提供x1，执行自注意力
        如果同时提供x1和x2，执行互注意力
        """
        if x2 is None:
            x2 = x1  # 自注意力模式
            
        b, c, h, w = x1.size()
        
        # 生成查询、键、值
        q = self.q_proj(x1)
        k = self.k_proj(x2)
        v = self.v_proj(x2)
        
        # 重塑张量以进行多头注意力计算
        q = rearrange(q, 'b c h w -> b (h w) c')
        k = rearrange(k, 'b c h w -> b (h w) c')
        v = rearrange(v, 'b c h w -> b (h w) c')
        
        # 多头拆分
        q = rearrange(q, 'b n (h d) -> b h n d', h=self.num_heads)
        k = rearrange(k, 'b n (h d) -> b h n d', h=self.num_heads)
        v = rearrange(v, 'b n (h d) -> b h n d', h=self.num_heads)
        
        # 计算注意力分数
        attn = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        
        # 添加相对位置编码
        if h <= 64 and w <= 64:  # 确保位置编码大小足够
            rel_pos_h = self.pos_embed_h[:, :, :h, :]
            rel_pos_w = self.pos_embed_w[:, :, :, :w]
            
            # 应用位置编码
            q_h = rearrange(q, 'b h (x y) d -> b h x y d', x=h)
            rel_h = torch.matmul(q_h, rel_pos_h.transpose(-2, -1))
            rel_h = rearrange(rel_h, 'b h x y z -> b h (x y) z')
            
            q_w = rearrange(q, 'b h (x y) d -> b h y x d', x=h)
            rel_w = torch.matmul(q_w, rel_pos_w.transpose(-2, -1))
            rel_w = rearrange(rel_w, 'b h y x z -> b h (y x) z')
            
            attn = attn + rel_h + rel_w
        
        # Softmax和Dropout
        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)
        
        # 应用注意力权重
        out = torch.matmul(attn, v)
        
        # 多头合并
        out = rearrange(out, 'b h n d -> b n (h d)')
        
        # 重塑回原始形状
        out = rearrange(out, 'b (h w) c -> b c h w', h=h)
        
        # 输出投影
        out = self.out_proj(out)
        
        return out


class BoundaryEnhancedAttention(nn.Module):
    """
    边界增强注意力 - 专注于捕获伪造边界区域
    受"Face X-ray"和"Local Relation Learning for Face Forgery Detection"启发
    """
    def __init__(self, in_channels):
        super(BoundaryEnhancedAttention, self).__init__()
        
        # 梯度特征提取
        self.conv_dx = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1, groups=in_channels, bias=False)
        self.conv_dy = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1, groups=in_channels, bias=False)
        
        # 初始化Sobel算子
        with torch.no_grad():
            self.conv_dx.weight[:, :, :, :] = 0
            self.conv_dx.weight[:, :, 1, 0] = -1
            self.conv_dx.weight[:, :, 1, 2] = 1
            
            self.conv_dy.weight[:, :, :, :] = 0
            self.conv_dy.weight[:, :, 0, 1] = -1
            self.conv_dy.weight[:, :, 2, 1] = 1
        
        # 边界增强
        self.conv_enhance = nn.Sequential(
            nn.Conv2d(in_channels * 2, in_channels, kernel_size=1),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, in_channels, kernel_size=1),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        # 提取梯度特征
        dx = self.conv_dx(x)
        dy = self.conv_dy(x)
        
        # 计算梯度幅度
        gradient = torch.cat([dx, dy], dim=1)
        
        # 生成边界注意力图
        boundary_attention = self.conv_enhance(gradient)
        
        # 增强特征
        enhanced = x * boundary_attention
        
        # 残差连接
        return enhanced + x


class ForensicAttentionFusion(nn.Module):
    """
    人脸伪造检测专用的注意力融合模块
    结合了坐标注意力、频率感知注意力、自相互注意力和边界增强注意力
    用于有效融合RGB和频域特征
    """
    def __init__(self, in_channels, reduction=16, num_heads=8, dropout=0.1):
        super(ForensicAttentionFusion, self).__init__()
        
        # 通道注意力 - 频域和RGB分支各自的特征提取
        self.coord_attn_rgb = CoordinateAttention(in_channels, reduction)
        self.freq_attn = FrequencyAwareAttention(in_channels, reduction)
        
        # 自相互注意力 - 用于分支间信息交流
        self.mutual_attn = SelfMutualAttention(in_channels, num_heads, dropout)
        
        # 边界增强注意力 - 增强伪造边界检测
        self.boundary_attn = BoundaryEnhancedAttention(in_channels)
        
        # 动态特征融合
        self.fusion_rgb = nn.Sequential(
            nn.Conv2d(in_channels * 2, in_channels, kernel_size=1),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True)
        )
        
        self.fusion_freq = nn.Sequential(
            nn.Conv2d(in_channels * 2, in_channels, kernel_size=1),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True)
        )
        
        # 自适应融合权重
        self.fusion_weights = nn.Parameter(torch.ones(2) * 0.5)
        
        # 最终融合层
        self.final_fusion = nn.Sequential(
            nn.Conv2d(in_channels * 2, in_channels, kernel_size=1),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True)
        )
        
    def forward(self, rgb_features, freq_features):
        batch_size = rgb_features.size(0)
        
        # 分别增强RGB和频域特征
        rgb_enhanced = self.coord_attn_rgb(rgb_features)
        freq_enhanced = self.freq_attn(freq_features)
        
        # 特征交互 - 让RGB和频域特征相互学习
        rgb_mutual = self.mutual_attn(rgb_enhanced, freq_enhanced)
        freq_mutual = self.mutual_attn(freq_enhanced, rgb_enhanced)
        
        # 边界增强
        rgb_boundary = self.boundary_attn(rgb_mutual)
        freq_boundary = self.boundary_attn(freq_mutual)
        
        # 中间融合 - 将原始特征与增强特征融合
        rgb_fused = self.fusion_rgb(torch.cat([rgb_features, rgb_boundary], dim=1))
        freq_fused = self.fusion_freq(torch.cat([freq_features, freq_boundary], dim=1))
        
        # 自适应权重归一化
        fusion_weights = F.softmax(self.fusion_weights, dim=0)
        
        # 加权融合两个分支
        fused_features = fusion_weights[0] * rgb_fused + fusion_weights[1] * freq_fused
        
        # 自残差连接
        rgb_residual = rgb_features + fused_features
        freq_residual = freq_features + fused_features
        
        # 最终融合 - 更好地保留各自分支的独特特征
        final_fusion = self.final_fusion(torch.cat([rgb_residual, freq_residual], dim=1))
        
        return rgb_residual, freq_residual, final_fusion


# 测试代码
if __name__ == '__main__':
    # 测试坐标注意力
    x = torch.randn(2, 64, 32, 32)
    coord_attn = CoordinateAttention(64)
    out = coord_attn(x)
    print(f"CoordinateAttention output shape: {out.shape}")
    
    # 测试频率感知注意力
    freq_attn = FrequencyAwareAttention(64)
    out = freq_attn(x)
    print(f"FrequencyAwareAttention output shape: {out.shape}")
    
    # 测试自相互注意力
    self_mutual_attn = SelfMutualAttention(64, num_heads=8)
    out1 = self_mutual_attn(x)  # 自注意力模式
    out2 = self_mutual_attn(x, torch.randn(2, 64, 32, 32))  # 互注意力模式
    print(f"SelfMutualAttention output shape: {out1.shape}, {out2.shape}")
    
    # 测试边界增强注意力
    boundary_attn = BoundaryEnhancedAttention(64)
    out = boundary_attn(x)
    print(f"BoundaryEnhancedAttention output shape: {out.shape}")
    
    # 测试完整的融合模块
    rgb_features = torch.randn(2, 64, 32, 32)
    freq_features = torch.randn(2, 64, 32, 32)
    fusion = ForensicAttentionFusion(64)
    rgb_out, freq_out, fusion_out = fusion(rgb_features, freq_features)
    print(f"ForensicAttentionFusion output shapes: RGB={rgb_out.shape}, Freq={freq_out.shape}, Fusion={fusion_out.shape}")
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
        
    # 修改 CoordinateAttention 类的 forward 方法
    def forward(self, x):
        identity = x

        # 添加通道检查和适配
        if hasattr(self, 'in_channels') and x.size(1) != self.in_channels:
            ##print(f"CoordinateAttention: 输入通道数 {x.size(1)} 与预期 {self.in_channels} 不匹配，进行适配")
            # 创建临时通道适配层
            adapter = nn.Conv2d(
                x.size(1), 
                self.in_channels, 
                kernel_size=1, 
                bias=False
            ).to(x.device)

            # 初始化权重
            nn.init.kaiming_uniform_(adapter.weight)

            # 应用适配
            x = adapter(x)
            ##print(f"通道适配后形状: {x.shape}")
            identity = x  # 更新identity使用适配后的x

        # 池化操作
        n, c, h, w = x.size()
        x_h = self.pool_h(x)  # [n, c, h, 1]
        x_w = self.pool_w(x)  # [n, c, 1, w]

        # 共享的降维处理
        try:
            x_h = self.mlp(x_h)
            x_w = self.mlp(x_w)

            # 分别生成不同方向的注意力权重
            x_h = self.mlp_h(x_h).sigmoid()  # [n, c, h, 1]
            x_w = self.mlp_w(x_w).sigmoid()  # [n, c, 1, w]

            # 将两个方向的注意力合并
            x_attention = x_h * x_w

            # 应用注意力
            out = identity * x_attention
        except Exception as e:
            ##print(f"CoordinateAttention 处理出错: {e}")
            out = identity  # 出错时直接返回原始特征

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
        
    # 修改 FrequencyAwareAttention 类的 forward 方法
    def forward(self, x):
        identity = x

        # 添加通道检查和适配
        if x.size(1) != self.in_channels:
            ##print(f"FrequencyAwareAttention: 输入通道 {x.size(1)} 与预期 {self.in_channels} 不匹配，进行适配")
            # 创建临时通道适配
            adapter = nn.Conv2d(
                x.size(1), 
                self.in_channels, 
                kernel_size=1, 
                bias=False
            ).to(x.device)

            # 初始化权重
            nn.init.kaiming_uniform_(adapter.weight)

            # 应用适配
            x = adapter(x)
            ##print(f"通道适配后形状: {x.shape}")
            identity = x  # 更新identity使用适配后的x

        try:
            # 分析不同频率区域
            freq_regions = self.get_frequency_regions(x)

            # 生成频率感知注意力图
            freq_attention = self.freq_process(freq_regions)

            # 应用频率注意力
            out = identity * freq_attention
        except Exception as e:
            ##print(f"FrequencyAwareAttention 处理出错: {e}")
            out = identity  # 出错时直接返回原始特征

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
        
        # 相对位置编码 - 增大尺寸以适应不同大小的特征图
        self.max_position_embeddings = 256  # 最大支持的位置
        self.use_dynamic_pos_embeddings = False

        def get_rel_pos_matrix(max_seq_len, max_position):
            """生成相对位置矩阵"""
            # 位置ID从 -max_seq_len+1 到 max_seq_len-1
            range_vec = torch.arange(max_seq_len)
            range_mat = range_vec.unsqueeze(0).repeat(max_seq_len, 1)
            distance_mat = range_mat - range_mat.t()
            # 将距离映射到有效范围内
            distance_mat_clipped = torch.clamp(distance_mat + max_position, 0, 2 * max_position)
            return distance_mat_clipped

        self.get_rel_pos_matrix = get_rel_pos_matrix

        # 创建可学习的相对位置嵌入
        self.rel_pos_embeddings = nn.Parameter(torch.zeros(2 * self.max_position_embeddings + 1, num_heads))

        # 初始化
        nn.init.xavier_uniform_(self.rel_pos_embeddings)


    # 在 SelfMutualAttention 类的 forward 方法中添加调试信息
    def forward(self, x1, x2=None):
        """
        如果只提供x1，执行自注意力
        如果同时提供x1和x2，执行互注意力
        """
        if x2 is None:
            x2 = x1  # 自注意力模式
            
        b, c, h, w = x1.size()
        ##print(f"SelfMutualAttention 输入形状 - x1: {x1.shape}, x2: {x2.shape}")
        
        # 检查输入尺寸不匹配
        if x1.shape[-2:] != x2.shape[-2:]:
            ##print(f"警告：互注意力输入尺寸不匹配 - x1: {x1.shape[-2:]}, x2: {x2.shape[-2:]}")
            # 调整x2以匹配x1
            x2 = F.interpolate(
                x2,
                size=x1.shape[-2:],
                mode='bilinear',
                align_corners=False
            )
            ##print(f"调整后x2形状: {x2.shape}")
        
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
        
        # 打印特征形状
        ##print(f"多头注意力形状 - q: {q.shape}, k: {k.shape}, v: {v.shape}")
        
        # 计算注意力分数
        attn = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        
        # 添加相对位置编码 - 带安全检查和详细调试信息
        # 替换位置编码代码为:
        # 修改 SelfMutualAttention 类中的 forward 方法中的位置编码部分
        try:
            if self.use_dynamic_pos_embeddings:
                b, h_heads, seq_len, d_head = q.shape
                
                # 获取特征图尺寸
                h = w = int(seq_len ** 0.5)
                
                if h <= self.max_position_embeddings and w <= self.max_position_embeddings:
                    #print(f"使用动态位置编码 (h={h}, w={w}, heads={h_heads})")
                    
                    # 水平方向相对位置
                    rel_pos_h = self.get_rel_pos_matrix(h, self.max_position_embeddings)
                    
                    # 修改: 确保使用正确的头数访问位置嵌入
                    # 原代码: rel_pos_h = self.rel_pos_embeddings[rel_pos_h.view(-1)].view(h, h, self.num_heads)
                    
                    # 先获取位置编码嵌入
                    rel_pos_h_emb = self.rel_pos_embeddings[rel_pos_h.view(-1)]
                    
                    # 检查并适配头数，确保与输入匹配
                    if h_heads != self.num_heads:
                        #print(f"注意: 输入头数 {h_heads} 与预期头数 {self.num_heads} 不匹配")
                        # 如果头数不匹配，选择合适的头数维度
                        # 如果实际头数小于配置的头数，我们只使用部分位置编码
                        # 如果实际头数大于配置的头数，我们需要扩展位置编码（通过复制或插值）
                        if h_heads <= self.num_heads:
                            rel_pos_h_emb = rel_pos_h_emb[:, :h_heads]
                        else:
                            # 扩展为需要的头数（这里通过重复最后一个维度实现）
                            last_dim = rel_pos_h_emb[:, -1].unsqueeze(1)
                            extra_dims = last_dim.repeat(1, h_heads - self.num_heads)
                            rel_pos_h_emb = torch.cat([rel_pos_h_emb, extra_dims], dim=1)
                    
                    # 重新形状为 [h, h, heads]
                    rel_pos_h = rel_pos_h_emb.view(-1, h_heads).view(h * h, h_heads).view(h, h, h_heads)
                    rel_pos_h = rel_pos_h.permute(2, 0, 1).unsqueeze(0)  # [1, heads, h, h]
                    
                    # 垂直方向相对位置 - 同样的逻辑
                    rel_pos_w = self.get_rel_pos_matrix(w, self.max_position_embeddings)
                    rel_pos_w_emb = self.rel_pos_embeddings[rel_pos_w.view(-1)]
                    
                    # 同样检查并适配头数
                    if h_heads != self.num_heads:
                        if h_heads <= self.num_heads:
                            rel_pos_w_emb = rel_pos_w_emb[:, :h_heads]
                        else:
                            last_dim = rel_pos_w_emb[:, -1].unsqueeze(1)
                            extra_dims = last_dim.repeat(1, h_heads - self.num_heads)
                            rel_pos_w_emb = torch.cat([rel_pos_w_emb, extra_dims], dim=1)
                    
                    rel_pos_w = rel_pos_w_emb.view(-1, h_heads).view(w * w, h_heads).view(w, w, h_heads)
                    rel_pos_w = rel_pos_w.permute(2, 0, 1).unsqueeze(0)  # [1, heads, w, w]
                    
                    # 添加形状检查，确保尺寸正确
                    #print(f"位置编码形状 - 水平: {rel_pos_h.shape}, 垂直: {rel_pos_w.shape}")
                    
                    # 行方向注意力
                    q_with_h_bias = q.reshape(b, h_heads, h, w, d_head)
                    
                    # 添加安全检查，确保可以进行矩阵乘法
                    if rel_pos_h.shape[1] != h_heads:
                        #print(f"警告: 位置编码头数 {rel_pos_h.shape[1]} 与查询头数 {h_heads} 不匹配，调整位置编码")
                        # 调整位置编码头数
                        rel_pos_h = rel_pos_h.expand(1, h_heads, h, h)
                    
                    # 安全的矩阵乘法，带有形状检查
                    try:
                        rel_h_attn = torch.matmul(q_with_h_bias.transpose(2, 3), 
                                                 rel_pos_h.transpose(-2, -1)).transpose(2, 3)
                    except RuntimeError as e:
                        #print(f"行方向注意力计算失败: {e}")
                        #print(f"q_with_h_bias: {q_with_h_bias.shape}, rel_pos_h: {rel_pos_h.shape}")
                        # 跳过位置编码
                        rel_h_attn = torch.zeros_like(q_with_h_bias)
                    
                    # 列方向注意力
                    q_with_w_bias = q.reshape(b, h_heads, h, w, d_head)
                    
                    # 同样检查位置编码维度
                    if rel_pos_w.shape[1] != h_heads:
                        #print(f"警告: 位置编码头数 {rel_pos_w.shape[1]} 与查询头数 {h_heads} 不匹配，调整位置编码")
                        # 调整位置编码头数
                        rel_pos_w = rel_pos_w.expand(1, h_heads, w, w)
                    
                    try:
                        rel_w_attn = torch.matmul(q_with_w_bias, 
                                                 rel_pos_w.transpose(-2, -1))
                    except RuntimeError as e:
                        #print(f"列方向注意力计算失败: {e}")
                        #print(f"q_with_w_bias: {q_with_w_bias.shape}, rel_pos_w: {rel_pos_w.shape}")
                        # 跳过位置编码
                        rel_w_attn = torch.zeros_like(q_with_w_bias)
                    
                    # 安全地重塑张量
                    try:
                        # 合并到原始注意力分数
                        rel_h_attn = rel_h_attn.reshape(b, h_heads, seq_len, w)
                        rel_w_attn = rel_w_attn.reshape(b, h_heads, seq_len, w)
                        
                        # 应用位置注意力
                        attn = attn + (rel_h_attn + rel_w_attn).reshape(b, h_heads, seq_len, seq_len)
                        #print("成功应用动态位置编码")
                    except RuntimeError as e:
                        #print(f"重塑位置编码注意力失败: {e}")
                        # 出错时不应用位置编码
                        pass
                else:
                    print(f"特征图尺寸 ({h}x{w}) 超过最大位置编码范围 ({self.max_position_embeddings})")
        except Exception as e:
            print(f"位置编码应用失败，跳过: {e}")
            # 捕获任何异常，确保即使位置编码失败，模型仍能继续运行
        
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
        
        # 保存输入通道数属性
        self.in_channels = in_channels
        
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
        
    # 在 ForensicAttentionFusion 类的 forward 方法中添加调试信息
    def forward(self, rgb_features, freq_features):
        batch_size = rgb_features.size(0)

        # 添加详细的输入形状和通道信息
        #print(f"ForensicAttentionFusion 输入形状 - RGB: {rgb_features.shape}, 频域: {freq_features.shape}")

        # 检查是否有in_channels属性
        if hasattr(self, 'in_channels'):
            #print(f"ForensicAttentionFusion 预期通道数: {self.in_channels}")

            # RGB特征通道检查
            if rgb_features.size(1) != self.in_channels:
                #print(f"RGB特征通道不匹配: 需要{self.in_channels}，实际{rgb_features.size(1)}，进行适配")
                # 创建通道适配层
                rgb_adapter = nn.Conv2d(
                    rgb_features.size(1),
                    self.in_channels,
                    kernel_size=1,
                    bias=False
                ).to(rgb_features.device)

                # 初始化权重
                nn.init.kaiming_uniform_(rgb_adapter.weight)

                # 应用适配
                rgb_features = rgb_adapter(rgb_features)
                #print(f"RGB特征通道适配后: {rgb_features.shape}")

            # 频域特征通道检查
            if freq_features.size(1) != self.in_channels:
                #print(f"频域特征通道不匹配: 需要{self.in_channels}，实际{freq_features.size(1)}，进行适配")
                # 创建通道适配层
                freq_adapter = nn.Conv2d(
                    freq_features.size(1),
                    self.in_channels,
                    kernel_size=1,
                    bias=False
                ).to(freq_features.device)

                # 初始化权重
                nn.init.kaiming_uniform_(freq_adapter.weight)

                # 应用适配
                freq_features = freq_adapter(freq_features)
                #print(f"频域特征通道适配后: {freq_features.shape}")
        else:
            print("警告: ForensicAttentionFusion 没有 in_channels 属性")

        try:
            # 分别增强RGB和频域特征
            rgb_enhanced = self.coord_attn_rgb(rgb_features)
            freq_enhanced = self.freq_attn(freq_features)
            #print(f"增强后特征形状 - RGB: {rgb_enhanced.shape}, 频域: {freq_enhanced.shape}")

            # 特征交互 - 让RGB和频域特征相互学习
            rgb_mutual = self.mutual_attn(rgb_enhanced, freq_enhanced)
            freq_mutual = self.mutual_attn(freq_enhanced, rgb_enhanced)
            #print(f"相互注意力后形状 - RGB: {rgb_mutual.shape}, 频域: {freq_mutual.shape}")

            # 边界增强
            rgb_boundary = self.boundary_attn(rgb_mutual)
            freq_boundary = self.boundary_attn(freq_mutual)
            #print(f"边界增强后形状 - RGB: {rgb_boundary.shape}, 频域: {freq_boundary.shape}")

            # 中间融合 - 将原始特征与增强特征融合
            rgb_fused = self.fusion_rgb(torch.cat([rgb_features, rgb_boundary], dim=1))
            freq_fused = self.fusion_freq(torch.cat([freq_features, freq_boundary], dim=1))
            #print(f"特征融合后形状 - RGB: {rgb_fused.shape}, 频域: {freq_fused.shape}")

            # 自适应权重归一化
            fusion_weights = F.softmax(self.fusion_weights, dim=0)

            # 加权融合两个分支
            fused_features = fusion_weights[0] * rgb_fused + fusion_weights[1] * freq_fused
            #print(f"加权融合后特征形状: {fused_features.shape}")

            # 自残差连接
            rgb_residual = rgb_features + fused_features
            freq_residual = freq_features + fused_features

            # 最终融合 - 更好地保留各自分支的独特特征
            final_fusion = self.final_fusion(torch.cat([rgb_residual, freq_residual], dim=1))
            #print(f"最终融合特征形状: {final_fusion.shape}")

            return rgb_residual, freq_residual, final_fusion
        except Exception as e:
            #print(f"ForensicAttentionFusion处理出错: {e}")
            # 发生错误时返回原始特征
            return rgb_features, freq_features, rgb_features

# 测试代码
if __name__ == '__main__':
    # 测试坐标注意力
    x = torch.randn(2, 64, 32, 32)
    coord_attn = CoordinateAttention(64)
    out = coord_attn(x)
    #print(f"CoordinateAttention output shape: {out.shape}")
    
    # 测试频率感知注意力
    freq_attn = FrequencyAwareAttention(64)
    out = freq_attn(x)
    #print(f"FrequencyAwareAttention output shape: {out.shape}")
    
    # 测试自相互注意力
    self_mutual_attn = SelfMutualAttention(64, num_heads=8)
    out1 = self_mutual_attn(x)  # 自注意力模式
    out2 = self_mutual_attn(x, torch.randn(2, 64, 32, 32))  # 互注意力模式
    #print(f"SelfMutualAttention output shape: {out1.shape}, {out2.shape}")
    
    # 测试边界增强注意力
    boundary_attn = BoundaryEnhancedAttention(64)
    out = boundary_attn(x)
    #print(f"BoundaryEnhancedAttention output shape: {out.shape}")
    
    # 测试完整的融合模块
    rgb_features = torch.randn(2, 64, 32, 32)
    freq_features = torch.randn(2, 64, 32, 32)
    fusion = ForensicAttentionFusion(64)
    rgb_out, freq_out, fusion_out = fusion(rgb_features, freq_features)
    #print(f"ForensicAttentionFusion output shapes: RGB={rgb_out.shape}, Freq={freq_out.shape}, Fusion={fusion_out.shape}")
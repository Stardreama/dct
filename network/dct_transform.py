import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

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
        """前向传播，将RGB图像转换为DCT系数"""
        try:
            # 输入验证和形状获取
            if not isinstance(x, torch.Tensor):
                print("输入不是有效的张量")
                return torch.zeros(1, self.out_channels, 256, 256, device=x.device if hasattr(x, 'device') else None)

            batch_size, channels, height, width = x.shape

            # 添加输入验证和形状修正
            if height != width:
                print(f"输入图像不是正方形：{height}x{width}，进行调整")
                # 裁剪为正方形
                min_size = min(height, width)
                x = x[:, :, :min_size, :min_size]
                height = width = min_size

            # 将输入图像从RGB转换到YCbCr
            if channels == 3:
                # 标准RGB到YCbCr转换矩阵
                transform_matrix = torch.tensor([
                    [0.299, 0.587, 0.114],
                    [-0.169, -0.331, 0.500],
                    [0.500, -0.419, -0.081]
                ], dtype=torch.float32, device=x.device)

                # 确保值范围在[0,1]内
                x_norm = x / 255.0 if x.max() > 1.0 else x

                # 正确的维度转换
                # 1. 转置到[B,H,W,C]格式以便于通道级矩阵乘法
                rgb_hwc = x_norm.permute(0, 2, 3, 1)  # [B,3,H,W] -> [B,H,W,3]

                # 2. 进行矩阵乘法
                ycbcr_hwc = torch.matmul(rgb_hwc, transform_matrix.t())

                # 3. 应用偏置(Cb和Cr需要+0.5以使范围在[0,1])
                offset = torch.tensor([0, 0.5, 0.5], dtype=torch.float32, device=x.device)
                ycbcr_hwc = ycbcr_hwc + offset

                # 4. 转回[B,C,H,W]格式
                ycbcr = ycbcr_hwc.permute(0, 3, 1, 2)  # [B,H,W,3] -> [B,3,H,W]

                # 验证并拆分通道
                if ycbcr.shape == x.shape:
                    y = ycbcr[:, 0:1, :, :]
                    cb = ycbcr[:, 1:2, :, :]
                    cr = ycbcr[:, 2:3, :, :]

                    # 应用DCT变换到每个通道
                    y_dct = self.apply_dct(y)
                    cb_dct = self.apply_dct(cb)
                    cr_dct = self.apply_dct(cr)

                    # 合并结果
                    return torch.cat([y_dct, cb_dct, cr_dct], dim=1)
                else:
                    print(f"YCbCr转换后形状异常: {ycbcr.shape}，期望{x.shape}")
                    # 回退到直接处理RGB通道
                    r_dct = self.apply_dct(x[:, 0:1, :, :])
                    g_dct = self.apply_dct(x[:, 1:2, :, :])
                    b_dct = self.apply_dct(x[:, 2:3, :, :])
                    return torch.cat([r_dct, g_dct, b_dct], dim=1)
            else:
                print(f"警告: 输入通道数 {channels} 不是3，直接应用DCT")
                # 如果不是3通道，对每个通道分别应用DCT
                dct_features = []
                channels_per_output = self.out_channels // max(1, min(3, channels))

                for i in range(min(3, channels)):
                    channel_dct = self.apply_dct(x[:, i:i+1, :, :])
                    # 确保通道数正确
                    if channel_dct.shape[1] != channels_per_output:
                        if channel_dct.shape[1] > channels_per_output:
                            channel_dct = channel_dct[:, :channels_per_output, :, :]
                        else:
                            padding = torch.zeros(batch_size, 
                                                channels_per_output - channel_dct.shape[1],
                                                height, width, device=x.device)
                            channel_dct = torch.cat([channel_dct, padding], dim=1)
                    dct_features.append(channel_dct)

                # 如果通道数少于3，填充剩余通道
                while len(dct_features) < 3:
                    zeros = torch.zeros(batch_size, channels_per_output, height, width, device=x.device)
                    dct_features.append(zeros)

                # 合并所有通道
                combined_dct = torch.cat(dct_features[:3], dim=1)

                # 确保输出通道数正确
                if combined_dct.shape[1] != self.out_channels:
                    if combined_dct.shape[1] > self.out_channels:
                        combined_dct = combined_dct[:, :self.out_channels, :, :]
                    else:
                        padding = torch.zeros(batch_size,
                                            self.out_channels - combined_dct.shape[1],
                                            height, width, device=x.device)
                        combined_dct = torch.cat([combined_dct, padding], dim=1)

                return combined_dct

        except Exception as e:
            print(f"dct_transform.py中DCTTransform.forward函数DCT特征提取失败: {e}")
            import traceback
            traceback.print_exc()
            # 返回零填充的DCT特征
            return torch.zeros(batch_size, self.out_channels, height, width, device=x.device)
# 在network/dct_transform.py中修改

class MultiScaleFrequencyExtractor(nn.Module):
    """多尺度频率特征提取器"""
    def __init__(self, in_channels=3, out_channels=63, pyramid_levels=3):
        super(MultiScaleFrequencyExtractor, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.pyramid_levels = pyramid_levels
        
        # 确保输出通道数能被3整除
        assert out_channels % 3 == 0, "输出通道数必须是3的倍数"
        
        # 每个通道的DCT系数数量
        self.dct_channels = out_channels // 3
        
    def rgb_to_ycbcr(self, rgb_image):
        """
        安全将RGB图像转换为YCbCr格式

        Args:
            rgb_image: RGB格式图像，形状为[B,3,H,W]

        Returns:
            YCbCr格式图像，形状为[B,3,H,W]，如果转换失败则返回None
        """
        # 安全检查：确保输入是有效的张量
        if not isinstance(rgb_image, torch.Tensor):
            print("输入不是有效的张量")
            return None

        # 确保输入是4D张量 [B,C,H,W]
        input_ndim = rgb_image.dim()
        if input_ndim != 4:
            if input_ndim == 3:  # [C,H,W] -> [1,C,H,W]
                rgb_image = rgb_image.unsqueeze(0)
            else:
                print(f"无效的输入维度: {input_ndim}，期望4维")
                return None

        # 获取维度并记录
        batch_size, channels, height, width = rgb_image.shape
        print(f"RGB图像形状: {rgb_image.shape}")

        # 确保是3通道输入
        if channels != 3:
            print(f"输入应为3通道RGB图像，但得到{channels}通道")
            if channels == 1:
                # 灰度图转RGB
                rgb_image = rgb_image.expand(-1, 3, -1, -1)
            else:
                # 截取前3个通道
                rgb_image = rgb_image[:, :3, :, :]
            print(f"调整后形状: {rgb_image.shape}")

        # 确保值范围在[0,1]内
        if rgb_image.max() > 1.0:
            rgb_image = rgb_image / 255.0

        try:
            # 标准RGB到YCbCr转换矩阵
            # 参考ITU-R BT.601标准
            transform_matrix = torch.tensor([
                [0.299, 0.587, 0.114],
                [-0.169, -0.331, 0.500],
                [0.500, -0.419, -0.081]
            ], dtype=torch.float32, device=rgb_image.device)

            # 重要修复: 正确的维度转换
            # 1. 转置到[B,H,W,C]格式以便于通道级矩阵乘法
            rgb_hwc = rgb_image.permute(0, 2, 3, 1)  # [B,3,H,W] -> [B,H,W,3]

            # 2. 进行矩阵乘法
            ycbcr_hwc = torch.matmul(rgb_hwc, transform_matrix.t())

            # 3. 应用偏置(Cb和Cr需要+0.5以使范围在[0,1])
            offset = torch.tensor([0, 0.5, 0.5], dtype=torch.float32, device=rgb_image.device)
            ycbcr_hwc = ycbcr_hwc + offset

            # 4. 转回[B,C,H,W]格式
            ycbcr = ycbcr_hwc.permute(0, 3, 1, 2)  # [B,H,W,3] -> [B,3,H,W]

            # 验证输出形状
            if ycbcr.shape != rgb_image.shape:
                print(f"警告: YCbCr形状{ycbcr.shape}与输入RGB形状{rgb_image.shape}不匹配")
                return None

            # 正常转换，输出调试信息
            print(f"YCbCr转换成功: {ycbcr.shape}")
            return ycbcr

        except RuntimeError as e:
            print(f"RGB到YCbCr转换出错: {e}")
            # 发生错误时返回原图，而不是None
            print("返回原始RGB图像作为备用")
            return rgb_image
    
    def extract_dct_features(self, img):
        """
        从RGB图像中提取DCT特征
        Args:
            img: RGB图像，形状为 [B,3,H,W]
        Returns:
            DCT特征，形状为 [B,out_channels,H,W]
        """
        try:
            # 输入检查
            if img is None or not isinstance(img, torch.Tensor):
                print("提供的输入无效")
                return None

            # 获取输入尺寸
            orig_shape = img.shape
            batch_size = orig_shape[0] if len(orig_shape) > 3 else 1
            height = orig_shape[-2]
            width = orig_shape[-1]

            # 确保4D张量 [B,C,H,W]
            if len(orig_shape) == 3:
                img = img.unsqueeze(0)

            # 转换到YCbCr颜色空间
            ycbcr = self.rgb_to_ycbcr(img)

            # 如果颜色转换失败，使用原始RGB
            if ycbcr is None:
                print("YCbCr转换失败，使用RGB通道进行DCT")
                channels = 3
                # 确保img是3通道
                if img.size(1) != 3:
                    if img.size(1) == 1:
                        img = img.expand(-1, 3, -1, -1)
                    else:
                        img = img[:, :3, :, :]
                channel_inputs = [img[:, i:i+1, :, :] for i in range(channels)]
            else:
                # 正常使用YCbCr通道
                channels = 3
                channel_inputs = [ycbcr[:, i:i+1, :, :] for i in range(channels)]

            # 初始化输出特征图
            dct_features_list = []

            # 对每个通道分别应用DCT
            for c in range(channels):
                try:
                    # 获取单一通道
                    channel = channel_inputs[c]  # [B,1,H,W]

                    # 调用DCT变换处理
                    channel_dct = self.apply_dct(channel)
                    dct_features_list.append(channel_dct)

                except Exception as e:
                    print(f"通道 {c} DCT处理失败: {e}")
                    # 失败时添加零填充
                    zeros = torch.zeros(batch_size, self.out_channels//3, height, width, device=img.device)
                    dct_features_list.append(zeros)

            # 合并所有通道的DCT特征
            try:
                all_features = torch.cat(dct_features_list, dim=1)

                # 确保输出通道数正确
                current_channels = all_features.size(1)
                if current_channels != self.out_channels:
                    print(f"调整最终DCT特征通道数: {current_channels} -> {self.out_channels}")
                    if current_channels > self.out_channels:
                        all_features = all_features[:, :self.out_channels, :, :]
                    else:
                        padding = torch.zeros(batch_size, self.out_channels - current_channels, 
                                                all_features.shape[2], all_features.shape[3], 
                                                device=all_features.device)
                        all_features = torch.cat([all_features, padding], dim=1)

                return all_features

            except Exception as e:
                print(f"合并DCT特征失败: {e}")
                # 返回零特征图
                return torch.zeros(batch_size, self.out_channels, height, width, device=img.device)

        except Exception as e:
            print(f"dct_transform.py中MultiScaleFrequencyExtractor.extract_dct_features函数DCT特征提取失败: {e}")
            import traceback
            traceback.print_exc()

            # 获取形状信息以创建正确尺寸的零张量
            if isinstance(img, torch.Tensor):
                if img.dim() >= 3:
                    b = img.size(0) if img.dim() == 4 else 1
                    h, w = img.size(-2), img.size(-1)
                    return torch.zeros(b, self.out_channels, h, w, device=img.device)

            # 完全找不到形状信息时的默认值
            return torch.zeros(1, self.out_channels, 256, 256, device=img.device if isinstance(img, torch.Tensor) else None)
        
    def apply_dct(self, x):
        """
        将DCT变换应用于图像块
        Args:
            x: 输入张量，[B,1,H,W]
        Returns:
            DCT特征，[B,out_channels/3,H,W]
        """
        # 安全检查
        if x is None or not isinstance(x, torch.Tensor):
            print("DCT输入无效")
            return None

        if x.dim() != 4 or x.size(1) != 1:
            print(f"DCT输入形状错误: {x.shape}，期望为[B,1,H,W]")
            # 尝试修复形状
            if x.dim() == 3:
                x = x.unsqueeze(0)
            if x.size(1) > 1:
                x = x[:, 0:1, :, :]
        batch_size, _, height, width = x.shape
        out_channels_per_band = self.out_channels // 3
        
        # 将图像分割成8x8块
        block_size = 8
        
        # 为处理整个图像，先对长和宽进行填充，使它们可以被8整除
        pad_h = (block_size - height % block_size) % block_size
        pad_w = (block_size - width % block_size) % block_size
        if pad_h > 0 or pad_w > 0:
            x = F.pad(x, (0, pad_w, 0, pad_h))
            
        padded_height, padded_width = x.shape[2], x.shape[3]
        
        # 准备DCT变换矩阵（仅初始化一次）
        if not hasattr(self, 'dct_matrix') or self.dct_matrix.shape[0] != block_size:
            # 创建DCT矩阵
            dct_m = torch.zeros(block_size, block_size, device=x.device)
            for i in range(block_size):
                scale = torch.sqrt(torch.tensor(1.0 / block_size)) if i == 0 else torch.sqrt(torch.tensor(2.0 / block_size))
                for j in range(block_size):
                    dct_m[i, j] = scale * torch.cos(torch.tensor(np.pi * (2 * j + 1) * i / (2 * block_size)))
            self.dct_matrix = dct_m
            self.dct_matrix_t = dct_m.t()
        
        # 2D-DCT变换的函数
        def dct_2d(block):
            """计算2D DCT变换"""
            return torch.matmul(torch.matmul(self.dct_matrix, block), self.dct_matrix_t)
        
        # 创建用于存储结果的张量
        dct_features = torch.zeros(batch_size, out_channels_per_band, padded_height, padded_width, device=x.device)
        
        try:
            # 根据DCT频率构建一个掩码，用于选择特定频率分量
            dct_mask = torch.zeros(block_size, block_size, device=x.device)
            
            # 选择低频成分 - 对每个通道可选择不同数量的系数
            coef_count = min(out_channels_per_band, block_size*block_size)
            idx = 0
            for i in range(block_size):
                for j in range(block_size):
                    if idx < coef_count:
                        dct_mask[i, j] = 1
                        idx += 1
            
            # 对图像块应用DCT变换
            for b in range(batch_size):
                for i in range(0, padded_height, block_size):
                    for j in range(0, padded_width, block_size):
                        # 提取一个块
                        block = x[b, 0, i:i+block_size, j:j+block_size]
                        
                        # 应用DCT变换
                        dct_block = dct_2d(block)
                        
                        # 保留低频成分
                        dct_coeff = dct_block * dct_mask
                        
                        # 重塑系数到输出通道
                        # 这里仅使用非零系数
                        flat_coeff = dct_coeff.flatten()[:out_channels_per_band]
                        
                        # 将系数重排为通道
                        for c in range(len(flat_coeff)):
                            if c < out_channels_per_band:
                                dct_features[b, c, i:i+block_size, j:j+block_size] = flat_coeff[c]
        except Exception as e:
            print(f"DCT块处理失败: {e}")
        
        # 裁剪回原始尺寸
        if pad_h > 0 or pad_w > 0:
            dct_features = dct_features[:, :, :height, :width]
            
        return dct_features
    
        
    def dct_transform(self, x):
        """执行二维DCT变换"""
        try:
            # 标准化到[-1,1]
            x = 2.0 * x - 1.0
            
            # 使用PyTorch的FFT进行DCT
            x = torch.fft.rfft2(x, norm='ortho')
            
            # 获取幅度
            return torch.abs(x)
        except Exception as e:
            print(f"DCT变换失败: {e}")
            # 出错时返回原始输入
            return x
    
    def collect_dct_coefficients(self, dct_blocks, block_size):
        """收集重要的DCT系数"""
        # 获取块的数量
        num_blocks = dct_blocks.size(0)
        
        # 仅保留左上角的低频系数 - 通常更重要
        coef_size = min(8, block_size)  # 最多保留8x8系数
        dct_coefs = dct_blocks[:, :, :coef_size, :coef_size]
        
        # 将系数展平为特征向量
        return dct_coefs.view(num_blocks, -1)
    
    def forward(self, x):
        """前向传播"""
        # 提取DCT特征
        dct_features = self.extract_dct_features(x)
        return dct_features

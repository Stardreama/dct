"""
增强的数据转换模块

针对人脸伪造检测的专用数据转换和增强技术
结合传统和频域增强方法，提高模型对各类伪造的鲁棒性

"""
import torch
import numpy as np
import random
import math
from PIL import Image, ImageFilter, ImageEnhance, ImageOps
from io import BytesIO
import cv2
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
from typing import List, Dict, Tuple, Optional, Union, Any


# 保留原始的基本转换以保持兼容性
mesonet_data_transforms = {
    'train': transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize([0.5]*3, [0.5]*3)
    ]),
    'val': transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize([0.5] * 3, [0.5] * 3)
    ]),
    'test': transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize([0.5] * 3, [0.5] * 3)
    ]),
}


class ForensicJPEGArtifacts:
    """模拟JPEG压缩伪影，用于数据增强"""
    def __init__(self, quality_range=(60, 95), p=0.5):
        self.quality_range = quality_range
        self.p = p
        
    def __call__(self, img):
        if random.random() > self.p:
            return img
            
        # 随机选择JPEG质量
        quality = random.randint(self.quality_range[0], self.quality_range[1])
        
        # 将PIL图像转为BytesIO, 应用JPEG压缩并重新加载
        buffer = BytesIO()
        img.save(buffer, format='JPEG', quality=quality)
        buffer.seek(0)
        img = Image.open(buffer)
        
        return img


class ForensicNoiseSuppression:
    """噪声抑制变换，模拟降噪算法处理"""
    def __init__(self, p=0.5, strength_range=(5, 15)):
        self.p = p
        self.strength_range = strength_range
    
    def __call__(self, img):
        if random.random() > self.p:
            return img
            
        # 转换为numpy数组
        img_array = np.array(img)
        
        # 随机选择强度
        strength = random.randint(self.strength_range[0], self.strength_range[1])
        
        # 应用非局部均值去噪(NLM)
        if len(img_array.shape) == 3:  # 彩色图像
            img_array = cv2.fastNlMeansDenoisingColored(
                img_array, None, strength, strength, 7, 21)
        else:  # 灰度图像
            img_array = cv2.fastNlMeansDenoising(
                img_array, None, strength, 7, 21)
        
        # 转回PIL图像
        img = Image.fromarray(img_array)
        
        return img


class FakeRegionBlending:
    """模拟伪造区域的边界模糊/重叠效应"""
    def __init__(self, p=0.5, blend_range=(0.1, 0.3)):
        self.p = p
        self.blend_range = blend_range
        
    def __call__(self, img, mask=None):
        if random.random() > self.p or mask is None:
            return img
        
        # 确保mask是二值化的
        mask_np = np.array(mask)
        if mask_np.max() > 1:
            mask_np = mask_np / 255.0
            
        # 创建边界模糊效果
        blend_width = int(random.uniform(self.blend_range[0], self.blend_range[1]) * img.size[0])
        
        # 膨胀和腐蚀操作创建边界区域
        kernel = np.ones((blend_width, blend_width), np.uint8)
        dilated = cv2.dilate(mask_np, kernel, iterations=1)
        eroded = cv2.erode(mask_np, kernel, iterations=1)
        
        # 边界区域是膨胀后与原始区域的差
        boundary = dilated - mask_np
        
        # 边界区域应用高斯模糊
        img_np = np.array(img)
        blurred_img = cv2.GaussianBlur(img_np, (blend_width*2+1, blend_width*2+1), 0)
        
        # 在边界区域混合原图和模糊图
        blend_mask = np.expand_dims(boundary, axis=-1) if len(img_np.shape) == 3 else boundary
        blended = img_np * (1 - blend_mask) + blurred_img * blend_mask
        
        return Image.fromarray(blended.astype(np.uint8))


class LocalColorDistortion:
    """局部颜色扭曲，模拟伪造区域与原始区域的色调不一致"""
    def __init__(self, p=0.5, scale_range=(0.8, 1.2)):
        self.p = p
        self.scale_range = scale_range
        
    def __call__(self, img, mask=None):
        if random.random() > self.p:
            return img

        try:
            # 转换为numpy数组
            img_np = np.array(img).astype(np.float32)

            # 检查图像形状
            if len(img_np.shape) != 3 or img_np.shape[2] != 3:
                #print(f"警告: 颜色变换收到形状异常的图像 {img_np.shape}，返回原图")
                return img

            # 确保我们有正确的3维数组 [H, W, C]
            if img_np.shape[-1] != 3 and len(img_np.shape) > 3:
                img_np = img_np.squeeze()  # 移除多余的维度
                if img_np.shape[-1] != 3:
                    #print(f"警告: 颜色变换结果形状异常 {img_np.shape}，返回原图")
                    return img

            if mask is None:
                # 随机选择一个区域
                h, w = img_np.shape[:2]
                y1 = random.randint(0, h//2)
                x1 = random.randint(0, w//2)
                y2 = random.randint(y1+h//4, h)
                x2 = random.randint(x1+w//4, w)
                
                # 创建一个掩码
                mask_np = np.zeros((h, w), dtype=np.float32)
                mask_np[y1:y2, x1:x2] = 1
                
                # 应用羽化效果到掩码边缘
                mask_np = cv2.GaussianBlur(mask_np, (21, 21), 11)
            else:
                mask_np = np.array(mask).astype(np.float32)
                if mask_np.max() > 1:
                    mask_np = mask_np / 255.0
            
            # 根据颜色通道随机扭曲
            channels = []
            for c in range(min(3, img_np.shape[2])):
                # 为每个通道选择独立的扭曲因子
                factor = random.uniform(self.scale_range[0], self.scale_range[1])
                
                # 应用到区域
                channel = img_np[:, :, c]
                distorted = channel * factor
                distorted = np.clip(distorted, 0, 255)
                
                # 融合回原图
                if len(mask_np.shape) == 2:
                    mask_3d = np.expand_dims(mask_np, axis=2)
                else:
                    mask_3d = mask_np
                    
                blended = channel * (1 - mask_3d) + distorted * mask_3d
                channels.append(blended)
                

            # 合并通道前检查
            if len(channels) == 3:  # RGB
                result = np.stack(channels, axis=2)
            elif len(channels) == 1:  # 灰度
                result = channels[0]

            # 确保结果形状正确
            if result.ndim > 3:
                result = result.squeeze()  # 移除大小为1的维度

            if result.ndim != 3 or result.shape[2] != 3:
                #print(f"警告: 颜色变换结果形状异常 {result.shape}，返回原图")
                return img

            return Image.fromarray(result.astype(np.uint8))
        except Exception as e:
            #print(f"颜色变换出错: {e}")
            return img  # 发生错误时返回原始图像


class FrequencyDomainTransform:
    """频域变换增强，加强对频域特征的识别能力"""
    def __init__(self, p=0.5):
        self.p = p
        
    def _apply_dct_numpy(self, img):
        """应用DCT变换,修改频域系数,再逆变换"""
        # 转换为numpy数组
        img_np = np.array(img).astype(np.float32)
        
        result = np.zeros_like(img_np)
        
        # 对每个通道单独处理
        for c in range(min(3, img_np.shape[2])):
            channel = img_np[:, :, c]
            
            # 应用DCT变换
            dct = cv2.dct(channel)
            
            # 随机修改不同频段的系数(突出或抑制)
            # 低频: 对应伪造区域的整体颜色/亮度
            # 高频: 对应伪造区域的细节/噪声
            
            # 随机选择是突出高频还是低频
            if random.random() > 0.5:
                # 增强高频(乘以放大因子)
                h, w = dct.shape
                high_freq_mask = np.ones((h, w), dtype=np.float32)
                high_freq_mask[0:h//4, 0:w//4] = 0.5  # 低频区域降低影响
                dct = dct * high_freq_mask * random.uniform(1.1, 1.5)
            else:
                # 增强低频(更微妙的变化)
                h, w = dct.shape
                low_freq_mask = np.ones((h, w), dtype=np.float32) * 0.9
                low_freq_mask[0:h//4, 0:w//4] = random.uniform(1.1, 1.3)  # 低频区域微增强
                dct = dct * low_freq_mask
            
            # 逆DCT变换
            idct = cv2.idct(dct)
            
            # 确保值在有效范围
            idct = np.clip(idct, 0, 255)
            
            result[:, :, c] = idct
        
        return Image.fromarray(result.astype(np.uint8))
        
    def _apply_dct_numpy(self, img):
        """应用DCT变换,修改频域系数,再逆变换"""
        try:
            # 转换为numpy数组
            img_np = np.array(img).astype(np.float32)

            # 检查图像形状
            if len(img_np.shape) != 3 or img_np.shape[2] != 3:
                #print(f"警告: DCT变换收到形状异常的图像 {img_np.shape}，跳过处理")
                return img

            result = np.zeros_like(img_np)

            # 对每个通道单独处理
            for c in range(3):
                channel = img_np[:, :, c]

                # 应用DCT变换
                dct = cv2.dct(channel)

                # 随机修改不同频段的系数(突出或抑制)
                # 低频: 对应伪造区域的整体颜色/亮度
                # 高频: 对应伪造区域的细节/噪声

                # 随机选择是突出高频还是低频
                if random.random() > 0.5:
                    # 增强高频(乘以放大因子)
                    h, w = dct.shape
                    high_freq_mask = np.ones((h, w), dtype=np.float32)
                    high_freq_mask[0:h//4, 0:w//4] = 0.5  # 低频区域降低影响
                    dct = dct * high_freq_mask * random.uniform(1.1, 1.5)
                else:
                    # 增强低频(更微妙的变化)
                    h, w = dct.shape
                    low_freq_mask = np.ones((h, w), dtype=np.float32) * 0.9
                    low_freq_mask[0:h//4, 0:w//4] = random.uniform(1.1, 1.3)  # 低频区域微增强
                    dct = dct * low_freq_mask

                # 逆DCT变换
                idct = cv2.idct(dct)

                # 确保值在有效范围
                idct = np.clip(idct, 0, 255)

                result[:, :, c] = idct

            # 确保结果是正确的形状和类型
            if result.shape != img_np.shape:
                #print(f"警告: DCT结果形状 {result.shape} 与输入 {img_np.shape} 不匹配，返回原图")
                return img

            return Image.fromarray(result.astype(np.uint8))
        except Exception as e:
            #print(f"DCT numpy处理出错: {e}")
            # 发生错误时返回原始图像
            return img
    
    def __call__(self, img):
        if random.random() > self.p:
            return img
            
        return self._apply_dct_modifier(img)


class FakeFeatureSimulator:
    """模拟伪造特征,如伪造脸部的平滑度,质量不一致等"""
    def __init__(self, p=0.5):
        self.p = p
        
    def __call__(self, img, mask=None):
        if random.random() > self.p:
            return img
            
        # 如果有掩码,认为掩码区域是伪造区域
        # 如果没有,随机选择一个区域作为"伪造区域"
        img_np = np.array(img)
        h, w = img_np.shape[:2]
        
        if mask is None:
            # 创建随机多边形掩码
            points = []
            num_points = random.randint(3, 7)
            center_x, center_y = w // 2, h // 2
            for i in range(num_points):
                angle = 2 * np.pi * i / num_points
                r = random.uniform(0.2, 0.4) * min(w, h) / 2
                x = int(center_x + r * np.cos(angle) + random.uniform(-r/2, r/2))
                y = int(center_y + r * np.sin(angle) + random.uniform(-r/2, r/2))
                points.append([x, y])
                
            # 创建掩码
            mask_np = np.zeros((h, w), dtype=np.uint8)
            pts = np.array(points, dtype=np.int32).reshape((-1, 1, 2))
            cv2.fillPoly(mask_np, [pts], 255)
            
            # 应用羽化
            mask_np = cv2.GaussianBlur(mask_np, (21, 21), 11)
        else:
            mask_np = np.array(mask)
            
        if mask_np.max() > 1:
            mask_np = mask_np / 255.0
        
        # 随机选择一种伪造特征模拟:
        feature_type = random.choice(['smooth', 'quality', 'noise', 'boundary'])
        
        if feature_type == 'smooth':
            # 过度平滑,伪造区域通常更平滑
            blurred = cv2.GaussianBlur(img_np, (9, 9), 0)
            
            # 将掩码扩展到3通道(如果需要)
            if len(img_np.shape) == 3 and len(mask_np.shape) == 2:
                mask_3d = np.repeat(mask_np[:, :, np.newaxis], 3, axis=2)
            else:
                mask_3d = mask_np
                
            # 应用模糊
            result = img_np * (1 - mask_3d) + blurred * mask_3d
            
        elif feature_type == 'quality':
            # 质量下降,如JPEG压缩痕迹
            _, buffer = cv2.imencode('.jpg', img_np, [cv2.IMWRITE_JPEG_QUALITY, random.randint(50, 75)])
            degraded = cv2.imdecode(buffer, cv2.IMREAD_UNCHANGED)
            
            # 将掩码扩展到3通道(如果需要)
            if len(img_np.shape) == 3 and len(mask_np.shape) == 2:
                mask_3d = np.repeat(mask_np[:, :, np.newaxis], 3, axis=2)
            else:
                mask_3d = mask_np
                
            # 应用质量降低
            result = img_np * (1 - mask_3d) + degraded * mask_3d
            
        elif feature_type == 'noise':
            # 添加轻微噪声
            noise = np.random.normal(0, random.uniform(3, 10), img_np.shape).astype(np.float32)
            noisy = np.clip(img_np + noise, 0, 255).astype(np.uint8)
            
            # 将掩码扩展到3通道(如果需要)
            if len(img_np.shape) == 3 and len(mask_np.shape) == 2:
                mask_3d = np.repeat(mask_np[:, :, np.newaxis], 3, axis=2)
            else:
                mask_3d = mask_np
                
            # 应用噪声
            result = img_np * (1 - mask_3d) + noisy * mask_3d
            
        else:  # 'boundary'
            # 强调边界
            # 检测原始边界
            edges = cv2.Canny(img_np, 100, 200)
            
            # 扩展边界
            kernel = np.ones((3, 3), np.uint8)
            dilated_edges = cv2.dilate(edges, kernel, iterations=1)
            
            # 边界颜色轻微变化
            result = img_np.copy()
            
            # 在检测到的边界上应用轻微的色调变换
            if len(img_np.shape) == 3:  # 彩色图像
                for c in range(3):
                    factor = random.uniform(0.93, 1.07)
                    channel = result[:, :, c]
                    channel[dilated_edges > 0] = np.clip(channel[dilated_edges > 0] * factor, 0, 255)
            else:  # 灰度图像
                factor = random.uniform(0.93, 1.07)
                result[dilated_edges > 0] = np.clip(result[dilated_edges > 0] * factor, 0, 255)
        
        return Image.fromarray(result.astype(np.uint8))


class AdvancedForensicTransforms:
    """面向伪造检测的先进数据增强转换集合"""
    
    @staticmethod
    def get_transform(mode='train', input_size=256, use_dct=False, **kwargs):
        """
        获取训练/验证/测试转换
        
        Args:
            mode: 'train', 'val' 或 'test'
            input_size: 输入尺寸
            use_dct: 是否包含DCT转换
            **kwargs: 其他配置项
        """
        # 通用转换(调整大小和标准化)
        common_transforms = [
            transforms.Resize((input_size, input_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.5]*3, [0.5]*3)
        ]
        
        if mode == 'train':
            # 训练阶段使用更多数据增强
            train_transforms = [
                # 随机裁剪和翻转
                transforms.RandomResizedCrop(input_size, scale=(0.8, 1.0)),
                transforms.RandomHorizontalFlip(),
                
                # 颜色抖动
                transforms.ColorJitter(
                    brightness=0.2, 
                    contrast=0.2, 
                    saturation=0.2, 
                    hue=0.1
                ),
                
                # 高级增强
                ForensicJPEGArtifacts(quality_range=(60, 95), p=0.3),
                FrequencyDomainTransform(p=0.3),
                FakeFeatureSimulator(p=0.3)
            ]
            
            # 添加程度更轻的RandAugment
            if kwargs.get('use_randaugment', True):
                from torchvision.transforms import RandAugment
                train_transforms.append(
                    RandAugment(num_ops=2, magnitude=5)  # 轻微增强,避免破坏伪造特征
                )
            
            # 组合转换
            return transforms.Compose(train_transforms + common_transforms)
            
        elif mode in ['val', 'test']:
            # 验证/测试阶段保持简单
            return transforms.Compose(common_transforms)
    
    @staticmethod
    def get_paired_transform(mode='train', input_size=256, use_dct=False, **kwargs):
        """获取图像-掩码对的转换(保持一致性)"""
        # 训练阶段使用特殊变换
        if mode == 'train':
            class PairedTransform:
                def __init__(self, input_size=256, **kwargs):
                    self.input_size = input_size
                    
                    # 伪造特征增强器
                    self.fake_enhancer = FakeFeatureSimulator(p=0.3)
                    self.color_distortion = LocalColorDistortion(p=0.3)
                    self.boundary_blender = FakeRegionBlending(p=0.3)
                    
                    # 基本几何变换
                    self.resized_crop = False
                    self.flip = False
                    self.angle = 0
                    
                def __call__(self, img, mask):
                    # 记录原始尺寸
                    w, h = img.size
                    
                    # 1. 首先应用相同的几何变换
                    
                    # 随机调整裁剪(两者保持一致)
                    if random.random() > 0.5:
                        self.resized_crop = True
                        scale = random.uniform(0.8, 1.0)
                        crop_h = int(h * scale)
                        crop_w = int(w * scale)
                        i = random.randint(0, h - crop_h)
                        j = random.randint(0, w - crop_w)
                        img = TF.crop(img, i, j, crop_h, crop_w)
                        mask = TF.crop(mask, i, j, crop_h, crop_w)
                    
                    # 随机水平翻转(两者保持一致)
                    if random.random() > 0.5:
                        self.flip = True
                        img = TF.hflip(img)
                        mask = TF.hflip(mask)
                    
                    # 随机旋转(两者保持一致)
                    if random.random() > 0.7:
                        self.angle = random.choice([0, 90, 180, 270])
                        img = TF.rotate(img, self.angle)
                        mask = TF.rotate(mask, self.angle)
                    
                    # 2. 专门针对伪造检测的数据增强
                    
                    # 伪造特征模拟(使用掩码区域)
                    img = self.fake_enhancer(img, mask)
                    
                    # 局部颜色扭曲
                    img = self.color_distortion(img, mask)
                    
                    # 伪造区域边界混合
                    img = self.boundary_blender(img, mask)
                    
                    # 3. 常规处理
                    
                    # 调整大小
                    img = TF.resize(img, (self.input_size, self.input_size))
                    mask = TF.resize(mask, (self.input_size, self.input_size))
                    
                    # 转换为张量
                    img = TF.to_tensor(img)
                    mask = TF.to_tensor(mask)
                    
                    # 标准化图像(掩码保留原始值)
                    img = TF.normalize(img, [0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
                    
                    return img, mask
                    
            return PairedTransform(input_size, **kwargs)
        
        else:
            # 验证/测试阶段使用简单转换
            class SimplePairedTransform:
                def __init__(self, input_size=256):
                    self.input_size = input_size
                    
                def __call__(self, img, mask):
                    # 调整大小
                    img = TF.resize(img, (self.input_size, self.input_size))
                    mask = TF.resize(mask, (self.input_size, self.input_size))
                    
                    # 转换为张量
                    img = TF.to_tensor(img)
                    mask = TF.to_tensor(mask)
                    
                    # 标准化图像(掩码保留原始值)
                    img = TF.normalize(img, [0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
                    
                    return img, mask
                    
            return SimplePairedTransform(input_size)
            
    @staticmethod
    def get_dual_transform(mode='train', input_size=256, **kwargs):
        """获取RGB和DCT双输入的转换"""
        # 导入DCT变换模块
        from network.dct_transform import MultiScaleFrequencyExtractor

        # 获取基础转换
        base_transform = AdvancedForensicTransforms.get_transform(
            mode, input_size, **kwargs)

        class DualInputTransform:
            def __init__(self, base_transform, input_size=256):
                self.base_transform = base_transform
                self.input_size = input_size
                self.dct_extractor = None  # 懒加载

            def _get_dct_extractor(self, device):
                if self.dct_extractor is None:
                    self.dct_extractor = MultiScaleFrequencyExtractor(
                        in_channels=3, out_channels=63
                    ).to(device)
                return self.dct_extractor

            def __call__(self, img):
                # RGB输入使用基础转换
                rgb_tensor = self.base_transform(img)

                # 使用专业的DCT提取器
                with torch.no_grad():  # 推理模式
                    dct_extractor = self._get_dct_extractor(rgb_tensor.device)
                    dct_tensor = dct_extractor(rgb_tensor.unsqueeze(0)).squeeze(0)

                return rgb_tensor, dct_tensor

        return DualInputTransform(base_transform, input_size)
    
    # 在 transform.py 中的 AdvancedForensicTransforms 类中添加

    @staticmethod
    def get_complete_transform(mode='train', input_size=256, **kwargs):
        """获取完整的转换流水线: (img, mask) -> (rgb_tensor, dct_tensor, mask_tensor)"""
        # 导入DCT变换模块
        from network.dct_transform import MultiScaleFrequencyExtractor

        # 获取成对转换
        paired_transform = AdvancedForensicTransforms.get_paired_transform(
            mode, input_size, **kwargs)

        class CompleteTransform:
            def __init__(self, paired_transform, input_size=256):
                self.paired_transform = paired_transform
                self.input_size = input_size
                self.dct_extractor = None  # 懒加载

            def _get_dct_extractor(self, device):
                if self.dct_extractor is None:
                    self.dct_extractor = MultiScaleFrequencyExtractor(
                        in_channels=3, out_channels=63
                    ).to(device)
                return self.dct_extractor

            def __call__(self, img, mask=None):
                # 应用成对变换
                rgb_tensor, mask_tensor = self.paired_transform(img, mask)

                # 使用专业的DCT提取器
                with torch.no_grad():  # 推理模式
                    dct_extractor = self._get_dct_extractor(rgb_tensor.device)
                    dct_tensor = dct_extractor(rgb_tensor.unsqueeze(0)).squeeze(0)

                return rgb_tensor, dct_tensor, mask_tensor

        return CompleteTransform(paired_transform, input_size)


# 导出接口
forensic_transforms = AdvancedForensicTransforms()

# 使用示例:
# train_transform = forensic_transforms.get_transform(mode='train', input_size=256, use_dct=True)
# val_transform = forensic_transforms.get_transform(mode='val', input_size=256)
# train_paired_transform = forensic_transforms.get_paired_transform(mode='train', input_size=256)
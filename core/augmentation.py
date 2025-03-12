import torch
import numpy as np
from PIL import Image, ImageFilter, ImageEnhance, ImageOps
import cv2
from io import BytesIO
import random
from random import random as rand_float
import logging
import torchvision.transforms as transforms

class ForensicAugmenter:
    """伪造检测数据增强器，提供各种数据增强方法"""
    def __init__(self, config=None):
        """初始化增强器"""
        self.config = config
        self.enabled = config is not None and hasattr(config, 'DATA_AUGMENTATION') and config.DATA_AUGMENTATION.ENABLED
        self.logger = logging.getLogger("ForensicAugmenter")
        
        # 记录可用的增强方法
        self.available_augmentations = []
        if self.enabled:
            for aug_name in ['COLOR_JITTER', 'NOISE', 'BLUR', 'JPEG_COMPRESSION', 'CUTOUT', 'RANDOM_ROTATION', 'RANDOM_CROP']:
                if hasattr(self.config.DATA_AUGMENTATION, aug_name):
                    aug_config = getattr(self.config.DATA_AUGMENTATION, aug_name)
                    if getattr(aug_config, 'ENABLED', False):
                        self.available_augmentations.append(aug_name)
            
    def should_apply_mixup(self):
        """判断是否应用MixUp增强"""
        if not self.enabled:
            return False
            
        if hasattr(self.config.DATA_AUGMENTATION, 'MIXUP') and self.config.DATA_AUGMENTATION.MIXUP.ENABLED:
            prob = self.config.DATA_AUGMENTATION.MIXUP.PROBABILITY
            return rand_float() < prob
        return False
    
    def apply_color_jitter(self, img):
        """应用颜色抖动增强"""
        if not self.enabled or not hasattr(self.config.DATA_AUGMENTATION, 'COLOR_JITTER'):
            return img
            
        try:
            cfg = self.config.DATA_AUGMENTATION.COLOR_JITTER
            
            if not getattr(cfg, 'ENABLED', False) or rand_float() >= cfg.PROBABILITY:
                return img
                
            # 亮度调整
            brightness = getattr(cfg, 'BRIGHTNESS', 0.2)
            if brightness > 0 and rand_float() < 0.8:
                enhancer = ImageEnhance.Brightness(img)
                factor = rand_float() * brightness * 2 + (1 - brightness)
                img = enhancer.enhance(factor)
                
            # 对比度调整
            contrast = getattr(cfg, 'CONTRAST', 0.2)
            if contrast > 0 and rand_float() < 0.8:
                enhancer = ImageEnhance.Contrast(img)
                factor = rand_float() * contrast * 2 + (1 - contrast)
                img = enhancer.enhance(factor)
                
            # 饱和度调整
            saturation = getattr(cfg, 'SATURATION', 0.2)
            if saturation > 0 and rand_float() < 0.8:
                enhancer = ImageEnhance.Color(img)
                factor = rand_float() * saturation * 2 + (1 - saturation)
                img = enhancer.enhance(factor)
                
            # 色调调整
            hue = getattr(cfg, 'HUE', 0.1)
            if hue > 0 and rand_float() < 0.4:
                if rand_float() < 0.5:
                    img = ImageOps.posterize(img, int(rand_float() * 4) + 4)
                else:
                    # 使用HSV空间调整色调
                    img_np = np.array(img)
                    img_hsv = cv2.cvtColor(img_np, cv2.COLOR_RGB2HSV).astype(np.float32)
                    hue_shift = (rand_float() * 2 - 1) * hue * 180
                    img_hsv[:, :, 0] = (img_hsv[:, :, 0] + hue_shift) % 180
                    img_hsv = np.clip(img_hsv, 0, 255).astype(np.uint8)
                    img_rgb = cv2.cvtColor(img_hsv, cv2.COLOR_HSV2RGB)
                    img = Image.fromarray(img_rgb)
                
            return img
        except Exception as e:
            self.logger.warning(f"颜色抖动增强失败: {e}")
            return img

    def apply_random_noise(self, img):
        """应用随机噪声增强"""
        if not self.enabled or not hasattr(self.config.DATA_AUGMENTATION, 'NOISE'):
            return img
            
        try:
            cfg = self.config.DATA_AUGMENTATION.NOISE
            
            if not getattr(cfg, 'ENABLED', False) or rand_float() >= cfg.PROBABILITY:
                return img
                
            # 转换为numpy数组
            img_np = np.array(img).astype(np.float32)
            
            # 选择噪声类型
            noise_type = random.choice(['gaussian', 'poisson', 'salt_pepper']) if hasattr(cfg, 'NOISE_TYPES') else 'gaussian'
            
            if noise_type == 'gaussian':
                # 高斯噪声
                mean = getattr(cfg, 'GAUSSIAN_MEAN', 0)
                std = getattr(cfg, 'GAUSSIAN_STD', 0.1)
                noise = np.random.normal(mean, std * 255, img_np.shape)
                img_np = img_np + noise
            elif noise_type == 'salt_pepper':
                # 椒盐噪声
                s_vs_p = 0.5  # 盐比椒的比例
                amount = getattr(cfg, 'SALT_PEPPER_AMOUNT', 0.05)
                
                # 盐噪声 - 白点
                salt_mask = np.random.random(img_np.shape[:2]) < amount * s_vs_p
                img_np[salt_mask] = 255
                
                # 椒噪声 - 黑点
                pepper_mask = np.random.random(img_np.shape[:2]) < amount * (1 - s_vs_p)
                img_np[pepper_mask] = 0
            
            # 裁剪值到有效范围
            img_np = np.clip(img_np, 0, 255).astype(np.uint8)
            
            # 转回PIL图像
            img = Image.fromarray(img_np)
            
            return img
        except Exception as e:
            self.logger.warning(f"随机噪声增强失败: {e}")
            return img

    def apply_blur(self, img):
        """应用模糊效果增强"""
        if not self.enabled or not hasattr(self.config.DATA_AUGMENTATION, 'BLUR'):
            return img
            
        try:
            cfg = self.config.DATA_AUGMENTATION.BLUR
            
            if not getattr(cfg, 'ENABLED', False) or rand_float() >= cfg.PROBABILITY:
                return img
                
            # 简化为只使用高斯模糊
            radius = getattr(cfg, 'GAUSSIAN_RADIUS', 2)
            radius_value = rand_float() * radius
            img = img.filter(ImageFilter.GaussianBlur(radius=radius_value))
            
            return img
        except Exception as e:
            self.logger.warning(f"模糊效果增强失败: {e}")
            return img

    def apply_jpeg_compression(self, img):
        """应用JPEG压缩效果，模拟压缩伪影"""
        if not self.enabled or not hasattr(self.config.DATA_AUGMENTATION, 'JPEG_COMPRESSION'):
            return img
            
        try:
            cfg = self.config.DATA_AUGMENTATION.JPEG_COMPRESSION
            
            if not getattr(cfg, 'ENABLED', False) or rand_float() >= cfg.PROBABILITY:
                return img
                
            # 获取质量范围
            quality_min = getattr(cfg, 'QUALITY_MIN', 30)
            quality_max = getattr(cfg, 'QUALITY_MAX', 90)
            
            # 随机JPEG质量
            quality = int(rand_float() * (quality_max - quality_min) + quality_min)
            
            # 保存为JPEG并重新加载
            buffer = BytesIO()
            img.save(buffer, format="JPEG", quality=quality)
            buffer.seek(0)
            img = Image.open(buffer).convert('RGB')
            
            return img
        except Exception as e:
            self.logger.warning(f"JPEG压缩增强失败: {e}")
            return img
    
    def apply_cutout(self, img):
        """应用Cutout增强，随机遮挡图像部分区域"""
        if not self.enabled or not hasattr(self.config.DATA_AUGMENTATION, 'CUTOUT'):
            return img
            
        try:
            cfg = self.config.DATA_AUGMENTATION.CUTOUT
            
            if not getattr(cfg, 'ENABLED', False) or rand_float() >= cfg.PROBABILITY:
                return img
                
            img_np = np.array(img)
            h, w, c = img_np.shape
            
            # 获取参数
            n_holes = getattr(cfg, 'HOLES', 1)
            length_range = getattr(cfg, 'LENGTH_RANGE', [10, 40])
            
            for _ in range(n_holes):
                # 随机位置
                y = np.random.randint(h)
                x = np.random.randint(w)
                
                # 随机大小
                length = int(rand_float() * (length_range[1] - length_range[0]) + length_range[0])
                
                # 确保不超出图像边界
                y1 = np.clip(y - length // 2, 0, h)
                y2 = np.clip(y + length // 2, 0, h)
                x1 = np.clip(x - length // 2, 0, w)
                x2 = np.clip(x + length // 2, 0, w)
                
                # 填充黑色
                img_np[y1:y2, x1:x2, :] = 0
                    
            img = Image.fromarray(img_np)
            
            return img
        except Exception as e:
            self.logger.warning(f"Cutout增强失败: {e}")
            return img
    
    def apply_random_rotation(self, img, mask=None):
        """应用随机旋转增强"""
        if not self.enabled or not hasattr(self.config.DATA_AUGMENTATION, 'RANDOM_ROTATION'):
            return img, mask
            
        try:
            cfg = self.config.DATA_AUGMENTATION.RANDOM_ROTATION
            
            if not getattr(cfg, 'ENABLED', False) or rand_float() >= cfg.PROBABILITY:
                return img, mask
                
            # 获取最大旋转角度
            max_angle = getattr(cfg, 'MAX_ANGLE', 10)
            angle = (rand_float() * 2 - 1) * max_angle  # -max_angle到max_angle
            
            # 旋转图像
            img = img.rotate(angle, Image.BILINEAR, expand=False)
            
            # 同样处理掩码
            if mask is not None:
                mask = mask.rotate(angle, Image.NEAREST, expand=False)
                
            return img, mask
        except Exception as e:
            self.logger.warning(f"随机旋转增强失败: {e}")
            return img, mask
    
    def apply_random_crop(self, img, mask=None):
        """应用随机裁剪增强"""
        if not self.enabled or not hasattr(self.config.DATA_AUGMENTATION, 'RANDOM_CROP'):
            return img, mask
            
        try:
            cfg = self.config.DATA_AUGMENTATION.RANDOM_CROP
            
            if not getattr(cfg, 'ENABLED', False) or rand_float() >= cfg.PROBABILITY:
                return img, mask
                
            # 原始尺寸
            w, h = img.size
            
            # 获取裁剪参数
            scale_range = getattr(cfg, 'SCALE', [0.8, 1.0])
            ratio_range = getattr(cfg, 'RATIO', [0.8, 1.2])
            
            # 随机裁剪比例
            scale = rand_float() * (scale_range[1] - scale_range[0]) + scale_range[0]
            ratio = rand_float() * (ratio_range[1] - ratio_range[0]) + ratio_range[0]
            
            # 计算新尺寸
            new_h = int(h * scale)
            new_w = int(new_h * ratio)
            
            # 确保在图像范围内
            new_h = min(new_h, h)
            new_w = min(new_w, w)
            
            # 随机位置
            top = int(rand_float() * (h - new_h)) if h > new_h else 0
            left = int(rand_float() * (w - new_w)) if w > new_w else 0
            
            # 裁剪
            img = img.crop((left, top, left + new_w, top + new_h))
            # 调整回原始尺寸
            img = img.resize((w, h), Image.BILINEAR)
            
            # 同样处理掩码
            if mask is not None:
                mask = mask.crop((left, top, left + new_w, top + new_h))
                mask = mask.resize((w, h), Image.NEAREST)
                
            return img, mask
        except Exception as e:
            self.logger.warning(f"随机裁剪增强失败: {e}")
            return img, mask
    
    def apply_mixup(self, img1, mask1, label1, get_another_sample_fn):
        """应用MixUp数据增强"""
        if not self.enabled or not hasattr(self.config.DATA_AUGMENTATION, 'MIXUP'):
            return img1, mask1, label1
            
        try:
            cfg = self.config.DATA_AUGMENTATION.MIXUP
            
            if not getattr(cfg, 'ENABLED', False) or rand_float() >= cfg.PROBABILITY:
                return img1, mask1, label1
            
            # 获取第二个样本    
            img2, mask2, label2 = get_another_sample_fn()
            if img2 is None:
                return img1, mask1, label1
                
            # 确保尺寸一致
            img2 = img2.resize(img1.size, Image.BILINEAR)
            if mask2 is not None:
                mask2 = mask2.resize(mask1.size, Image.NEAREST)
            else:
                mask2 = Image.new('L', mask1.size, 0)
                
            # 获取混合系数
            alpha_range = getattr(cfg, 'ALPHA_RANGE', [0.2, 0.8])
            lam = rand_float() * (alpha_range[1] - alpha_range[0]) + alpha_range[0]
            
            # 混合图像
            img1_np = np.array(img1).astype(np.float32)
            img2_np = np.array(img2).astype(np.float32)
            mixed_img = (lam * img1_np + (1 - lam) * img2_np).astype(np.uint8)
            
            # 混合掩码
            mask1_np = np.array(mask1).astype(np.float32)
            mask2_np = np.array(mask2).astype(np.float32)
            mixed_mask = (lam * mask1_np + (1 - lam) * mask2_np).astype(np.uint8)
            
            # 混合标签
            mixed_label = lam * label1 + (1 - lam) * label2
            
            return Image.fromarray(mixed_img), Image.fromarray(mixed_mask), mixed_label
        except Exception as e:
            self.logger.warning(f"MixUp增强失败: {e}")
            return img1, mask1, label1
            
    def apply_augmentation_sequence(self, img, mask=None):
        """应用一系列增强操作"""
        # 几何变换 (需要同时处理图像和掩码)
        img, mask = self.apply_random_rotation(img, mask)
        img, mask = self.apply_random_crop(img, mask)
        
        # 图像内容变换 (仅处理图像)
        img = self.apply_color_jitter(img)
        img = self.apply_random_noise(img)
        img = self.apply_blur(img)
        img = self.apply_jpeg_compression(img)
        img = self.apply_cutout(img)
        
        return img, mask


# 帮助函数：验证配置
def verify_augmentation_config(config):
    """验证数据增强配置是否有效"""
    if not hasattr(config, 'DATA_AUGMENTATION'):
        print("警告: 配置中缺少DATA_AUGMENTATION部分")
        return False
        
    augmentation = config.DATA_AUGMENTATION
    if not getattr(augmentation, 'ENABLED', False):
        print("数据增强未启用")
        return False
        
    # 验证必要的增强配置
    required_augmentations = ['COLOR_JITTER', 'NOISE', 'BLUR', 'JPEG_COMPRESSION']
    
    for aug in required_augmentations:
        if not hasattr(augmentation, aug):
            print(f"警告: 缺少{aug}配置")
        elif not hasattr(getattr(augmentation, aug), 'ENABLED'):
            print(f"警告: {aug}配置缺少ENABLED字段")
            
    return True


# 创建默认增强配置
def create_default_augmentation_config():
    """创建包含默认增强设置的配置"""
    from easydict import EasyDict
    
    config = EasyDict()
    config.DATA_AUGMENTATION = EasyDict()
    config.DATA_AUGMENTATION.ENABLED = True
    
    # 颜色抖动
    config.DATA_AUGMENTATION.COLOR_JITTER = EasyDict()
    config.DATA_AUGMENTATION.COLOR_JITTER.ENABLED = True
    config.DATA_AUGMENTATION.COLOR_JITTER.PROBABILITY = 0.8
    config.DATA_AUGMENTATION.COLOR_JITTER.BRIGHTNESS = 0.2
    config.DATA_AUGMENTATION.COLOR_JITTER.CONTRAST = 0.2
    config.DATA_AUGMENTATION.COLOR_JITTER.SATURATION = 0.2
    config.DATA_AUGMENTATION.COLOR_JITTER.HUE = 0.1
    
    # 噪声
    config.DATA_AUGMENTATION.NOISE = EasyDict()
    config.DATA_AUGMENTATION.NOISE.ENABLED = True
    config.DATA_AUGMENTATION.NOISE.PROBABILITY = 0.5
    config.DATA_AUGMENTATION.NOISE.GAUSSIAN_MEAN = 0
    config.DATA_AUGMENTATION.NOISE.GAUSSIAN_STD = 0.05
    
    # 模糊
    config.DATA_AUGMENTATION.BLUR = EasyDict()
    config.DATA_AUGMENTATION.BLUR.ENABLED = True
    config.DATA_AUGMENTATION.BLUR.PROBABILITY = 0.4
    config.DATA_AUGMENTATION.BLUR.GAUSSIAN_RADIUS = 1.5
    
    # JPEG压缩
    config.DATA_AUGMENTATION.JPEG_COMPRESSION = EasyDict()
    config.DATA_AUGMENTATION.JPEG_COMPRESSION.ENABLED = True
    config.DATA_AUGMENTATION.JPEG_COMPRESSION.PROBABILITY = 0.5
    config.DATA_AUGMENTATION.JPEG_COMPRESSION.QUALITY_MIN = 50
    config.DATA_AUGMENTATION.JPEG_COMPRESSION.QUALITY_MAX = 95
    
    # Cutout
    config.DATA_AUGMENTATION.CUTOUT = EasyDict()
    config.DATA_AUGMENTATION.CUTOUT.ENABLED = True
    config.DATA_AUGMENTATION.CUTOUT.PROBABILITY = 0.5
    config.DATA_AUGMENTATION.CUTOUT.HOLES = 3
    config.DATA_AUGMENTATION.CUTOUT.LENGTH_RANGE = [20, 60]

    # MixUp
    config.DATA_AUGMENTATION.MIXUP = EasyDict()
    config.DATA_AUGMENTATION.MIXUP.ENABLED = True
    config.DATA_AUGMENTATION.MIXUP.PROBABILITY = 0.2
    config.DATA_AUGMENTATION.MIXUP.ALPHA_RANGE = [0.2, 0.8]
    
    return config


def create_tensor_transforms(config=None):
    """创建标准的张量转换管道"""
    if config is None:
        config = {
            'RESIZE': (256, 256),
            'NORMALIZE_MEAN': [0.5, 0.5, 0.5],
            'NORMALIZE_STD': [0.5, 0.5, 0.5],
            'GRAY_NORMALIZE_MEAN': [0.5],
            'GRAY_NORMALIZE_STD': [0.5]
        }
        
    # RGB图像转换
    rgb_transform = transforms.Compose([
        transforms.Resize(config['RESIZE']),
        transforms.ToTensor(),
        transforms.Normalize(config['NORMALIZE_MEAN'], config['NORMALIZE_STD'])
    ])
    
    # 灰度图像转换
    gray_transform = transforms.Compose([
        transforms.Resize(config['RESIZE']),
        transforms.ToTensor(),
        transforms.Normalize(config['GRAY_NORMALIZE_MEAN'], config['GRAY_NORMALIZE_STD'])
    ])
    
    # 返回转换组合
    return {
        'rgb': rgb_transform,
        'gray': gray_transform
    }


def visualize_augmentations(image_path, config=None, num_samples=5, save_path=None):
    """可视化数据增强效果"""
    import matplotlib.pyplot as plt
    
    # 加载图像
    try:
        image = Image.open(image_path).convert('RGB')
    except Exception as e:
        print(f"无法加载图像 {image_path}: {e}")
        return
    
    # 创建增强器
    if config is None:
        config = create_default_augmentation_config()
    
    augmenter = ForensicAugmenter(config)
    
    # 创建一个matplotlib图形
    plt.figure(figsize=(num_samples * 4, 4))
    
    # 显示原始图像
    plt.subplot(1, num_samples + 1, 1)
    plt.imshow(image)
    plt.title('原始图像')
    plt.axis('off')
    
    # 生成并显示增强后的样本
    for i in range(num_samples):
        # 复制原始图像
        img = image.copy()
        mask = Image.new('L', image.size, 0)  # 创建空白掩码用于演示
        
        # 应用一系列增强
        img, mask = augmenter.apply_augmentation_sequence(img, mask)
        
        # 显示增强后的图像
        plt.subplot(1, num_samples + 1, i + 2)
        plt.imshow(img)
        plt.title(f'样本 {i+1}')
        plt.axis('off')
    
    plt.tight_layout()
    
    # 保存或显示
    if save_path:
        plt.savefig(save_path, dpi=300)
        print(f"可视化结果已保存至: {save_path}")
    else:
        plt.show()
    
    plt.close()


def get_augmentation_pipeline(config=None, is_train=True):
    """获取数据增强管道，便于外部调用"""
    if not is_train:
        # 如果不是训练模式，只返回标准张量转换
        transforms_dict = create_tensor_transforms()
        return transforms_dict['rgb']
    
    augmenter = ForensicAugmenter(config)
    
    def augment_fn(img, mask=None):
        """应用增强管道"""
        # 应用一系列增强
        img, mask = augmenter.apply_augmentation_sequence(img, mask)
        
        # 应用张量转换
        transforms_dict = create_tensor_transforms()
        img_tensor = transforms_dict['rgb'](img)
        mask_tensor = transforms_dict['gray'](mask) if mask is not None else None
        
        if mask is not None:
            return img_tensor, mask_tensor
        else:
            return img_tensor
    
    return augment_fn


if __name__ == "__main__":
    # 测试增强器
    import sys
    import os
    
    # 解析命令行参数
    if len(sys.argv) > 1:
        image_path = sys.argv[1]
        
        if not os.path.exists(image_path):
            print(f"图像路径不存在: {image_path}")
            sys.exit(1)
            
        # 创建默认配置
        config = create_default_augmentation_config()
        
        # 可视化增强效果
        visualize_augmentations(
            image_path,
            config,
            num_samples=5, 
            save_path='augmentation_samples.png'
        )
        
        print("数据增强示例已生成")
    else:
        print("用法: python augmentation.py <图像路径>")
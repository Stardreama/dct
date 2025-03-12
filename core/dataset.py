import os
import torch
import numpy as np
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms
import logging
from pathlib import Path
import random
# 导入增强的数据转换
from network.transform import (
    forensic_transforms, 
    ForensicJPEGArtifacts, 
    FrequencyDomainTransform, 
    FakeFeatureSimulator
)

# 导入DCT变换
from network.dct_transform import DCTTransform, MultiScaleFrequencyExtractor

class BaseForensicDataset(Dataset):
    """
    基础伪造检测数据集
    提供共享的数据加载和掩码处理功能
    """
    def __init__(self, img_paths, dataset_type, 
                 aug_transform=None, tensor_transform=None, 
                 config=None, return_path=False):
        """
        初始化数据集
        
        Args:
            img_paths: 数据集根目录路径
            dataset_type: 数据集类型('train', 'val', 'test')
            aug_transform: 数据增强转换
            tensor_transform: 张量转换
            config: 配置参数
            return_path: 是否在__getitem__中返回图片路径
        """
        self.img_paths = img_paths
        self.dataset_type = dataset_type
        self.aug_transform = aug_transform
        self.tensor_transform = tensor_transform
        self.config = config
        self.return_path = return_path
        
        # 是否为训练模式
        self.is_train = dataset_type == 'train'
        
        # 灰度图像转换
        self.tensor_transform_gray = transforms.Compose([
            transforms.Resize(256),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5])])
            
        # 读取索引文件
        self.sample_list = []
        self.typepath = os.path.join(img_paths, f"{self.dataset_type}.txt")
        if os.path.exists(self.typepath):
            with open(self.typepath) as f:
                lines = f.readlines()
                for line in lines:
                    self.sample_list.append(line.strip())
        else:
            logging.error(f"找不到索引文件: {self.typepath}")
        
        # 设置日志记录器
        self.logger = logging.getLogger(f"{self.__class__.__name__}_{dataset_type}")
        
    def _extract_forgery_type(self, img_path):
        """
        提取图像的伪造类型
        
        Args:
            img_path: 图像路径
            
        Returns:
            str: 伪造类型
        """
        if "real" in img_path.lower() or "original" in img_path.lower():
            return "real"
            
        if "fake" not in img_path.lower():
            return "real"
            
        # 尝试从路径中提取伪造类型
        path_parts = img_path.split(os.sep)
        fake_idx = path_parts.index("fake") if "fake" in path_parts else -1
        
        if fake_idx != -1 and fake_idx + 1 < len(path_parts):
            return path_parts[fake_idx + 1]
        
        # 根据常见伪造类型关键词识别
        if "deepfakes" in img_path.lower():
            return "deepfakes"
        elif "face2face" in img_path.lower() or "f2f" in img_path.lower():
            return "face2face"
        elif "faceswap" in img_path.lower():
            return "faceswap"
        elif "neuraltextures" in img_path.lower() or "nt" in img_path.lower():
            return "neuraltextures"
        elif "gan" in img_path.lower() or "stylegan" in img_path.lower():
            return "gan"
        else:
            return "unknown_fake"
    
    def _load_mask_for_path(self, img_path, img_size):
        """
        通用掩码加载方法
        
        Args:
            img_path: 图像路径
            img_size: 图像尺寸，用于创建空白掩码
            
        Returns:
            PIL.Image: 加载的掩码图像
        """
        try:
            if "fake" in img_path:
                # 提取文件路径组件
                path_components = img_path.split(os.path.sep)
                
                # 找出关键部分
                fake_idx = path_components.index("fake") if "fake" in path_components else -1
                
                if fake_idx != -1 and fake_idx + 1 < len(path_components):
                    fake_type = path_components[fake_idx + 1]
                    video_id = path_components[fake_idx + 2] if fake_idx + 2 < len(path_components) else ""
                    frame_name = path_components[-1]
                    
                    # 构建掩码路径
                    try:
                        dataset_idx = path_components.index("dataset") if "dataset" in path_components else -1
                        if dataset_idx != -1:
                            base_dir = os.path.join(*path_components[:dataset_idx + 1])
                            mask_path = os.path.join(base_dir, "mask", fake_type, video_id, frame_name)
                        else:
                            # 如果找不到dataset目录，尝试从fake目录构建
                            base_dir = os.path.join(*path_components[:fake_idx])
                            mask_path = os.path.join(base_dir, "mask", fake_type, video_id, frame_name)
                    except Exception as e:
                        self.logger.warning(f"掩码路径构建错误: {e}")
                        return Image.new('L', img_size, 0)
                    
                    if os.path.exists(mask_path):
                        return Image.open(mask_path).convert('L')
                    else:
                        # 掩码不存在，创建空白掩码
                        self.logger.debug(f"找不到掩码: {mask_path}")
                        return Image.new('L', img_size, 0)
                else:
                    return Image.new('L', img_size, 0)
            else:
                # 真实图像，使用空白掩码
                return Image.new('L', img_size, 0)
        except Exception as e:
            self.logger.warning(f"无法加载掩码 {img_path}, {e}")
            # 创建空白掩码
            return Image.new('L', img_size, 0)
    
    def _analyze_dataset_stats(self):
        """
        通用数据集统计分析
        
        Returns:
            dict: 数据集统计信息
        """
        real_count = 0
        fake_count = 0
        fake_types = {}
        
        for item in self.sample_list:
            parts = item.split(' ')
            if len(parts) < 2:
                continue
                
            img_path = parts[0]
            label = int(parts[1]) if parts[1].isdigit() else 0
            
            if label == 0:  # 真实图像
                real_count += 1
                forgery_type = "real"
            else:  # 伪造图像
                fake_count += 1
                forgery_type = self._extract_forgery_type(img_path)
            
            # 增加计数
            if forgery_type not in fake_types:
                fake_types[forgery_type] = 0
            fake_types[forgery_type] += 1
        
        stats = {
            "total": len(self.sample_list),
            "real": real_count,
            "fake": fake_count,
            "forgery_types": fake_types
        }
        
        return stats
        
    def _safe_image_loading(self, img_path, fallback_index=None):
        """
        安全地加载图像，处理可能的异常
        
        Args:
            img_path: 图像路径
            fallback_index: 如果加载失败，回退到的样本索引
            
        Returns:
            PIL.Image 或者 None 如果无法加载
        """
        try:
            return Image.open(img_path).convert('RGB')
        except Exception as e:
            self.logger.warning(f"错误: 无法加载图像 {img_path}, {e}")
            if fallback_index is not None and 0 <= fallback_index < len(self.sample_list):
                # 尝试加载其他图像
                fallback_idx = (fallback_index + 1) % len(self.sample_list)
                fallback_path = self.sample_list[fallback_idx].split(' ')[0]
                self.logger.info(f"尝试加载替代图像: {fallback_path}")
                return self._safe_image_loading(fallback_path)  # 防止递归过深
            return None
    
    def print_dataset_info(self, stats=None):
        """
        打印数据集统计信息
        
        Args:
            stats: 数据集统计信息，如果为None则重新计算
        """
        if stats is None:
            stats = self._analyze_dataset_stats()
            
        if stats["total"] == 0:
            self.logger.warning(f"数据集 '{self.dataset_type}' 为空!")
            return
            
        print(f"\n数据集 '{self.dataset_type}' 统计:")
        print(f"- 总样本数: {stats['total']}")
        print(f"- 真实图像: {stats['real']} ({stats['real']/stats['total']*100:.1f}%)")
        print(f"- 伪造图像: {stats['fake']} ({stats['fake']/stats['total']*100:.1f}%)")
        print("- 伪造类型分布:")
        
        # 按数量排序展示伪造类型
        sorted_types = sorted(
            [(t, c) for t, c in stats['forgery_types'].items() if t != "real"], 
            key=lambda x: x[1], 
            reverse=True
        )
        
        for ft, count in sorted_types:
            if ft != "real":  # 避免重复显示真实样本
                print(f"  * {ft}: {count} 样本")
        print("")
            
    def __len__(self):
        """返回数据集长度"""
        return len(self.sample_list)

    # 添加DCT提取器相关方法
    def _get_dct_extractor(self, device='cpu'):
        """懒加载DCT特征提取器"""
        if self.dct_extractor is None:
            # 检查配置中的DCT设置
            dct_config = {}
            if self.config and hasattr(self.config, 'DCT_TRANSFORM'):
                dct_config = self.config.DCT_TRANSFORM
                
            # 设置默认值
            out_channels = dct_config.get('OUT_CHANNELS', 12) if isinstance(dct_config, dict) else 12
            multi_scale = dct_config.get('MULTI_SCALE', True) if isinstance(dct_config, dict) else True
            
            # 创建提取器
            if multi_scale:
                self.dct_extractor = MultiScaleFrequencyExtractor(in_channels=3, out_channels=out_channels)
            else:
                self.dct_extractor = DCTTransform(in_channels=3, out_channels=out_channels)
                
            self.dct_extractor = self.dct_extractor.to(device)
            
        return self.dct_extractor
        

class TrainingForensicDataset(BaseForensicDataset):
    """训练用的增强数据集"""
    def __init__(self, img_paths, type, dataset_type,
                 aug_transform=None, tensor_transform=None, config=None):
        """
        初始化训练数据集
        
        Args:
            img_paths: 数据集根目录路径
            type: 数据集类型标识符(用于兼容旧代码)
            dataset_type: 数据集类型('train', 'val', 'test')
            aug_transform: 数据增强转换
            tensor_transform: 张量转换 
            config: 配置参数
        """
        super().__init__(img_paths, dataset_type, aug_transform, tensor_transform, config)
        self.type = type
        
        # 分析并打印数据集信息
        stats = self._analyze_dataset_stats()
        self.print_dataset_info(stats)
        
    def __getitem__(self, index):
        """获取单个训练样本"""
        if index >= len(self.sample_list):
            raise IndexError(f"索引 {index} 超出范围 (0-{len(self.sample_list)-1})")
            
        item = self.sample_list[index]
        parts = item.split(' ')
        if len(parts) < 2:
            self.logger.warning(f"样本格式错误: {item}")
            # 回退到下一个样本
            return self.__getitem__((index + 1) % len(self.sample_list))
            
        img_path = parts[0]
        label = int(parts[1])
        
        # 安全加载图像
        img = self._safe_image_loading(img_path, index)
        if img is None:
            # 如果加载失败，返回一个全黑图像
            img = Image.new('RGB', (256, 256), (0, 0, 0))
        
        # 使用辅助方法加载掩码
        img_mask = self._load_mask_for_path(img_path, img.size)
            
        # 仅在训练阶段应用数据增强
        if self.is_train:
            try:
                # 导入增强器
                from core.augmentation import ForensicAugmenter
                
                # 创建增强器实例
                augmenter = ForensicAugmenter(self.config)
                
                # 应用增强
                if random.random() < 0.5:
                    img = img.transpose(Image.FLIP_LEFT_RIGHT)
                    img_mask = img_mask.transpose(Image.FLIP_LEFT_RIGHT)
                    
                # 调用增强器方法
                img = augmenter.apply_color_jitter(img)
                img = augmenter.apply_random_noise(img)
                img = augmenter.apply_blur(img)
                img = augmenter.apply_jpeg_compression(img)
                img = augmenter.apply_cutout(img)
                
                # 几何变换需要同时应用到掩码
                img, img_mask = augmenter.apply_random_rotation(img, img_mask)
                img, img_mask = augmenter.apply_random_crop(img, img_mask)
                
                # 可以选择性应用MixUp增强
                if hasattr(augmenter, 'should_apply_mixup') and augmenter.should_apply_mixup():
                    # 创建获取随机样本的函数
                    def get_random_sample():
                        rand_idx = random.randint(0, len(self.sample_list) - 1)
                        rand_item = self.sample_list[rand_idx]
                        rand_parts = rand_item.split(' ')
                        rand_path = rand_parts[0]
                        rand_label = int(rand_parts[1])
                        
                        rand_img = self._safe_image_loading(rand_path)
                        rand_mask = self._load_mask_for_path(rand_path, rand_img.size if rand_img else (256, 256))
                        
                        return rand_img, rand_mask, rand_label
                    
                    # 应用MixUp
                    img, img_mask, label = augmenter.apply_mixup(img, img_mask, label, get_random_sample)
                
            except (ImportError, AttributeError) as e:
                self.logger.debug(f"无法应用高级增强: {e}")

        # 应用用户提供的增强
        if self.aug_transform is not None:
            img = self.aug_transform(img)

        # 张量转换
        if self.tensor_transform is not None:
            img = self.tensor_transform(img)
            img_mask = self.tensor_transform_gray(img_mask)
        else:
            # 默认转换为张量
            to_tensor = transforms.ToTensor()
            img = to_tensor(img)
            img_mask = to_tensor(img_mask)

        return img, img_mask, label


class TestForensicDataset(BaseForensicDataset):
    """测试用的数据集"""
    def __init__(self, dataset_dir, split="test", transform=None, return_path=False):
        """
        初始化测试数据集
        
        Args:
            dataset_dir: 数据集根目录
            split: 数据集划分类型('train', 'val', 'test')
            transform: 数据变换
            return_path: 是否返回文件路径
        """
        super().__init__(dataset_dir, split, None, transform, None, return_path)
        
        # 收集伪造类型和样本信息
        self.forgery_types = set()
        self.samples = []
        
        for item in self.sample_list:
            parts = item.strip().split()
            if len(parts) < 2:
                self.logger.warning(f"无效的样本行: {item}")
                continue
                
            img_path, label = parts[0], parts[1]
            
            # 分析伪造类型
            forgery_type = self._extract_forgery_type(img_path)
            
            self.forgery_types.add(forgery_type)
            self.samples.append((img_path, int(label), forgery_type))
        
        # 统计数据集信息
        stats = self._analyze_dataset_stats()
        self.print_dataset_info(stats)
        
    def __getitem__(self, index):
        """获取单个测试样本"""
        if index >= len(self.samples):
            raise IndexError(f"索引 {index} 超出范围 (0-{len(self.samples)-1})")
            
        img_path, label, forgery_type = self.samples[index]
        
        # 安全加载图像
        img = self._safe_image_loading(img_path, index)
        if img is None:
            # 如果加载失败，返回一个全黑图像
            img = Image.new('RGB', (256, 256), (0, 0, 0))
        
        # 加载掩码
        mask = self._load_mask_for_path(img_path, img.size)
            
        # 应用转换
        if self.tensor_transform:
            img = self.tensor_transform(img)
            mask = self.tensor_transform_gray(mask)
        else:
            # 默认转换为张量
            to_tensor = transforms.ToTensor()
            img = to_tensor(img)
            mask = to_tensor(mask)
            
        if self.return_path:
            return img, mask, label, img_path, forgery_type
        else:
            return img, mask, label


class EnhancedForensicDataset(BaseForensicDataset):
    """增强的伪造检测数据集，支持掩码和DCT特征"""
    def __init__(self, img_paths, dataset_type, 
                 config=None, return_path=False, use_dct=True):
        """
        Args:
            img_paths: 数据集根目录
            dataset_type: 'train', 'val' 或 'test'
            config: 配置参数
            return_path: 是否返回文件路径
            use_dct: 是否提取DCT特征
        """
        # 基础初始化
        super().__init__(img_paths, dataset_type, None, None, config, return_path)
        self.use_dct = use_dct
        
        # 使用增强的转换
        if dataset_type == 'train':
            self.transform = forensic_transforms.get_transform(
                mode='train', 
                input_size=256, 
                use_dct=use_dct, 
                **config.DATA_AUGMENTATION if hasattr(config, 'DATA_AUGMENTATION') else {}
            )
            self.paired_transform = forensic_transforms.get_paired_transform(
                mode='train',
                input_size=256,
                **config.DATA_AUGMENTATION if hasattr(config, 'DATA_AUGMENTATION') else {}
            )
        else:
            self.transform = forensic_transforms.get_transform(
                mode=dataset_type,
                input_size=256,
                use_dct=use_dct
            )
            self.paired_transform = forensic_transforms.get_paired_transform(
                mode=dataset_type,
                input_size=256
            )
        
        # 分析并打印数据集信息
        stats = self._analyze_dataset_stats()
        self.print_dataset_info(stats)
    
    def __getitem__(self, index):
        """获取数据集样本，支持DCT特征和掩码"""
        if index >= len(self.sample_list):
            raise IndexError(f"索引 {index} 超出范围 (0-{len(self.sample_list)-1})")
            
        item = self.sample_list[index]
        parts = item.split(' ')
        if len(parts) < 2:
            self.logger.warning(f"样本格式错误: {item}")
            # 回退到下一个样本
            return self.__getitem__((index + 1) % len(self.sample_list))
            
        img_path = parts[0]
        label = int(parts[1])
        
        # 安全加载图像
        img = self._safe_image_loading(img_path, index)
        if img is None:
            # 如果加载失败，返回一个全黑图像
            img = Image.new('RGB', (256, 256), (0, 0, 0))
        
        # 加载掩码
        img_mask = self._load_mask_for_path(img_path, img.size)
        
        # 应用成对转换 - 同时处理图像和掩码
        if self.paired_transform:
            img, img_mask = self.paired_transform(img, img_mask)
        
        # 如果需要DCT特征
        if self.use_dct:
            try:
                # 使用设备无关的方式生成DCT特征
                with torch.no_grad():
                    # 确保图像是张量格式
                    if not torch.is_tensor(img):
                        img_tensor = transforms.ToTensor()(img).unsqueeze(0)
                    else:
                        img_tensor = img.unsqueeze(0) if img.dim() == 3 else img
                    
                    # 提取DCT特征
                    dct_extractor = self._get_dct_extractor(img_tensor.device)
                    dct_features = dct_extractor(img_tensor).squeeze(0)
                    
                    return img, dct_features, img_mask, label
            except Exception as e:
                self.logger.warning(f"DCT特征提取失败: {e}")
                # 返回空DCT特征
                dct_features = torch.zeros((12, 256, 256), device=img.device if torch.is_tensor(img) else 'cpu')
                return img, dct_features, img_mask, label
        else:
            # 不需要DCT特征
            return img, img_mask, label


# 更新数据加载器创建函数
def create_forensic_data_loaders(config, use_enhanced_dataset=True):
    """创建数据加载器，支持掩码和DCT双输入"""
    # 提取配置参数
    train_path = config.TRAIN_PATH
    val_path = config.VAL_PATH
    test_path = config.TEST_PATH
    batch_size = config.BATCH_SIZE
    num_workers = config.WORKERS if hasattr(config, 'WORKERS') else 4
    
    # 检查是否使用DCT
    use_dct = True
    if hasattr(config, 'DCT_TRANSFORM') and isinstance(config.DCT_TRANSFORM, dict):
        use_dct = config.DCT_TRANSFORM.get('ENABLED', True)
    
    # 创建增强数据集
    if use_enhanced_dataset:
        train_dataset = EnhancedForensicDataset(
            img_paths=train_path,
            dataset_type='train',
            config=config,
            use_dct=use_dct
        )
        
        val_dataset = EnhancedForensicDataset(
            img_paths=val_path,
            dataset_type='val',
            config=config,
            use_dct=use_dct
        )
        
        test_dataset = EnhancedForensicDataset(
            img_paths=test_path,
            dataset_type='test',
            config=config,
            return_path=True,
            use_dct=use_dct
        )
    else:
        # 使用原始数据集（保留原代码）
        # ...
    
    # 创建数据加载器
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True
    )
    
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return train_loader, val_loader, test_loader


if __name__ == "__main__":
    # 简单的测试代码，用于验证dataset模块
    import argparse
    import yaml
    import easydict
    
    parser = argparse.ArgumentParser(description="测试数据集模块")
    parser.add_argument('--config', type=str, default='config.yaml', help='配置文件路径')
    args = parser.parse_args()
    
    # 加载配置
    with open(args.config, 'r') as stream:
        config = yaml.safe_load(stream)
    config = easydict.EasyDict(config)
    
    # 测试数据集
    print("测试训练集...")
    train_dataset = TrainingForensicDataset(
        img_paths=config.TRAIN_PATH,
        type=config.TYPE if hasattr(config, 'TYPE') else "",
        dataset_type='train',
        tensor_transform=transforms.ToTensor(),
        config=config
    )
    
    print(f"训练集大小: {len(train_dataset)}")
    if len(train_dataset) > 0:
        sample = train_dataset[0]
        print(f"样本形状: 图像={sample[0].shape}, 掩码={sample[1].shape}, 标签={sample[2]}")
    
    # 测试测试集
    print("\n测试测试集...")
    test_dataset = TestForensicDataset(
        dataset_dir=config.TEST_PATH,
        return_path=True
    )
    
    print(f"测试集大小: {len(test_dataset)}")
    if len(test_dataset) > 0:
        sample = test_dataset[0]
        if len(sample) > 3:
            print(f"样本: 图像={sample[0].shape}, 掩码={sample[1].shape}, 标签={sample[2]}")
            print(f"路径: {sample[3]}")
            print(f"伪造类型: {sample[4]}")
    
    # 测试增强数据集
    print("\n测试增强数据集...")
    enhanced_dataset = EnhancedForensicDataset(
        img_paths=config.TRAIN_PATH,
        dataset_type='train',
        config=config,
        use_dct=True
    )
    
    print(f"增强数据集大小: {len(enhanced_dataset)}")
    if len(enhanced_dataset) > 0:
        sample = enhanced_dataset[0]
        print(f"样本形状: 图像={sample[0].shape}, DCT特征={sample[1].shape}, "
              f"掩码={sample[2].shape}, 标签={sample[3]}")
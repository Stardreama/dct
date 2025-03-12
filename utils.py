import torch
import os
import numpy as np
import logging
from PIL import Image
import sys
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from pathlib import Path
import cv2
from tqdm import tqdm

# 保留import mesonet_data_transforms
from network.transform import mesonet_data_transforms


def setup_logger(logger_path, log_file, logger_name):
    """设置日志记录器

    Args:
        logger_path: 日志文件目录
        log_file: 日志文件名
        logger_name: 日志记录器名称
    
    Returns:
        logging.Logger: 配置好的日志记录器
    """
    logger = logging.getLogger(logger_name)
    file_formatter = logging.Formatter('%(asctime)s - %(message)s')
    console_formatter = logging.Formatter('%(message)s')

    # 创建目录（如果不存在）
    if not os.path.exists(logger_path):
        os.makedirs(logger_path)

    # 文件处理器
    file_handler = logging.FileHandler(os.path.join(logger_path, log_file))
    file_handler.setFormatter(file_formatter)
    logger.addHandler(file_handler)

    # 控制台处理器
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)

    logger.setLevel(logging.DEBUG)
    
    return logger


def set_seed(seed):
    """设置随机种子以确保可重现性

    Args:
        seed: 随机种子值
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def evaluate(model, config, test_path, batch_size=16):
    """评估模型性能 (兼容旧版本)

    Args:
        model: 模型实例
        config: 配置对象
        test_path: 测试数据路径
        batch_size: 批量大小

    Returns:
        float: 准确率
    """
    from core.evaluation import ModelEvaluator
    from core.dataset import TestForensicDataset
    import torchvision.transforms as transforms
    from torch.utils.data import DataLoader
    
    # 数据变换
    test_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])
    
    # 创建数据集和加载器
    test_dataset = TestForensicDataset(
        dataset_dir=test_path, 
        split="test", 
        transform=test_transform,
        return_path=False
    )
    
    test_loader = DataLoader(
        test_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=4
    )
    
    # 使用ModelEvaluator进行评估
    device = next(model.model.parameters()).device
    evaluator = ModelEvaluator()
    metrics = evaluator.evaluate_model(model.model, test_loader, device)
    
    return metrics['accuracy']


def load_model_weights(model, path, strict=True):
    """加载模型权重，处理各种情况

    Args:
        model: 模型实例
        path: 权重文件路径
        strict: 是否严格加载权重
        
    Returns:
        模型实例: 加载权重后的模型
    """
    if not os.path.exists(path):
        print(f"警告: 模型权重文件不存在: {path}")
        return model
        
    try:
        checkpoint = torch.load(path, map_location='cpu')
        
        # 处理不同保存格式
        if 'model' in checkpoint:
            model.load_state_dict(checkpoint['model'], strict=strict)
        elif 'state_dict' in checkpoint:
            model.load_state_dict(checkpoint['state_dict'], strict=strict)
        else:
            model.load_state_dict(checkpoint, strict=strict)
            
        print(f"成功加载模型权重: {path}")
        return model
    except Exception as e:
        print(f"加载模型权重出错: {e}")
        return model


def save_checkpoint(state, is_best, save_path, filename='checkpoint.pth'):
    """保存检查点和最佳模型

    Args:
        state: 要保存的状态字典
        is_best: 是否为最佳模型
        save_path: 保存目录
        filename: 检查点文件名
    """
    if not os.path.exists(save_path):
        os.makedirs(save_path)
        
    # 保存检查点
    checkpoint_path = os.path.join(save_path, filename)
    torch.save(state, checkpoint_path)
    
    # 如果是最佳模型，单独保存一份
    if is_best:
        best_path = os.path.join(save_path, 'model_best.pth')
        torch.save(state, best_path)


# 保留一些工具函数
class AverageMeter(object):
    """跟踪平均值和当前值"""
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


# 向前兼容：提供简化的接口以兼容已有代码
def get_forensic_evaluator(save_dir=None):
    """获取伪造检测评估器（兼容原有代码）"""
    from core.evaluation import ModelEvaluator
    return ModelEvaluator(save_dir=save_dir)


def get_forensic_visualizer(save_dir=None):
    """获取伪造检测可视化器（兼容原有代码）"""
    from core.visualization import ForensicVisualizer
    return ForensicVisualizer(save_dir=save_dir)


def get_forensic_dataset(img_paths, dataset_type, transform=None, config=None, return_path=False):
    """获取伪造检测数据集（兼容原有代码）"""
    from core.dataset import TestForensicDataset
    return TestForensicDataset(img_paths, split=dataset_type, transform=transform, return_path=return_path)


# 创建一个类似于FaceDataset的评估数据集类 (保留以兼容旧代码)
class EvalDataset(torch.utils.data.Dataset):
    """兼容旧版本的评估数据集类"""
    def __init__(self, img_paths, dataset_type, transform=None):
        from core.dataset import BaseForensicDataset
        
        # 提示使用新的数据集类
        print("注意: EvalDataset已被弃用，请使用TestForensicDataset")
        
        self.base_dataset = BaseForensicDataset(img_paths, dataset_type, None, transform, None, False)
        self.transform = transform

    def __getitem__(self, index):
        # 只返回图像和标签，不返回掩码
        img, mask, label = self.base_dataset[index]
        return img, label

    def __len__(self):
        return len(self.base_dataset)


# 以下为没有在核心模块中实现的功能或需要保持兼容性的功能

class FFDataset(torch.utils.data.Dataset):
    """
    Face Forensics数据集，保留以兼容旧代码
    """
    def __init__(self, dataset_root, frame_num=300, size=299, augment=True):
        import torchvision.transforms as transforms
        
        self.data_root = dataset_root
        self.frame_num = frame_num
        self.train_list = self.collect_image(self.data_root)
        if augment:
            self.transform = transforms.Compose([
                transforms.RandomHorizontalFlip(p=0.5), 
                transforms.ToTensor()
            ])
            print("Augment True!")
        else:
            self.transform = transforms.ToTensor()
        self.max_val = 1.
        self.min_val = -1.
        self.size = size

    def collect_image(self, root):
        image_path_list = []
        print(root)
        for split in os.listdir(root):
            split_root = os.path.join(root, split)
            print(split_root)
            img_list = os.listdir(split_root)
            np.random.shuffle(img_list)
            img_list = img_list if len(img_list) < self.frame_num else img_list[:self.frame_num]
            for img in img_list:
                img_path = os.path.join(split_root, img)
                image_path_list.append(img_path)
        return image_path_list

    def read_image(self, path):
        img = Image.open(path)
        return img

    def resize_image(self, image, size):
        img = image.resize((size, size))
        return img

    def __getitem__(self, index):
        image_path = self.train_list[index]
        img = self.read_image(image_path)
        img = self.resize_image(img,size=self.size)
        img = self.transform(img)
        img = img * (self.max_val - self.min_val) + self.min_val
        return img

    def __len__(self):
        return len(self.train_list)


def get_dataset(name = 'train', size=299, root='/data/yike/FF++_std_c40_300frames/', frame_num=300, augment=True):
    """获取FF++数据集（兼容旧代码）"""
    root = os.path.join(root, name)
    fake_root = os.path.join(root,'fake')

    fake_list = ['Deepfakes', 'Face2Face', 'FaceSwap', 'NeuralTextures']
    
    total_len = len(fake_list)
    dset_lst = []
    for i in range(total_len):
        fake = os.path.join(fake_root , fake_list[i])
        dset = FFDataset(fake, frame_num, size, augment)
        dset.size = size
        dset_lst.append(dset)
    return torch.utils.data.ConcatDataset(dset_lst), total_len


# 一个简单的函数，用于快速设置基本的数据加载器
def create_simple_dataloaders(config):
    """创建基本的数据加载器"""
    from core.dataset import create_forensic_data_loaders
    return create_forensic_data_loaders(config, use_enhanced_dataset=True)


# 导出主要接口
__all__ = [
    'setup_logger',
    'set_seed',
    'evaluate',
    'load_model_weights',
    'save_checkpoint',
    'AverageMeter',
    'get_forensic_evaluator',
    'get_forensic_visualizer',
    'get_forensic_dataset',
    'create_simple_dataloaders',
    'EvalDataset',
    'FFDataset',
    'get_dataset'
]


if __name__ == "__main__":
    print("utils.py: 伪造检测工具模块")
    print("可导入的主要功能:")
    for func in __all__:
        print(f" - {func}")
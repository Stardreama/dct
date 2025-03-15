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
import json
import time
from easydict import EasyDict
import torchvision.transforms as transforms

# 导入新组件
from network.transform import mesonet_data_transforms
from models import create_model
from core.evaluation import ModelEvaluator, find_optimal_threshold, BoundaryEvaluator
from core.visualization import ForensicVisualizer
from core.dataset import create_forensic_data_loaders, EnhancedForensicDataset
from core.augmentation import ForensicAugmenter, DCTFeatureAugmenter, create_default_augmentation_config


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


def evaluate(model, config, test_path, batch_size=16, evaluate_boundary=False, evaluate_freq=False):
    """评估模型性能 (增强版)

    Args:
        model: 模型实例
        config: 配置对象
        test_path: 测试数据路径
        batch_size: 批量大小
        evaluate_boundary: 是否评估边界检测性能
        evaluate_freq: 是否进行频域分析

    Returns:
        dict: 评估指标
    """
    # 使用core.dataset中的函数创建数据加载器
    _, _, test_loader = create_forensic_data_loaders(config)
    
    # 获取设备
    device = next(model.parameters()).device
    
    # 使用ModelEvaluator评估
    evaluator = ModelEvaluator(save_dir=os.path.join(config.OUTPUT_DIR, 'evaluation') if hasattr(config, 'OUTPUT_DIR') else None)
    
    # 执行评估
    metrics = evaluator.evaluate_model(model, test_loader, device, evaluate_freq=evaluate_freq)
    
    # 如果需要评估边界检测
    if evaluate_boundary:
        boundary_metrics = evaluator.evaluate_boundary_detection(
            model, test_loader, device,
            save_dir=os.path.join(config.OUTPUT_DIR, 'boundary_evaluation') if hasattr(config, 'OUTPUT_DIR') else None
        )
        metrics['boundary_metrics'] = boundary_metrics
    
    # 保存评估结果
    if hasattr(config, 'OUTPUT_DIR'):
        # 创建可视化器
        visualizer = ForensicVisualizer(save_dir=os.path.join(config.OUTPUT_DIR, 'visualizations'))
        
        # 更新评估报告
        visualizer.update_evaluation_report(
            metrics, 
            boundary_metrics=metrics.get('boundary_metrics'), 
            freq_analysis=metrics.get('frequency_analysis'),
            save_dir=os.path.join(config.OUTPUT_DIR, 'evaluation')
        )
    
    return metrics


def load_model_weights(model, path, strict=True):
    """加载模型权重，处理各种情况，支持新的模型架构

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
            model_weights = checkpoint['model']
        elif 'state_dict' in checkpoint:
            model_weights = checkpoint['state_dict']
        else:
            model_weights = checkpoint
            
        # 处理权重键名不匹配情况
        if not strict:
            model_dict = model.state_dict()
            # 过滤掉不匹配的键
            filtered_weights = {k: v for k, v in model_weights.items() if k in model_dict}
            # 如果缺少过多权重，发出警告
            if len(filtered_weights) < len(model_dict) * 0.9:  # 少于90%匹配
                missing = len(model_dict) - len(filtered_weights)
                print(f"警告: 权重加载不完整，缺少 {missing} 个参数")
            model_weights = filtered_weights
            
        # 加载权重
        load_result = model.load_state_dict(model_weights, strict=strict)
        
        if not strict and load_result.missing_keys:
            print(f"信息: 未加载的权重: {len(load_result.missing_keys)} 个")
            # 打印部分缺失的键作为示例
            num_to_print = min(5, len(load_result.missing_keys))
            if num_to_print > 0:
                print(f"示例缺失键: {load_result.missing_keys[:num_to_print]}")
                
        print(f"成功加载模型权重: {path}")
        return model
    except Exception as e:
        print(f"加载模型权重出错: {e}")
        return model


def save_checkpoint(state, is_best, save_path, filename='checkpoint.pth', save_optimizers=True):
    """保存检查点和最佳模型，支持新的模型结构

    Args:
        state: 要保存的状态字典
        is_best: 是否为最佳模型
        save_path: 保存目录
        filename: 检查点文件名
        save_optimizers: 是否保存优化器状态
    """
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    
    # 如果不需要保存优化器状态，则移除相关键
    if not save_optimizers:
        state_copy = state.copy()
        keys_to_remove = ['optimizer', 'optimizer_state_dict']
        for key in keys_to_remove:
            if key in state_copy:
                del state_copy[key]
        state = state_copy
    
    # 添加时间戳和模型信息
    if 'timestamp' not in state:
        state['timestamp'] = time.strftime('%Y-%m-%d_%H-%M-%S')
    
    if 'model_info' not in state and 'model' in state and hasattr(state['model'], '__class__'):
        state['model_info'] = {
            'type': state['model'].__class__.__name__,
            'config': state.get('config', {})
        }
    
    # 保存检查点
    checkpoint_path = os.path.join(save_path, filename)
    torch.save(state, checkpoint_path)
    
    # 如果是最佳模型，单独保存一份
    if is_best:
        best_path = os.path.join(save_path, 'model_best.pth')
        torch.save(state, best_path)
        
        # 创建精简版模型权重文件，仅包含模型权重
        model_only_path = os.path.join(save_path, 'model_best_weights_only.pth')
        if 'model' in state:
            if hasattr(state['model'], 'state_dict'):
                torch.save(state['model'].state_dict(), model_only_path)
            else:
                torch.save(state['model'], model_only_path)
        elif 'state_dict' in state:
            torch.save(state['state_dict'], model_only_path)


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
    """创建基本的数据加载器，使用core.dataset中的函数"""
    return create_forensic_data_loaders(config, use_enhanced_dataset=True)


def create_model_from_checkpoint(config, model_path, model_type=None, mode=None, device='cuda'):
    """从检查点创建并加载模型

    Args:
        config: 配置对象
        model_path: 模型检查点路径
        model_type: 模型类型，如果不指定则从配置读取
        mode: 模型模式，如果不指定则从配置读取
        device: 设备

    Returns:
        加载了权重的模型
    """
    # 从配置或参数确定模型类型和模式
    if not model_type and hasattr(config, 'MODEL_CONFIG'):
        model_type = config.MODEL_CONFIG.TYPE
    else:
        model_type = model_type or 'enhanced'
        
    if not mode and hasattr(config, 'MODEL_CONFIG'):
        mode = config.MODEL_CONFIG.MODE
    else:
        mode = mode or 'Both'
    
    # 确定其他参数
    if hasattr(config, 'MODEL_CONFIG'):
        img_size = config.MODEL_CONFIG.IMG_SIZE if hasattr(config.MODEL_CONFIG, 'IMG_SIZE') else 256
        num_classes = config.MODEL_CONFIG.NUM_CLASSES if hasattr(config.MODEL_CONFIG, 'NUM_CLASSES') else 2
    else:
        img_size = 256
        num_classes = 2
    
    # 使用models.py中的create_model函数
    model = create_model(config, model_type, num_classes, img_size, mode)
    model = model.to(device)
    
    # 加载权重
    if model_path and os.path.exists(model_path):
        # 尝试严格加载，如果失败则非严格加载
        try:
            load_model_weights(model, model_path, strict=True)
        except Exception as e:
            print(f"严格加载失败: {e}，尝试非严格加载...")
            load_model_weights(model, model_path, strict=False)
    
    return model


def visualize_model_predictions(model, dataset, device, num_samples=8, save_dir=None):
    """可视化模型预测结果，使用ForensicVisualizer

    Args:
        model: 模型实例
        dataset: 数据集
        device: 设备
        num_samples: 可视化样本数量
        save_dir: 保存目录
        
    Returns:
        可视化结果保存路径
    """
    # 创建数据加载器，批量大小为num_samples
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=num_samples, shuffle=True, num_workers=2
    )
    
    # 获取一批数据
    batch = next(iter(dataloader))
    
    # 处理不同批次格式
    if len(batch) >= 4 and isinstance(batch[3], str):  # 带路径
        if isinstance(batch[1], torch.Tensor) and batch[1].dim() == 4 and batch[1].size(1) <= 64:
            # (img, dct, mask, path)
            inputs, dct_inputs, masks, paths = batch[:4]
            labels = batch[4] if len(batch) > 4 else None
        else:
            # (img, mask, label, path)
            inputs, masks, labels, paths = batch[:4]
            dct_inputs = None
    elif len(batch) == 3:
        if isinstance(batch[1], torch.Tensor) and batch[1].dim() == 4 and batch[1].size(1) <= 64:
            # (img, dct, label)
            inputs, dct_inputs, labels = batch
            masks = None
        else:
            # (img, mask, label)
            inputs, masks, labels = batch
            dct_inputs = None
    elif len(batch) == 2:
        # (img, label)
        inputs, labels = batch
        masks = None
        dct_inputs = None
    else:
        raise ValueError(f"不支持的批次格式: {[type(b) for b in batch]}")
    
    # 移动到设备
    inputs = inputs.to(device)
    if dct_inputs is not None:
        dct_inputs = dct_inputs.to(device)
    if masks is not None:
        masks = masks.to(device)
    if labels is not None:
        labels = labels.to(device)
    
    # 模型推理
    model.eval()
    with torch.no_grad():
        # 前向传播
        if dct_inputs is not None:
            outputs = model(inputs, dct_inputs)
        else:
            outputs = model(inputs)
        
        # 处理输出
        if isinstance(outputs, tuple):
            mask_preds, class_outputs = outputs
        else:
            class_outputs = outputs
            mask_preds = torch.zeros((inputs.size(0), 1, inputs.size(2), inputs.size(3)), device=device)
    
    # 创建可视化器
    visualizer = ForensicVisualizer(save_dir=save_dir)
    
    # 如果有掩码，进行掩码可视化
    if masks is not None:
        visualizer.visualize_masks(
            inputs.cpu().numpy(), 
            masks.cpu().numpy(), 
            mask_preds.cpu().numpy(),
            save_path=os.path.join(save_dir, 'mask_predictions.png') if save_dir else None
        )
    
    # 可视化注意力图（如果模型支持）
    try:
        visualizer.visualize_attention_maps(
            model, 
            inputs,
            device=device,
            save_dir=save_dir
        )
    except Exception as e:
        print(f"注意力图可视化失败: {e}")
    
    # 如果模型支持DCT特征，进行可视化
    try:
        visualizer.visualize_dct_features(
            model,
            inputs,
            device=device,
            save_dir=save_dir
        )
    except Exception as e:
        print(f"DCT特征可视化失败: {e}")
    
    return save_dir if save_dir else "可视化未保存"


def apply_model_to_image(model, img_path, device='cuda', visualize=True, save_dir=None):
    """对单张图像应用模型，包括边界预测

    Args:
        model: 模型实例
        img_path: 图像路径
        device: 设备
        visualize: 是否可视化结果
        save_dir: 可视化保存目录

    Returns:
        dict: 包含预测结果的字典
    """
    # 加载图像
    try:
        img = Image.open(img_path).convert('RGB')
    except Exception as e:
        print(f"无法加载图像 {img_path}: {e}")
        return None
    
    # 标准转换
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])
    
    # 转换图像
    img_tensor = transform(img).unsqueeze(0).to(device)
    
    # 检查是否需要DCT特征
    dct_tensor = None
    if hasattr(model, 'dct_extractor') or hasattr(model, 'needs_dct'):
        try:
            # 使用DCT变换提取特征
            from network.dct_transform import MultiScaleFrequencyExtractor
            dct_extractor = MultiScaleFrequencyExtractor(in_channels=3, out_channels=12).to(device)
            with torch.no_grad():
                dct_tensor = dct_extractor(img_tensor)
        except Exception as e:
            print(f"utils.py中apply_model_to_image函数DCT特征提取失败: {e}")
    
    # 模型推理
    model.eval()
    with torch.no_grad():
        # 前向传播
        if dct_tensor is not None:
            outputs = model(img_tensor, dct_tensor)
        else:
            outputs = model(img_tensor)
        
        # 处理输出
        if isinstance(outputs, tuple):
            mask_preds, class_outputs = outputs
        else:
            class_outputs = outputs
            mask_preds = None
        
        # 获取预测和概率
        if class_outputs.size(1) > 1:
            probs = torch.softmax(class_outputs, dim=1)[:, 1].cpu().numpy()
            _, preds = torch.max(class_outputs, 1)
            preds = preds.cpu().numpy()
        else:
            probs = torch.sigmoid(class_outputs).squeeze().cpu().numpy()
            preds = (probs >= 0.5).astype(int)
    
    # 结果字典
    result = {
        'pred': preds[0] if isinstance(preds, np.ndarray) else preds.item(),
        'prob': float(probs[0]) if isinstance(probs, np.ndarray) else float(probs),
        'mask': mask_preds.cpu().numpy()[0] if mask_preds is not None else None
    }
    
    # 可视化
    if visualize:
        # 创建可视化器
        visualizer = ForensicVisualizer(save_dir=save_dir)
        
        # 如果有掩码，可视化掩码和边界
        if mask_preds is not None:
            try:
                boundary_img = visualizer.visualize_mask_and_boundary(
                    img_tensor.squeeze(0).permute(1, 2, 0).cpu().numpy(),
                    mask_preds.squeeze(0).cpu().numpy(),
                    save_path=os.path.join(save_dir, 'boundary.png') if save_dir else None
                )
                result['boundary_img'] = boundary_img
            except Exception as e:
                print(f"边界可视化失败: {e}")
    
    return result


# 创建配置辅助函数
def create_default_config():
    """创建包含默认设置的配置"""
    config = EasyDict()
    
    # 基本配置
    config.GPUS = 1
    config.LOG_DIR = "log/"
    config.OUTPUT_DIR = "output/"
    config.WORKERS = 8
    config.PRINT_FREQ = 1000
    config.TRAIN_PATH = "dataset/train"
    config.VAL_PATH = "dataset/val"
    config.TEST_PATH = "dataset/test"
    config.BATCH_SIZE = 32
    config.EPOCHES = 100
    
    # 模型配置
    config.MODEL_CONFIG = EasyDict()
    config.MODEL_CONFIG.TYPE = "forensics"  # 'enhanced', 'f3net', 'forensics'
    config.MODEL_CONFIG.MODE = "Both"  # 'RGB', 'FAD', 'Both'
    config.MODEL_CONFIG.IMG_SIZE = 256
    config.MODEL_CONFIG.NUM_CLASSES = 2
    
    # 数据增强配置
    aug_config = create_default_augmentation_config()
    config.DATA_AUGMENTATION = aug_config.DATA_AUGMENTATION
    config.DCT_TRANSFORM = aug_config.DCT_TRANSFORM
    
    return config


# 导出主要接口
__all__ = [
    'setup_logger',
    'set_seed',
    'evaluate',
    'load_model_weights',
    'create_model_from_checkpoint',
    'save_checkpoint',
    'visualize_model_predictions',
    'apply_model_to_image',
    'create_simple_dataloaders',
    'get_forensic_evaluator',
    'get_forensic_visualizer',
    'create_default_config',
    'AverageMeter'
]


if __name__ == "__main__":
    print("utils.py: 伪造检测工具模块")
    print("可导入的主要功能:")
    for func in __all__:
        print(f" - {func}")
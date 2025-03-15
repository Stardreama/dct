#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
测试训练系统的各个组件

此脚本测试训练系统的各个核心组件，包括：
- 配置加载
- 数据集创建
- 模型初始化
- 损失函数
- 前向传播
- 反向传播
"""

import os
import sys
import yaml
import easydict
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from pathlib import Path
import logging
import argparse
import traceback

# 设置项目根目录
PROJECT_ROOT = Path(__file__).parent
sys.path.append(str(PROJECT_ROOT))

# 设置日志格式
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.StreamHandler()
    ]
)

logger = logging.getLogger('test_train')

def test_config_loading():
    """测试配置加载"""
    logger.info("==== 测试配置加载 ====")
    
    config_path = PROJECT_ROOT / "config.yaml"
    
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        config = easydict.EasyDict(config)
        logger.info(f"配置加载成功，包含 {len(config.keys())} 个顶级键")
        
        # 检查必要的配置项
        required_keys = ['MODEL_CONFIG', 'TRAIN_PATH', 'OUTPUT_DIR', 'BATCH_SIZE']
        missing_keys = [key for key in required_keys if key not in config]
        
        if missing_keys:
            logger.warning(f"配置文件缺少以下必要项: {missing_keys}")
        else:
            logger.info("所有必要配置项均已存在")
        
        return config
    except Exception as e:
        logger.error(f"配置加载失败: {e}")
        return None

def test_dataset_creation(config):
    """测试数据集创建"""
    logger.info("==== 测试数据集创建 ====")
    
    try:
        from core.dataset import create_forensic_data_loaders, EnhancedForensicDataset
        
        # 检查数据集路径
        train_path = Path(config.TRAIN_PATH)
        if not train_path.exists():
            logger.warning(f"训练数据集路径不存在: {train_path}")
            logger.info("尝试创建测试数据...")
            
            # 创建一个小的模拟数据集用于测试
            dummy_data = create_dummy_dataset()
            if dummy_data:
                logger.info("成功创建模拟数据集")
                
                # 更新配置中的数据集路径
                config.TRAIN_PATH = str(dummy_data / "train")
                config.VAL_PATH = str(dummy_data / "val")
                config.TEST_PATH = str(dummy_data / "test")
            else:
                logger.error("无法创建模拟数据集")
                return False
        
        # 尝试创建数据加载器
        logger.info("创建数据加载器...")
        try:
            train_loader, val_loader, test_loader = create_forensic_data_loaders(
                config, use_enhanced_dataset=True
            )
            
            logger.info(f"数据加载器创建成功")
            logger.info(f"训练集大小: {len(train_loader.dataset)}")
            logger.info(f"验证集大小: {len(val_loader.dataset)}")
            logger.info(f"测试集大小: {len(test_loader.dataset)}")
            
            # 测试一个批次的数据
            if len(train_loader) > 0:
                batch = next(iter(train_loader))
                logger.info(f"批次类型: {type(batch)}, 批次长度: {len(batch)}")
                
                # 检查批次内容
                if len(batch) >= 3:
                    img, mask, label = batch[:3]
                    logger.info(f"图像形状: {img.shape}")
                    logger.info(f"掩码形状: {mask.shape}")
                    logger.info(f"标签形状: {label.shape}")
                
                if len(batch) >= 4 and torch.is_tensor(batch[1]) and batch[1].dim() == 4:
                    logger.info(f"DCT特征形状: {batch[1].shape}")
                
            return train_loader, val_loader, test_loader
        
        except Exception as e:
            logger.error(f"数据加载器创建失败: {e}")
            traceback.print_exc()
            return None
    
    except ImportError as e:
        logger.error(f"导入数据集模块失败: {e}")
        return None

def create_dummy_dataset():
    """创建临时的模拟数据集用于测试"""
    try:
        import cv2
        from PIL import Image
        
        # 创建临时目录
        dummy_dir = PROJECT_ROOT / "dummy_dataset"
        dummy_dir.mkdir(exist_ok=True)
        
        # 创建训练/验证/测试目录
        train_dir = dummy_dir / "train"
        val_dir = dummy_dir / "val" 
        test_dir = dummy_dir / "test"
        
        train_dir.mkdir(exist_ok=True)
        val_dir.mkdir(exist_ok=True)
        test_dir.mkdir(exist_ok=True)
        
        # 创建模拟图像
        for i in range(10):
            # 创建随机图像
            img_array = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
            img = Image.fromarray(img_array)
            
            # 创建随机掩码
            mask_array = np.zeros((256, 256), dtype=np.uint8)
            if i % 2 == 1:  # 一半是伪造图像
                # 创建圆形掩码
                cv2.circle(mask_array, (128, 128), 64, 255, -1)
            mask = Image.fromarray(mask_array)
            
            # 保存图像和掩码
            img_path = train_dir / f"image_{i}.jpg"
            mask_path = train_dir / f"mask_{i}.png"
            img.save(img_path)
            mask.save(mask_path)
            
            # 复制到验证和测试集
            img.save(val_dir / f"image_{i}.jpg")
            mask.save(val_dir / f"mask_{i}.png")
            img.save(test_dir / f"image_{i}.jpg")
            mask.save(test_dir / f"mask_{i}.png")
        
        # 创建索引文件
        for split_dir in [train_dir, val_dir, test_dir]:
            with open(split_dir / f"{split_dir.name}.txt", 'w') as f:
                for i in range(10):
                    label = 1 if i % 2 == 1 else 0
                    f.write(f"{split_dir}/image_{i}.jpg {label}\n")
        
        return dummy_dir
    except Exception as e:
        logger.error(f"创建模拟数据集失败: {e}")
        traceback.print_exc()
        return None

def test_model_creation(config):
    """测试模型创建"""
    logger.info("==== 测试模型创建 ====")
    
    try:
        from models import create_model
        
        # 获取模型配置
        model_config = config.MODEL_CONFIG if hasattr(config, 'MODEL_CONFIG') else None
        model_type = model_config.TYPE if model_config and hasattr(model_config, 'TYPE') else 'forensics'
        mode = model_config.MODE if model_config and hasattr(model_config, 'MODE') else 'Both'
        num_classes = model_config.NUM_CLASSES if model_config and hasattr(model_config, 'NUM_CLASSES') else 2
        img_size = model_config.IMG_SIZE if model_config and hasattr(model_config, 'IMG_SIZE') else 256
        
        # 创建模型
        logger.info(f"创建模型，类型: {model_type}, 模式: {mode}")
        model = create_model(
            config=config,
            model_type=model_type,
            num_classes=num_classes,
            img_size=img_size,
            mode=mode
        )
        
        # 检查模型
        if model is None:
            logger.error("模型创建失败，返回为None")
            return None
        
        logger.info(f"模型创建成功，类型: {type(model).__name__}")
        
        # 获取模型参数数量
        param_count = sum(p.numel() for p in model.parameters())
        logger.info(f"模型参数数量: {param_count:,}")
        
        # 测试模型设备转移
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"将模型转移到设备: {device}")
        model = model.to(device)
        
        # 验证模型是否成功转移到设备
        for param in model.parameters():
            if param.device != device:
                logger.error(f"模型转移到设备失败，参数依然在 {param.device}")
                return None
            break
        
        logger.info("模型成功转移到设备")
        return model, device
    
    except Exception as e:
        logger.error(f"模型创建失败: {e}")
        traceback.print_exc()
        return None

def test_trainer_creation(config, model=None):
    """测试训练器创建"""
    logger.info("==== 测试训练器创建 ====")
    
    try:
        from trainer import create_trainer
        
        # 设置GPU IDs
        gpu_ids = []
        if torch.cuda.is_available():
            gpu_ids = [0]  # 仅使用第一个GPU进行测试
            
        # 获取模型参数
        model_config = config.MODEL_CONFIG if hasattr(config, 'MODEL_CONFIG') else None
        model_type = model_config.TYPE if model_config and hasattr(model_config, 'TYPE') else 'forensics'
        mode = model_config.MODE if model_config and hasattr(model_config, 'MODE') else 'Both'
        
        # 创建训练器
        logger.info(f"创建训练器，使用GPU IDs: {gpu_ids}")
        trainer = create_trainer(
            config=config,
            gpu_ids=gpu_ids,
            model_type=model_type,
            mode=mode
        )
        
        # 检查训练器
        if trainer is None:
            logger.error("训练器创建失败，返回为None")
            return None
        
        logger.info(f"训练器创建成功，类型: {type(trainer).__name__}")
        
        # 检查关键组件
        components = [
            ('model', hasattr(trainer, 'model')),
            ('optimizer', hasattr(trainer, 'optimizer')),
            ('scheduler', hasattr(trainer, 'scheduler')),
            ('cls_loss_fn', hasattr(trainer, 'cls_loss_fn')),
            ('mask_loss_fn', hasattr(trainer, 'mask_loss_fn')),
            ('freq_loss_fn', hasattr(trainer, 'freq_loss_fn'))
        ]
        
        for name, exists in components:
            status = "✓" if exists else "✗"
            logger.info(f"组件 {name}: {status}")
        
        # 测试compute_losses方法
        if hasattr(trainer, 'compute_losses'):
            logger.info("测试compute_losses方法...")
            
            # 创建模拟数据
            batch_size = 2
            device = trainer.device
            
            # 单输出场景
            dummy_output = torch.randn(batch_size, 2).to(device)  # 二分类
            dummy_labels = torch.randint(0, 2, (batch_size,)).to(device)
            
            try:
                losses = trainer.compute_losses(dummy_output, dummy_labels)
                logger.info(f"单输出compute_losses成功，返回 {len(losses)} 个值")
                
                # 多输出场景
                dummy_mask_output = torch.sigmoid(torch.randn(batch_size, 1, 64, 64)).to(device)
                dummy_cls_output = torch.randn(batch_size, 2).to(device)
                dummy_masks = torch.randint(0, 2, (batch_size, 1, 64, 64), dtype=torch.float).to(device)
                
                losses = trainer.compute_losses((dummy_mask_output, dummy_cls_output), 
                                               dummy_labels, dummy_masks)
                logger.info(f"多输出compute_losses成功，返回 {len(losses)} 个值")
                
            except Exception as e:
                logger.error(f"compute_losses方法测试失败: {e}")
                traceback.print_exc()
        
        return trainer
    
    except Exception as e:
        logger.error(f"训练器创建失败: {e}")
        traceback.print_exc()
        return None

def test_forward_backward(trainer, data_loaders):
    """测试前向传播和反向传播"""
    logger.info("==== 测试前向传播和反向传播 ====")
    
    if trainer is None or data_loaders is None:
        logger.error("缺少训练器或数据加载器，无法进行测试")
        return False
    
    train_loader, _, _ = data_loaders
    
    try:
        # 获取一个批次
        batch = next(iter(train_loader))
        trainer.model.train()
        
        # 处理不同的批次格式
        inputs, labels, masks = None, None, None
        dct_features = None
        
        if len(batch) == 2:
            inputs, labels = batch
        elif len(batch) == 3:
            if isinstance(batch[1], torch.Tensor) and batch[1].dim() == 4 and batch[1].size(1) <= 64:
                # RGB + DCT + 标签格式
                inputs, dct_features, labels = batch
            else:
                # RGB + 标签 + 掩码格式
                inputs, labels, masks = batch
        elif len(batch) >= 4:
            if isinstance(batch[1], torch.Tensor) and batch[1].dim() == 4 and batch[1].size(1) <= 64:
                # RGB + DCT + 掩码 + 标签格式
                inputs, dct_features, masks, labels = batch[:4]
            else:
                # 其他格式，默认处理前3个元素
                inputs, labels, masks = batch[:3]
        
        # 移动到设备
        device = trainer.device
        inputs = inputs.to(device)
        labels = labels.to(device)
        
        if masks is not None:
            masks = masks.to(device)
        if dct_features is not None:
            dct_features = dct_features.to(device)
        
        logger.info(f"输入形状: {inputs.shape}")
        if dct_features is not None:
            logger.info(f"DCT特征形状: {dct_features.shape}")
        if masks is not None:
            logger.info(f"掩码形状: {masks.shape}")
        logger.info(f"标签形状: {labels.shape}")
        
        # 前向传播
        logger.info("执行前向传播...")
        trainer.optimizer.zero_grad()
        
        try:
            if dct_features is not None:
                outputs = trainer.model(inputs, dct_features)
            else:
                outputs = trainer.model(inputs)
            
            # 输出形状信息
            if isinstance(outputs, tuple):
                logger.info(f"模型输出是一个元组，包含 {len(outputs)} 个元素")
                for i, out in enumerate(outputs):
                    logger.info(f"输出[{i}] 形状: {out.shape}")
            else:
                logger.info(f"模型输出形状: {outputs.shape}")
            
            # 计算损失
            logger.info("计算损失...")
            loss, cls_loss, mask_loss, freq_loss = trainer.compute_losses(outputs, labels, masks)
            
            logger.info(f"总损失: {loss.item():.4f}")
            logger.info(f"分类损失: {cls_loss.item():.4f}")
            if mask_loss is not None:
                logger.info(f"掩码损失: {mask_loss.item():.4f}")
            if freq_loss is not None:
                logger.info(f"频域损失: {freq_loss.item():.4f}")
            
            # 反向传播
            logger.info("执行反向传播...")
            loss.backward()
            
            # 检查梯度
            has_grad = False
            for name, param in trainer.model.named_parameters():
                if param.requires_grad and param.grad is not None:
                    has_grad = True
                    break
            
            if has_grad:
                logger.info("梯度计算成功")
            else:
                logger.warning("没有参数接收到梯度")
            
            # 优化器步进
            trainer.optimizer.step()
            
            logger.info("前向传播和反向传播测试成功")
            return True
        
        except Exception as e:
            logger.error(f"前向/反向传播测试失败: {e}")
            traceback.print_exc()
            return False
    
    except Exception as e:
        logger.error(f"无法获取训练批次: {e}")
        traceback.print_exc()
        return False

def test_training_epoch(trainer, data_loaders):
    """测试一个训练epoch"""
    logger.info("==== 测试训练epoch ====")
    
    if trainer is None or data_loaders is None:
        logger.error("缺少训练器或数据加载器，无法进行测试")
        return False
    
    train_loader, val_loader, _ = data_loaders
    
    try:
        # 限制为很少的步骤进行测试
        logger.info("创建少量样本的测试加载器...")
        limited_loader = torch.utils.data.DataLoader(
        torch.utils.data.Subset(
            train_loader.dataset,
            indices=torch.randperm(len(train_loader.dataset))[:min(10, len(train_loader.dataset))].tolist()
        ),
        batch_size=train_loader.batch_size,
        shuffle=False,  # 移除shuffle参数
        num_workers=0
    )
        
        # 测试train_epoch方法
        logger.info("执行训练epoch...")
        epoch_metrics = trainer.train_epoch(limited_loader, epoch=0)
        
        logger.info(f"训练epoch完成，返回指标: {epoch_metrics}")
        
        # 如果有验证集，测试validate方法
        if val_loader is not None and len(val_loader) > 0:
            logger.info("执行验证...")
            val_metrics, early_stop = trainer.validate(val_loader, epoch=0)
            logger.info(f"验证完成，返回指标: {val_metrics}")
            logger.info(f"早停信号: {early_stop}")
        
        return True
    
    except Exception as e:
        logger.error(f"训练epoch测试失败: {e}")
        traceback.print_exc()
        return False

def run_tests():
    """运行所有测试"""
    # 测试配置加载
    config = test_config_loading()
    if config is None:
        logger.error("配置加载测试失败，中止后续测试")
        return
    
    # 测试数据集创建
    data_loaders = test_dataset_creation(config)
    
    # 测试模型创建
    model_result = test_model_creation(config)
    if model_result is not None:
        model, device = model_result
    else:
        model = None
    
    # 测试训练器创建
    trainer = test_trainer_creation(config, model)
    
    # 如果有数据加载器和训练器，测试前向/反向传播
    if data_loaders is not None and trainer is not None:
        test_forward_backward(trainer, data_loaders)
        
        # 测试训练epoch
        test_training_epoch(trainer, data_loaders)
    
    logger.info("所有测试完成")

if __name__ == "__main__":
    run_tests()
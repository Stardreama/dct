import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
import math
import time
import logging
from pathlib import Path
from torch.cuda.amp import autocast, GradScaler
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts, OneCycleLR

# 导入自定义模块
from models import F3Net, EnhancedF3Net
from core.visualization import ForensicVisualizer  
from core.evaluation import ModelEvaluator, find_optimal_threshold
from core.dataset import create_forensic_data_loaders
from core.augmentation import ForensicAugmenter


def initModel(mod, gpu_ids):
    """初始化模型到指定GPU"""
    mod = mod.to(f'cuda:{gpu_ids[0]}')
    mod = nn.DataParallel(mod, gpu_ids)
    return mod


# 焦点损失函数 - 更关注难分类样本
class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        
    def forward(self, inputs, targets):
        BCE_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-BCE_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * BCE_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


# 边缘感知损失函数 - 提高掩码边缘准确性
class EdgeAwareLoss(nn.Module):
    def __init__(self, reduction='mean'):
        super(EdgeAwareLoss, self).__init__()
        self.reduction = reduction
        
        # 定义Sobel算子用于边缘检测
        self.sobel_x = nn.Conv2d(1, 1, kernel_size=3, stride=1, padding=1, bias=False)
        self.sobel_y = nn.Conv2d(1, 1, kernel_size=3, stride=1, padding=1, bias=False)
        
        # 初始化Sobel算子权重
        sobel_x_weights = torch.FloatTensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]).unsqueeze(0).unsqueeze(0)
        sobel_y_weights = torch.FloatTensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]]).unsqueeze(0).unsqueeze(0)
        
        self.sobel_x.weight.data = sobel_x_weights
        self.sobel_y.weight.data = sobel_y_weights
        
        # 冻结参数
        for param in self.sobel_x.parameters():
            param.requires_grad = False
        for param in self.sobel_y.parameters():
            param.requires_grad = False
            
    def detect_edges(self, x):
        # 检测图像边缘
        edge_x = self.sobel_x(x)
        edge_y = self.sobel_y(x)
        edge = torch.sqrt(edge_x ** 2 + edge_y ** 2 + 1e-8)
        return edge
        
    def forward(self, pred, target):
        # 计算边缘
        pred_edges = self.detect_edges(pred)
        target_edges = self.detect_edges(target)
        
        # 计算边缘MSE损失
        edge_loss = F.mse_loss(pred_edges, target_edges, reduction=self.reduction)
        
        # 计算标准MSE损失
        mse_loss = F.mse_loss(pred, target, reduction=self.reduction)
        
        # 结合边缘损失和标准损失
        total_loss = mse_loss + 0.5 * edge_loss
        return total_loss


# 频谱一致性损失 - 关注频域特征
class FrequencyConsistencyLoss(nn.Module):
    def __init__(self, reduction='mean'):
        super(FrequencyConsistencyLoss, self).__init__()
        self.reduction = reduction
        
    def forward(self, pred, target):
        # 应用DCT变换
        pred_freq = torch.fft.rfft2(pred, norm="ortho")
        target_freq = torch.fft.rfft2(target, norm="ortho")
        
        # 计算频域差异
        pred_magnitude = torch.abs(pred_freq)
        target_magnitude = torch.abs(target_freq)
        
        # 均方差损失
        freq_loss = F.mse_loss(pred_magnitude, target_magnitude, reduction=self.reduction)
        return freq_loss


# EMA更新器 - 跟踪模型参数的指数移动平均
class ModelEMA:
    def __init__(self, model, decay=0.9999, device=None):
        self.module = model.module if hasattr(model, 'module') else model
        self.decay = decay
        self.device = device
        
        # 创建EMA模型
        self.ema_model = type(model)(model.config, mode=model.mode, device=model.device)
        self.ema_model.eval()
        
        # 初始化EMA权重
        for param_ema, param in zip(self.ema_model.parameters(), self.module.parameters()):
            param_ema.data.copy_(param.data)
            param_ema.requires_grad_(False)
            
    def update(self, model):
        module = model.module if hasattr(model, 'module') else model
        
        with torch.no_grad():
            for ema_param, param in zip(self.ema_model.parameters(), module.parameters()):
                ema_param.data.mul_(self.decay).add_(param.data, alpha=1 - self.decay)
                
    def update_attr(self, model):
        # 更新属性
        for k, v in model.__dict__.items():
            if not k.startswith('_') and not callable(v):
                if k not in self.ema_model.__dict__:
                    self.ema_model.__dict__[k] = v


# 增强版训练器
class EnhancedTrainer:
    def __init__(self, config, gpu_ids, mode='Both', pretrained_path=None):
        """
        初始化训练器
        
        Args:
            config: 配置对象
            gpu_ids: GPU ID列表
            mode: 模型模式 ('Both', 'RGB', 'DCT')
            pretrained_path: 预训练权重路径
        """
        self.config = config
        self.gpu_ids = gpu_ids
        self.mode = mode
        self.device = torch.device(f'cuda:{gpu_ids[0]}') if gpu_ids else torch.device('cpu')
        
        # 设置日志
        self.logger = logging.getLogger("Trainer")
        
        # 使用增强版模型
        self.model = EnhancedF3Net(config, mode=mode, device=self.device)
        
        # 如果有多GPU，使用DataParallel
        if len(gpu_ids) > 1:
            self.model = initModel(self.model, gpu_ids)
            
        # 复合损失函数
        self.cls_loss_fn = FocalLoss(alpha=0.25, gamma=2.0)
        self.mask_loss_fn = EdgeAwareLoss(reduction='mean')
        self.freq_loss_fn = FrequencyConsistencyLoss(reduction='mean')
        
        # 损失权重
        self.cls_weight = config.LOSS.CLS_WEIGHT if hasattr(config.LOSS, 'CLS_WEIGHT') else 1.0
        self.mask_weight = config.LOSS.MASK_WEIGHT if hasattr(config.LOSS, 'MASK_WEIGHT') else 0.5
        self.freq_weight = config.LOSS.FREQ_WEIGHT if hasattr(config.LOSS, 'FREQ_WEIGHT') else 0.3
        
        # 使用改进的优化器 (AdamW)
        self.optimizer = torch.optim.AdamW(
            filter(lambda p: p.requires_grad, self.model.parameters()),
            lr=config.OPTIMIZER.LR,
            betas=(config.OPTIMIZER.BETA1, config.OPTIMIZER.BETA2),
            weight_decay=config.OPTIMIZER.WEIGHT_DECAY
        )
        
        # 学习率调度器
        if config.LR_SCHEDULER.NAME == 'cosine':
            self.scheduler = CosineAnnealingWarmRestarts(
                self.optimizer,
                T_0=config.LR_SCHEDULER.COSINE.T_MAX,
                T_mult=1,
                eta_min=config.LR_SCHEDULER.COSINE.ETA_MIN
            )
        elif config.LR_SCHEDULER.NAME == 'onecycle':
            self.scheduler = OneCycleLR(
                self.optimizer,
                max_lr=config.OPTIMIZER.LR * 10,
                total_steps=config.TRAINING.TOTAL_STEPS,
                pct_start=0.3,
                div_factor=10.0,
                final_div_factor=1000.0
            )
        else:
            self.scheduler = None
            
        # 混合精度训练
        self.use_amp = config.TRAINING.MIXED_PRECISION if hasattr(config.TRAINING, 'MIXED_PRECISION') else False
        if self.use_amp:
            self.scaler = GradScaler()
            
        # EMA模型
        self.use_ema = config.TRAINING.EMA.ENABLED if hasattr(config.TRAINING, 'EMA') and hasattr(config.TRAINING.EMA, 'ENABLED') else False
        if self.use_ema:
            self.ema = ModelEMA(
                self.model, 
                decay=config.TRAINING.EMA.DECAY if hasattr(config.TRAINING.EMA, 'DECAY') else 0.9998, 
                device=self.device
            )
            
        # 梯度裁剪值
        self.grad_clip_norm = config.TRAINING.CLIP_GRAD_NORM if hasattr(config.TRAINING, 'CLIP_GRAD_NORM') else None
        
        # 创建可视化器和评估器
        self.results_dir = Path(config.OUTPUT_DIR) if hasattr(config, 'OUTPUT_DIR') else Path('./results')
        self.results_dir.mkdir(exist_ok=True, parents=True)
        self.visualizer = ForensicVisualizer(save_dir=self.results_dir / 'visualizations', logger=self.logger)
        self.evaluator = ModelEvaluator(save_dir=self.results_dir / 'evaluation', logger=self.logger)
        
        # 训练历史记录
        self.history = {
            'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': [],
            'train_cls_loss': [], 'train_mask_loss': [], 'train_freq_loss': [], 
            'lr': []
        }
        
        # 训练信息
        self.current_epoch = 0
        self.current_step = 0
        self.best_val_loss = float('inf')
        self.best_val_acc = 0
        
        # 加载预训练权重(如果有)
        if pretrained_path and os.path.exists(pretrained_path):
            self.load(pretrained_path)
            
    def prepare_data_loaders(self):
        """准备数据加载器"""
        # 使用统一的数据加载器创建函数
        return create_forensic_data_loaders(self.config)
        
    def train_epoch(self, train_loader, epoch):
        """
        训练一个完整的epoch
        
        Args:
            train_loader: 训练数据加载器
            epoch: 当前epoch
            
        Returns:
            dict: 训练指标
        """
        self.model.train()
        total_loss = 0
        total_cls_loss = 0
        total_mask_loss = 0
        total_freq_loss = 0
        correct = 0
        total = 0
        
        # 使用进度条跟踪训练
        with torch.set_grad_enabled(True):
            for batch_idx, (inputs, masks, labels) in enumerate(train_loader):
                # 递增步数
                self.current_step += 1
                
                # 移到设备
                inputs, masks, labels = inputs.to(self.device), masks.to(self.device), labels.to(self.device)
                
                # 混合精度训练
                if self.use_amp:
                    with autocast():
                        mask_pred, outputs = self.model(inputs)
                        
                        # 计算分类损失
                        loss_cls = self.cls_loss_fn(outputs, labels)
                        
                        # 计算掩码损失
                        loss_mask = self.mask_loss_fn(mask_pred, masks)
                        
                        # 计算频域一致性损失
                        loss_freq = self.freq_loss_fn(mask_pred, masks)
                        
                        # 总损失
                        loss = self.cls_weight * loss_cls + self.mask_weight * loss_mask + self.freq_weight * loss_freq
                    
                    # 梯度缩放
                    self.optimizer.zero_grad()
                    self.scaler.scale(loss).backward()
                    
                    # 梯度裁剪
                    if self.grad_clip_norm is not None:
                        self.scaler.unscale_(self.optimizer)
                        nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip_norm)
                        
                    # 更新权重
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                
                else:
                    # 标准训练流程
                    mask_pred, outputs = self.model(inputs)
                    
                    # 计算分类损失
                    loss_cls = self.cls_loss_fn(outputs, labels)
                    
                    # 计算掩码损失
                    loss_mask = self.mask_loss_fn(mask_pred, masks)
                    
                    # 计算频域一致性损失
                    loss_freq = self.freq_loss_fn(mask_pred, masks)
                    
                    # 总损失
                    loss = self.cls_weight * loss_cls + self.mask_weight * loss_mask + self.freq_weight * loss_freq
                    
                    self.optimizer.zero_grad()
                    loss.backward()
                    
                    # 梯度裁剪
                    if self.grad_clip_norm is not None:
                        nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip_norm)
                        
                    self.optimizer.step()
                
                # 学习率调度
                if self.scheduler is not None:
                    self.scheduler.step()
                    
                # 更新EMA模型
                if self.use_ema and self.current_step % 10 == 0:  # 每10步更新一次EMA模型
                    self.ema.update(self.model)
                
                # 累计损失和准确率
                total_loss += loss.item()
                total_cls_loss += loss_cls.item()
                total_mask_loss += loss_mask.item()
                total_freq_loss += loss_freq.item()
                
                # 计算准确度
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                
                # 显示进度
                if batch_idx % 100 == 0:
                    acc = correct / total
                    self.logger.info(f'Epoch: {epoch} [{batch_idx*len(inputs)}/{len(train_loader.dataset)} '
                                    f'({100. * batch_idx / len(train_loader):.0f}%)]\t'
                                    f'Loss: {loss.item():.6f}\t Acc: {acc:.4f}')
        
        # 计算平均指标
        avg_loss = total_loss / len(train_loader)
        avg_cls_loss = total_cls_loss / len(train_loader)
        avg_mask_loss = total_mask_loss / len(train_loader)
        avg_freq_loss = total_freq_loss / len(train_loader)
        accuracy = correct / total
        current_lr = self.optimizer.param_groups[0]['lr']
        
        # 更新训练历史
        self.history['train_loss'].append(avg_loss)
        self.history['train_cls_loss'].append(avg_cls_loss)
        self.history['train_mask_loss'].append(avg_mask_loss)
        self.history['train_freq_loss'].append(avg_freq_loss)
        self.history['train_acc'].append(accuracy)
        self.history['lr'].append(current_lr)
        
        # 返回训练指标
        return {
            'loss': avg_loss,
            'cls_loss': avg_cls_loss,
            'mask_loss': avg_mask_loss,
            'freq_loss': avg_freq_loss,
            'acc': accuracy,
            'lr': current_lr
        }
        
    def validate(self, val_loader):
        """
        模型验证
        
        Args:
            val_loader: 验证数据加载器
            
        Returns:
            dict: 验证指标
        """
        self.model.eval()
        
        # 选择要使用的模型
        eval_model = self.ema.ema_model if self.use_ema else self.model
        
        # 使用评估器进行验证
        metrics = self.evaluator.evaluate_model(eval_model, val_loader, self.device)
        
        # 提取关键指标
        val_loss = metrics['confusion_matrix'].sum() - (metrics['confusion_matrix'][0, 0] + metrics['confusion_matrix'][1, 1])
        val_loss = val_loss / metrics['confusion_matrix'].sum()  # 将错误数量归一化为损失
        val_acc = metrics['accuracy']
        
        # 更新训练历史
        self.history['val_loss'].append(val_loss)
        self.history['val_acc'].append(val_acc)
        
        # 更新最佳模型
        if val_acc > self.best_val_acc:
            self.best_val_acc = val_acc
            self.save(self.results_dir / 'best_acc_model.pth')
        
        if val_loss < self.best_val_loss:
            self.best_val_loss = val_loss
            self.save(self.results_dir / 'best_loss_model.pth')
        
        # 返回验证指标
        return {
            'loss': val_loss,
            'acc': val_acc,
            'auc': metrics['auc']
        }
    
    def test(self, test_loader):
        """
        测试模型性能
        
        Args:
            test_loader: 测试数据加载器
            
        Returns:
            dict: 测试指标
        """
        # 选择要使用的模型
        eval_model = self.ema.ema_model if self.use_ema else self.model
        
        # 使用评估器进行测试，同时保存原始预测
        metrics = self.evaluator.evaluate_model(eval_model, test_loader, self.device, return_predictions=True)
        
        # 保存测试结果
        self.evaluator.save_evaluation_results(metrics, self.results_dir / 'test_results')
        
        # 创建可视化报告
        if 'predictions' in metrics:
            self.visualizer.create_evaluation_report(metrics, self.results_dir / 'test_results')
        
            # 可视化部分预测样本
            if metrics['predictions']['paths'] is not None:
                img_paths = metrics['predictions']['paths'][:10]  # 限制为前10个样本
                mask_preds = metrics['predictions']['masks_pred'][:10]
                true_labels = metrics['predictions']['labels'][:10]
                pred_labels = metrics['predictions']['preds'][:10]
                probabilities = metrics['predictions']['probs'][:10]
                
                self.visualizer.visualize_predictions(
                    img_paths, mask_preds, true_labels, pred_labels, probabilities, 
                    save_path=self.results_dir / 'test_results/sample_predictions.png'
                )
        
        return metrics
    
    def train(self, epochs, train_loader, val_loader=None, test_loader=None):
        """
        训练模型
        
        Args:
            epochs: 总epochs数
            train_loader: 训练数据加载器
            val_loader: 验证数据加载器
            test_loader: 测试数据加载器
            
        Returns:
            dict: 最终训练历史
        """
        start_time = time.time()
        self.logger.info(f"开始训练，总计{epochs}轮...")

        for epoch in range(self.current_epoch, epochs):
            self.current_epoch = epoch
            
            # 训练一个epoch
            train_metrics = self.train_epoch(train_loader, epoch)
            
            # 保存当前模型
            self.save(self.results_dir / f'checkpoint_epoch_{epoch:03d}.pth')
            
            # 验证当前模型
            if val_loader:
                val_metrics = self.validate(val_loader)
                
                # 打印验证指标
                self.logger.info(f"Epoch {epoch}: 训练损失: {train_metrics['loss']:.4f}, "
                                f"验证损失: {val_metrics['loss']:.4f}, "
                                f"训练准确率: {train_metrics['acc']:.4f}, "
                                f"验证准确率: {val_metrics['acc']:.4f}, "
                                f"验证AUC: {val_metrics['auc']:.4f}")
            else:
                # 只打印训练指标
                self.logger.info(f"Epoch {epoch}: 训练损失: {train_metrics['loss']:.4f}, "
                                f"训练准确率: {train_metrics['acc']:.4f}")
            
            # 定期可视化训练进度
            if (epoch + 1) % 5 == 0 or epoch == epochs - 1:
                self.visualizer.plot_training_curves(self.history, 
                                                    save_path=self.results_dir / f'training_curves_epoch_{epoch:03d}.png')
                
        # 训练结束，进行最终测试
        if test_loader:
            self.logger.info("训练完成，进行最终测试...")
            test_metrics = self.test(test_loader)
            self.logger.info(f"测试结果: 准确率: {test_metrics['accuracy']:.4f}, AUC: {test_metrics['auc']:.4f}")
        
        total_time = time.time() - start_time
        self.logger.info(f"训练完成，总耗时: {total_time/60:.2f} 分钟")
        
        return self.history

    def save(self, path):
        """保存模型"""
        if self.use_ema:
            # 保存EMA模型
            torch.save({
                'model': self.model.state_dict(),
                'ema_model': self.ema.ema_model.state_dict(),
                'optimizer': self.optimizer.state_dict(),
                'scheduler': self.scheduler.state_dict() if self.scheduler else None,
                'scaler': self.scaler.state_dict() if self.use_amp else None,
                'epoch': self.current_epoch,
                'step': self.current_step,
                'best_val_loss': self.best_val_loss,
                'best_val_acc': self.best_val_acc,
                'history': self.history
            }, path)
        else:
            # 仅保存普通模型
            torch.save({
                'model': self.model.state_dict(),
                'optimizer': self.optimizer.state_dict(),
                'scheduler': self.scheduler.state_dict() if self.scheduler else None,
                'scaler': self.scaler.state_dict() if self.use_amp else None,
                'epoch': self.current_epoch,
                'step': self.current_step,
                'best_val_loss': self.best_val_loss,
                'best_val_acc': self.best_val_acc,
                'history': self.history
            }, path)

    def load(self, path):
        """加载模型"""
        checkpoint = torch.load(path)
        
        # 加载模型权重
        if 'model' in checkpoint:
            self.model.load_state_dict(checkpoint['model'])
        else:
            # 向后兼容旧格式
            self.model.load_state_dict(checkpoint)
            
        # 加载EMA模型(如果存在)
        if self.use_ema and 'ema_model' in checkpoint:
            self.ema.ema_model.load_state_dict(checkpoint['ema_model'])
            
        # 加载优化器状态(如果存在)
        if 'optimizer' in checkpoint:
            self.optimizer.load_state_dict(checkpoint['optimizer'])
            
        # 加载调度器状态(如果存在)
        if self.scheduler is not None and 'scheduler' in checkpoint and checkpoint['scheduler'] is not None:
            self.scheduler.load_state_dict(checkpoint['scheduler'])
            
        # 加载梯度缩放器状态(如果存在)
        if self.use_amp and 'scaler' in checkpoint and checkpoint['scaler'] is not None:
            self.scaler.load_state_dict(checkpoint['scaler'])
            
        # 加载训练状态
        if 'epoch' in checkpoint:
            self.current_epoch = checkpoint['epoch'] + 1  # 从下一个epoch继续
        if 'step' in checkpoint:
            self.current_step = checkpoint['step']
        if 'best_val_loss' in checkpoint:
            self.best_val_loss = checkpoint['best_val_loss']
        if 'best_val_acc' in checkpoint:
            self.best_val_acc = checkpoint['best_val_acc']
        if 'history' in checkpoint:
            self.history = checkpoint['history']


# 保留原始Trainer类以保持向后兼容性
class Trainer(nn.Module): 
    def __init__(self, config, gpu_ids, mode='Both', pretrained_path=None):
        super(Trainer, self).__init__()
        self.device = torch.device('cuda:{}'.format(gpu_ids[0])) if gpu_ids else torch.device('cpu')
        self.model = F3Net(config, mode=mode, device=self.device)
        # self.model = initModel(self.model, gpu_ids)
        self.loss_fn = nn.BCEWithLogitsLoss()
        self.loss_fn = nn.CrossEntropyLoss()
        self.loss_mse = nn.MSELoss(reduction='mean')
        self.optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, self.model.parameters()),
                                              lr=0.0002, betas=(0.9, 0.999))
        # self.optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, self.model.parameters()),
        #                                         lr=0.002, momentum=0.9, weight_decay=0)

    def set_input(self, input, mask, label):
        self.input = input.to(self.device)
        self.label = label.to(self.device)
        self.mask = mask.to(self.device)

    def forward(self, x):
        mask_pred, out = self.model(x)
        return mask_pred, out

    def optimize_weight(self, config):
        mask_pred, out = self.model(self.input)
        self.loss_cla = self.loss_fn(out.squeeze(1), self.label)  # classify loss
        self.loss_mask = self.loss_mse(mask_pred, self.mask)
        self.loss = self.loss_cla + self.loss_mask

        self.optimizer.zero_grad()
        self.loss.backward()
        self.optimizer.step()
        return self.loss

    def save(self, path):
        torch.save(self.model.state_dict(), path)

    def load(self, path):
        state_dict = torch.load(path)
        self.model.load_state_dict(state_dict)
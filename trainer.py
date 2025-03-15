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
from tqdm import tqdm

# 更新导入 - 使用 core 中的模块
from models import EnhancedF3Net, DeepForensicsNet, create_model
from core.visualization import ForensicVisualizer  
from core.evaluation import ModelEvaluator, find_optimal_threshold, BoundaryEvaluator, FrequencyDomainEvaluator
from core.dataset import create_forensic_data_loaders, EnhancedForensicDataset
from core.augmentation import ForensicAugmenter, DCTFeatureAugmenter, get_dual_input_pipeline
from utils import save_checkpoint, load_model_weights, create_default_config, evaluate


def ensure_same_device(tensor_a, tensor_b):
    """确保两个张量在同一设备上"""
    if tensor_a.device != tensor_b.device:
        #print(f"张量设备不匹配: {tensor_a.device} vs {tensor_b.device}，执行迁移")
        return tensor_b.to(tensor_a.device)
    return tensor_b

def ensure_tensor_on_device(tensor, device):
    """确保张量位于指定设备上"""
    if tensor.device != device:
        #print(f"将张量从 {tensor.device} 移至 {device}")
        return tensor.to(device)
    return tensor


def initModel(mod, gpu_ids):
    """初始化模型到指定GPU"""
    mod = mod.to(f'cuda:{gpu_ids[0]}')
    mod = nn.DataParallel(mod, gpu_ids)
    return mod

# 添加预热调度器的实现
class GradualWarmupScheduler(torch.optim.lr_scheduler._LRScheduler):
    """
    预热学习率调度器
    首先在预热阶段逐步增加学习率，然后使用常规调度器
    
    Args:
        optimizer: 优化器
        multiplier: 预热结束时学习率相对于初始值的倍数
        total_epoch: 预热的总epoch数
        after_scheduler: 预热结束后使用的调度器
    """
    def __init__(self, optimizer, multiplier, total_epoch, after_scheduler=None):
        self.multiplier = multiplier
        self.total_epoch = total_epoch
        self.after_scheduler = after_scheduler
        self.finished = False
        super().__init__(optimizer)
        
    def get_lr(self):
        if self.last_epoch > self.total_epoch:
            if self.after_scheduler:
                if not self.finished:
                    self.after_scheduler.base_lrs = [base_lr * self.multiplier for base_lr in self.base_lrs]
                    self.finished = True
                return self.after_scheduler.get_lr()
            return [base_lr * self.multiplier for base_lr in self.base_lrs]
            
        return [base_lr * ((self.multiplier - 1.) * self.last_epoch / self.total_epoch + 1.) for base_lr in self.base_lrs]
    
    def step(self, epoch=None):
        if self.finished and self.after_scheduler:
            if epoch is None:
                self.after_scheduler.step(None)
            else:
                self.after_scheduler.step(epoch - self.total_epoch)
        else:
            return super().step(epoch)

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


# Dice损失用于掩码预测 - 添加此损失函数更适合掩码和边界检测
class DiceLoss(nn.Module):
    def __init__(self, smooth=1.0):
        super(DiceLoss, self).__init__()
        self.smooth = smooth
        
    def forward(self, pred, target):
        # 展平预测和目标
        pred_flat = pred.view(-1)
        target_flat = target.view(-1)
        
        # 计算交集
        intersection = (pred_flat * target_flat).sum()
        
        # 计算Dice系数
        dice = (2. * intersection + self.smooth) / (pred_flat.sum() + target_flat.sum() + self.smooth)
        
        # 返回损失值
        return 1 - dice


class BCEDiceLoss(nn.Module):
    def __init__(self, weight_bce=0.5, weight_dice=0.5):
        super(BCEDiceLoss, self).__init__()
        self.bce_loss = nn.BCEWithLogitsLoss()
        self.dice_loss = DiceLoss()
        self.weight_bce = weight_bce
        self.weight_dice = weight_dice
        
    def forward(self, pred, target):
        # 添加安全检查与值限制
        pred = torch.clamp(pred, min=0.0, max=1.0)
        bce_loss = self.bce_loss(pred, target)
        dice_loss = self.dice_loss(pred, target)
        return self.weight_bce * bce_loss + self.weight_dice * dice_loss



class EdgeAwareLoss(nn.Module):
    def __init__(self, reduction='mean', edge_weight=0.8):
        super(EdgeAwareLoss, self).__init__()
        self.reduction = reduction
        self.edge_weight = edge_weight
        
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
        
        # 结合边缘损失和标准损失，增加边缘权重
        total_loss = (1 - self.edge_weight) * mse_loss + self.edge_weight * edge_loss
        return total_loss


# 更新频谱一致性损失 - 处理频域特征
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
        
        # 加权处理不同频率成分
        # 低频区域权重更高
        h, w = pred_magnitude.shape[-2], pred_magnitude.shape[-1]
        y_grid, x_grid = torch.meshgrid(torch.arange(h), torch.arange(w), indexing='ij')
        y_grid, x_grid = y_grid.to(pred.device), x_grid.to(pred.device)
        
        # 计算到中心点的距离，作为频率的指标
        center_y, center_x = h // 2, w // 2
        dist = torch.sqrt((y_grid - center_y) ** 2 + (x_grid - center_x) ** 2) + 1e-8
        # 归一化距离
        dist = dist / torch.max(dist)
        
        # 构建频率权重，低频权重更高
        freq_weight = torch.exp(-2.0 * dist).unsqueeze(0).unsqueeze(0)
        
        # 应用权重
        weighted_diff = ((pred_magnitude - target_magnitude) ** 2) * freq_weight
        
        if self.reduction == 'mean':
            freq_loss = torch.mean(weighted_diff)
        else:
            freq_loss = torch.sum(weighted_diff)
        
        return freq_loss

class ModelEMA:
    def __init__(self, model, decay=0.9999, device=None):
        self.module = model.module if hasattr(model, 'module') else model
        self.decay = decay
        self.device = device
        
        # 禁用EMA模型跟踪梯度
        self.ema_has_module = hasattr(model, 'module')
        
        # 使用更可靠的方法创建EMA模型
        try:
            # 首先尝试深度复制模型
            import copy
            self.ema_model = copy.deepcopy(self.module)
            if device:
                self.ema_model = self.ema_model.to(device)
                
        except Exception as e:
            #print(f"无法深度复制模型，尝试其他方法: {e}")
            
            # 如果深度复制失败，尝试使用相同参数创建模型
            try:
                # 假设模型有config属性
                if hasattr(self.module, 'config'):
                    model_cls = self.module.__class__
                    self.ema_model = model_cls(self.module.config)
                    
                    # 复制模型权重
                    self.ema_model.load_state_dict(self.module.state_dict())
                    
                    if device:
                        self.ema_model = self.ema_model.to(device)
                else:
                    #print("模型没有config属性，无法创建EMA模型")
                    # 临时解决方案：禁用EMA
                    self.ema_model = None
                    
            except Exception as e:
                #print(f"创建EMA模型失败，禁用EMA: {e}")
                self.ema_model = None
                
        # 如果EMA模型成功创建，设置为评估模式并禁用梯度
        if self.ema_model is not None:
            self.ema_model.eval()
            for param in self.ema_model.parameters():
                param.requires_grad_(False)
            
    def update(self, model):
        """更新EMA模型权重"""
        # 如果EMA模型创建失败，跳过更新
        if self.ema_model is None:
            return
            
        with torch.no_grad():
            # 获取当前模型的状态
            msd = model.module.state_dict() if self.ema_has_module else model.state_dict()
            
            # 更新EMA模型权重
            for k, ema_v in self.ema_model.state_dict().items():
                if k in msd:
                    model_v = msd[k].detach()
                    ema_v.copy_(self.decay * ema_v + (1. - self.decay) * model_v)
                
    def update_attr(self, model):
        # 更新属性
        for k, v in model.__dict__.items():
            if not k.startswith('_') and not callable(v):
                if k not in self.ema_model.__dict__:
                    self.ema_model.__dict__[k] = v

# 更新增强版训练器以适配新模型架构
class EnhancedTrainer:
    def __init__(self, config, gpu_ids, model_type='enhanced', mode='Both', pretrained_path=None):
        """
        初始化训练器
        
        Args:
            config: 配置对象
            gpu_ids: GPU ID列表
            model_type: 模型类型 ('enhanced', 'f3net', 'forensics')
            mode: 模型模式 ('Both', 'RGB', 'FAD')
            pretrained_path: 预训练权重路径
        """
        self.config = config
        self.gpu_ids = gpu_ids
        self.mode = mode
        self.device = torch.device(f'cuda:{gpu_ids[0]}') if gpu_ids else torch.device('cpu')
        
        # 设置日志
        self.logger = logging.getLogger("EnhancedTrainer")
        
        # 使用工厂函数创建模型
        if hasattr(config, 'MODEL_CONFIG'):
            model_config = config.MODEL_CONFIG
            model_type = model_config.TYPE if hasattr(model_config, 'TYPE') else model_type
            mode = model_config.MODE if hasattr(model_config, 'MODE') else mode
            img_size = model_config.IMG_SIZE if hasattr(model_config, 'IMG_SIZE') else 256
            num_classes = model_config.NUM_CLASSES if hasattr(model_config, 'NUM_CLASSES') else 2
        else:
            img_size = 256
            num_classes = 2
            
        # 创建模型后
        self.model = create_model(config, model_type, num_classes, img_size, mode)

        # 确保模型完全转移到指定设备
        self.model = self.model.to(self.device)

        # 打印检查设备是否正确
        #print(f"模型设备检查: {next(self.model.parameters()).device}")
        
        # 如果有多GPU，使用DataParallel
        if len(gpu_ids) > 1:
            self.model = initModel(self.model, gpu_ids)
            
        # 设置保存目录
        self.results_dir = Path(config.OUTPUT_DIR) if hasattr(config, 'OUTPUT_DIR') else Path("output")
        self.results_dir.mkdir(exist_ok=True, parents=True)
            
        # 使用来自 core 文件夹的损失函数
        self.setup_loss_functions()
        
        # 设置优化器
        self.setup_optimizer()
            
        # 设置学习率调度器
        self.setup_scheduler()
            
        # 设置混合精度训练
        self.setup_mixed_precision()
            
        # 设置EMA模型
        self.setup_ema()
            
        # 设置训练参数
        self.setup_training_params()
        
        # 使用核心文件夹中的评估器和可视化器
        self.setup_evaluation_tools()
        
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
    
    def setup_loss_functions(self):
        """设置多任务损失函数"""
        # 使用已定义的损失函数类，而不是从core.losses导入
        # 这些类已在trainer.py顶部定义
        
        # 获取损失函数配置
        if hasattr(self.config, 'MULTI_TASK'):
            multi_task = self.config.MULTI_TASK
            
            # 分类损失
            if hasattr(multi_task, 'CLASSIFICATION') and multi_task.CLASSIFICATION.ENABLED:
                # 获取类别权重
                class_weights = multi_task.CLASSIFICATION.get('CLASS_WEIGHTS', [1.0, 1.0])
                # 先创建张量，但不指定设备
                self.cls_weights_raw = class_weights
                # 创建无权重的损失函数
                self.cls_loss_fn = nn.CrossEntropyLoss(reduction='mean')
                self.cls_weight = multi_task.CLASSIFICATION.get('WEIGHT', 1.0)
            else:
                # 默认也创建无权重的损失函数
                self.cls_loss_fn = nn.CrossEntropyLoss(reduction='mean')
                self.cls_weight = 1.0
            
                
            # 掩码损失
            if hasattr(multi_task, 'MASK') and multi_task.MASK.ENABLED:
                mask_cfg = multi_task.MASK
                if mask_cfg.LOSS_TYPE == 'dice':
                    self.mask_loss_fn = DiceLoss(
                        smooth=mask_cfg.DICE_SMOOTH if hasattr(mask_cfg, 'DICE_SMOOTH') else 1.0
                    )
                elif mask_cfg.LOSS_TYPE == 'dice_bce':
                    self.mask_loss_fn = BCEDiceLoss(weight_bce=0.5, weight_dice=0.5)
                else:
                    self.mask_loss_fn = EdgeAwareLoss(reduction='mean', edge_weight=0.7)
                self.mask_weight = mask_cfg.WEIGHT if hasattr(mask_cfg, 'WEIGHT') else 0.5
            else:
                self.mask_loss_fn = EdgeAwareLoss(reduction='mean')
                self.mask_weight = 0.5
                
            # 频域损失
            self.freq_loss_fn = FrequencyConsistencyLoss(reduction='mean')
            self.freq_weight = 0.3  # 默认权重
            
            # 对比损失
            if hasattr(multi_task, 'CONTRASTIVE') and multi_task.CONTRASTIVE.ENABLED:
                # 如果需要对比损失，可以在这里实现或导入
                self.contrastive_loss_fn = None  # 这里需要实现ContrastiveLoss类
                contrastive_cfg = multi_task.CONTRASTIVE
                self.contrastive_weight = contrastive_cfg.WEIGHT if hasattr(contrastive_cfg, 'WEIGHT') else 0.2
            else:
                self.contrastive_loss_fn = None
                self.contrastive_weight = 0.0
        else:
            # 默认损失函数，使用已在trainer.py顶部定义的类
            self.cls_loss_fn = FocalLoss(alpha=0.25, gamma=2.0)
            self.mask_loss_fn = EdgeAwareLoss(reduction='mean')
            self.freq_loss_fn = FrequencyConsistencyLoss(reduction='mean')
            self.cls_weight = 1.0
            self.mask_weight = 0.5
            self.freq_weight = 0.3
            self.contrastive_loss_fn = None
            self.contrastive_weight = 0.0

        # 更新掩码损失函数定义
        # 从这样的代码:
        # self.mask_criterion = nn.BCELoss()
        
        # 修改为:
        if hasattr(self.config, 'MULTI_TASK') and hasattr(self.config.MULTI_TASK, 'MASK') and self.config.MULTI_TASK.MASK.get('ENABLED', False):
            mask_loss_type = self.config.MULTI_TASK.MASK.get('LOSS_TYPE', 'dice_bce').lower()
            #print(f"使用掩码损失类型: {mask_loss_type}")

            if mask_loss_type == 'bce':
                # 使用BCEWithLogitsLoss代替BCELoss，以支持混合精度
                self.mask_criterion = nn.BCEWithLogitsLoss()
                #print("使用BCEWithLogitsLoss作为掩码损失")
            elif mask_loss_type == 'dice':
                from core.losses import DiceLoss
                self.mask_criterion = DiceLoss()
                #print("使用DiceLoss作为掩码损失")
            elif mask_loss_type == 'dice_bce':
                from core.losses import DiceBCELoss
                self.mask_criterion = DiceBCELoss()
                #print("使用DiceBCELoss作为掩码损失")
            else:
                # 默认安全的选择
                self.mask_criterion = nn.BCEWithLogitsLoss()
                #print(f"未知的掩码损失类型 '{mask_loss_type}'，使用默认的BCEWithLogitsLoss")
        else:
            self.mask_criterion = None
            #print("掩码损失被禁用")
    
    def setup_optimizer(self):
        """设置优化器，支持分层学习率"""
        if hasattr(self.config, 'OPTIMIZER'):
            opt_cfg = self.config.OPTIMIZER
            
            # 检查是否使用分层学习率
            if hasattr(opt_cfg, 'LAYER_DECAY') and opt_cfg.LAYER_DECAY.ENABLED:
                layer_cfg = opt_cfg.LAYER_DECAY
                
                # 根据不同模块设置不同学习率
                backbone_params = []
                fusion_params = []
                attention_params = []
                classifier_params = []
                other_params = []
                
                # 分类模型参数
                for name, param in self.model.named_parameters():
                    if 'backbone' in name:
                        backbone_params.append(param)
                    elif 'fusion' in name or 'feature_fusion' in name:
                        fusion_params.append(param)
                    elif 'attention' in name:
                        attention_params.append(param)
                    elif 'classifier' in name or 'fc' in name:
                        classifier_params.append(param)
                    else:
                        other_params.append(param)
                
                # 参数组配置
                param_groups = [
                    {'params': backbone_params, 
                     'lr': opt_cfg.LR * layer_cfg.BACKBONE},
                    {'params': fusion_params, 
                     'lr': opt_cfg.LR * layer_cfg.FUSION},
                    {'params': attention_params, 
                     'lr': opt_cfg.LR * layer_cfg.ATTENTION},
                    {'params': classifier_params, 
                     'lr': opt_cfg.LR * layer_cfg.CLASSIFIER},
                    {'params': other_params}
                ]
                
                # 创建优化器
                if opt_cfg.NAME.lower() == 'sgd':
                    self.optimizer = torch.optim.SGD(
                        param_groups,
                        lr=opt_cfg.LR,
                        momentum=opt_cfg.MOMENTUM if hasattr(opt_cfg, 'MOMENTUM') else 0.9,
                        weight_decay=opt_cfg.WEIGHT_DECAY if hasattr(opt_cfg, 'WEIGHT_DECAY') else 0.0001,
                        nesterov=True
                    )
                elif opt_cfg.NAME.lower() == 'adam':
                    self.optimizer = torch.optim.Adam(
                        param_groups,
                        lr=opt_cfg.LR,
                        betas=(opt_cfg.BETA1 if hasattr(opt_cfg, 'BETA1') else 0.9, 
                              opt_cfg.BETA2 if hasattr(opt_cfg, 'BETA2') else 0.999),
                        weight_decay=opt_cfg.WEIGHT_DECAY if hasattr(opt_cfg, 'WEIGHT_DECAY') else 0.0001
                    )
                else:  # 默认AdamW
                    self.optimizer = torch.optim.AdamW(
                        param_groups,
                        lr=opt_cfg.LR,
                        betas=(opt_cfg.BETA1 if hasattr(opt_cfg, 'BETA1') else 0.9, 
                              opt_cfg.BETA2 if hasattr(opt_cfg, 'BETA2') else 0.999),
                        weight_decay=opt_cfg.WEIGHT_DECAY if hasattr(opt_cfg, 'WEIGHT_DECAY') else 0.0001
                    )
            else:
                # 标准优化器
                if opt_cfg.NAME.lower() == 'sgd':
                    self.optimizer = torch.optim.SGD(
                        filter(lambda p: p.requires_grad, self.model.parameters()),
                        lr=opt_cfg.LR,
                        momentum=opt_cfg.MOMENTUM if hasattr(opt_cfg, 'MOMENTUM') else 0.9,
                        weight_decay=opt_cfg.WEIGHT_DECAY if hasattr(opt_cfg, 'WEIGHT_DECAY') else 0.0001,
                        nesterov=True
                    )
                elif opt_cfg.NAME.lower() == 'adam':
                    self.optimizer = torch.optim.Adam(
                        filter(lambda p: p.requires_grad, self.model.parameters()),
                        lr=opt_cfg.LR,
                        betas=(opt_cfg.BETA1 if hasattr(opt_cfg, 'BETA1') else 0.9, 
                              opt_cfg.BETA2 if hasattr(opt_cfg, 'BETA2') else 0.999),
                        weight_decay=opt_cfg.WEIGHT_DECAY if hasattr(opt_cfg, 'WEIGHT_DECAY') else 0.0001
                    )
                else:  # 默认AdamW
                    self.optimizer = torch.optim.AdamW(
                        filter(lambda p: p.requires_grad, self.model.parameters()),
                        lr=opt_cfg.LR,
                        betas=(opt_cfg.BETA1 if hasattr(opt_cfg, 'BETA1') else 0.9, 
                              opt_cfg.BETA2 if hasattr(opt_cfg, 'BETA2') else 0.999),
                        weight_decay=opt_cfg.WEIGHT_DECAY if hasattr(opt_cfg, 'WEIGHT_DECAY') else 0.0001
                    )
        else:
            # 默认AdamW
            self.optimizer = torch.optim.AdamW(
                filter(lambda p: p.requires_grad, self.model.parameters()),
                lr=0.0002,
                betas=(0.9, 0.999),
                weight_decay=0.0001
            )
    
    def setup_scheduler(self):
        """设置学习率调度器"""
        if hasattr(self.config, 'LR_SCHEDULER'):
            lr_cfg = self.config.LR_SCHEDULER
            
            # 根据配置类型选择调度器
            if lr_cfg.NAME.lower() == 'cosine':
                # 余弦退火学习率
                self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                    self.optimizer,
                    T_max=lr_cfg.COSINE.T_MAX if hasattr(lr_cfg, 'COSINE') and hasattr(lr_cfg.COSINE, 'T_MAX') else 100,
                    eta_min=lr_cfg.COSINE.ETA_MIN if hasattr(lr_cfg, 'COSINE') and hasattr(lr_cfg.COSINE, 'ETA_MIN') else 0.00001
                )
            elif lr_cfg.NAME.lower() == 'step':
                # 阶梯式学习率
                self.scheduler = torch.optim.lr_scheduler.StepLR(
                    self.optimizer,
                    step_size=lr_cfg.STEP.STEPS[0] if hasattr(lr_cfg, 'STEP') and hasattr(lr_cfg.STEP, 'STEPS') else 30,
                    gamma=lr_cfg.STEP.GAMMA if hasattr(lr_cfg, 'STEP') and hasattr(lr_cfg.STEP, 'GAMMA') else 0.1
                )
            elif lr_cfg.NAME.lower() == 'multistep':
                # 多阶梯式学习率
                self.scheduler = torch.optim.lr_scheduler.MultiStepLR(
                    self.optimizer,
                    milestones=lr_cfg.MULTISTEP.MILESTONES if hasattr(lr_cfg, 'MULTISTEP') and hasattr(lr_cfg.MULTISTEP, 'MILESTONES') else [30, 60, 90],
                    gamma=lr_cfg.MULTISTEP.GAMMA if hasattr(lr_cfg, 'MULTISTEP') and hasattr(lr_cfg.MULTISTEP, 'GAMMA') else 0.1
                )
            elif lr_cfg.NAME.lower() == 'plateau':
                # 根据验证结果调整学习率
                self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                    self.optimizer,
                    mode='min',
                    factor=lr_cfg.PLATEAU.FACTOR if hasattr(lr_cfg, 'PLATEAU') and hasattr(lr_cfg.PLATEAU, 'FACTOR') else 0.5,
                    patience=lr_cfg.PLATEAU.PATIENCE if hasattr(lr_cfg, 'PLATEAU') and hasattr(lr_cfg.PLATEAU, 'PATIENCE') else 5,
                    min_lr=lr_cfg.PLATEAU.MIN_LR if hasattr(lr_cfg, 'PLATEAU') and hasattr(lr_cfg.PLATEAU, 'MIN_LR') else 0.00001
                )
            else:
                # 默认使用余弦退火
                self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                    self.optimizer,
                    T_max=100,
                    eta_min=0.00001
                )
                
        # 检查是否使用预热
        if hasattr(lr_cfg, 'WARMUP') and lr_cfg.WARMUP.ENABLED:
            warmup_cfg = lr_cfg.WARMUP
            # 使用本地定义的GradualWarmupScheduler而不是从core.scheduler导入
            self.scheduler = GradualWarmupScheduler(
                self.optimizer, 
                multiplier=warmup_cfg.MULTIPLIER if hasattr(warmup_cfg, 'MULTIPLIER') else 1.0,
                total_epoch=warmup_cfg.EPOCHS if hasattr(warmup_cfg, 'EPOCHS') else 5,
                after_scheduler=self.scheduler
            )
        else:
            # 默认学习率调度器
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=100,
                eta_min=0.00001
            )
    
    def setup_mixed_precision(self):
        """设置混合精度训练"""
        self.use_amp = False
        self.scaler = None
        
        if hasattr(self.config, 'TRAINING') and hasattr(self.config.TRAINING, 'MIXED_PRECISION'):
            self.use_amp = self.config.TRAINING.MIXED_PRECISION
            if self.use_amp:
                self.scaler = torch.amp.GradScaler('cuda')
    
    def setup_ema(self):
        """设置EMA模型"""
        self.use_ema = False
        self.ema = None
        
        if hasattr(self.config, 'TRAINING') and hasattr(self.config.TRAINING, 'EMA') and self.config.TRAINING.EMA.ENABLED:
            self.use_ema = True
            ema_cfg = self.config.TRAINING.EMA
            self.ema = ModelEMA(
                self.model,
                decay=ema_cfg.DECAY if hasattr(ema_cfg, 'DECAY') else 0.999,
                device=self.device
            )
    
    def setup_training_params(self):
        """设置训练参数"""
        # 梯度裁剪值
        self.clip_grad_norm = 0.0
        if hasattr(self.config, 'TRAINING') and hasattr(self.config.TRAINING, 'CLIP_GRAD_NORM'):
            self.clip_grad_norm = self.config.TRAINING.CLIP_GRAD_NORM
        
        # 早停设置
        self.early_stopping = False
        self.patience = 0
        self.min_delta = 0
        self.best_metric = float('inf')
        self.no_improve_count = 0
        
        if hasattr(self.config, 'TRAINING') and hasattr(self.config.TRAINING, 'EARLY_STOPPING') and self.config.TRAINING.EARLY_STOPPING.ENABLED:
            self.early_stopping = True
            early_cfg = self.config.TRAINING.EARLY_STOPPING
            self.patience = early_cfg.PATIENCE if hasattr(early_cfg, 'PATIENCE') else 10
            self.min_delta = early_cfg.MIN_DELTA if hasattr(early_cfg, 'MIN_DELTA') else 0.001
    
    def setup_evaluation_tools(self):
        """设置评估工具，从core文件夹中导入"""
        # 使用来自core的评估器和可视化器
        self.evaluator = ModelEvaluator(save_dir=self.results_dir / "evaluation")
        self.visualizer = ForensicVisualizer(save_dir=self.results_dir / "visualizations")
    
    def train_epoch(self, train_loader, epoch):
        """
        训练一个epoch，使用core目录中的功能
        
        Args:
            train_loader: 训练数据加载器
            epoch: 当前epoch
            
        Returns:
            训练损失和准确率
        """
        # 更新当前epoch
        self.current_epoch = epoch
        self.model.train()
        
        # 初始化指标追踪器
        total_loss = 0.0
        total_cls_loss = 0.0
        total_mask_loss = 0.0
        total_freq_loss = 0.0
        total_correct = 0
        total_samples = 0
        
        # 打印进度条
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}, Training")
        
        for i, batch in enumerate(pbar):
            self.current_step += 1
            
            # 处理批次数据 - 使用core.dataset中的格式
            inputs, labels, masks, dct_features = None, None, None, None
            
            # 根据批次格式处理输入
            if len(batch) == 2:  # 仅图像和标签
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
            else:
                self.logger.warning(f"未知的批次格式: {len(batch)}")
                continue
            
            # 移动到设备
            inputs = inputs.to(self.device)
            labels = labels.to(self.device)
            
            if masks is not None:
                masks = masks.to(self.device)
            if dct_features is not None:
                dct_features = dct_features.to(self.device)
            
            # 清零梯度
            self.optimizer.zero_grad()
            
            # 自动混合精度处理
            if hasattr(self, 'amp_enabled') and self.amp_enabled:
                # 使用 autocast 上下文管理器
                with torch.cuda.amp.autocast():
                    if dct_features is not None:
                        outputs = self.model(inputs, dct_features)
                    else:
                        outputs = self.model(inputs)

                    # 分离输出
                    if isinstance(outputs, tuple) and len(outputs) == 2:
                        # 分类和掩码输出
                        cls_outputs, mask_outputs = outputs
                        outputs = cls_outputs  # 分类输出用于主要损失计算
                    else:
                        mask_outputs = None

                    # 使用ensure_same_device函数确保所有张量在同一设备上
                    labels = ensure_same_device(outputs, labels)

                    # 计算分类损失
                    cls_loss = self.cls_loss_fn(outputs, labels)

                    # 计算掩码损失 (如果有)
                    if hasattr(self, 'mask_loss_fn') and self.mask_loss_fn is not None and mask_outputs is not None and masks is not None:
                        # 确保张量在同一设备上
                        masks_device = ensure_same_device(mask_outputs, masks)

                        # 检查是否需要logits处理
                        if isinstance(self.mask_loss_fn, nn.BCELoss):
                            # 如果是BCELoss，手动添加sigmoid (不推荐，应该用BCEWithLogitsLoss)
                            mask_outputs_sigmoid = torch.sigmoid(mask_outputs)
                            try:
                                mask_loss = self.mask_loss_fn(mask_outputs_sigmoid, masks_device)
                            except Exception as e:
                                #print(f"计算掩码损失(sigmoid后)错误: {e}")
                                mask_loss = torch.tensor(0.0, device=self.device)
                        else:
                            # 适合混合精度的损失函数(如BCEWithLogitsLoss)
                            try:
                                mask_loss = self.mask_loss_fn(mask_outputs, masks_device)
                            except Exception as e:
                                #print(f"计算掩码损失错误: {e}")
                                mask_loss = torch.tensor(0.0, device=self.device)

                        # 添加掩码损失到总损失
                        mask_weight = self.mask_weight
                        loss = cls_loss + mask_weight * mask_loss
                    else:
                        mask_loss = torch.tensor(0.0, device=self.device)
                        loss = cls_loss

                # 使用ScalerContext进行反向传播
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()

            else:
                # 检查输入和模型设备是否匹配
                model_device = next(self.model.parameters()).device
                if inputs.device != model_device:
                    #print(f"警告: 输入设备 {inputs.device} 与模型设备 {model_device} 不匹配，尝试修复")
                    inputs = inputs.to(model_device)
                    if dct_features is not None:
                        dct_features = dct_features.to(model_device)
                        
                # 前向传播 - 处理不同的输入格式
                if dct_features is not None:
                    outputs = self.model(inputs, dct_features)
                else:
                    outputs = self.model(inputs)
                # 使用compute_losses计算损失
                loss, cls_loss, mask_loss, freq_loss = self.compute_losses(outputs, labels, masks)
                
                # 反向传播和优化
                loss.backward()
                if self.clip_grad_norm > 0:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clip_grad_norm)
                self.optimizer.step()
            
            # 更新EMA模型
            if self.use_ema:
                self.ema.update(self.model)
            
            # 计算分类准确率 - 处理不同的输出格式
            if isinstance(outputs, tuple):
                # 如果是多任务输出，取出分类输出
                class_outputs = outputs[1] if len(outputs) >= 2 else outputs[0]
            else:
                class_outputs = outputs
                
            try:
                _, predicted = torch.max(class_outputs.data, 1)
                correct = (predicted == labels).sum().item()
            except:
                # 二分类模型，使用sigmoid
                predicted = (torch.sigmoid(class_outputs) > 0.5).float()
                correct = (predicted.squeeze() == labels).sum().item()
            
            # 更新统计信息
            batch_size = labels.size(0)
            total_correct += correct
            total_samples += batch_size
            total_loss += loss.item() * batch_size
            total_cls_loss += cls_loss.item() * batch_size
            
            if mask_loss is not None:
                total_mask_loss += mask_loss.item() * batch_size
            if freq_loss is not None:
                total_freq_loss += freq_loss.item() * batch_size
            
            # 更新进度条
            pbar.set_postfix({
                'loss': loss.item(), 
                'cls_loss': cls_loss.item(),
                'mask_loss': mask_loss.item() if mask_loss is not None else 0,
                'acc': correct / batch_size
            })
            
            # 清理内存
            torch.cuda.empty_cache()
        
        # 计算epoch平均指标
        epoch_loss = total_loss / total_samples
        epoch_cls_loss = total_cls_loss / total_samples
        epoch_mask_loss = total_mask_loss / total_samples
        epoch_freq_loss = total_freq_loss / total_samples
        epoch_acc = total_correct / total_samples
        
        # 更新学习率
        if self.scheduler:
            if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                self.scheduler.step(epoch_loss)
            else:
                self.scheduler.step()
        
        # 获取当前学习率
        lr = self.optimizer.param_groups[0]['lr']
        
        # 更新历史记录
        self.history['train_loss'].append(epoch_loss)
        self.history['train_acc'].append(epoch_acc)
        self.history['train_cls_loss'].append(epoch_cls_loss)
        self.history['train_mask_loss'].append(epoch_mask_loss)
        self.history['train_freq_loss'].append(epoch_freq_loss)
        self.history['lr'].append(lr)
        
        # 打印epoch结果
        print(f"Epoch {epoch} - Train Loss: {epoch_loss:.4f}, Acc: {epoch_acc:.4f}, LR: {lr:.6f}")
        
        return {
            'loss': epoch_loss,
            'cls_loss': epoch_cls_loss,
            'mask_loss': epoch_mask_loss,
            'freq_loss': epoch_freq_loss,
            'accuracy': epoch_acc
        }
    
    # 在trainer.py的validate函数中修复早停逻辑
    def validate(self, val_loader, epoch=None):
        """在验证集上评估模型"""
        # 使用EMA模型进行评估
        model_to_eval = self.ema.ema_model if self.use_ema else self.model
        
        # 使用ModelEvaluator进行评估
        metrics = self.evaluator.evaluate_model(model_to_eval, val_loader, self.device)
        
        # 更新历史记录
        self.history['val_loss'].append(metrics.get('loss', 0))
        self.history['val_acc'].append(metrics.get('accuracy', 0))
        
        # 打印epoch结果
        if epoch is not None:
            print(f"Epoch {epoch} - Val Loss: {metrics.get('loss', 0):.4f}, "
                  f"Acc: {metrics.get('accuracy', 0):.4f}, AUC: {metrics.get('auc', 0):.4f}")
        
        # 检查是否需要早停
        if self.early_stopping:
            current_metric = metrics.get('loss', float('inf'))
            if current_metric < self.best_metric - self.min_delta:
                self.best_metric = current_metric
                self.no_improve_count = 0
                
                # 保存最佳模型
                self.save(os.path.join(self.results_dir, 'best_model.pth'), is_best=True)
                print(f"保存新的最佳模型，验证损失: {self.best_metric:.4f}")
            else:
                self.no_improve_count += 1
                print(f"验证指标未改善，当前连续未改善次数: {self.no_improve_count}/{self.patience}")
                    
            if self.no_improve_count >= self.patience:
                print(f"早停触发! {self.patience} 轮未改善。")
                return metrics, True  # 第二个返回值为True表示触发早停
        
        return metrics, False
    
    def compute_losses(self, outputs, labels, masks=None):
        """
        计算多任务损失

        Args:
            outputs: 模型输出，可能是单一输出或包含多个元素的元组
            labels: 真实标签
            masks: 真实掩码（可选）

        Returns:
            总损失, 分类损失, 掩码损失, 频域损失
        """
        # 初始化损失值
        mask_loss = None
        freq_loss = None
        contrastive_loss = None

        # 处理不同的输出格式
        if isinstance(outputs, tuple):
            # 多任务输出
            if len(outputs) == 2:  # 例如: (mask_preds, class_outputs)
                mask_preds, class_outputs = outputs
            elif len(outputs) >= 3:  # 例如: (mask_preds, class_outputs, features)
                mask_preds, class_outputs, features = outputs[:3]
            else:
                mask_preds = None
                class_outputs = outputs[0]
        else:
            # 单任务输出（只有分类）
            mask_preds = None
            class_outputs = outputs

        # 确保张量在同一设备上
        device = class_outputs.device
        labels = ensure_tensor_on_device(labels, device)
        if masks is not None:
            masks = ensure_tensor_on_device(masks, device)

        # 在计算分类损失前，动态设置权重到正确设备
        if hasattr(self, 'cls_weights_raw'):
            weight_tensor = torch.tensor(self.cls_weights_raw, dtype=torch.float, device=device)
            # 重新创建带权重的损失函数在正确设备上
            self.cls_loss_fn = nn.CrossEntropyLoss(weight=weight_tensor, reduction='mean')
        
        # 计算分类损失
        try:
            cls_loss = self.cls_loss_fn(class_outputs, labels)
        except Exception as e:
            print(f"计算分类损失错误: {e}")
            cls_loss = torch.tensor(0.0, device=device)

        # 如果有掩码预测和真实掩码，计算掩码损失
        if mask_preds is not None and masks is not None and hasattr(self, 'mask_loss_fn'):
            try:
                # 确保掩码形状匹配
                if mask_preds.shape != masks.shape:
                    mask_preds = F.interpolate(mask_preds, size=masks.shape[2:], 
                                             mode='bilinear', align_corners=False)
                # 添加这一行确保值在[0,1]范围内
                mask_preds = torch.sigmoid(mask_preds)
                # 应用掩码损失函数
                mask_loss = self.mask_loss_fn(mask_preds, masks)
            except Exception as e:
                print(f"计算掩码损失错误: {e}")
                mask_loss = torch.tensor(0.0, device=device)
        else:
            mask_loss = torch.tensor(0.0, device=device)

        # 计算频域损失（如果启用）
        if hasattr(self, 'freq_loss_fn') and hasattr(self.model, 'get_frequency_features'):
            try:
                pred_freq, true_freq = self.model.get_frequency_features()
                if pred_freq is not None and true_freq is not None:
                    # 确保频域特征在同一设备上
                    device = class_outputs.device  # 使用主设备作为参考
                    pred_freq = pred_freq.to(device)
                    true_freq = true_freq.to(device)
                    freq_loss = self.freq_loss_fn(pred_freq, true_freq)
                else:
                    freq_loss = torch.tensor(0.0, device=device)
            except Exception as e:
                print(f"计算频域损失错误: {e}")
                freq_loss = torch.tensor(0.0, device=device)
        else:
            freq_loss = torch.tensor(0.0, device=device)

        # 计算对比损失（如果启用）
        if hasattr(self, 'contrastive_loss_fn') and self.contrastive_loss_fn is not None and isinstance(outputs, tuple) and len(outputs) >= 3:
            try:
                features = outputs[2]
                contrastive_loss = self.contrastive_loss_fn(features, labels)
            except Exception as e:
                print(f"计算对比损失错误: {e}")
                contrastive_loss = torch.tensor(0.0, device=device)
        else:
            contrastive_loss = torch.tensor(0.0, device=device)

        # 计算总损失
        total_loss = self.cls_weight * cls_loss

        if mask_loss is not None:
            total_loss += self.mask_weight * mask_loss

        if freq_loss is not None:
            total_loss += self.freq_weight * freq_loss

        if contrastive_loss is not None:
            total_loss += self.contrastive_weight * contrastive_loss

        return total_loss, cls_loss, mask_loss, freq_loss
    
    def test(self, test_loader):
        """测试模型，使用评估器进行完整评估"""
        # 使用EMA模型进行测试(如果有)
        if self.use_ema:
            eval_model = self.ema.ema_model
        else:
            eval_model = self.model
        
        # 使用utils中的evaluate函数，它调用ModelEvaluator
        metrics = evaluate(eval_model, self.config, test_loader=test_loader, 
                          evaluate_boundary=True, evaluate_freq=True)
        
        # 打印主要结果
        self.logger.info(f"测试结果: 准确率: {metrics.get('accuracy', 0):.4f}, AUC: {metrics.get('auc', 0):.4f}")
        
        # 如果有掩码指标，也打印
        if 'mask_metrics' in metrics:
            mask_metrics = metrics['mask_metrics']
            self.logger.info(f"掩码评估: IoU: {mask_metrics.get('mean_iou', 0):.4f}, "
                           f"Dice: {mask_metrics.get('mean_dice', 0):.4f}")
            
        # 如果有边界指标，也打印
        if 'boundary_metrics' in metrics:
            boundary_metrics = metrics['boundary_metrics']
            self.logger.info(f"边界评估: F1: {boundary_metrics.get('boundary_f1', 0):.4f}, "
                           f"IoU: {boundary_metrics.get('boundary_iou', 0):.4f}")
        
        return metrics
    
    def train(self, epochs, train_loader, val_loader=None, test_loader=None):
        """
        训练模型，使用core模块进行评估和可视化
        
        Args:
            epochs: 总epochs数
            train_loader: 训练数据加载器
            val_loader: 验证数据加载器
            test_loader: 测试数据加载器
            
        Returns:
            dict: 最终训练历史
        """
        # 确保模型在正确设备上
        self.ensure_model_on_device()
        
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
                val_metrics, stop_early = self.validate(val_loader, epoch)
                
                # 检查是否是最佳模型
                if val_metrics['accuracy'] > self.best_val_acc:
                    self.best_val_acc = val_metrics['accuracy']
                    self.save(self.results_dir / 'best_acc_model.pth', is_best=True)
                    self.logger.info(f"新的最佳验证准确率: {self.best_val_acc:.4f}")
                
                if val_metrics.get('loss', float('inf')) < self.best_val_loss:
                    self.best_val_loss = val_metrics.get('loss', float('inf'))
                    self.save(self.results_dir / 'best_loss_model.pth', is_best=True)
                    self.logger.info(f"新的最佳验证损失: {self.best_val_loss:.4f}")
                
                # 判断是否需要早停
                if stop_early:
                    self.logger.info("早停触发，终止训练")
                    break
            
                # 定期可视化训练进度
                if (epoch + 1) % 5 == 0 or epoch == epochs - 1:
                    # 绘制训练曲线
                    self.visualizer.plot_training_curves(
                        self.history,
                        save_path=self.results_dir / 'training_curves.png'
                    )
                
                # 显示一些预测结果
                if val_loader:
                    # 使用核心可视化功能显示一批预测结果
                    self.visualize_predictions(val_loader, epoch)
        
        # 训练结束，进行最终测试
        if test_loader:
            self.logger.info("训练完成，进行最终测试...")
            test_metrics = self.test(test_loader)
            
            # 创建完整的评估报告
            self.visualizer.create_evaluation_report(test_metrics, self.results_dir / 'final_evaluation')
        
        total_time = time.time() - start_time
        self.logger.info(f"训练完成，总耗时: {total_time/60:.2f} 分钟")
        
        return self.history
    
    def visualize_predictions(self, val_loader, epoch):
        """可视化验证集上的预测结果"""
        # 使用EMA模型(如果有)
        model_to_eval = self.ema.ema_model if self.use_ema else self.model
        model_to_eval.eval()
        
        # 获取一批数据
        batch = next(iter(val_loader))
        
        # 处理批次格式
        if len(batch) >= 4 and isinstance(batch[3], str):  # 如果第四个元素是字符串(路径)
            if isinstance(batch[1], torch.Tensor) and batch[1].dim() == 4 and batch[1].size(1) <= 64:
                # RGB + DCT + 掩码 + 路径格式
                inputs, dct_inputs, masks, paths = batch[:4]
                labels = batch[4] if len(batch) > 4 else None
            else:
                # RGB + 掩码 + 标签 + 路径格式
                inputs, masks, labels, paths = batch[:4]
                dct_inputs = None
        else:
            # 处理没有路径的情况
            if len(batch) == 3:
                if isinstance(batch[1], torch.Tensor) and batch[1].dim() == 4 and batch[1].size(1) <= 64:
                    inputs, dct_inputs, labels = batch
                    masks = None
                else:
                    inputs, masks, labels = batch
                    dct_inputs = None
                paths = ["unknown"] * len(inputs)
            elif len(batch) >= 4:
                # 同上处理...
                if isinstance(batch[1], torch.Tensor) and batch[1].dim() == 4 and batch[1].size(1) <= 64:
                    inputs, dct_inputs, masks, labels = batch[:4]
                else:
                    inputs, masks, labels, _ = batch[:4]
                    dct_inputs = None
                paths = ["unknown"] * len(inputs)
            else:
                self.logger.warning("无法解析批次格式")
                return
        
        # 推理
        with torch.no_grad():
            # 移动数据到设备
            inputs = inputs.to(self.device)
            if dct_inputs is not None:
                dct_inputs = dct_inputs.to(self.device)
            if masks is not None:
                masks = masks.to(self.device)
            labels = labels.to(self.device)
            
            # 模型推理
            if dct_inputs is not None:
                outputs = model_to_eval(inputs, dct_inputs)
            else:
                outputs = model_to_eval(inputs)
            
            # 处理输出
            if isinstance(outputs, tuple):
                mask_preds, class_outputs = outputs[:2]
            else:
                class_outputs = outputs
                mask_preds = None
                
            # 获取预测概率和标签
            if class_outputs.size(1) > 1:  # 多分类情况
                probs = F.softmax(class_outputs, dim=1)[:, 1].detach().cpu().numpy()
                _, preds = torch.max(class_outputs, 1)
            else:  # 二分类情况
                probs = torch.sigmoid(class_outputs).squeeze().detach().cpu().numpy()
                preds = (probs >= 0.5).astype(np.int64)
                
        # 从批次中提取可视化样本的数量
        n_samples = min(8, len(inputs))
        
        # 使用可视化器进行可视化
        if mask_preds is not None and masks is not None:
            # 可视化掩码预测
            self.visualizer.visualize_masks(
                inputs.cpu().numpy()[:n_samples], 
                masks.cpu().numpy()[:n_samples],
                mask_preds.detach().cpu().numpy()[:n_samples],
                save_path=self.results_dir / f'epoch_{epoch}_masks.png'
            )
            
            # 可视化边界检测
            self.visualizer.visualize_boundary_detection(
                inputs.cpu().numpy()[:n_samples],
                mask_preds.detach().cpu().numpy()[:n_samples],
                masks.cpu().numpy()[:n_samples],
                save_dir=self.results_dir / f'epoch_{epoch}_boundaries'
            )
        
        # 可视化注意力图(如果可用)
        try:
            self.visualizer.visualize_attention_maps(
                model_to_eval,
                inputs[:n_samples],
                device=self.device,
                save_dir=self.results_dir / f'epoch_{epoch}_attention'
            )
        except:
            self.logger.debug("无法可视化注意力图")
        
        # 可视化DCT特征(如果可用)
        try:
            self.visualizer.visualize_dct_features(
                model_to_eval,
                inputs[:n_samples],
                device=self.device,
                save_dir=self.results_dir / f'epoch_{epoch}_dct'
            )
        except:
            self.logger.debug("无法可视化DCT特征")

    def save(self, path, is_best=False):
        """
        保存模型检查点
        
        Args:
            path: 保存路径
            is_best: 是否为最佳模型
        """
        # 创建保存目录
        os.makedirs(os.path.dirname(path), exist_ok=True)
        
        # 准备检查点数据
        checkpoint = {
            'epoch': self.current_epoch,
            'step': self.current_step,
            'model': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'scheduler': self.scheduler.state_dict() if self.scheduler else None,
            'config': self.config,
            'best_val_loss': self.best_val_loss,
            'best_val_acc': self.best_val_acc,
            'history': self.history,
            'timestamp': time.strftime('%Y-%m-%d_%H-%M-%S')
        }
        
        # 添加EMA模型状态(如果有)
        if self.use_ema:
            checkpoint['ema_model'] = self.ema.ema_model.state_dict()
        
        # 添加混合精度训练状态(如果有)
        if self.use_amp and self.scaler:
            checkpoint['scaler'] = self.scaler.state_dict()
        
        # 使用utils的save_checkpoint函数保存
        save_checkpoint(checkpoint, is_best, os.path.dirname(path), filename=os.path.basename(path))
        self.logger.info(f"模型检查点已保存至 {path}")

    def load(self, path):
        """
        加载模型检查点
        
        Args:
            path: 检查点路径
        """
        if not os.path.exists(path):
            self.logger.warning(f"检查点不存在: {path}")
            return False
            
        try:
            # 加载检查点，使用utils中的函数
            self.logger.info(f"加载检查点: {path}")
            checkpoint = torch.load(path, map_location=self.device)
            
            # 加载模型状态
            if 'model' in checkpoint:
                self.model.load_state_dict(checkpoint['model'])
                self.logger.info("模型权重已加载")
            else:
                # 向后兼容旧格式
                self.model.load_state_dict(checkpoint)
                self.logger.info("模型权重已加载(旧格式)")
                
            # 加载EMA模型(如果存在)
            if self.use_ema and 'ema_model' in checkpoint:
                self.ema.ema_model.load_state_dict(checkpoint['ema_model'])
                self.logger.info("EMA模型已加载")
                
            # 加载优化器状态(如果存在)
            if 'optimizer' in checkpoint:
                try:
                    self.optimizer.load_state_dict(checkpoint['optimizer'])
                    self.logger.info("优化器状态已加载")
                except Exception as e:
                    self.logger.warning(f"优化器状态加载失败, 使用初始化状态: {e}")
                    
            # 加载调度器状态(如果存在)
            if self.scheduler is not None and 'scheduler' in checkpoint and checkpoint['scheduler'] is not None:
                try:
                    self.scheduler.load_state_dict(checkpoint['scheduler'])
                    self.logger.info("调度器状态已加载")
                except Exception as e:
                    self.logger.warning(f"调度器状态加载失败, 使用初始化状态: {e}")
                    
            # 加载混合精度缩放器状态(如果存在)
            if self.use_amp and self.scaler and 'scaler' in checkpoint and checkpoint['scaler'] is not None:
                try:
                    self.scaler.load_state_dict(checkpoint['scaler'])
                    self.logger.info("混合精度缩放器已加载")
                except Exception as e:
                    self.logger.warning(f"混合精度缩放器加载失败, 使用初始化状态: {e}")
                    
            # 加载训练状态
            if 'epoch' in checkpoint:
                self.current_epoch = checkpoint['epoch'] + 1  # 从下一个epoch继续
                self.logger.info(f"恢复训练, 起始epoch: {self.current_epoch}")
                
            if 'step' in checkpoint:
                self.current_step = checkpoint['step']
                
            if 'best_val_loss' in checkpoint:
                self.best_val_loss = checkpoint['best_val_loss']
                
            if 'best_val_acc' in checkpoint:
                self.best_val_acc = checkpoint['best_val_acc']
                
            if 'history' in checkpoint:
                self.history = checkpoint['history']
                
            return True
            
        except Exception as e:
            self.logger.error(f"加载检查点失败: {e}")
            import traceback
            self.logger.debug(traceback.format_exc())
            return False

    # 添加新方法：使用预训练权重初始化
    def initialize_from_pretrained(self, pretrained_path, strict=False):
        """
        从预训练权重初始化模型（不加载训练状态）
        
        Args:
            pretrained_path: 预训练权重路径
            strict: 是否严格加载权重
        """
        if not os.path.exists(pretrained_path):
            self.logger.warning(f"预训练权重不存在: {pretrained_path}")
            return False
            
        # 使用utils中的load_model_weights函数
        from utils import load_model_weights
        try:
            load_model_weights(self.model, pretrained_path, strict=strict)
            self.logger.info(f"从预训练权重初始化模型: {pretrained_path}")
            return True
        except Exception as e:
            self.logger.error(f"加载预训练权重失败: {e}")
            return False
    
    # 添加数据加载方法，使用core.dataset中的函数
    def prepare_data_loaders(self, config):
        """
        准备数据加载器
        
        Args:
            config: 配置对象
            
        Returns:
            训练、验证和测试数据加载器
        """
        from core.dataset import create_forensic_data_loaders
        
        self.logger.info("准备数据加载器...")
        try:
            train_loader, val_loader, test_loader = create_forensic_data_loaders(
                config, use_enhanced_dataset=True
            )
            
            self.logger.info(f"训练集: {len(train_loader.dataset)} 样本")
            self.logger.info(f"验证集: {len(val_loader.dataset)} 样本")
            self.logger.info(f"测试集: {len(test_loader.dataset)} 样本")
            
            return train_loader, val_loader, test_loader
        except Exception as e:
            self.logger.error(f"准备数据加载器失败: {e}")
            import traceback
            self.logger.error(traceback.format_exc())
            raise
    
    # 新增辅助方法：打印模型结构和参数量
    def print_model_summary(self):
        """打印模型结构和参数量"""
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        
        self.logger.info(f"模型类型: {self.model.__class__.__name__}")
        self.logger.info(f"总参数量: {total_params:,}")
        self.logger.info(f"可训练参数量: {trainable_params:,}")
        self.logger.info(f"使用设备: {self.device}")
        self.logger.info(f"模式: {self.mode}")
        
        # 输出模型结构
        try:
            from torchsummary import summary
            input_size = (3, 256, 256)  # 假设默认输入大小
            summary_str = summary(self.model, input_size, device=self.device, verbose=0)
            self.logger.debug(f"模型结构:\n{summary_str}")
        except ImportError:
            self.logger.debug("未安装torchsummary，跳过模型结构打印")
        except Exception as e:
            self.logger.debug(f"打印模型结构出错: {e}")

    def ensure_model_on_device(self):
        """确保模型的所有部分都在同一设备上"""
        device = self.device
        #print(f"强制将模型所有部分移至 {device}")

        # 更新损失函数的设备
        if hasattr(self, 'cls_weights_data'):
            self.cls_loss_fn.weight = self.cls_weights_data.to(device)
    
        # 递归检查并修复所有子模块
        for name, module in self.model.named_modules():
            for param_name, param in module.named_parameters(recurse=False):
                if param.device != device:
                    #print(f"警告: 参数 {name}.{param_name} 在 {param.device}，正在移至 {device}")
                    param.data = param.data.to(device)

        return self.model    

# 删除旧的 Trainer 类定义，只保留 EnhancedTrainer 类和兼容层

class Trainer(EnhancedTrainer):
    """兼容旧代码的训练器 - 只是 EnhancedTrainer 的别名"""
    def __init__(self, config, gpu_ids, model_type='enhanced', mode='Both', pretrained_path=None):
        super().__init__(config, gpu_ids, model_type, mode, pretrained_path)


# 添加便于快速创建训练器的工厂函数
def create_trainer(config, gpu_ids=None, model_type=None, mode=None, pretrained_path=None):
    """
        创建训练器工厂函数
        
        Args:
            config: 配置对象
            gpu_ids: GPU ID列表
            model_type: 模型类型
            mode: 模型模式
            pretrained_path: 预训练权重路径
            
        Returns:
            训练器实例
        """
    # 设置默认值
    if gpu_ids is None:
        gpu_ids = [0] if torch.cuda.is_available() else []
        
    # 从配置中获取模型类型和模式
    if model_type is None and hasattr(config, 'MODEL_CONFIG'):
        model_type = config.MODEL_CONFIG.TYPE
        
    if mode is None and hasattr(config, 'MODEL_CONFIG'):
        mode = config.MODEL_CONFIG.MODE
        
    # 设置默认值
    model_type = model_type or 'enhanced'
    mode = mode or 'Both'
    
    # 创建训练器
    return EnhancedTrainer(config, gpu_ids, model_type, mode, pretrained_path)
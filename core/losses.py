import torch
import torch.nn as nn
import torch.nn.functional as F

class DiceLoss(nn.Module):
    """适合混合精度训练的Dice损失函数"""
    def __init__(self, smooth=1.0, reduction='mean'):
        super(DiceLoss, self).__init__()
        self.smooth = smooth
        self.reduction = reduction
        
    def forward(self, logits, targets):
        # 应用sigmoid到logits
        probs = torch.sigmoid(logits)
        
        # 扁平化预测和目标
        batch_size = logits.shape[0]
        probs = probs.view(batch_size, -1)
        targets = targets.view(batch_size, -1)
        
        # 计算交集和总和
        intersection = torch.sum(probs * targets, dim=1)
        sum_p = torch.sum(probs, dim=1)
        sum_g = torch.sum(targets, dim=1)
        
        # 计算Dice系数
        dice = (2.0 * intersection + self.smooth) / (sum_p + sum_g + self.smooth)
        
        # 计算损失
        dice_loss = 1.0 - dice
        
        # 应用reduction
        if self.reduction == 'mean':
            return torch.mean(dice_loss)
        elif self.reduction == 'sum':
            return torch.sum(dice_loss)
        else:
            return dice_loss


class DiceBCELoss(nn.Module):
    """结合Dice损失和BCE损失，适合混合精度训练"""
    def __init__(self, dice_weight=1.0, bce_weight=1.0, smooth=1.0):
        super(DiceBCELoss, self).__init__()
        self.dice_weight = dice_weight
        self.bce_weight = bce_weight
        self.dice_loss = DiceLoss(smooth=smooth)
        self.bce_loss = nn.BCEWithLogitsLoss()  # 使用BCEWithLogitsLoss代替BCELoss
        
    def forward(self, logits, targets):
        dice_loss = self.dice_loss(logits, targets)
        bce_loss = self.bce_loss(logits, targets)
        
        # 结合两种损失
        total_loss = self.dice_weight * dice_loss + self.bce_weight * bce_loss
        return total_loss


class FocalLoss(nn.Module):
    """适合混合精度训练的Focal损失"""
    def __init__(self, gamma=2.0, alpha=0.25, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction
        
    def forward(self, logits, targets):
        # 应用sigmoid到logits
        probs = torch.sigmoid(logits)
        
        # 计算BCE损失
        bce_loss = F.binary_cross_entropy_with_logits(
            logits, targets, reduction='none')
        
        # 应用Focal项
        p_t = probs * targets + (1 - probs) * (1 - targets)
        loss = bce_loss * ((1 - p_t) ** self.gamma)
        
        # 应用alpha加权
        if self.alpha > 0:
            alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
            loss = alpha_t * loss
            
        # 应用reduction
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss
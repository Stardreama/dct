import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import pandas as pd
from PIL import Image
import cv2
from pathlib import Path
import torch
import os
import logging
from sklearn.metrics import confusion_matrix
import itertools
from typing import List, Dict, Any, Tuple, Optional, Union
import torchvision.transforms as transforms  # 添加这行导入

class ForensicVisualizer:
    """
    可视化工具类
    提供各种可视化方法
    """
    def __init__(self, save_dir=None, dpi=300, logger=None):
        """
        初始化可视化器
        
        Args:
            save_dir: 保存结果的目录
            dpi: 图像分辨率
            logger: 日志记录器
        """
        self.save_dir = Path(save_dir) if save_dir else None
        if self.save_dir:
            self.save_dir.mkdir(exist_ok=True, parents=True)
        self.dpi = dpi
        self.logger = logger or logging.getLogger(__name__)
    
    def plot_training_curves(self, history: Dict[str, List], save_path=None):
        """
        绘制训练曲线
        
        Args:
            history: 包含训练指标的字典，键为指标名，值为按轮次排列的列表
            save_path: 保存路径，如果不提供则使用默认设置
        """
        # 创建图形
        plt.figure(figsize=(15, 10))
        
        # 1. 绘制损失曲线
        plt.subplot(2, 2, 1)
        if 'train_loss' in history:
            plt.plot(history['train_loss'], label='训练损失')
        if 'val_loss' in history:
            plt.plot(history['val_loss'], label='验证损失')
        plt.title('损失曲线')
        plt.xlabel('轮次')
        plt.ylabel('损失')
        plt.legend()
        
        # 2. 绘制组件损失曲线
        plt.subplot(2, 2, 2)
        for loss_type in ['train_cls_loss', 'train_mask_loss', 'train_freq_loss']:
            if loss_type in history:
                plt.plot(history[loss_type], 
                         label=loss_type.replace('train_', '').replace('_', ' ').title())
        plt.title('损失组件')
        plt.xlabel('轮次')
        plt.ylabel('损失')
        plt.legend()
        
        # 3. 绘制准确率曲线
        plt.subplot(2, 2, 3)
        for acc_key in ['val_acc', 'train_acc', 'test_acc']:
            if acc_key in history:
                plt.plot(history[acc_key], 
                         label=acc_key.replace('_', ' ').title())
        plt.title('准确率')
        plt.xlabel('轮次')
        plt.ylabel('准确率')
        plt.legend()
        
        # 4. 绘制学习率曲线
        plt.subplot(2, 2, 4)
        if 'lr' in history:
            plt.plot(history['lr'], label='学习率')
        plt.title('学习率')
        plt.xlabel('轮次')
        plt.ylabel('LR')
        plt.yscale('log')  # 对数尺度更便于观察学习率变化
        plt.legend()
        
        # 保存图表
        plt.tight_layout()
        save_path = save_path or (self.save_dir / 'training_curves.png' if self.save_dir else None)
        if save_path:
            plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
            self.logger.info(f"训练曲线已保存至: {save_path}")
        plt.close()
    
    def plot_roc_curve(self, fpr, tpr, auc_score, save_path=None):
        """
        绘制ROC曲线
        
        Args:
            fpr: 假阳性率
            tpr: 真阳性率
            auc_score: AUC分数
            save_path: 保存路径
        """
        plt.figure(figsize=(10, 8))
        plt.plot(fpr, tpr, color='darkorange', lw=2, 
                label=f'ROC曲线 (AUC = {auc_score:.4f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('假阳性率')
        plt.ylabel('真阳性率')
        plt.title('ROC曲线分析')
        plt.legend(loc="lower right")
        
        # 保存图表
        save_path = save_path or (self.save_dir / 'roc_curve.png' if self.save_dir else None)
        if save_path:
            plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
            self.logger.info(f"ROC曲线已保存至: {save_path}")
        plt.close()
    
    def plot_pr_curve(self, precision, recall, ap_score, save_path=None):
        """
        绘制Precision-Recall曲线
        
        Args:
            precision: 精确率
            recall: 召回率
            ap_score: 平均精度分数
            save_path: 保存路径
        """
        plt.figure(figsize=(10, 8))
        plt.plot(recall, precision, color='blue', lw=2, 
                label=f'PR曲线 (AP = {ap_score:.4f})')
        plt.xlabel('召回率')
        plt.ylabel('精确率')
        plt.ylim([0.0, 1.05])
        plt.xlim([0.0, 1.0])
        plt.title('Precision-Recall曲线')
        plt.legend(loc="lower left")
        
        # 保存图表
        save_path = save_path or (self.save_dir / 'pr_curve.png' if self.save_dir else None)
        if save_path:
            plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
            self.logger.info(f"PR曲线已保存至: {save_path}")
        plt.close()
    
    def plot_confusion_matrix(self, cm, classes=['真实', '伪造'], save_path=None, normalize=False):
        """
        绘制混淆矩阵
        
        Args:
            cm: 混淆矩阵
            classes: 类别名称
            save_path: 保存路径
            normalize: 是否归一化
        """
        if normalize:
            cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
            fmt = '.2f'
        else:
            fmt = 'd'
            
        plt.figure(figsize=(10, 8))
        
        # 使用seaborn改善可视化效果
        sns.heatmap(cm, annot=True, fmt=fmt, cmap="Blues", 
                   xticklabels=classes, yticklabels=classes)
        
        plt.ylabel('真实标签')
        plt.xlabel('预测标签')
        plt.title('混淆矩阵')
        
        # 保存图表
        save_path = save_path or (self.save_dir / 'confusion_matrix.png' if self.save_dir else None)
        if save_path:
            plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
            self.logger.info(f"混淆矩阵已保存至: {save_path}")
        plt.close()
    
    def visualize_predictions(self, img_paths, mask_preds, true_labels, pred_labels, 
                              probabilities, save_path=None, num_samples=None):
        """
        可视化预测结果
        
        Args:
            img_paths: 图像路径列表
            mask_preds: 预测掩码列表
            true_labels: 真实标签列表
            pred_labels: 预测标签列表
            probabilities: 预测概率列表
            save_path: 保存路径
            num_samples: 可视化的样本数量，如果为None则全部可视化
        """
        if num_samples is not None:
            img_paths = img_paths[:num_samples]
            mask_preds = mask_preds[:num_samples]
            true_labels = true_labels[:num_samples]
            pred_labels = pred_labels[:num_samples]
            probabilities = probabilities[:num_samples]
            
        num_samples = len(img_paths)
        fig, axes = plt.subplots(num_samples, 3, figsize=(15, num_samples * 4))
        
        # 处理单样本情况
        if num_samples == 1:
            axes = np.expand_dims(axes, axis=0)
        
        for i in range(num_samples):
            img_path = img_paths[i]
            mask_pred = mask_preds[i][0] if len(mask_preds[i].shape) > 2 else mask_preds[i]  # 取第一个通道
            true_label = true_labels[i]
            pred_label = pred_labels[i]
            prob = probabilities[i]
            
            # 加载原始图像
            try:
                img = Image.open(img_path).convert('RGB')
                img = img.resize((256, 256))
                
                # 显示原始图像
                axes[i, 0].imshow(img)
                axes[i, 0].set_title(f"真实标签: {'伪造' if true_label else '真实'}")
                axes[i, 0].axis('off')
                
                # 显示预测掩码
                axes[i, 1].imshow(mask_pred, cmap='jet')
                axes[i, 1].set_title("预测掩码")
                axes[i, 1].axis('off')
                
                # 显示带掩码的原始图像
                img_np = np.array(img)
                mask_np = cv2.resize(mask_pred, (256, 256))
                mask_np = np.expand_dims(mask_np, axis=2)
                mask_np = np.repeat(mask_np, 3, axis=2)
                mask_np = (mask_np * [0, 0, 255]).astype(np.uint8)
                
                # 创建掩码叠加图像
                alpha = 0.5
                overlay = cv2.addWeighted(img_np, 1, mask_np, alpha, 0)
                
                axes[i, 2].imshow(overlay)
                axes[i, 2].set_title(f"预测: {'伪造' if pred_label else '真实'} ({prob:.4f})")
                axes[i, 2].axis('off')
            except Exception as e:
                self.logger.warning(f"可视化样本 {img_path} 出错: {e}")
                # 显示空白
                for j in range(3):
                    axes[i, j].imshow(np.zeros((256, 256, 3), dtype=np.uint8))
                    axes[i, j].set_title("加载失败")
                    axes[i, j].axis('off')
        
        plt.tight_layout()
        
        # 保存图表
        save_path = save_path or (self.save_dir / 'prediction_samples.png' if self.save_dir else None)
        if save_path:
            plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
            self.logger.info(f"预测样本可视化已保存至: {save_path}")
        plt.close()
    
    def plot_forgery_type_performance(self, type_analysis, save_path=None):
        """
        绘制不同伪造类型的性能对比
        
        Args:
            type_analysis: 按伪造类型划分的性能分析
            save_path: 保存路径
        """
        # 准备数据
        types = list(type_analysis.keys())
        accuracies = [data['accuracy'] for data in type_analysis.values()]
        aucs = [data.get('auc', 0) for data in type_analysis.values()]
        counts = [data['count'] for data in type_analysis.values()]
        
        # 创建数据框
        df = pd.DataFrame({
            '伪造类型': types,
            '样本数量': counts,
            '准确率': accuracies,
            'AUC分数': aucs
        })
        
        # 创建图形
        plt.figure(figsize=(12, 8))
        
        # 绘制准确率条形图
        plt.subplot(2, 1, 1)
        ax = sns.barplot(x='伪造类型', y='准确率', data=df)
        plt.title("不同伪造类型的准确率")
        plt.xticks(rotation=45)
        
        # 添加样本数量标签
        for i, p in enumerate(ax.patches):
            ax.annotate(f"n={counts[i]}", 
                       (p.get_x() + p.get_width() / 2., p.get_height()), 
                       ha='center', va='bottom', 
                       fontsize=9)
        
        # 绘制AUC分数条形图
        plt.subplot(2, 1, 2)
        ax = sns.barplot(x='伪造类型', y='AUC分数', data=df)
        plt.title("不同伪造类型的AUC分数")
        plt.xticks(rotation=45)
        
        # 添加样本数量标签
        for i, p in enumerate(ax.patches):
            ax.annotate(f"n={counts[i]}", 
                       (p.get_x() + p.get_width() / 2., p.get_height()), 
                       ha='center', va='bottom', 
                       fontsize=9)
        
        plt.tight_layout()
        
        # 保存图表
        save_path = save_path or (self.save_dir / 'forgery_type_performance.png' if self.save_dir else None)
        if save_path:
            plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
            self.logger.info(f"伪造类型性能对比已保存至: {save_path}")
        plt.close()
    
    def visualize_masks(self, images, true_masks, pred_masks, save_path=None, num_samples=8):
        """
        可视化掩码预测结果
        
        Args:
            images: 原始图像列表或张量
            true_masks: 真实掩码列表或张量  
            pred_masks: 预测掩码列表或张量
            save_path: 保存路径
            num_samples: 可视化的样本数量
        """
        # 限制样本数
        n = min(len(images), num_samples)
        
        # 转换为numpy数组
        if torch.is_tensor(images):
            images = images.cpu().numpy()
        if torch.is_tensor(true_masks):
            true_masks = true_masks.cpu().numpy()
        if torch.is_tensor(pred_masks):
            pred_masks = pred_masks.cpu().numpy()
            
        # 创建图形
        fig, axes = plt.subplots(n, 3, figsize=(15, n * 4))
        
        for i in range(n):
            # 处理图像
            img = images[i].transpose(1, 2, 0) if images[i].shape[0] == 3 else images[i]
            img = (img - img.min()) / (img.max() - img.min())  # 归一化
            
            # 处理掩码
            true_mask = true_masks[i].squeeze()
            pred_mask = pred_masks[i].squeeze()
            
            # 显示原始图像
            axes[i, 0].imshow(img)
            axes[i, 0].set_title("原始图像")
            axes[i, 0].axis('off')
            
            # 显示真实掩码
            axes[i, 1].imshow(true_mask, cmap='jet', vmin=0, vmax=1)
            axes[i, 1].set_title("真实掩码")
            axes[i, 1].axis('off')
            
            # 显示预测掩码
            axes[i, 2].imshow(pred_mask, cmap='jet', vmin=0, vmax=1)
            axes[i, 2].set_title("预测掩码")
            axes[i, 2].axis('off')
        
        plt.tight_layout()
        
        # 保存图表
        save_path = save_path or (self.save_dir / 'mask_comparison.png' if self.save_dir else None)
        if save_path:
            plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
            self.logger.info(f"掩码比较已保存至: {save_path}")
        plt.close()

    def visualize_feature_maps(self, feature_maps, layer_names=None, save_path=None):
        """
        可视化特征图
        
        Args:
            feature_maps: 特征图列表或张量（N, C, H, W）
            layer_names: 层名称列表
            save_path: 保存路径
        """
        # 确定特征图数量和通道数
        n_layers = len(feature_maps)
        
        if layer_names is None:
            layer_names = [f"Layer {i+1}" for i in range(n_layers)]
            
        # 对每一层可视化
        for layer_idx, (feat_map, name) in enumerate(zip(feature_maps, layer_names)):
            # 转换为numpy
            if torch.is_tensor(feat_map):
                feat_map = feat_map.detach().cpu().numpy()
                
            # 选择要可视化的通道
            n_channels = min(16, feat_map.shape[1])  # 最多显示16个通道
            
            # 创建图形
            rows = int(np.sqrt(n_channels))
            cols = int(np.ceil(n_channels / rows))
            fig, axes = plt.subplots(rows, cols, figsize=(3*cols, 3*rows))
            
            # 压平axes数组以便索引
            if rows*cols > 1:
                axes = axes.flatten()
            else:
                axes = [axes]
                
            # 绘制每个通道
            for i in range(n_channels):
                channel_data = feat_map[0, i]  # 取第一个样本，第i个通道
                
                # 归一化
                if channel_data.max() > channel_data.min():
                    channel_data = (channel_data - channel_data.min()) / (channel_data.max() - channel_data.min())
                
                axes[i].imshow(channel_data, cmap='viridis')
                axes[i].set_title(f"Channel {i+1}")
                axes[i].axis('off')
                
            # 隐藏多余的子图
            for i in range(n_channels, len(axes)):
                axes[i].axis('off')
                
            plt.suptitle(f"Feature Maps - {name}")
            plt.tight_layout()
            
            # 保存图表
            if save_path:
                if self.save_dir:
                    # 创建特征图目录
                    feat_dir = self.save_dir / 'feature_maps'
                    feat_dir.mkdir(exist_ok=True, parents=True)
                    layer_path = feat_dir / f"layer_{layer_idx+1}_{name.replace(' ', '_')}.png"
                    plt.savefig(layer_path, dpi=self.dpi, bbox_inches='tight')
                else:
                    base_path = Path(save_path)
                    layer_path = base_path.with_stem(f"{base_path.stem}_layer_{layer_idx+1}")
                    plt.savefig(layer_path, dpi=self.dpi, bbox_inches='tight')
            plt.close()
            
    def create_evaluation_report(self, metrics, save_dir=None):
        """
        创建评估报告，包括所有图形和指标
        
        Args:
            metrics: 评估指标字典
            save_dir: 保存目录
        """
        save_dir = Path(save_dir) if save_dir else self.save_dir
        if not save_dir:
            self.logger.warning("未指定保存目录，无法创建评估报告")
            return
            
        save_dir.mkdir(exist_ok=True, parents=True)
        
        # 绘制ROC曲线
        if 'roc_data' in metrics:
            self.plot_roc_curve(
                metrics['roc_data']['fpr'],
                metrics['roc_data']['tpr'],
                metrics.get('auc', 0),
                save_path=save_dir / 'roc_curve.png'
            )
        
        # 绘制PR曲线
        if 'pr_data' in metrics:
            self.plot_pr_curve(
                metrics['pr_data']['precision'],
                metrics['pr_data']['recall'],
                metrics.get('ap', 0),
                save_path=save_dir / 'pr_curve.png'
            )
        
        # 绘制混淆矩阵
        if 'confusion_matrix' in metrics:
            self.plot_confusion_matrix(
                metrics['confusion_matrix'],
                save_path=save_dir / 'confusion_matrix.png'
            )
        
        # 绘制伪造类型性能
        if 'type_analysis' in metrics:
            self.plot_forgery_type_performance(
                metrics['type_analysis'],
                save_path=save_dir / 'forgery_type_performance.png'
            )
            
        # 保存指标摘要
        metrics_summary = {k: v for k, v in metrics.items() 
                          if not isinstance(v, dict) and not isinstance(v, np.ndarray)}
        
        # 添加掩码评估指标
        if 'mask_metrics' in metrics:
            for k, v in metrics['mask_metrics'].items():
                metrics_summary[f'mask_{k}'] = v
                
        # 保存为CSV
        pd.DataFrame([metrics_summary]).to_csv(save_dir / 'metrics_summary.csv', index=False)
        
        # 创建一个HTML报告
        self._create_html_report(metrics, save_dir)
        
    def _create_html_report(self, metrics, save_dir):
        """创建HTML格式的报告"""
        html_content = [
            "<!DOCTYPE html>",
            "<html>",
            "<head>",
            "    <title>伪造检测评估报告</title>",
            "    <style>",
            "        body { font-family: Arial, sans-serif; margin: 20px; }",
            "        h1, h2 { color: #2c3e50; }",
            "        .metric { margin-bottom: 5px; }",
            "        .section { margin-bottom: 30px; }",
            "        img { max-width: 100%; border: 1px solid #ddd; }",
            "        table { border-collapse: collapse; width: 100%; }",
            "        th, td { border: 1px solid #ddd; padding: 8px; text-align: left; }",
            "        th { background-color: #f2f2f2; }",
            "    </style>",
            "</head>",
            "<body>",
            "    <h1>伪造检测评估报告</h1>"
        ]
        
        # 基本指标部分
        html_content.extend([
            "    <div class='section'>",
            "        <h2>基本性能指标</h2>"
        ])
        
        # 添加主要指标
        main_metrics = ['accuracy', 'auc', 'ap', 'specificity', 'sensitivity', 'precision']
        for metric in main_metrics:
            if metric in metrics:
                html_content.append(f"        <div class='metric'><b>{metric.title()}:</b> {metrics[metric]:.4f}</div>")
        
        html_content.append("    </div>")
        
        # 掩码评估部分
        if 'mask_metrics' in metrics:
            html_content.extend([
                "    <div class='section'>",
                "        <h2>掩码评估指标</h2>"
            ])
            
            for k, v in metrics['mask_metrics'].items():
                html_content.append(f"        <div class='metric'><b>{k.replace('_', ' ').title()}:</b> {v:.4f}</div>")
                
            html_content.append("    </div>")
        
        # 图像部分
        html_content.extend([
            "    <div class='section'>",
            "        <h2>性能图表</h2>",
            "        <div style='display: flex; flex-wrap: wrap; gap: 20px;'>"
        ])
        
        # 添加所有图像
        images = ['roc_curve.png', 'pr_curve.png', 'confusion_matrix.png', 'forgery_type_performance.png']
        for img in images:
            if (save_dir / img).exists():
                img_path = img  # 相对路径
                html_content.append(f"            <div style='flex: 1; min-width: 300px;'><img src='{img_path}' alt='{img}'></div>")
                
        html_content.extend([
            "        </div>",
            "    </div>",
            "</body>",
            "</html>"
        ])
        
        # 保存HTML文件
        with open(save_dir / 'evaluation_report.html', 'w') as f:
            f.write('\n'.join(html_content))
            
        self.logger.info(f"评估报告已保存至: {save_dir / 'evaluation_report.html'}")

    def visualize_attention_maps(self, model, images, device='cuda', save_dir=None, layer_names=None):
        """
        可视化模型的注意力图
        
        Args:
            model: 模型实例
            images: 输入图像列表或张量
            device: 设备
            save_dir: 保存目录
            layer_names: 层名称列表
        """
        # 确保save_dir存在
        if save_dir:
            save_dir = Path(save_dir) / 'attention_maps'
            save_dir.mkdir(exist_ok=True, parents=True)
            
        # 转换为张量
        if not torch.is_tensor(images):
            transform = transforms.ToTensor()
            images_tensor = torch.stack([transform(img) for img in images])
        else:
            images_tensor = images
            
        # 移动到设备
        images_tensor = images_tensor.to(device)
        model = model.to(device)
        model.eval()
        
        with torch.no_grad():
            # 尝试获取注意力图
            try:
                # 检查是否支持get_attention_maps方法
                if hasattr(model, 'get_attention_maps'):
                    attention_maps = model.get_attention_maps(images_tensor)
                # 否则尝试从ForensicAttentionFusion模块获取
                elif (hasattr(model, 'feature_fusion') and 
                      hasattr(model.feature_fusion, 'get_attention_maps')):
                    # 先做一次前向传播
                    _ = model(images_tensor)
                    # 然后获取注意力图
                    attention_maps = model.feature_fusion.get_attention_maps()
                # 直接执行前向传播，看是否生成attention_map属性
                else:
                    _ = model(images_tensor)
                    # 尝试从多个可能的位置获取注意力图
                    attention_maps = None
                    for attr_name in ['last_attention_map', 'attention_map', 'attn_weights']:
                        for module in [model, 
                                      getattr(model, 'model', None), 
                                      getattr(model, 'feature_fusion', None),
                                      getattr(model, 'boundary_module', None)]:
                            if module is not None and hasattr(module, attr_name):
                                attention_maps = getattr(module, attr_name)
                                break
                        if attention_maps is not None:
                            break
                            
                if attention_maps is None:
                    self.logger.warning("无法获取注意力图，模型可能不支持")
                    return None
                    
            except Exception as e:
                self.logger.error(f"获取注意力图失败: {e}")
                return None
                
        # 可视化每张图的注意力图
        results = []
        for i, (img, attn_map) in enumerate(zip(images_tensor, attention_maps)):
            # 注意力图可能有多个头或层
            if attn_map.dim() > 2:
                # 如果是多头注意力，取平均
                if attn_map.dim() == 3:  # [num_heads, h, w]
                    attn_map = attn_map.mean(0)  # [h, w]
                elif attn_map.dim() == 4:  # [batch, num_heads, h, w]
                    attn_map = attn_map.mean(1)  # [batch, h, w]
                    
            # 生成可视化图像
            img_np = img.permute(1, 2, 0).cpu().numpy()
            img_np = (img_np - img_np.min()) / (img_np.max() - img_np.min())
            
            attn_np = attn_map.cpu().numpy()
            attn_np = (attn_np - attn_np.min()) / (attn_np.max() - attn_np.min() + 1e-8)
            
            # 生成热力图叠加
            viz_img = self.visualize_attention(img_np, attn_np, 
                                               save_path=f"{save_dir}/attention_map_{i}.png" if save_dir else None)
            results.append(viz_img)
            
        return results
        
    def visualize_dct_features(self, model, images, device='cuda', save_dir=None):
        """
        可视化DCT特征
        
        Args:
            model: 模型实例
            images: 输入图像列表或张量
            device: 设备
            save_dir: 保存目录
        """
        # 确保save_dir存在
        if save_dir:
            save_dir = Path(save_dir) / 'dct_features'
            save_dir.mkdir(exist_ok=True, parents=True)
            
        # 转换为张量
        if not torch.is_tensor(images):
            transform = transforms.ToTensor()
            images_tensor = torch.stack([transform(img) for img in images])
        else:
            images_tensor = images
            
        # 移动到设备
        images_tensor = images_tensor.to(device)
        model = model.to(device)
        model.eval()
        
        with torch.no_grad():
            # 尝试获取DCT特征
            try:
                # 直接使用模型提取DCT特征
                if hasattr(model, 'FAD_head'):
                    dct_features = model.FAD_head(images_tensor)
                elif hasattr(model, 'dct_extractor'):
                    dct_features = model.dct_extractor(images_tensor)
                else:
                    # 导入DCT变换模块，创建临时提取器
                    from network.dct_transform import MultiScaleFrequencyExtractor
                    dct_extractor = MultiScaleFrequencyExtractor(in_channels=3, out_channels=12).to(device)
                    dct_features = dct_extractor(images_tensor)
                    
            except Exception as e:
                self.logger.error(f"获取DCT特征失败: {e}")
                return None
                
        # 可视化每张图的DCT特征
        results = []
        for i, (img, dct_feat) in enumerate(zip(images_tensor, dct_features)):
            # 准备可视化
            img_np = img.permute(1, 2, 0).cpu().numpy()
            img_np = (img_np - img_np.min()) / (img_np.max() - img_np.min())
            
            # DCT特征可能有多个通道
            n_channels = min(12, dct_feat.size(0))
            
            # 创建可视化图表
            plt.figure(figsize=(15, 10))
            
            # 显示原始图像
            plt.subplot(4, 4, 1)
            plt.imshow(img_np)
            plt.title("原始图像")
            plt.axis('off')
            
            # 显示每个DCT通道
            for ch in range(n_channels):
                plt.subplot(4, 4, ch + 2)
                dct_ch = dct_feat[ch].cpu().numpy()
                dct_ch = (dct_ch - dct_ch.min()) / (dct_ch.max() - dct_ch.min() + 1e-8)
                plt.imshow(dct_ch, cmap='viridis')
                plt.title(f"DCT通道 {ch+1}")
                plt.axis('off')
            
            # 保存或显示
            if save_dir:
                plt.savefig(f"{save_dir}/dct_features_{i}.png", dpi=self.dpi, bbox_inches='tight')
                plt.close()
                results.append(f"{save_dir}/dct_features_{i}.png")
            else:
                plt.tight_layout()
                plt.show()
                results.append(plt.gcf())
                plt.close()
                
        return results
        
    def visualize_boundary_detection(self, images, pred_masks, true_masks=None, save_dir=None):
        """
        可视化边界检测结果
        
        Args:
            images: 输入图像列表或张量
            pred_masks: 预测掩码列表或张量
            true_masks: 真实掩码列表或张量（可选）
            save_dir: 保存目录
        """
        # 确保save_dir存在
        if save_dir:
            save_dir = Path(save_dir) / 'boundary_detection'
            save_dir.mkdir(exist_ok=True, parents=True)
            
        # 转换为numpy数组
        if torch.is_tensor(images):
            images = images.cpu().numpy()
        if torch.is_tensor(pred_masks):
            pred_masks = pred_masks.cpu().numpy()
        if true_masks is not None and torch.is_tensor(true_masks):
            true_masks = true_masks.cpu().numpy()
            
        # 限制样本数量
        n_samples = min(len(images), 8)
        
        results = []
        for i in range(n_samples):
            # 处理图像格式
            img = images[i]
            if img.shape[0] == 3:  # 通道在前，转换到后面
                img = img.transpose(1, 2, 0)
            img = (img - img.min()) / (img.max() - img.min())
            
            # 处理掩码
            pred_mask = pred_masks[i].squeeze()
            true_mask = true_masks[i].squeeze() if true_masks is not None else None
            
            # 二值化掩码并提取边界
            pred_mask_bin = (pred_mask > 0.5).astype(np.uint8) * 255
            pred_boundary = cv2.Canny(pred_mask_bin, 100, 200)
            
            if true_mask is not None:
                true_mask_bin = (true_mask > 0.5).astype(np.uint8) * 255
                true_boundary = cv2.Canny(true_mask_bin, 100, 200)
                
                # 计算IoU
                intersection = np.logical_and(pred_mask_bin > 0, true_mask_bin > 0).sum()
                union = np.logical_or(pred_mask_bin > 0, true_mask_bin > 0).sum()
                iou = intersection / union if union > 0 else 0
                
                # 计算边界IoU
                pred_boundary_points = set(zip(*np.where(pred_boundary > 0)))
                true_boundary_points = set(zip(*np.where(true_boundary > 0)))
                
                boundary_intersection = len(pred_boundary_points.intersection(true_boundary_points))
                boundary_union = len(pred_boundary_points.union(true_boundary_points))
                boundary_iou = boundary_intersection / boundary_union if boundary_union > 0 else 0
            
            # 创建可视化图像
            plt.figure(figsize=(15, 10))
            
            # 子图数量取决于是否有真实掩码
            n_cols = 4 if true_mask is not None else 3
            
            # 显示原始图像
            plt.subplot(1, n_cols, 1)
            plt.imshow(img)
            plt.title("原始图像")
            plt.axis('off')
            
            # 显示预测掩码
            plt.subplot(1, n_cols, 2)
            plt.imshow(pred_mask, cmap='jet', vmin=0, vmax=1)
            plt.title("预测掩码")
            plt.axis('off')
            
            # 显示预测边界叠加在原始图像上
            plt.subplot(1, n_cols, 3)
            img_with_boundary = img.copy()
            img_with_boundary[pred_boundary > 0] = [1, 0, 0]  # 红色边界
            plt.imshow(img_with_boundary)
            plt.title("预测边界")
            plt.axis('off')
            
            # 如果有真实掩码，显示真实边界
            if true_mask is not None:
                plt.subplot(1, n_cols, 4)
                img_with_true_boundary = img.copy()
                img_with_true_boundary[true_boundary > 0] = [0, 1, 0]  # 绿色边界
                plt.imshow(img_with_true_boundary)
                plt.title(f"真实边界\nIoU: {iou:.3f}\n边界IoU: {boundary_iou:.3f}")
                plt.axis('off')
                
            # 保存或显示
            if save_dir:
                plt.savefig(f"{save_dir}/boundary_{i}.png", dpi=self.dpi, bbox_inches='tight')
                plt.close()
                results.append(f"{save_dir}/boundary_{i}.png")
            else:
                plt.tight_layout()
                plt.show()
                results.append(plt.gcf())
                plt.close()
                
        return results
        
    def visualize_multi_model_comparison(self, images, predictions_dict, true_labels=None, save_dir=None):
        """
        可视化多个模型预测结果的比较
        
        Args:
            images: 输入图像列表或张量
            predictions_dict: 模型名称到预测结果的字典，每个元素包含掩码、概率等
            true_labels: 真实标签列表（可选）
            save_dir: 保存目录
        """
        # 确保save_dir存在
        if save_dir:
            save_dir = Path(save_dir) / 'model_comparison'
            save_dir.mkdir(exist_ok=True, parents=True)
            
        # 转换为numpy数组
        if torch.is_tensor(images):
            images = images.cpu().numpy()
            
        # 模型数量
        n_models = len(predictions_dict)
        model_names = list(predictions_dict.keys())
        
        # 限制样本数量
        n_samples = min(len(images), 6)
        
        results = []
        for i in range(n_samples):
            # 处理图像格式
            img = images[i]
            if img.shape[0] == 3:  # 通道在前，转换到后面
                img = img.transpose(1, 2, 0)
            img = (img - img.min()) / (img.max() - img.min())
            
            # 创建可视化图表
            plt.figure(figsize=(4 + 4 * n_models, 8))
            
            # 显示原始图像
            plt.subplot(2, n_models + 1, 1)
            plt.imshow(img)
            title = "原始图像"
            if true_labels is not None:
                title += f"\n真实标签: {'伪造' if true_labels[i] == 1 else '真实'}"
            plt.title(title)
            plt.axis('off')
            
            # 为每个模型显示预测结果
            for j, model_name in enumerate(model_names):
                predictions = predictions_dict[model_name]
                
                # 获取该模型对当前样本的预测
                pred_mask = predictions['masks'][i] if 'masks' in predictions else None
                pred_prob = predictions['probs'][i] if 'probs' in predictions else None
                pred_label = predictions['preds'][i] if 'preds' in predictions else None
                
                # 显示预测掩码
                if pred_mask is not None:
                    plt.subplot(2, n_models + 1, j + 2)
                    if isinstance(pred_mask, np.ndarray):
                        plt.imshow(pred_mask.squeeze(), cmap='jet', vmin=0, vmax=1)
                    else:  # 张量
                        plt.imshow(pred_mask.squeeze().cpu().numpy(), cmap='jet', vmin=0, vmax=1)
                    plt.title(f"{model_name}\n掩码")
                    plt.axis('off')
                
                # 显示掩码叠加图
                if pred_mask is not None:
                    plt.subplot(2, n_models + 1, n_models + j + 3)
                    # 创建掩码叠加图像
                    if isinstance(pred_mask, np.ndarray):
                        mask_np = pred_mask.squeeze()
                    else:  # 张量
                        mask_np = pred_mask.squeeze().cpu().numpy()
                        
                    mask_rgb = cv2.applyColorMap(np.uint8(mask_np * 255), cv2.COLORMAP_JET)
                    mask_rgb = cv2.cvtColor(mask_rgb, cv2.COLOR_BGR2RGB) / 255.0
                    
                    overlay = img * 0.7 + mask_rgb * 0.3
                    
                    plt.imshow(overlay)
                    title = f"预测: {'伪造' if pred_label == 1 else '真实'}"
                    if pred_prob is not None:
                        title += f"\n概率: {pred_prob:.3f}"
                    plt.title(title)
                    plt.axis('off')
            
            # 保存或显示
            if save_dir:
                plt.savefig(f"{save_dir}/comparison_{i}.png", dpi=self.dpi, bbox_inches='tight')
                plt.close()
                results.append(f"{save_dir}/comparison_{i}.png")
            else:
                plt.tight_layout()
                plt.show()
                results.append(plt.gcf())
                plt.close()
                
        return results
        
    def update_evaluation_report(self, metrics, boundary_metrics=None, freq_analysis=None, save_dir=None):
        """
        更新评估报告，包括边界检测和频域分析
        
        Args:
            metrics: 基本评估指标
            boundary_metrics: 边界检测评估指标
            freq_analysis: 频域分析结果
            save_dir: 保存目录
        """
        # 已有代码的增强版
        save_dir = Path(save_dir) if save_dir else self.save_dir
        if not save_dir:
            self.logger.warning("未指定保存目录，无法创建评估报告")
            return
            
        save_dir.mkdir(exist_ok=True, parents=True)
        
        # 合并指标
        complete_metrics = dict(metrics)
        if boundary_metrics:
            complete_metrics['boundary_metrics'] = boundary_metrics
        if freq_analysis:
            complete_metrics['frequency_analysis'] = freq_analysis
        
        # 调用原有方法创建基本报告
        self.create_evaluation_report(complete_metrics, save_dir)
        
        # 添加边界检测评估部分到HTML报告
        if boundary_metrics:
            self._append_to_html_report(save_dir / 'evaluation_report.html', self._create_boundary_html(boundary_metrics))
            
        # 添加频域分析部分到HTML报告
        if freq_analysis:
            self._append_to_html_report(save_dir / 'evaluation_report.html', self._create_frequency_html(freq_analysis))
            
    def _create_boundary_html(self, boundary_metrics):
        """创建边界检测评估的HTML内容"""
        html_content = [
            "    <div class='section'>",
            "        <h2>边界检测评估</h2>",
            "        <div class='metrics-container'>"
        ]
        
        for k, v in boundary_metrics.items():
            html_content.append(f"            <div class='metric'><b>{k.replace('_', ' ').title()}:</b> {v:.4f}</div>")
            
        html_content.extend([
            "        </div>",
            "        <div class='chart-container'>",
            "            <img src='boundary_metrics_chart.png' alt='边界检测指标图表'>",
            "        </div>",
            "    </div>"
        ])
        
        return html_content
        
    def _create_frequency_html(self, freq_analysis):
        """创建频域分析的HTML内容"""
        html_content = [
            "    <div class='section'>",
            "        <h2>频域分析</h2>",
            "        <div class='metrics-container'>"
        ]
        
        if isinstance(freq_analysis, dict) and 'stats' in freq_analysis:
            for k, v in freq_analysis['stats'].items():
                html_content.append(f"            <div class='metric'><b>{k.title()}:</b> {v:.4f}</div>")
                
        html_content.extend([
            "        </div>",
            "        <div class='chart-container'>",
            "            <img src='frequency_analysis/frequency_response.png' alt='频率响应'>",
            "        </div>",
            "    </div>"
        ])
        
        return html_content
        
    def _append_to_html_report(self, html_path, content_lines):
        """向HTML报告添加内容"""
        if not html_path.exists():
            self.logger.warning(f"HTML报告不存在: {html_path}")
            return
            
        with open(html_path, 'r') as f:
            content = f.readlines()
            
        # 在</body>标签前插入内容
        insert_pos = -1
        for i, line in enumerate(content):
            if '</body>' in line:
                insert_pos = i
                break
                
        if insert_pos >= 0:
            new_content = content[:insert_pos] + [line + '\n' for line in content_lines] + content[insert_pos:]
            
            with open(html_path, 'w') as f:
                f.writelines(new_content)
        else:
            self.logger.warning("无法找到HTML报告中的</body>标签位置")

# 便于测试的主函数
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='测试可视化模块')
    parser.add_argument('--save_dir', type=str, default='./visualization_test', help='保存目录')
    args = parser.parse_args()
    
    # 设置日志
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    logger = logging.getLogger('test_visualizer')
    
    # 创建可视化器
    visualizer = ForensicVisualizer(save_dir=args.save_dir, logger=logger)
    
    # 模拟数据
    history = {
        'train_loss': np.random.rand(50) * 0.5 + 0.5,
        'val_loss': np.random.rand(50) * 0.3 + 0.3,
        'train_cls_loss': np.random.rand(50) * 0.3 + 0.3,
        'train_mask_loss': np.random.rand(50) * 0.2 + 0.2,
        'val_acc': np.random.rand(50) * 0.3 + 0.7,
        'lr': [0.001 * (0.9 ** i) for i in range(50)]
    }
    
    # 绘制训练曲线
    visualizer.plot_training_curves(history)
    
    # 模拟ROC数据
    fpr = np.linspace(0, 1, 100)
    tpr = np.power(fpr, 0.3)  # 模拟曲线
    auc_score = 0.85
    visualizer.plot_roc_curve(fpr, tpr, auc_score)
    
    # 模拟PR曲线数据
    recall = np.linspace(0, 1, 100)
    precision = 1 - np.power(recall, 2)  # 模拟曲线
    ap_score = 0.78
    visualizer.plot_pr_curve(precision, recall, ap_score)
    
    # 模拟混淆矩阵
    cm = np.array([[85, 15], [10, 90]])
    visualizer.plot_confusion_matrix(cm)
    
    logger.info("可视化模块测试完成。所有图表已保存至指定目录。")
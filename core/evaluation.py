import torch
import numpy as np
from sklearn.metrics import (
    roc_auc_score, precision_recall_curve, average_precision_score,
    roc_curve, confusion_matrix, classification_report, accuracy_score
)
from tqdm import tqdm
import os
from pathlib import Path
import logging
import pandas as pd
import time
from collections import defaultdict
import warnings
from typing import List, Dict, Any, Tuple, Optional, Union
import json
import cv2
import matplotlib.pyplot as plt
import seaborn as sns
from einops import rearrange
import torchvision.transforms as transforms  # 添加这一行导入
class ModelEvaluator:
    """模型评估工具类，统一评估逻辑"""
    def __init__(self, save_dir=None, logger=None):
        """初始化评估器"""
        self.save_dir = Path(save_dir) if save_dir else None
        if self.save_dir:
            self.save_dir.mkdir(exist_ok=True, parents=True)
        self.logger = logger or logging.getLogger(__name__)
    
    def evaluate_model(self, model, data_loader, device, threshold=0.5, return_predictions=False, evaluate_freq=False):
        """通用模型评估方法"""
        model.eval()
        
        # 初始化结果收集器
        all_preds, all_probs, all_labels = [], [], []
        all_paths, all_types = [], []
        all_masks_pred, all_masks_true = [], []
        all_images = []
        all_attention_maps = []
        
        # 计时器
        start_time = time.time()
        
        with torch.no_grad():
            for batch in tqdm(data_loader, desc="评估中", ncols=100):
                # 处理各种批次格式
                if len(batch) == 3:
                    inputs, labels, masks = batch
                    paths, types = None, None
                elif len(batch) == 4:
                    # 检查第二个元素是否为DCT特征
                    if isinstance(batch[1], torch.Tensor) and batch[1].dim() == 4:
                        # 双输入模式: (img, dct_features, labels)
                        inputs, dct_inputs, labels = batch[:3]
                        masks = None if len(batch) <= 3 else batch[3]
                        paths, types = None, None
                    else:
                        # 标准模式: (img, mask, labels, paths)
                        inputs, masks, labels, paths = batch
                        dct_inputs = None
                        types = None
                elif len(batch) >= 5:
                    if isinstance(batch[1], torch.Tensor) and batch[1].dim() == 4:
                        # 双输入模式带掩码和路径: (img, dct_features, masks, labels, paths)
                        inputs, dct_inputs, masks, labels, paths = batch[:5]
                        types = None if len(batch) <= 5 else batch[5]
                    else:
                        # 标准模式: (img, masks, labels, paths, types)
                        inputs, masks, labels, paths, types = batch[:5]
                        dct_inputs = None
                else:
                    self.logger.error(f"不支持的批次格式: {[type(b) for b in batch]}")
                    continue
                
                # 移动到设备
                inputs = inputs.to(device)
                if masks is not None:
                    masks = masks.to(device)
                if dct_inputs is not None:
                    dct_inputs = dct_inputs.to(device)
                labels = labels.to(device)
                
                # 保存原始图像用于频域分析
                if evaluate_freq:
                    all_images.extend(inputs.cpu())
                
                # 获取预测结果
                try:
                    # 尝试使用双输入模式
                    if dct_inputs is not None:
                        outputs = model(inputs, dct_inputs)
                    else:
                        outputs = model(inputs)
                    
                    # 检查输出格式
                    if isinstance(outputs, tuple) and len(outputs) == 2:
                        mask_preds, class_outputs = outputs
                    else:
                        class_outputs = outputs
                        mask_preds = torch.zeros((inputs.size(0), 1, inputs.size(2), inputs.size(3)), device=device) if masks is not None else None
                        
                except Exception as e:
                    self.logger.error(f"模型推理错误: {e}")
                    class_outputs = torch.zeros((inputs.size(0), 2), device=device)
                    mask_preds = torch.zeros((inputs.size(0), 1, inputs.size(2), inputs.size(3)), device=device) if masks is not None else None
                
                # 获取概率和预测值
                if not isinstance(class_outputs, torch.Tensor):
                    self.logger.warning(f"非标准输出格式: {type(class_outputs)}")
                    continue
                    
                if class_outputs.dim() > 1 and class_outputs.size(1) > 1:
                    probs = torch.softmax(class_outputs, dim=1)[:, 1].cpu().numpy()
                else:
                    probs = torch.sigmoid(class_outputs).squeeze().cpu().numpy()
                    
                preds = (probs >= threshold).astype(int)
                
                # 收集结果
                all_preds.extend(preds)
                all_probs.extend(probs)
                all_labels.extend(labels.cpu().numpy())
                
                if mask_preds is not None and masks is not None:
                    all_masks_pred.extend(mask_preds.cpu().numpy())
                    all_masks_true.extend(masks.cpu().numpy())
                
                # 尝试获取注意力图
                try:
                    if hasattr(model, 'get_attention_maps'):
                        attn_maps = model.get_attention_maps(inputs, dct_inputs)
                        if attn_maps is not None:
                            all_attention_maps.extend(attn_maps.cpu().numpy())
                except:
                    pass
                    
                if paths is not None:
                    all_paths.extend(paths)
                if types is not None:
                    all_types.extend(types)
        
        # 计算总推理时间
        total_time = time.time() - start_time
        fps = len(all_labels) / total_time if total_time > 0 else 0
        
        # 计算分类评估指标
        metrics = self.calculate_metrics(all_labels, all_preds, all_probs)
        
        # 计算掩码评估指标
        if len(all_masks_pred) > 0 and len(all_masks_true) > 0:
            try:
                mask_metrics = self.evaluate_masks(all_masks_pred, all_masks_true)
                metrics['mask_metrics'] = mask_metrics
                
                # 添加边界评估
                boundary_metrics = BoundaryEvaluator.calculate_boundary_metrics(all_masks_pred, all_masks_true)
                metrics['boundary_metrics'] = boundary_metrics
            except Exception as e:
                self.logger.error(f"掩码评估错误: {e}")
        
        # 添加伪造类型分析
        if len(all_types) > 0:
            type_analysis = self.analyze_by_forgery_type(all_types, all_labels, all_preds, all_probs)
            metrics['type_analysis'] = type_analysis
        
        # 添加性能指标
        metrics['inference_time'] = total_time
        metrics['fps'] = fps
        
        # 添加注意力图统计
        if len(all_attention_maps) > 0:
            metrics['attention_stats'] = {
                'mean': float(np.mean(all_attention_maps)),
                'std': float(np.std(all_attention_maps)),
                'max': float(np.max(all_attention_maps)),
                'min': float(np.min(all_attention_maps))
            }
            
        # 频域分析
        if evaluate_freq and len(all_images) > 0:
            try:
                # 随机采样最多10张图像进行分析
                if len(all_images) > 10:
                    indices = np.random.choice(len(all_images), 10, replace=False)
                    sample_images = [all_images[i] for i in indices]
                else:
                    sample_images = all_images
                    
                freq_results = FrequencyDomainEvaluator.analyze_frequency_response(
                    model, sample_images, device, 
                    save_dir=self.save_dir / 'frequency_analysis' if self.save_dir else None
                )
                metrics['frequency_analysis'] = {
                    'performed': True,
                    'stats': {
                        'mean': float(np.mean(freq_results['avg_freq_response'])),
                        'std': float(np.std(freq_results['avg_freq_response'])),
                        'max': float(np.max(freq_results['avg_freq_response'])),
                        'min': float(np.min(freq_results['avg_freq_response']))
                    }
                }
            except Exception as e:
                self.logger.error(f"频域分析错误: {e}")
                metrics['frequency_analysis'] = {'performed': False, 'error': str(e)}
            
        # 是否返回原始预测
        if return_predictions:
            metrics['predictions'] = {
                'preds': all_preds,
                'probs': all_probs,
                'labels': all_labels,
                'masks_pred': all_masks_pred if len(all_masks_pred) > 0 else None,
                'masks_true': all_masks_true if len(all_masks_true) > 0 else None,
                'attention_maps': all_attention_maps if len(all_attention_maps) > 0 else None,
                'paths': all_paths if len(all_paths) > 0 else None,
                'types': all_types if len(all_types) > 0 else None
            }
            
        return metrics
    
    def calculate_metrics(self, labels, preds, probs):
        """计算各种分类评估指标"""
        labels = np.array(labels)
        preds = np.array(preds)
        probs = np.array(probs)
        
        # 基本指标
        accuracy = accuracy_score(labels, preds)
        
        # ROC曲线和AUC
        if len(np.unique(labels)) > 1:
            fpr, tpr, _ = roc_curve(labels, probs)
            auc_score = roc_auc_score(labels, probs)
            precision, recall, _ = precision_recall_curve(labels, probs)
            ap = average_precision_score(labels, probs)
        else:
            fpr, tpr = np.array([0, 1]), np.array([0, 1])
            precision, recall = np.array([0, 1]), np.array([0, 1])
            auc_score, ap = 0.5, 0.5
        
        # 混淆矩阵
        cm = confusion_matrix(labels, preds)
        
        # 分类报告
        report = classification_report(labels, preds, output_dict=True)
        
        metrics = {
            'accuracy': accuracy,
            'auc': auc_score,
            'ap': ap,
            'confusion_matrix': cm,
            'classification_report': report,
            'roc_data': {'fpr': fpr, 'tpr': tpr},
            'pr_data': {'precision': precision, 'recall': recall}
        }
        
        # 计算精度和召回率
        if len(cm) > 1:
            tn, fp, fn, tp = cm.ravel()
            specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
            sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
            precision_val = tp / (tp + fp) if (tp + fp) > 0 else 0
            
            metrics.update({
                'specificity': specificity,
                'sensitivity': sensitivity,
                'precision': precision_val
            })
        
        return metrics
    
    def evaluate_masks(self, pred_masks, true_masks):
        """评估掩码预测质量"""
        pred_masks = np.array(pred_masks)
        true_masks = np.array(true_masks)
        
        # 二值化掩码（阈值0.5）
        pred_masks_bin = pred_masks > 0.5
        true_masks_bin = true_masks > 0.5
        
        # 计算IoU (交并比)
        intersection = np.logical_and(pred_masks_bin, true_masks_bin).sum(axis=(1,2,3))
        union = np.logical_or(pred_masks_bin, true_masks_bin).sum(axis=(1,2,3))
        
        # 避免除零错误
        iou_scores = np.divide(intersection, union, out=np.zeros_like(intersection, dtype=float), where=union!=0)
        mean_iou = np.mean(iou_scores)
        
        # 计算Dice系数
        numerator = 2 * intersection
        denominator = pred_masks_bin.sum(axis=(1,2,3)) + true_masks_bin.sum(axis=(1,2,3))
        dice_scores = np.divide(numerator, denominator, out=np.zeros_like(numerator, dtype=float), where=denominator!=0)
        mean_dice = np.mean(dice_scores)
        
        # 计算像素准确率
        pixel_acc = np.mean((pred_masks_bin == true_masks_bin).astype(float))
        
        return {
            'mean_iou': float(mean_iou),
            'mean_dice': float(mean_dice),
            'pixel_accuracy': float(pixel_acc)
        }
    
    def analyze_by_forgery_type(self, types, labels, preds, probs):
        """按伪造类型分析性能"""
        type_data = {}
        
        # 收集每种类型的数据
        for t, l, p, prob in zip(types, labels, preds, probs):
            if t not in type_data:
                type_data[t] = {
                    'count': 0,
                    'correct': 0,
                    'probs': [],
                    'labels': [],
                    'preds': []
                }
                
            type_data[t]['count'] += 1
            type_data[t]['correct'] += (p == l)
            type_data[t]['probs'].append(prob)
            type_data[t]['labels'].append(l)
            type_data[t]['preds'].append(p)
        
        # 计算每种类型的性能指标
        analysis = {}
        for t, data in type_data.items():
            if len(data['labels']) > 0:
                labels_array = np.array(data['labels'])
                preds_array = np.array(data['preds'])
                probs_array = np.array(data['probs'])
                
                # 计算基本指标
                accuracy = data['correct'] / data['count']
                
                # 计算AUC
                auc_value = 0.5  # 默认值
                if len(np.unique(labels_array)) > 1:
                    try:
                        auc_value = roc_auc_score(labels_array, probs_array)
                    except ValueError:
                        pass
                
                analysis[t] = {
                    'count': data['count'],
                    'accuracy': float(accuracy),
                    'auc': float(auc_value)
                }
                
                # 如果样本够多，添加更详细的指标
                if data['count'] > 10:
                    try:
                        cm = confusion_matrix(labels_array, preds_array)
                        if len(cm) > 1:
                            tn, fp, fn, tp = cm.ravel()
                            specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
                            sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
                            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
                            f1 = 2 * precision * sensitivity / (precision + sensitivity) if (precision + sensitivity) > 0 else 0
                            
                            analysis[t].update({
                                'specificity': float(specificity),
                                'sensitivity': float(sensitivity),
                                'precision': float(precision),
                                'f1_score': float(f1)
                            })
                    except Exception:
                        pass
        
        return analysis
    
    def save_evaluation_results(self, metrics, save_dir=None):
        """保存评估结果为CSV和JSON"""
        save_dir = Path(save_dir or self.save_dir)
        if not save_dir:
            self.logger.warning("未指定保存目录，无法保存评估结果")
            return
            
        save_dir.mkdir(exist_ok=True, parents=True)
        
        # 1. 保存基本指标
        basic_metrics = {
            'accuracy': metrics['accuracy'],
            'auc': metrics['auc'],
            'ap': metrics['ap'],
            'inference_time': metrics.get('inference_time', 0),
            'fps': metrics.get('fps', 0)
        }
        
        # 添加混淆矩阵相关指标
        cm = metrics['confusion_matrix']
        if len(cm) > 1:
            try:
                tn, fp, fn, tp = cm.ravel()
                basic_metrics.update({
                    'true_positive': int(tp),
                    'false_positive': int(fp),
                    'false_negative': int(fn),
                    'true_negative': int(tn),
                    'specificity': float(tn / (tn + fp)) if (tn + fp) > 0 else 0,
                    'sensitivity': float(tp / (tp + fn)) if (tp + fn) > 0 else 0,
                    'precision': float(tp / (tp + fp)) if (tp + fp) > 0 else 0,
                })
            except Exception as e:
                self.logger.error(f"计算混淆矩阵指标错误: {e}")
        
        # 添加掩码指标
        if 'mask_metrics' in metrics:
            basic_metrics.update({
                'mask_iou': metrics['mask_metrics']['mean_iou'],
                'mask_dice': metrics['mask_metrics']['mean_dice'],
                'mask_pixel_accuracy': metrics['mask_metrics']['pixel_accuracy']
            })
            
        # 保存基本指标
        pd.DataFrame([basic_metrics]).to_csv(save_dir / 'basic_metrics.csv', index=False)
        
        # 2. 保存分类报告
        if 'classification_report' in metrics:
            df_report = pd.DataFrame(metrics['classification_report']).transpose()
            df_report.to_csv(save_dir / 'classification_report.csv')
        
        # 3. 保存ROC和PR曲线数据
        roc_data = pd.DataFrame({
            'fpr': metrics['roc_data']['fpr'],
            'tpr': metrics['roc_data']['tpr']
        })
        roc_data.to_csv(save_dir / 'roc_curve_data.csv', index=False)
        
        pr_data = pd.DataFrame({
            'precision': metrics['pr_data']['precision'],
            'recall': metrics['pr_data']['recall']
        })
        pr_data.to_csv(save_dir / 'pr_curve_data.csv', index=False)
        
        # 4. 保存伪造类型分析
        if 'type_analysis' in metrics:
            type_metrics = []
            for t, data in metrics['type_analysis'].items():
                data_copy = data.copy()
                data_copy['type'] = t
                type_metrics.append(data_copy)
                
            pd.DataFrame(type_metrics).to_csv(save_dir / 'forgery_type_analysis.csv', index=False)
                
        return True
    
    def evaluate_boundary_detection(self, model, data_loader, device, save_dir=None):
        """评估边界检测性能"""
        model.eval()
        
        # 确保save_dir存在
        if save_dir:
            save_dir = Path(save_dir) / 'boundary_evaluation'
            save_dir.mkdir(exist_ok=True, parents=True)
        
        all_metrics = []
        sample_idx = 0
        
        with torch.no_grad():
            for batch in tqdm(data_loader, desc="评估边界检测", ncols=100):
                # 处理不同的批次格式
                if len(batch) >= 3:
                    if isinstance(batch[1], torch.Tensor) and batch[1].dim() == 4:
                        # 双输入模式
                        inputs, dct_inputs, masks = batch[:3]
                        labels = batch[3] if len(batch) > 3 else None
                    else:
                        # 标准模式
                        inputs, masks, labels = batch[:3]
                        dct_inputs = None
                else:
                    self.logger.error(f"不支持的批次格式: {len(batch)}")
                    continue
                
                # 移动到设备
                inputs = inputs.to(device)
                if dct_inputs is not None:
                    dct_inputs = dct_inputs.to(device)
                masks = masks.to(device)
                
                # 获取预测结果
                try:
                    if dct_inputs is not None:
                        outputs = model(inputs, dct_inputs)
                    else:
                        outputs = model(inputs)
                    
                    # 检查输出格式
                    if isinstance(outputs, tuple) and len(outputs) == 2:
                        mask_preds, _ = outputs
                    else:
                        mask_preds = outputs
                        
                except Exception as e:
                    self.logger.error(f"模型推理错误: {e}")
                    continue
                
                # 批量处理
                for i in range(len(inputs)):
                    # 提取单张图像和掩码
                    img = inputs[i]
                    pred_mask = mask_preds[i]
                    true_mask = masks[i]
                    
                    # 计算边界指标
                    metrics = BoundaryEvaluator.calculate_boundary_metrics(
                        pred_mask.unsqueeze(0).cpu().numpy(), 
                        true_mask.unsqueeze(0).cpu().numpy()
                    )
                    
                    # 保存可视化结果
                    if save_dir:
                        img_path = save_dir / f'sample_{sample_idx}.png'
                        ForensicVisualizer.visualize_mask_and_boundary(
                            img, pred_mask, true_mask, save_path=img_path
                        )
                        
                    all_metrics.append(metrics)
                    sample_idx += 1
                    
                    # 限制样本数量
                    if sample_idx >= 50:
                        break
                
                # 如果已处理足够样本，退出循环
                if sample_idx >= 50:
                    break
        
        # 计算平均指标
        avg_metrics = {}
        for key in all_metrics[0].keys():
            avg_metrics[key] = float(np.mean([m[key] for m in all_metrics]))
        
        # 保存结果
        if save_dir:
            # 保存指标
            with open(save_dir / 'boundary_metrics.json', 'w') as f:
                json.dump(avg_metrics, f, indent=2)
                
            # 创建可视化图表
            plt.figure(figsize=(10, 6))
            sns.barplot(x=list(avg_metrics.keys()), y=list(avg_metrics.values()))
            plt.title('Boundary Detection Metrics')
            plt.ylabel('Score')
            plt.xticks(rotation=45)
            plt.tight_layout()
            plt.savefig(save_dir / 'boundary_metrics_chart.png')
            plt.close()
        
        return avg_metrics


def find_optimal_threshold(probs, labels, metric='f1'):
    """寻找最优决策阈值"""
    thresholds = np.linspace(0.01, 0.99, 20)  # 减少搜索点数
    best_score = 0
    best_threshold = 0.5
    
    for threshold in thresholds:
        preds = (np.array(probs) >= threshold).astype(int)
        
        # 计算混淆矩阵
        cm = confusion_matrix(labels, preds)
        if len(cm) < 2:  # 只有一个类别
            continue
            
        tn, fp, fn, tp = cm.ravel()
        
        # 计算各种指标
        if metric == 'balanced_acc':
            sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
            specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
            score = (sensitivity + specificity) / 2
        elif metric == 'f1':
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            score = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        else:  # accuracy
            score = (tp + tn) / (tp + tn + fp + fn)
        
        if score > best_score:
            best_score = score
            best_threshold = threshold
    
    return best_threshold


class ForensicMetrics:
    """伪造检测的专用评估指标计算器"""
    @staticmethod
    def calculate_full_metrics(labels, preds, probs):
        """计算全面的评估指标"""
        labels = np.array(labels)
        preds = np.array(preds)
        probs = np.array(probs)
        
        # 基本指标
        accuracy = accuracy_score(labels, preds)
        
        # 混淆矩阵
        cm = confusion_matrix(labels, preds)
        if len(cm) > 1:
            tn, fp, fn, tp = cm.ravel()
        else:
            if labels[0] == 1:  # 所有样本都是正类
                tp, fp, fn, tn = (preds == 1).sum(), 0, (preds == 0).sum(), 0
            else:  # 所有样本都是负类
                tp, fp, fn, tn = 0, (preds == 1).sum(), 0, (preds == 0).sum()
        
        # 计算各种比率
        metrics = {
            'accuracy': float(accuracy),
            'true_positives': int(tp),
            'false_positives': int(fp),
            'true_negatives': int(tn),
            'false_negatives': int(fn)
        }
        
        # 添加比率指标
        metrics['sensitivity'] = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0  # 召回率
        metrics['specificity'] = float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0  # 特异性
        metrics['precision'] = float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0  # 精确率
        
        # F1分数
        if metrics['precision'] + metrics['sensitivity'] > 0:
            metrics['f1_score'] = float(2 * metrics['precision'] * metrics['sensitivity'] / 
                                      (metrics['precision'] + metrics['sensitivity']))
        else:
            metrics['f1_score'] = 0.0
            
        # 计算平衡准确率
        metrics['balanced_accuracy'] = float((metrics['sensitivity'] + metrics['specificity']) / 2)
        
        # 尝试计算AUC
        if len(np.unique(labels)) > 1:
            try:
                metrics['auc'] = float(roc_auc_score(labels, probs))
                metrics['ap'] = float(average_precision_score(labels, probs))
            except Exception:
                metrics['auc'] = 0.5
                metrics['ap'] = 0.5
        else:
            metrics['auc'] = 0.5
            metrics['ap'] = 0.5
        
        return metrics


class EnsembleEvaluator:
    """评估模型集成的性能"""
    @staticmethod
    def evaluate_ensemble(models, data_loader, device, ensemble_type='average', weights=None, threshold=0.5):
        """评估模型集成性能"""
        # 初始化变量
        all_labels, all_masks_true = [], []
        all_paths, all_types = [], []
        model_outputs = []
        
        # 依次获取每个模型的预测
        for i, model in enumerate(models):
            model.eval()
            current_probs, current_masks = [], []
            
            # 第一个模型收集标签和路径
            if i == 0:
                with torch.no_grad():
                    for batch in data_loader:
                        if len(batch) == 3:
                            inputs, masks, labels = batch
                            paths, types = None, None
                        elif len(batch) == 4:
                            inputs, masks, labels, paths = batch
                            types = None
                        else:
                            inputs, masks, labels, paths, types = batch
                        
                        # 获取预测
                        inputs = inputs.to(device)
                        try:
                            mask_preds, outputs = model(inputs)
                        except:
                            outputs = model(inputs)
                            mask_preds = torch.zeros_like(masks)
                        
                        # 获取概率
                        probs = torch.softmax(outputs, dim=1)[:, 1].cpu().numpy()
                        
                        # 收集结果
                        current_probs.extend(probs)
                        current_masks.extend(mask_preds.cpu().numpy())
                        all_labels.extend(labels.cpu().numpy())
                        all_masks_true.extend(masks.cpu().numpy())
                        
                        if paths is not None:
                            all_paths.extend(paths)
                        if types is not None:
                            all_types.extend(types)
            else:
                with torch.no_grad():
                    for batch in data_loader:
                        if len(batch) >= 3:
                            inputs = batch[0].to(device)
                            try:
                                mask_preds, outputs = model(inputs)
                            except:
                                outputs = model(inputs)
                                mask_preds = torch.zeros_like(batch[1])
                            
                            # 获取概率
                            probs = torch.softmax(outputs, dim=1)[:, 1].cpu().numpy()
                            
                            # 收集结果
                            current_probs.extend(probs)
                            current_masks.extend(mask_preds.cpu().numpy())
            
            # 添加到模型预测列表
            model_outputs.append({
                'probs': np.array(current_probs),
                'masks': np.array(current_masks)
            })
        
        # 执行集成（平均法）
        if ensemble_type == 'average' or weights is None:
            # 简单平均
            ensemble_probs = sum(m['probs'] for m in model_outputs) / len(model_outputs)
            ensemble_masks = sum(m['masks'] for m in model_outputs) / len(model_outputs)
        else:  # weighted
            # 归一化权重
            weights = np.array(weights) / np.sum(weights)
            ensemble_probs = sum(m['probs'] * w for m, w in zip(model_outputs, weights))
            ensemble_masks = sum(m['masks'] * w for m, w in zip(model_outputs, weights))
        
        ensemble_preds = (ensemble_probs >= threshold).astype(int)
        
        # 使用ModelEvaluator计算指标
        evaluator = ModelEvaluator()
        metrics = evaluator.calculate_metrics(all_labels, ensemble_preds, ensemble_probs)
        metrics['mask_metrics'] = evaluator.evaluate_masks(ensemble_masks, all_masks_true)
        
        # 按伪造类型分析
        if all_types:
            metrics['type_analysis'] = evaluator.analyze_by_forgery_type(
                all_types, all_labels, ensemble_preds, ensemble_probs)
        
        return metrics


class ErrorAnalyzer:
    """错误分析工具：识别和分析模型的误分类案例"""
    @staticmethod
    def find_error_samples(labels, preds, probs, paths=None, types=None, masks_pred=None, masks_true=None):
        """查找误分类样本"""
        labels = np.array(labels)
        preds = np.array(preds)
        
        # 找出不一致的样本索引
        false_positives = np.where((labels == 0) & (preds == 1))[0]
        false_negatives = np.where((labels == 1) & (preds == 0))[0]
        
        # 收集误分类样本详情
        fp_samples = []
        for idx in false_positives:
            sample = {
                'index': int(idx),
                'confidence': float(probs[idx]),
            }
            
            if paths is not None:
                sample['path'] = paths[idx]
            if types is not None:
                sample['type'] = types[idx]
                
            fp_samples.append(sample)
            
        fn_samples = []
        for idx in false_negatives:
            sample = {
                'index': int(idx),
                'confidence': float(probs[idx]),
            }
            
            if paths is not None:
                sample['path'] = paths[idx]
            if types is not None:
                sample['type'] = types[idx]
                
            fn_samples.append(sample)
        
        # 按信心排序（最不确定的排在前面）
        fp_samples.sort(key=lambda x: abs(x['confidence'] - 0.5))
        fn_samples.sort(key=lambda x: abs(x['confidence'] - 0.5))
        
        return fp_samples, fn_samples


class BoundaryEvaluator:
    """评估边界检测性能的专用类"""
    @staticmethod
    def calculate_boundary_metrics(pred_masks, true_masks, threshold=0.5):
        """计算边界检测指标"""
        # 准备容器
        metrics = {}
        all_precision = []
        all_recall = []
        all_f1 = []
        all_boundary_iou = []
        
        for pred, target in zip(pred_masks, true_masks):
            # 确保输入是numpy数组
            if torch.is_tensor(pred):
                pred = pred.cpu().numpy()
            if torch.is_tensor(target):
                target = target.cpu().numpy()
                
            # 移除额外维度
            pred = pred.squeeze()
            target = target.squeeze()
            
            # 二值化
            pred_binary = (pred > threshold).astype(np.uint8) * 255
            target_binary = (target > threshold).astype(np.uint8) * 255
            
            # 使用Canny找到边界
            pred_boundary = cv2.Canny(pred_binary, 100, 200)
            target_boundary = cv2.Canny(target_binary, 100, 200)
            
            # 计算准确率和召回率
            pred_boundary_points = np.where(pred_boundary > 0)
            target_boundary_points = np.where(target_boundary > 0)
            
            # 计算真阳性、假阳性、假阴性
            if len(pred_boundary_points[0]) == 0 and len(target_boundary_points[0]) == 0:
                # 如果都没有边界，则认为完美匹配
                precision = 1.0
                recall = 1.0
                f1 = 1.0
                boundary_iou = 1.0
            elif len(pred_boundary_points[0]) == 0:
                # 如果预测没有边界但真实有
                precision = 0.0
                recall = 0.0
                f1 = 0.0
                boundary_iou = 0.0
            elif len(target_boundary_points[0]) == 0:
                # 如果真实没有边界但预测有
                precision = 0.0
                recall = 0.0
                f1 = 0.0
                boundary_iou = 0.0
            else:
                # 计算交集和并集
                pred_boundary_set = set(zip(pred_boundary_points[0], pred_boundary_points[1]))
                target_boundary_set = set(zip(target_boundary_points[0], target_boundary_points[1]))
                
                # 计算真阳性
                tp = len(pred_boundary_set.intersection(target_boundary_set))
                
                # 计算精度和召回率
                precision = tp / len(pred_boundary_set) if len(pred_boundary_set) > 0 else 0
                recall = tp / len(target_boundary_set) if len(target_boundary_set) > 0 else 0
                
                # 计算F1分数
                f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
                
                # 计算边界IoU
                boundary_iou = tp / (len(pred_boundary_set) + len(target_boundary_set) - tp) if (len(pred_boundary_set) + len(target_boundary_set) - tp) > 0 else 0
            
            # 收集结果
            all_precision.append(precision)
            all_recall.append(recall)
            all_f1.append(f1)
            all_boundary_iou.append(boundary_iou)
        
        # 计算平均值
        metrics['boundary_precision'] = float(np.mean(all_precision))
        metrics['boundary_recall'] = float(np.mean(all_recall))
        metrics['boundary_f1'] = float(np.mean(all_f1))
        metrics['boundary_iou'] = float(np.mean(all_boundary_iou))
        
        return metrics


class FrequencyDomainEvaluator:
    """频域特征评估类"""
    @staticmethod
    def analyze_frequency_response(model, images, device, save_dir=None):
        """分析模型对不同频率的响应"""
        model.eval()
        
        # 准备频率响应记录容器
        freq_responses = []
        
        with torch.no_grad():
            # 为每张图像计算频率响应
            for img in images:
                # 确保图像是张量
                if not torch.is_tensor(img):
                    img = torch.tensor(img).float().to(device)
                if img.dim() == 3:
                    img = img.unsqueeze(0)  # 添加批次维度
                
                # 移动到设备
                img = img.to(device)
                
                # 使用模型提取特征
                if hasattr(model, 'extract_features'):
                    features = model.extract_features(img)
                    # 通常extract_features返回各个阶段的特征，我们使用最后一个
                    if isinstance(features, (list, tuple)):
                        feature_map = features[-1]
                    else:
                        feature_map = features
                else:
                    # 如果模型没有特定方法，尝试使用它的前向传播直到倒数第二层
                    # 这需要具体根据模型结构调整
                    feature_map, _ = model(img)
                
                # 分析特征图的频率响应
                feature_map = feature_map.cpu().numpy()
                
                # 对通道维度取平均
                avg_feature = np.mean(feature_map, axis=1)[0]  # [H, W]
                
                # 计算2D FFT
                freq_spectrum = np.fft.fft2(avg_feature)
                freq_spectrum_shifted = np.fft.fftshift(freq_spectrum)
                magnitude_spectrum = np.log(np.abs(freq_spectrum_shifted) + 1)
                
                # 记录频率响应
                freq_responses.append(magnitude_spectrum)
        
        # 计算平均频率响应
        avg_freq_response = np.mean(freq_responses, axis=0)
        
        # 如果提供了保存目录，保存可视化结果
        if save_dir:
            save_dir = Path(save_dir)
            save_dir.mkdir(exist_ok=True, parents=True)
            
            plt.figure(figsize=(10, 8))
            plt.imshow(avg_freq_response, cmap='viridis')
            plt.colorbar()
            plt.title('Average Frequency Response')
            plt.tight_layout()
            plt.savefig(save_dir / 'frequency_response.png')
            plt.close()
        
        return {
            'avg_freq_response': avg_freq_response,
            'freq_responses': freq_responses
        }


class ForensicVisualizer:
    """伪造检测结果可视化工具"""
    
    @staticmethod
    def visualize_attention(image, attention_map, save_path=None, alpha=0.7):
        """可视化注意力图"""
        # 确保输入是numpy数组
        if torch.is_tensor(image):
            image = image.permute(1, 2, 0).cpu().numpy()
        if torch.is_tensor(attention_map):
            attention_map = attention_map.squeeze().cpu().numpy()
            
        # 归一化图像
        image = (image - image.min()) / (image.max() - image.min())
        
        # 调整注意力图大小
        h, w = image.shape[:2]
        attention_map = cv2.resize(attention_map, (w, h))
        
        # 创建热力图
        heatmap = cv2.applyColorMap(np.uint8(255 * attention_map), cv2.COLORMAP_JET)
        heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
        
        # 融合图像
        output = np.float32(heatmap) / 255 * alpha + np.float32(image) * (1 - alpha)
        output = output / np.max(output) * 255
        output = np.uint8(output)
        
        if save_path:
            plt.figure(figsize=(12, 4))
            
            plt.subplot(1, 3, 1)
            plt.imshow(image)
            plt.title("原始图像")
            plt.axis('off')
            
            plt.subplot(1, 3, 2)
            plt.imshow(attention_map, cmap='jet')
            plt.title("注意力图")
            plt.axis('off')
            
            plt.subplot(1, 3, 3)
            plt.imshow(output)
            plt.title("叠加结果")
            plt.axis('off')
            
            plt.tight_layout()
            plt.savefig(save_path)
            plt.close()
            
        return output
    
    @staticmethod
    def visualize_mask_and_boundary(image, pred_mask, true_mask=None, save_path=None):
        """可视化掩码和边界预测"""
        # 确保输入是numpy数组
        if torch.is_tensor(image):
            image = image.permute(1, 2, 0).cpu().numpy()
        if torch.is_tensor(pred_mask):
            pred_mask = pred_mask.squeeze().cpu().numpy()
        if true_mask is not None and torch.is_tensor(true_mask):
            true_mask = true_mask.squeeze().cpu().numpy()
            
        # 归一化图像
        image = (image - image.min()) / (image.max() - image.min())
        
        # 二值化掩码
        pred_mask_bin = (pred_mask > 0.5).astype(np.uint8)
        
        # 使用Canny找到边界
        pred_boundary = cv2.Canny(pred_mask_bin * 255, 100, 200)
        
        # 准备绘图
        plt.figure(figsize=(15, 5))
        
        # 原始图像
        plt.subplot(1, 3 if true_mask is None else 4, 1)
        plt.imshow(image)
        plt.title("原始图像")
        plt.axis('off')
        
        # 预测掩码
        plt.subplot(1, 3 if true_mask is None else 4, 2)
        plt.imshow(pred_mask, cmap='jet', vmin=0, vmax=1)
        plt.title("预测掩码")
        plt.axis('off')
        
        # 预测边界
        plt.subplot(1, 3 if true_mask is None else 4, 3)
        # 创建RGB图像用于显示边界
        boundary_overlay = image.copy()
        for c in range(3):
            channel = boundary_overlay[:, :, c]
            channel[pred_boundary > 0] = 1 if c == 0 else 0  # 红色边界
        plt.imshow(boundary_overlay)
        plt.title("预测边界")
        plt.axis('off')
        
        # 真实掩码（如果有）
        if true_mask is not None:
            plt.subplot(1, 4, 4)
            plt.imshow(true_mask, cmap='jet', vmin=0, vmax=1)
            plt.title("真实掩码")
            plt.axis('off')
            
            # 计算边界IoU
            true_mask_bin = (true_mask > 0.5).astype(np.uint8)
            true_boundary = cv2.Canny(true_mask_bin * 255, 100, 200)
            
            # 添加IoU信息
            intersection = np.logical_and(pred_mask_bin, true_mask_bin).sum()
            union = np.logical_or(pred_mask_bin, true_mask_bin).sum()
            iou = intersection / union if union > 0 else 0
            
            plt.suptitle(f"IoU: {iou:.4f}")
            
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path)
            plt.close()
        else:
            plt.show()
        
        return boundary_overlay
    
    @staticmethod
    def visualize_frequency_features(model, image, save_path=None):
        """可视化频域特征"""
        # 确保图像是张量
        if not torch.is_tensor(image):
            # 转换为[C, H, W]格式的张量
            transform = transforms.ToTensor()
            image_tensor = transform(image).unsqueeze(0)  # [1, C, H, W]
        else:
            image_tensor = image.unsqueeze(0) if image.dim() == 3 else image
            
        # 获取设备
        device = next(model.parameters()).device
        image_tensor = image_tensor.to(device)
        
        # 提取DCT特征
        with torch.no_grad():
            if hasattr(model, 'FAD_head'):
                # 使用FAD_head提取DCT特征
                dct_features = model.FAD_head(image_tensor)
            elif hasattr(model, 'dct_extractor'):
                # 使用专用DCT提取器
                dct_features = model.dct_extractor(image_tensor)
            else:
                from network.dct_transform import MultiScaleFrequencyExtractor
                dct_extractor = MultiScaleFrequencyExtractor(in_channels=3, out_channels=12).to(device)
                dct_features = dct_extractor(image_tensor)
            
            # 获取特征的形状
            n, c, h, w = dct_features.shape
            
            # 将特征转换为cpu numpy数组
            dct_features = dct_features.cpu().numpy().squeeze()
            
            # 绘制每个通道的DCT特征
            num_channels = min(dct_features.shape[0], 12)  # 限制最多显示12个通道
            cols = 4
            rows = (num_channels + cols - 1) // cols
            
            plt.figure(figsize=(cols * 3, rows * 3))
            
            for i in range(num_channels):
                plt.subplot(rows, cols, i + 1)
                plt.imshow(dct_features[i], cmap='viridis')
                plt.title(f'DCT Channel {i+1}')
                plt.axis('off')
            
            plt.suptitle("Frequency Domain Features")
            plt.tight_layout()
            
            if save_path:
                plt.savefig(save_path)
                plt.close()
            else:
                plt.show()
            
            return dct_features
        
        return None


# 更新ModelEvaluator类，添加评估边界和频域的方法
def evaluate_boundary_detection(self, model, data_loader, device, save_dir=None):
    """评估边界检测性能"""
    model.eval()
    
    # 确保save_dir存在
    if save_dir:
        save_dir = Path(save_dir) / 'boundary_evaluation'
        save_dir.mkdir(exist_ok=True, parents=True)
    
    all_metrics = []
    sample_idx = 0
    
    with torch.no_grad():
        for batch in tqdm(data_loader, desc="评估边界检测", ncols=100):
            # 处理不同的批次格式
            if len(batch) >= 3:
                if isinstance(batch[1], torch.Tensor) and batch[1].dim() == 4:
                    # 双输入模式
                    inputs, dct_inputs, masks = batch[:3]
                    labels = batch[3] if len(batch) > 3 else None
                else:
                    # 标准模式
                    inputs, masks, labels = batch[:3]
                    dct_inputs = None
            else:
                self.logger.error(f"不支持的批次格式: {len(batch)}")
                continue
            
            # 移动到设备
            inputs = inputs.to(device)
            if dct_inputs is not None:
                dct_inputs = dct_inputs.to(device)
            masks = masks.to(device)
            
            # 获取预测结果
            try:
                if dct_inputs is not None:
                    outputs = model(inputs, dct_inputs)
                else:
                    outputs = model(inputs)
                
                # 检查输出格式
                if isinstance(outputs, tuple) and len(outputs) == 2:
                    mask_preds, _ = outputs
                else:
                    mask_preds = outputs
                    
            except Exception as e:
                self.logger.error(f"模型推理错误: {e}")
                continue
            
            # 批量处理
            for i in range(len(inputs)):
                # 提取单张图像和掩码
                img = inputs[i]
                pred_mask = mask_preds[i]
                true_mask = masks[i]
                
                # 计算边界指标
                metrics = BoundaryEvaluator.calculate_boundary_metrics(
                    pred_mask.unsqueeze(0).cpu().numpy(), 
                    true_mask.unsqueeze(0).cpu().numpy()
                )
                
                # 保存可视化结果
                if save_dir:
                    img_path = save_dir / f'sample_{sample_idx}.png'
                    ForensicVisualizer.visualize_mask_and_boundary(
                        img, pred_mask, true_mask, save_path=img_path
                    )
                    
                all_metrics.append(metrics)
                sample_idx += 1
                
                # 限制样本数量
                if sample_idx >= 50:
                    break
            
            # 如果已处理足够样本，退出循环
            if sample_idx >= 50:
                break
    
    # 计算平均指标
    avg_metrics = {}
    for key in all_metrics[0].keys():
        avg_metrics[key] = float(np.mean([m[key] for m in all_metrics]))
    
    # 保存结果
    if save_dir:
        # 保存指标
        with open(save_dir / 'boundary_metrics.json', 'w') as f:
            json.dump(avg_metrics, f, indent=2)
            
        # 创建可视化图表
        plt.figure(figsize=(10, 6))
        sns.barplot(x=list(avg_metrics.keys()), y=list(avg_metrics.values()))
        plt.title('Boundary Detection Metrics')
        plt.ylabel('Score')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(save_dir / 'boundary_metrics_chart.png')
        plt.close()
    
    return avg_metrics


# 将新定义的方法绑定到ModelEvaluator类
# 正确语法是保存原方法的引用，而不是直接替换
# 这是将类方法作为函数引用的正确方式

# 修复绑定evaluate_model方法
ModelEvaluator.evaluate_model = ModelEvaluator.evaluate_model  # 不需要修改，类已内置该方法

# 添加新方法
ModelEvaluator.evaluate_boundary_detection = evaluate_boundary_detection

# 或者更好的做法是，直接在类定义中添加该方法，而不是在外部定义

# 导出接口
__all__ = [
    'ModelEvaluator',
    'ForensicMetrics',
    'EnsembleEvaluator',
    'ErrorAnalyzer',
    'find_optimal_threshold',
    'BoundaryEvaluator',
    'FrequencyDomainEvaluator',
    'ForensicVisualizer'
]

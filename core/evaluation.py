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


class ModelEvaluator:
    """模型评估工具类，统一评估逻辑"""
    def __init__(self, save_dir=None, logger=None):
        """初始化评估器"""
        self.save_dir = Path(save_dir) if save_dir else None
        if self.save_dir:
            self.save_dir.mkdir(exist_ok=True, parents=True)
        self.logger = logger or logging.getLogger(__name__)
    
    def evaluate_model(self, model, data_loader, device, threshold=0.5, return_predictions=False):
        """通用模型评估方法"""
        model.eval()
        
        # 初始化结果收集器
        all_preds, all_probs, all_labels = [], [], []
        all_paths, all_types = [], []
        all_masks_pred, all_masks_true = [], []
        
        # 计时器
        start_time = time.time()
        
        with torch.no_grad():
            for batch in tqdm(data_loader, desc="评估中", ncols=100):
                if len(batch) == 3:
                    inputs, masks, labels = batch
                    paths, types = None, None
                elif len(batch) == 4:
                    inputs, masks, labels, paths = batch
                    types = None
                else:
                    inputs, masks, labels, paths, types = batch
                
                # 移动到设备
                inputs = inputs.to(device)
                masks = masks.to(device)
                labels = labels.to(device)
                
                # 获取预测结果
                try:
                    mask_preds, outputs = model(inputs)
                except:
                    try:
                        outputs = model(inputs)
                        mask_preds = torch.zeros_like(masks)
                    except Exception as e:
                        self.logger.error(f"模型推理错误: {e}")
                        outputs = torch.zeros((inputs.size(0), 2), device=device)
                        mask_preds = torch.zeros_like(masks)
                
                # 获取概率和预测值
                probs = torch.softmax(outputs, dim=1)[:, 1].cpu().numpy()
                preds = (probs >= threshold).astype(int)
                
                # 收集结果
                all_preds.extend(preds)
                all_probs.extend(probs)
                all_labels.extend(labels.cpu().numpy())
                all_masks_pred.extend(mask_preds.cpu().numpy())
                all_masks_true.extend(masks.cpu().numpy())
                
                if paths is not None:
                    all_paths.extend(paths)
                if types is not None:
                    all_types.extend(types)
        
        # 计算总推理时间
        total_time = time.time() - start_time
        fps = len(all_labels) / total_time if total_time > 0 else 0
        
        # 计算评估指标
        metrics = self.calculate_metrics(all_labels, all_preds, all_probs)
        
        # 计算掩码评估指标
        mask_metrics = self.evaluate_masks(all_masks_pred, all_masks_true)
        metrics['mask_metrics'] = mask_metrics
        
        # 添加伪造类型分析
        if len(all_types) > 0:
            type_analysis = self.analyze_by_forgery_type(all_types, all_labels, all_preds, all_probs)
            metrics['type_analysis'] = type_analysis
        
        # 添加性能指标
        metrics['inference_time'] = total_time
        metrics['fps'] = fps
            
        # 是否返回原始预测
        if return_predictions:
            metrics['predictions'] = {
                'preds': all_preds,
                'probs': all_probs,
                'labels': all_labels,
                'masks_pred': all_masks_pred,
                'masks_true': all_masks_true,
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


# 导出接口
__all__ = [
    'ModelEvaluator',
    'ForensicMetrics',
    'EnsembleEvaluator',
    'ErrorAnalyzer',
    'find_optimal_threshold',
]


# 测试代码
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='评估模块测试')
    parser.add_argument('--model_path', type=str, help='模型路径')
    parser.add_argument('--test_data', type=str, help='测试数据路径')
    parser.add_argument('--output_dir', type=str, default='./evaluation_results', help='结果保存目录')
    args = parser.parse_args()
    
    if args.model_path and args.test_data:
        print("测试评估模块...")
        evaluator = ModelEvaluator(save_dir=args.output_dir)
        print("评估模块可以使用。若要运行完整测试，请提供有效的模型和数据路径。")
    else:
        print("未提供模型和数据路径，仅验证模块可导入性。")
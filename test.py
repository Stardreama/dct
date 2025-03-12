import torch
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
osenvs = len(os.environ['CUDA_VISIBLE_DEVICES'].split(','))
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import yaml
import easydict
import numpy as np
from network.transform import mesonet_data_transforms
from trainer import EnhancedTrainer, Trainer
from tqdm import tqdm
import argparse
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from PIL import Image
from sklearn.metrics import (
    roc_auc_score, precision_recall_curve, average_precision_score,
    roc_curve, confusion_matrix, classification_report
)
import cv2
import time

# 从utils导入评估工具
from utils import ForensicsEvaluator, setup_logger


# 增强的测试数据集，提供更多细节分析
class EnhancedTestDataset(torch.utils.data.Dataset):
    def __init__(self, dataset_dir, split="test", transform=None, return_path=False):
        """
        增强的测试数据集，提供更详细的样本信息和分析
        
        Args:
            dataset_dir: 数据集根目录
            split: 数据集划分
            transform: 数据变换
            return_path: 是否返回文件路径
        """
        self.transform = transform
        self.return_path = return_path
        self.samples = []
        self.forgery_types = set()
        
        # 读取索引文件
        index_file = os.path.join(dataset_dir, f"{split}.txt")
        with open(index_file, "r") as f:
            for line in f:
                img_path, label = line.strip().split()
                
                # 分析伪造类型
                forgery_type = "real"
                if "fake" in img_path:
                    # 提取伪造类型
                    path_parts = img_path.split(os.sep)
                    fake_idx = path_parts.index("fake") if "fake" in path_parts else -1
                    if fake_idx != -1 and fake_idx + 1 < len(path_parts):
                        forgery_type = path_parts[fake_idx + 1]
                
                self.forgery_types.add(forgery_type)
                self.samples.append((img_path, int(label), forgery_type))
        
        # 统计数据集信息
        self.stats = self._analyze_dataset()
        
    def _analyze_dataset(self):
        """分析数据集统计信息"""
        stats = {
            "total": len(self.samples),
            "real": sum(1 for _, label, _ in self.samples if label == 0),
            "fake": sum(1 for _, label, _ in self.samples if label == 1),
            "forgery_types": {}
        }
        
        # 统计各伪造类型数量
        for _, label, forgery_type in self.samples:
            if forgery_type not in stats["forgery_types"]:
                stats["forgery_types"][forgery_type] = 0
            stats["forgery_types"][forgery_type] += 1
            
        return stats
    
    
    def __len__(self):
        return len(self.samples)


# 模型集成类
class ModelEnsemble:
    """模型集成类，支持多种集成策略"""
    def __init__(self, models, device, ensemble_type="average"):
        """
        初始化模型集成
        
        Args:
            models: 模型列表
            device: 计算设备
            ensemble_type: 集成类型 ("average", "weighted", "max")
        """
        self.models = models
        self.device = device
        self.ensemble_type = ensemble_type
        self.weights = [1.0/len(models)] * len(models)  # 默认平均权重
    
    def set_weights(self, weights):
        """设置模型权重"""
        assert len(weights) == len(self.models), "权重数量必须等于模型数量"
        self.weights = weights
    
    def predict(self, inputs):
        """集成预测"""
        all_probs = []
        all_masks = []
        
        with torch.no_grad():
            for model in self.models:
                model.eval()
                mask_pred, outputs = model(inputs)
                probs = torch.softmax(outputs, dim=1)
                
                all_masks.append(mask_pred)
                all_probs.append(probs)
        
        # 根据集成类型组合预测结果
        if self.ensemble_type == "average":
            # 简单平均
            ensemble_probs = torch.stack(all_probs).mean(dim=0)
            ensemble_mask = torch.stack(all_masks).mean(dim=0)
        elif self.ensemble_type == "weighted":
            # 加权平均
            weighted_probs = [prob * w for prob, w in zip(all_probs, self.weights)]
            weighted_masks = [mask * w for mask, w in zip(all_masks, self.weights)]
            
            ensemble_probs = torch.stack(weighted_probs).sum(dim=0)
            ensemble_mask = torch.stack(weighted_masks).sum(dim=0)
        elif self.ensemble_type == "max":
            # 最大值融合
            ensemble_probs = torch.stack(all_probs).max(dim=0)[0]
            ensemble_mask = torch.stack(all_masks).max(dim=0)[0]
        else:
            raise ValueError(f"不支持的集成类型: {self.ensemble_type}")
            
        return ensemble_mask, ensemble_probs


def test_model(model, test_loader, device, evaluator=None, save_dir=None, threshold=0.5):
    """
    全面评估模型性能
    
    Args:
        model: 待评估模型
        test_loader: 测试数据加载器
        device: 计算设备
        evaluator: 评估工具
        save_dir: 结果保存目录
        threshold: 决策阈值
        
    Returns:
        dict: 评估结果
    """
    if isinstance(model, ModelEnsemble):
        # 使用集成模型，无需额外设置
        pass
    else:
        model.eval()
    
    # 初始化结果收集器
    all_preds = []
    all_probs = []
    all_labels = []
    all_paths = []
    all_types = []
    all_masks_pred = []
    all_masks_true = []
    
    # 计时
    start_time = time.time()
    
    # 开始测试
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="评估中"):
            if len(batch) == 3:
                inputs, masks, labels = batch
                paths = None
                types = None
            else:
                inputs, masks, labels, paths, types = batch
            
            inputs = inputs.to(device)
            masks = masks.to(device)
            labels = labels.to(device)
            
            # 获取预测结果
            mask_preds, outputs = model(inputs)
            
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
    inference_time = time.time() - start_time
    fps = len(all_labels) / inference_time
    
    # 计算基本指标
    accuracy = np.mean(np.array(all_preds) == np.array(all_labels))
    auc_score = roc_auc_score(all_labels, all_probs)
    
    # 创建结果字典
    results = {
        'accuracy': accuracy,
        'auc': auc_score,
        'predictions': all_preds,
        'probabilities': all_probs,
        'labels': all_labels,
        'inference_time': inference_time,
        'fps': fps
    }
    
    # 如果有评估器，生成详细报告
    if evaluator:
        print("\n正在生成详细评估报告...")
        # 构建评估数据结构
        metrics = {
            'accuracy': accuracy,
            'auc': auc_score,
            'confusion_matrix': confusion_matrix(all_labels, all_preds),
            'classification_report': classification_report(all_labels, all_preds, output_dict=True),
            'roc_data': {'fpr': None, 'tpr': None},
            'pr_data': {'precision': None, 'recall': None}
        }
        
        # 计算ROC和PR曲线数据
        fpr, tpr, _ = roc_curve(all_labels, all_probs)
        precision, recall, _ = precision_recall_curve(all_labels, all_probs)
        ap = average_precision_score(all_labels, all_probs)
        
        metrics['roc_data'] = {'fpr': fpr, 'tpr': tpr}
        metrics['pr_data'] = {'precision': precision, 'recall': recall}
        metrics['ap'] = ap
        
        # 计算掩码评估指标
        mask_metrics = evaluator.evaluate_masks(all_masks_pred, all_masks_true)
        metrics['mask_metrics'] = mask_metrics
        
        # 生成详细报告
        evaluator.generate_evaluation_report(metrics, save_dir)
        
        # 可视化样例预测
        if len(all_paths) > 0:
            visualize_predictions(
                all_paths[:10], 
                all_masks_pred[:10], 
                all_labels[:10], 
                all_preds[:10], 
                all_probs[:10],
                save_dir
            )
            
        # 添加伪造类型分析
        if len(all_types) > 0:
            type_analysis = analyze_by_forgery_type(all_types, all_labels, all_preds, all_probs)
            results['type_analysis'] = type_analysis
            
            if save_dir:
                # 保存伪造类型分析结果
                save_type_analysis(type_analysis, save_dir)
        
        # 保存结果摘要
        if save_dir:
            save_summary(results, save_dir)
    
    return results


def visualize_predictions(img_paths, mask_preds, true_labels, pred_labels, probabilities, save_dir=None):
    """可视化预测结果"""
    if save_dir is None:
        return
        
    save_dir = Path(save_dir)
    save_dir.mkdir(exist_ok=True, parents=True)
    
    num_samples = len(img_paths)
    fig, axes = plt.subplots(num_samples, 3, figsize=(15, num_samples * 4))
    
    for i in range(num_samples):
        img_path = img_paths[i]
        mask_pred = mask_preds[i][0]  # 取第一个通道
        true_label = true_labels[i]
        pred_label = pred_labels[i]
        prob = probabilities[i]
        
        # 加载原始图像
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
    
    plt.tight_layout()
    plt.savefig(save_dir / "prediction_samples.png", dpi=300, bbox_inches='tight')
    plt.close()


def analyze_by_forgery_type(types, labels, preds, probs):
    """按伪造类型分析性能"""
    type_data = {}
    
    for t, l, p, prob in zip(types, labels, preds, probs):
        if t not in type_data:
            type_data[t] = {
                'count': 0,
                'correct': 0,
                'probs': [],
                'labels': []
            }
            
        type_data[t]['count'] += 1
        type_data[t]['correct'] += (p == l)
        type_data[t]['probs'].append(prob)
        type_data[t]['labels'].append(l)
    
    # 计算每种类型的性能指标
    analysis = {}
    for t, data in type_data.items():
        if len(data['labels']) > 0:
            analysis[t] = {
                'count': data['count'],
                'accuracy': data['correct'] / data['count'],
                'auc': roc_auc_score(data['labels'], data['probs']) if len(set(data['labels'])) > 1 else 0
            }
    
    return analysis


def save_type_analysis(type_analysis, save_dir):
    """保存伪造类型分析结果"""
    save_dir = Path(save_dir)
    
    # 准备数据
    types = list(type_analysis.keys())
    accuracies = [data['accuracy'] for data in type_analysis.values()]
    aucs = [data['auc'] for data in type_analysis.values()]
    counts = [data['count'] for data in type_analysis.values()]
    
    # 创建数据框
    df = pd.DataFrame({
        '伪造类型': types,
        '样本数量': counts,
        '准确率': accuracies,
        'AUC分数': aucs
    })
    
    # 保存CSV
    df.to_csv(save_dir / "forgery_type_analysis.csv", index=False)
    
    # 创建条形图
    plt.figure(figsize=(12, 8))
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
    plt.savefig(save_dir / "forgery_type_performance.png", dpi=300, bbox_inches='tight')
    plt.close()


def save_summary(results, save_dir):
    """保存测试结果摘要"""
    save_dir = Path(save_dir)
    
    with open(save_dir / "test_summary.txt", "w") as f:
        f.write(f"测试准确率: {results['accuracy']:.4f}\n")
        f.write(f"ROC AUC分数: {results['auc']:.4f}\n")
        f.write(f"推理时间: {results['inference_time']:.2f}秒\n")
        f.write(f"处理速度: {results['fps']:.2f} FPS\n")
        
        if 'type_analysis' in results:
            f.write("\n伪造类型性能分析:\n")
            for t, data in results['type_analysis'].items():
                f.write(f"- {t}: 准确率={data['accuracy']:.4f}, AUC={data['auc']:.4f}, 样本数={data['count']}\n")


def main(config):
    # 设置随机种子以提高可重复性
    seed = config.SEED if hasattr(config, 'SEED') else 42
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    # 参数设置
    test_path = config.TEST_PATH if hasattr(config, 'TEST_PATH') else "D:/project/DCT_RGB_HRNet/dataset"
    batch_size = config.BATCH_SIZE if hasattr(config, 'BATCH_SIZE') else 8
    mode = config.MODE if hasattr(config, 'MODE') else 'Both'
    
    # 创建结果保存目录
    timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime())
    result_dir = Path(f"results/test_{timestamp}")
    result_dir.mkdir(parents=True, exist_ok=True)
    
    # 设置日志
    logger = setup_logger(str(result_dir), 'test.log', 'test_logger')
    logger.info(f"开始测试, 配置: {config}")
    
    # 保存配置
    with open(result_dir / 'test_config.yaml', 'w') as f:
        yaml.dump(dict(config), f)
    
    # 设置设备
    device = torch.device(f'cuda:{0}' if torch.cuda.is_available() else 'cpu')
    logger.info(f"使用设备: {device}")
    
    # 创建数据转换
    test_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])
    
    # 创建测试数据集
    test_dataset = EnhancedTestDataset(
        test_path, 
        split="test", 
        transform=test_transform,
        return_path=True
    )
    
    logger.info(f"测试集统计: 总样本={test_dataset.stats['total']}, "
               f"真实样本={test_dataset.stats['real']}, "
               f"伪造样本={test_dataset.stats['fake']}")
               
    # 显示伪造类型分布
    for ftype, count in test_dataset.stats['forgery_types'].items():
        logger.info(f"伪造类型 '{ftype}': {count}样本")
    
    # 创建数据加载器
    test_loader = torch.utils.data.DataLoader(
        test_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=8,
        pin_memory=True
    )
    
    # 模型评估方式
    eval_mode = config.EVALUATION.MODE if hasattr(config, 'EVALUATION') and hasattr(config.EVALUATION, 'MODE') else "single"
    
    if eval_mode == "ensemble":
        # 集成评估多个模型
        model_paths = config.EVALUATION.MODEL_PATHS if hasattr(config.EVALUATION, 'MODEL_PATHS') else []
        ensemble_type = config.EVALUATION.ENSEMBLE_TYPE if hasattr(config.EVALUATION, 'ENSEMBLE_TYPE') else "average"
        
        if not model_paths:
            logger.error("集成模式需要提供模型路径列表")
            return
            
        # 加载多个模型
        logger.info(f"正在加载{len(model_paths)}个模型进行集成评估")
        models = []
        
        for path in model_paths:
            if not os.path.exists(path):
                logger.warning(f"模型路径不存在: {path}")
                continue
                
            try:
                # 使用增强版训练器
                model = EnhancedTrainer(config, [0], mode=mode)
                model.load(path)
                model.model.to(device)
                models.append(model.model)
                logger.info(f"成功加载模型: {path}")
            except Exception as e:
                logger.error(f"加载模型失败 {path}: {e}")
        
        if not models:
            logger.error("没有成功加载任何模型")
            return
            
        # 创建模型集成
        model = ModelEnsemble(models, device, ensemble_type)
        logger.info(f"创建{ensemble_type}集成模型，共{len(models)}个子模型")
        
    else:
        # 单模型评估
        model_path = config.MODEL_PATH if hasattr(config, 'MODEL_PATH') else None
        pretrained_path = config.PRETRAINED_PATH if hasattr(config, 'PRETRAINED_PATH') else './pretrained/xception-b5690688.pth'
        
        if not model_path or not os.path.exists(model_path):
            logger.error(f"模型路径不存在: {model_path}")
            return
            
        try:
            # 使用增强版训练器
            model = EnhancedTrainer(config, [0], mode=mode, pretrained_path=pretrained_path)
            model.load(model_path)
            model = model.model.to(device)
            logger.info(f"成功加载模型: {model_path}")
        except Exception as e:
            logger.error(f"加载模型失败: {e}")
            return
    
    # 创建评估器
    evaluator = ForensicsEvaluator(save_dir=result_dir)
    
    # 执行模型测试
    logger.info("开始模型评估...")
    results = test_model(
        model=model,
        test_loader=test_loader,
        device=device,
        evaluator=evaluator,
        save_dir=result_dir,
        threshold=0.5
    )
    
    # 输出测试结果
    logger.info(f"测试完成! 准确率: {results['accuracy']:.4f}, AUC: {results['auc']:.4f}")
    logger.info(f"推理时间: {results['inference_time']:.2f}秒, 处理速度: {results['fps']:.2f} FPS")
    
    # 如果包含伪造类型分析，输出详细结果
    if 'type_analysis' in results:
        logger.info("\n伪造类型性能分析:")
        type_analysis = results['type_analysis']
        for t, data in sorted(type_analysis.items(), key=lambda x: x[1]['accuracy'], reverse=True):
            logger.info(f"- {t}: 准确率={data['accuracy']:.4f}, AUC={data['auc']:.4f}, 样本数={data['count']}")
    
    logger.info(f"详细评估报告已保存至: {result_dir}")
    
    return results


if __name__ == '__main__':
    # 添加命令行参数解析
    parser = argparse.ArgumentParser(description='深度伪造检测模型测试工具')
    parser.add_argument('--config', type=str, default="./config.yaml", help='配置文件路径')
    parser.add_argument('--model_path', type=str, help='模型路径')
    parser.add_argument('--test_path', type=str, help='测试数据集路径')
    parser.add_argument('--batch_size', type=int, default=8, help='批次大小')
    parser.add_argument('--ensemble', action='store_true', help='使用模型集成')
    parser.add_argument('--mode', type=str, default='Both', choices=['RGB', 'FAD', 'Both'], help='模型模式')
    
    args = parser.parse_args()
    
    # 加载配置文件
    with open(args.config, 'r') as stream:
        config = yaml.safe_load(stream)
    config = easydict.EasyDict(config)
    
    # 命令行参数优先级高于配置文件
    if args.model_path:
        config.MODEL_PATH = args.model_path
    if args.test_path:
        config.TEST_PATH = args.test_path
    if args.batch_size:
        config.BATCH_SIZE = args.batch_size
    if args.mode:
        config.MODE = args.mode
        
    # 设置评估模式
    if not hasattr(config, 'EVALUATION'):
        config.EVALUATION = easydict.EasyDict()
    config.EVALUATION.MODE = "ensemble" if args.ensemble else "single"
    
    # 执行测试
    main(config)

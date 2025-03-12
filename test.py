import torch
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
osenvs = len(os.environ['CUDA_VISIBLE_DEVICES'].split(','))
import yaml
import easydict
import numpy as np
import time
import argparse
from pathlib import Path
from tqdm import tqdm

# 导入自定义模块
from utils import setup_logger, set_seed, load_model_weights
from trainer import EnhancedTrainer, Trainer
from core.dataset import TestForensicDataset
from core.evaluation import ModelEvaluator, EnsembleEvaluator
from core.visualization import ForensicVisualizer


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


def main(config):
    """测试主函数"""
    # 设置随机种子以提高可重复性
    seed = config.SEED if hasattr(config, 'SEED') else 42
    set_seed(seed)
    
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
    
    # 创建数据转换和数据集
    import torchvision.transforms as transforms
    test_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])
    
    # 创建测试数据集
    test_dataset = TestForensicDataset(
        dataset_dir=test_path, 
        split="test", 
        transform=test_transform,
        return_path=True
    )
    
    # 显示数据集统计信息
    logger.info(f"测试集统计: 总样本={len(test_dataset)}")
    
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
        
        # 创建集成评估器
        evaluator = EnsembleEvaluator()
        
        # 执行集成评估
        logger.info("开始集成评估...")
        metrics = evaluator.evaluate_ensemble(
            models=models,
            data_loader=test_loader,
            device=device,
            ensemble_type=ensemble_type
        )
        
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
        evaluator = ModelEvaluator(save_dir=result_dir)
        
        # 执行模型评估
        logger.info("开始模型评估...")
        metrics = evaluator.evaluate_model(
            model=model,
            data_loader=test_loader,
            device=device,
            return_predictions=True  # 返回原始预测用于可视化
        )
    
    # 输出测试结果
    logger.info(f"测试完成! 准确率: {metrics['accuracy']:.4f}, AUC: {metrics['auc']:.4f}")
    logger.info(f"推理时间: {metrics['inference_time']:.2f}秒, 处理速度: {metrics['fps']:.2f} FPS")
    
    # 保存评估结果
    evaluator.save_evaluation_results(metrics, result_dir)
    
    # 可视化评估结果
    visualizer = ForensicVisualizer(save_dir=result_dir)
    visualizer.create_evaluation_report(metrics)
    
    # 可视化样例预测
    if 'predictions' in metrics:
        # 选取一些样本进行可视化
        preds = metrics['predictions']
        sample_indices = np.random.choice(len(preds['labels']), min(10, len(preds['labels'])), replace=False)
        
        # 收集所需数据
        sample_paths = [preds['paths'][i] for i in sample_indices] if preds['paths'] else None
        sample_masks = [preds['masks_pred'][i] for i in sample_indices]
        sample_labels = [preds['labels'][i] for i in sample_indices]
        sample_preds = [preds['preds'][i] for i in sample_indices]
        sample_probs = [preds['probs'][i] for i in sample_indices]
        
        if sample_paths:
            visualizer.visualize_predictions(
                sample_paths, sample_masks, sample_labels, sample_preds, sample_probs,
                save_path=result_dir / "prediction_samples.png"
            )
    
    # 如果包含伪造类型分析，输出详细结果
    if 'type_analysis' in metrics:
        logger.info("\n伪造类型性能分析:")
        type_analysis = metrics['type_analysis']
        for t, data in sorted(type_analysis.items(), key=lambda x: x[1]['accuracy'], reverse=True):
            logger.info(f"- {t}: 准确率={data['accuracy']:.4f}, AUC={data['auc']:.4f}, 样本数={data['count']}")
        
        # 绘制伪造类型性能对比图
        visualizer.plot_forgery_type_performance(type_analysis)
    
    logger.info(f"详细评估报告已保存至: {result_dir}")
    
    return metrics


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
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

import torch
import yaml
import easydict
import argparse
from pathlib import Path
import time
import logging

# 导入自定义模块 - 更新导入以使用修改后的功能
from utils import setup_logger, set_seed, save_checkpoint, AverageMeter, create_default_config, evaluate
from trainer import create_trainer
from core.dataset import create_forensic_data_loaders
from core.evaluation import ModelEvaluator
from core.visualization import ForensicVisualizer
from models import create_model


def main(config):
    """训练主函数"""
    # 设置随机种子保证可重现性
    seed = config.SEED if hasattr(config, 'SEED') else 42
    set_seed(seed)
    
    # 创建实验目录 - 使用配置的模型名称和时间戳
    timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime())
    
    # 从配置中获取模型名称
    if hasattr(config, 'MODEL_CONFIG') and hasattr(config.MODEL_CONFIG, 'TYPE'):
        model_type = config.MODEL_CONFIG.TYPE
    elif hasattr(config, 'MODEL') and hasattr(config.MODEL, 'NAME'):
        model_type = config.MODEL.NAME
    else:
        model_type = "forensics"
        
    experiment_name = f"{model_type}_{timestamp}"
    output_dir = Path(config.OUTPUT_DIR if hasattr(config, 'OUTPUT_DIR') else './output')
    experiment_dir = output_dir / experiment_name
    experiment_dir.mkdir(parents=True, exist_ok=True)
    
    # 设置日志记录器
    logger = setup_logger(str(experiment_dir), 'train.log', 'trainer_logger')
    logger.info(f"实验名称: {experiment_name}")
    logger.info(f"配置: {config}")
    
    # 保存配置到实验目录
    with open(experiment_dir / 'config.yaml', 'w') as f:
        yaml.dump(dict(config), f)
    
    # 创建数据加载器 - 使用core.dataset中的函数
    logger.info("初始化数据集...")
    train_loader, val_loader, test_loader = create_forensic_data_loaders(config)
    
    # 设置设备和初始化参数
    gpu_ids = list(range(len(os.environ['CUDA_VISIBLE_DEVICES'].split(','))))
    
    # 从配置中获取模型参数
    model_config = config.MODEL_CONFIG if hasattr(config, 'MODEL_CONFIG') else None
    model_type = model_config.TYPE if model_config and hasattr(model_config, 'TYPE') else 'forensics'
    mode = model_config.MODE if model_config and hasattr(model_config, 'MODE') else 'Both'
    
    # 获取预训练路径
    pretrained_path = None
    if model_config and hasattr(model_config, 'PRETRAINED'):
        if isinstance(model_config.PRETRAINED, dict):
            # 如果有多个预训练模型可选
            if model_type.lower() == 'enhanced' or model_type.lower() == 'f3net':
                pretrained_path = model_config.PRETRAINED.get('XCEPTION', None)
            elif model_type.lower() == 'forensics':
                pretrained_path = model_config.PRETRAINED.get('HRNET', None)
        else:
            pretrained_path = model_config.PRETRAINED
    
    # 如果没有指定预训练路径，尝试使用默认路径
    if not pretrained_path and hasattr(config, 'MODEL') and hasattr(config.MODEL, 'PRETRAINED'):
        pretrained_path = config.MODEL.PRETRAINED
    
    # 初始化训练器 - 使用factory方法创建
    try:
        logger.info("初始化训练器...")
        trainer = create_trainer(
            config=config,
            gpu_ids=gpu_ids,
            model_type=model_type,
            mode=mode,
            pretrained_path=pretrained_path
        )
        
        # 打印模型信息
        trainer.print_model_summary()
        
        # 恢复训练检查点（如果存在）
        if hasattr(config, 'TRAINING') and hasattr(config.TRAINING, 'RESUME') and config.TRAINING.RESUME:
            resume_path = None
            if hasattr(config.TRAINING, 'RESUME_PATH'):
                resume_path = config.TRAINING.RESUME_PATH
                if os.path.exists(resume_path):
                    logger.info(f"从检查点恢复训练: {resume_path}")
                    trainer.load(resume_path)
                else:
                    logger.warning(f"检查点文件不存在: {resume_path}")
            else:
                logger.warning("配置中启用了RESUME但未指定RESUME_PATH，跳过恢复训练")
    except Exception as e:
        logger.error(f"初始化模型失败: {e}")
        import traceback
        logger.error(traceback.format_exc())
        raise e
    
    # 更新输出目录
    trainer.results_dir = experiment_dir
    
    # 开始训练 - 使用trainer的内部训练循环
    logger.info("开始训练...")
    try:
        # 使用trainer的train方法处理完整训练过程
        epochs = config.EPOCHES if hasattr(config, 'EPOCHES') else config.EPOCHS if hasattr(config, 'EPOCHS') else 100
        history = trainer.train(
            epochs=epochs, 
            train_loader=train_loader, 
            val_loader=val_loader,
            test_loader=test_loader
        )
        
        # 训练完成后的额外处理
        logger.info(f"训练完成。最佳验证准确率: {trainer.best_val_acc:.4f}")
        
        # 保存最终结果摘要
        with open(experiment_dir / "results_summary.txt", "w") as f:
            f.write(f"实验名称: {experiment_name}\n")
            f.write(f"模型类型: {model_type}\n")
            f.write(f"模型模式: {mode}\n")
            f.write(f"最佳验证准确率: {trainer.best_val_acc:.4f}\n")
            f.write(f"最佳验证损失: {trainer.best_val_loss:.4f}\n")
            
        return {
            'experiment_dir': experiment_dir,
            'best_val_acc': trainer.best_val_acc,
            'best_val_loss': trainer.best_val_loss
        }
        
    except Exception as e:
        logger.error(f"训练过程中出错: {e}")
        import traceback
        logger.error(traceback.format_exc())
        raise e


def parse_args():
    """解析命令行参数并加载配置"""
    parser = argparse.ArgumentParser(description='训练伪造检测模型')
    parser.add_argument('--config', type=str, default='./config.yaml', help='配置文件路径')
    parser.add_argument('--seed', type=int, default=42, help='随机种子')
    parser.add_argument('--epochs', type=int, help='训练轮数')
    parser.add_argument('--batch_size', type=int, help='批量大小')
    parser.add_argument('--lr', type=float, help='学习率')
    parser.add_argument('--output_dir', type=str, help='输出目录')
    parser.add_argument('--model_type', type=str, choices=['enhanced', 'f3net', 'forensics'], help='模型类型')
    parser.add_argument('--mode', type=str, choices=['RGB', 'FAD', 'Both'], help='模型模式')
    parser.add_argument('--resume', type=str, help='恢复训练的检查点路径')
    parser.add_argument('--mixed_precision', action='store_true', help='使用混合精度训练')
    parser.add_argument('--eval_only', action='store_true', help='仅评估模型')
    args = parser.parse_args()
    
    # 修改这里，明确指定UTF-8编码
    try:
        with open(args.config, 'r', encoding='utf-8') as stream:
            config = yaml.safe_load(stream)
        config = easydict.EasyDict(config)
    except Exception as e:
        print(f"无法加载配置文件 {args.config}: {e}")
        # 如果配置加载失败，使用默认配置
        config = create_default_config()
        
        
    # 使用命令行参数更新配置
    if args.seed:
        config.SEED = args.seed
    if args.epochs:
        config.EPOCHS = args.epochs
    if args.batch_size:
        config.BATCH_SIZE = args.batch_size
    if args.output_dir:
        config.OUTPUT_DIR = args.output_dir
    if args.model_type:
        if not hasattr(config, 'MODEL_CONFIG'):
            config.MODEL_CONFIG = easydict.EasyDict()
        config.MODEL_CONFIG.TYPE = args.model_type
    if args.mode:
        if not hasattr(config, 'MODEL_CONFIG'):
            config.MODEL_CONFIG = easydict.EasyDict()
        config.MODEL_CONFIG.MODE = args.mode
    if args.lr:
        if not hasattr(config, 'OPTIMIZER'):
            config.OPTIMIZER = easydict.EasyDict()
        config.OPTIMIZER.LR = args.lr
    if args.resume:
        if not hasattr(config, 'TRAINING'):
            config.TRAINING = easydict.EasyDict()
        config.TRAINING.RESUME = True
        config.TRAINING.RESUME_PATH = args.resume
    if args.mixed_precision:
        if not hasattr(config, 'TRAINING'):
            config.TRAINING = easydict.EasyDict()
        config.TRAINING.MIXED_PRECISION = True
        
    # 确保基本配置字段存在
    if not hasattr(config, 'EPOCHS'):
        config.EPOCHS = 100
    if not hasattr(config, 'SAVE_FREQ'):
        config.SAVE_FREQ = 5
    
    return args, config


def evaluate_model(config, model_path):
    """评估已训练的模型"""
    # 加载配置和模型路径
    logger = logging.getLogger("eval_logger")
    logger.info(f"评估模型: {model_path}")
    
    # 设置GPU
    gpu_ids = list(range(len(os.environ['CUDA_VISIBLE_DEVICES'].split(','))))
    
    # 创建数据加载器 - 仅使用测试数据集
    _, _, test_loader = create_forensic_data_loaders(config)
    
    # 从模型路径创建模型
    from utils import create_model_from_checkpoint
    device = f'cuda:{gpu_ids[0]}' if gpu_ids else 'cpu'
    model = create_model_from_checkpoint(config, model_path, device=device)
    
    # 评估模型
    logger.info("开始评估...")
    metrics = evaluate(model, config, test_loader=test_loader, 
                      evaluate_boundary=True, evaluate_freq=True)
    
    # 打印关键指标
    logger.info(f"评估结果:")
    logger.info(f"准确率: {metrics.get('accuracy', 0):.4f}")
    logger.info(f"AUC: {metrics.get('auc', 0):.4f}")
    if 'mask_metrics' in metrics:
        logger.info(f"掩码IoU: {metrics['mask_metrics'].get('mean_iou', 0):.4f}")
    if 'boundary_metrics' in metrics:
        logger.info(f"边界F1: {metrics['boundary_metrics'].get('boundary_f1', 0):.4f}")
    
    # 创建可视化
    output_dir = Path(config.OUTPUT_DIR) / "evaluation" / Path(model_path).stem
    output_dir.mkdir(parents=True, exist_ok=True)
    
    visualizer = ForensicVisualizer(save_dir=output_dir)
    visualizer.create_evaluation_report(metrics, save_dir=output_dir)
    
    logger.info(f"评估完成，结果已保存至: {output_dir}")
    return metrics


if __name__ == '__main__':
    # 解析命令行参数
    args, config = parse_args()
    
    # 仅评估模式
    if (args.eval_only):
        if not args.resume:
            print("错误: 请指定要评估的模型路径 (--resume)")
            exit(1)
        evaluate_model(config, args.resume)
    else:
        # 训练模式
        main(config)
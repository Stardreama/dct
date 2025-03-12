import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

import torch
import yaml
import easydict
import argparse
from pathlib import Path
import time
import logging

# 导入自定义模块
from utils import setup_logger, set_seed, save_checkpoint, AverageMeter
from trainer import EnhancedTrainer, Trainer
from core.dataset import create_forensic_data_loaders
from core.evaluation import ModelEvaluator
from core.visualization import ForensicVisualizer


def main(config):
    """训练主函数"""
    # 设置随机种子保证可重现性
    seed = config.SEED if hasattr(config, 'SEED') else 42
    set_seed(seed)
    
    # 创建实验目录
    timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime())
    experiment_name = f"{config.MODEL.NAME}_{timestamp}" if hasattr(config, 'MODEL') else f"experiment_{timestamp}"
    output_dir = Path(config.OUTPUT_DIR)
    experiment_dir = output_dir / experiment_name
    experiment_dir.mkdir(parents=True, exist_ok=True)
    
    # 设置日志记录器
    logger = setup_logger(str(experiment_dir), 'train.log', 'trainer_logger')
    logger.info(f"实验名称: {experiment_name}")
    logger.info(f"配置: {config}")
    
    # 保存配置到实验目录
    with open(experiment_dir / 'config.yaml', 'w') as f:
        yaml.dump(dict(config), f)
    
    # 创建数据加载器
    logger.info("初始化数据集...")
    train_loader, val_loader, test_loader = create_forensic_data_loaders(config)
    
    # 设置设备和初始化模型
    gpu_ids = list(range(len(os.environ['CUDA_VISIBLE_DEVICES'].split(','))))
    mode = config.MODEL.MODE if hasattr(config, 'MODEL') and hasattr(config.MODEL, 'MODE') else 'Both'
    pretrained_path = config.MODEL.PRETRAINED if hasattr(config, 'MODEL') and hasattr(config.MODEL, 'PRETRAINED') else './pretrained/xception-b5690688.pth'
    
    # 初始化训练器
    try:
        logger.info("初始化训练器...")
        if hasattr(config, 'USE_ENHANCED_TRAINER') and config.USE_ENHANCED_TRAINER:
            model = EnhancedTrainer(config, gpu_ids, mode=mode, pretrained_path=pretrained_path)
        else:
            model = Trainer(config, gpu_ids, mode=mode, pretrained_path=pretrained_path)
            
        # 恢复训练检查点（如果存在）
        if hasattr(config, 'TRAINING') and hasattr(config.TRAINING, 'RESUME') and config.TRAINING.RESUME:
            resume_path = config.TRAINING.RESUME_PATH
            if os.path.exists(resume_path):
                logger.info(f"从检查点恢复训练: {resume_path}")
                model.load(resume_path)
    except Exception as e:
        logger.error(f"初始化模型失败: {e}")
        raise e
    
    # 设置早停和评估器
    early_stopping_enabled = config.TRAINING.EARLY_STOPPING.ENABLED if hasattr(config, 'TRAINING') and hasattr(config.TRAINING, 'EARLY_STOPPING') else False
    patience = config.TRAINING.EARLY_STOPPING.PATIENCE if early_stopping_enabled else 10
    early_stop_counter = 0
    best_val_acc = 0.0
    best_epoch = 0
    
    # 创建可视化器和评估器
    visualizer = ForensicVisualizer(save_dir=experiment_dir)
    evaluator = ModelEvaluator(save_dir=experiment_dir / "evaluation")
    
    # 训练历史记录
    history = {
        'train_loss': [], 'val_loss': [], 'val_acc': [],
        'train_cls_loss': [], 'train_mask_loss': [], 'train_freq_loss': [],
        'lr': []
    }
    
    # 开始训练
    logger.info("开始训练...")
    try:
        for epoch in range(config.EPOCHS):
            # 训练阶段
            model.model.train()
            train_metrics = model.train_epoch(train_loader, epoch)
            
            # 验证阶段
            model.model.eval()
            device = next(model.model.parameters()).device
            val_metrics = evaluator.evaluate_model(model.model, val_loader, device)
            val_acc = val_metrics['accuracy']
            
            # 更新学习率
            current_lr = model.optimizer.param_groups[0]['lr']
            if hasattr(model, 'lr_scheduler'):
                model.lr_scheduler.step()
            
            # 更新历史记录
            history['train_loss'].append(train_metrics.get('total_loss', 0))
            history['train_cls_loss'].append(train_metrics.get('cls_loss', 0))
            history['train_mask_loss'].append(train_metrics.get('mask_loss', 0))
            history['train_freq_loss'].append(train_metrics.get('freq_loss', 0))
            history['val_acc'].append(val_acc)
            history['val_loss'].append(train_metrics.get('total_loss', 0))  # 用训练损失代替
            history['lr'].append(current_lr)
            
            # 记录训练信息
            logger.info(f'Epoch {epoch+1}/{config.EPOCHS} - '
                       f'Train Loss: {train_metrics.get("total_loss", 0):.4f}, '
                       f'Val Acc: {val_acc:.4f}, '
                       f'LR: {current_lr:.6f}')
            
            # 检查最佳性能
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_epoch = epoch
                # 保存最佳模型
                best_model_path = experiment_dir / "best_model.pth"
                model.save(str(best_model_path))
                logger.info(f"保存最佳模型 (acc={val_acc:.4f}) 到 {best_model_path}")
                early_stop_counter = 0
            else:
                early_stop_counter += 1
                
            # 保存检查点
            if epoch % config.SAVE_FREQ == 0 or epoch == config.EPOCHS - 1:
                checkpoint_path = experiment_dir / f"checkpoint_epoch_{epoch+1}.pth"
                model.save(str(checkpoint_path))
                
            # 可视化训练曲线
            if (epoch + 1) % 5 == 0 or epoch == config.EPOCHS - 1:
                visualizer.plot_training_curves(history)
                
            # 检查早停条件
            if early_stopping_enabled and early_stop_counter >= patience:
                logger.info(f"早停触发! 最佳验证准确率: {best_val_acc:.4f} at epoch {best_epoch+1}")
                break
        
        # 训练完成后的评估
        logger.info("训练完成。在测试集上评估最佳模型...")
        
        # 加载最佳模型
        best_model_path = experiment_dir / "best_model.pth"
        model.load(str(best_model_path))
        
        # 在测试集上评估
        model.model.eval()
        device = next(model.model.parameters()).device
        test_metrics = evaluator.evaluate_model(model.model, test_loader, device)
        test_acc = test_metrics['accuracy']
        
        # 保存评估结果
        evaluator.save_evaluation_results(test_metrics)
        
        # 创建评估报告
        visualizer.create_evaluation_report(test_metrics)
        
        # 保存最终结果摘要
        with open(experiment_dir / "results_summary.txt", "w") as f:
            f.write(f"实验名称: {experiment_name}\n")
            f.write(f"最佳验证准确率: {best_val_acc:.4f} (epoch {best_epoch+1})\n")
            f.write(f"最终测试准确率: {test_acc:.4f}\n")
            f.write(f"AUC分数: {test_metrics['auc']:.4f}\n")
            if 'mask_metrics' in test_metrics:
                f.write(f"掩码IoU: {test_metrics['mask_metrics']['mean_iou']:.4f}\n")
            
        logger.info(f"最终测试准确率: {test_acc:.4f} (最佳模型来自epoch {best_epoch+1})")
        logger.info(f"实验结果已保存至: {experiment_dir}")
        
        return {
            'experiment_dir': experiment_dir,
            'best_val_acc': best_val_acc,
            'best_epoch': best_epoch,
            'test_acc': test_acc
        }
        
    except Exception as e:
        logger.error(f"训练过程中出错: {e}", exc_info=True)
        raise e


if __name__ == '__main__':
    # 命令行参数解析
    parser = argparse.ArgumentParser(description='训练伪造检测模型')
    parser.add_argument('--config', type=str, default='./config.yaml', help='配置文件路径')
    parser.add_argument('--seed', type=int, help='随机种子')
    args = parser.parse_args()
    
    # 加载配置
    with open(args.config, 'r') as stream:
        config = yaml.safe_load(stream)
    config = easydict.EasyDict(config)
    
    # 添加命令行参数
    if args.seed:
        config.SEED = args.seed
        
    # 添加缺失的配置字段
    if not hasattr(config, 'EPOCHS'):
        config.EPOCHS = 30
    if not hasattr(config, 'SAVE_FREQ'):
        config.SAVE_FREQ = 5
    
    # 开始训练
    main(config)
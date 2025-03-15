import sys
import os
import torch
import logging
from pathlib import Path
import traceback
import importlib.util

# 添加项目根目录到路径
root_dir = Path(__file__).parent.parent
sys.path.append(str(root_dir))
print(f"添加路径: {root_dir}")

# 显示当前Python路径
print("Python搜索路径:")
for p in sys.path:
    print(f" - {p}")

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[logging.StreamHandler(), logging.FileHandler('device_check.log')]
)
logger = logging.getLogger('device_check')

def check_device(tensor, name):
    """检查张量的设备和形状"""
    if tensor is None:
        logger.info(f"{name} 是 None")
        return
        
    if not torch.is_tensor(tensor):
        logger.info(f"{name} 不是张量，类型为: {type(tensor)}")
        return
        
    logger.info(f"{name}:")
    logger.info(f"  - 设备: {tensor.device}")
    logger.info(f"  - 形状: {tensor.shape}")
    logger.info(f"  - 类型: {tensor.dtype}")
    if torch.isnan(tensor).any():
        logger.warning(f"  - 警告: {name} 包含NaN值!")
    if torch.isinf(tensor).any():
        logger.warning(f"  - 警告: {name} 包含无穷值!")

def check_file_exists(file_path):
    """检查文件是否存在"""
    path = Path(file_path)
    exists = path.exists()
    logger.info(f"检查文件 {path}: {'存在' if exists else '不存在'}")
    return exists

def load_module_from_path(module_name, file_path):
    """从文件路径加载模块"""
    try:
        spec = importlib.util.spec_from_file_location(module_name, file_path)
        if spec is None:
            logger.error(f"无法从 {file_path} 加载模块规范")
            return None
            
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        logger.info(f"成功从 {file_path} 加载模块 {module_name}")
        return module
    except Exception as e:
        logger.error(f"从 {file_path} 加载模块 {module_name} 失败: {e}")
        traceback.print_exc()
        return None

def debug_loss_computation():
    """调试损失计算过程中的设备问题"""
    try:
        # 检查训练器文件是否存在
        trainer_path = root_dir / "trainer.py"
        if not check_file_exists(trainer_path):
            logger.error(f"找不到训练器文件: {trainer_path}")
            
            # 尝试检查项目根目录下的所有Python文件
            logger.info("在项目根目录下查找Python文件:")
            for file in root_dir.glob("*.py"):
                logger.info(f" - {file}")
            
            # 如果没找到trainer.py，可能是目录结构问题
            logger.info("检查项目结构:")
            for dir_path in root_dir.iterdir():
                if dir_path.is_dir() and not dir_path.name.startswith('.'):
                    logger.info(f" - 目录: {dir_path}")
                    for file in dir_path.glob("*.py"):
                        logger.info(f"   - {file}")
            
            raise FileNotFoundError(f"找不到训练器文件: {trainer_path}")
        
        # 直接从文件加载trainer模块
        trainer_module = load_module_from_path('trainer', trainer_path)
        if trainer_module is None:
            logger.error("无法加载trainer模块，中止调试")
            return
        
        # 检查EnhancedTrainer类是否存在
        if not hasattr(trainer_module, 'EnhancedTrainer'):
            logger.error(f"trainer模块中不存在EnhancedTrainer类")
            logger.info(f"trainer模块包含以下内容: {dir(trainer_module)}")
            return
        
        EnhancedTrainer = trainer_module.EnhancedTrainer
        
        # 导入其他必要的模块
        import yaml
        import easydict
        
        # 加载配置
        config_path = root_dir / "config.yaml"
        if not check_file_exists(config_path):
            logger.error(f"找不到配置文件: {config_path}")
            return
            
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        config = easydict.EasyDict(config)
        
        logger.info("=== 开始调试损失计算 ===")
        
        # 检查每种损失函数及其设备
        logger.info("检查损失函数初始化...")
        
        # 创建临时设备和张量用于测试
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"当前默认设备: {device}")
        
        # 初始化常见损失函数并检查它们的内部设备
        loss_fns = {
            "CrossEntropyLoss": torch.nn.CrossEntropyLoss(),
            "BCELoss": torch.nn.BCELoss(),
            "BCEWithLogitsLoss": torch.nn.BCEWithLogitsLoss(),
            "DiceLoss": None  # 自定义损失可能需要单独导入
        }
        
        for name, loss_fn in loss_fns.items():
            if loss_fn is not None:
                logger.info(f"\n{name}:")
                # 检查损失函数是否有设备属性
                if hasattr(loss_fn, 'device'):
                    logger.info(f"  - 损失函数设备: {loss_fn.device}")
                else:
                    logger.info(f"  - 损失函数没有明确的device属性")
                
                # 检查损失函数的参数
                for param_name, param in vars(loss_fn).items():
                    if torch.is_tensor(param):
                        logger.info(f"  - 参数 {param_name} 设备: {param.device}")
        
        # 创建一个简化版本的trainer来测试compute_losses
        logger.info("\n创建简化版训练器用于测试...")
        
        class SimplifiedTrainer:
            def __init__(self):
                self.device = device
                self.cls_loss_fn = torch.nn.CrossEntropyLoss()
                self.mask_loss_fn = torch.nn.BCEWithLogitsLoss()
                self.cls_weight = 1.0
                self.mask_weight = 0.5
                self.freq_weight = 0.3
                
                # 将损失函数显式移至设备
                if hasattr(self.cls_loss_fn, 'to'):
                    self.cls_loss_fn = self.cls_loss_fn.to(self.device)
                if hasattr(self.mask_loss_fn, 'to'):
                    self.mask_loss_fn = self.mask_loss_fn.to(self.device)
                
                logger.info(f"训练器设备: {self.device}")
            
            def compute_losses(self, outputs, labels, masks=None):
                """简化版损失计算函数，增加详细的设备检查"""
                # 检查输入的设备
                logger.info("=== 检查compute_losses输入的设备 ===")
                if isinstance(outputs, tuple):
                    for i, out in enumerate(outputs):
                        check_device(out, f"outputs[{i}]")
                else:
                    check_device(outputs, "outputs")
                
                check_device(labels, "labels")
                check_device(masks, "masks")
                
                # 预处理逻辑
                mask_loss = None
                freq_loss = None
                
                # 处理不同的输出格式
                if isinstance(outputs, tuple):
                    # 多任务输出
                    if len(outputs) == 2:  # 例如: (mask_preds, class_outputs)
                        mask_preds, class_outputs = outputs
                        logger.info("识别为双输出格式: (mask_preds, class_outputs)")
                    elif len(outputs) >= 3:  # 例如: (mask_preds, class_outputs, features)
                        mask_preds, class_outputs, features = outputs[:3]
                        logger.info("识别为三输出格式: (mask_preds, class_outputs, features)")
                    else:
                        mask_preds = None
                        class_outputs = outputs[0]
                        logger.info("识别为单输出元组格式，取第一个元素作为class_outputs")
                else:
                    # 单任务输出（只有分类）
                    mask_preds = None
                    class_outputs = outputs
                    logger.info("识别为单输出张量格式")
                
                # 再次检查处理后的输出
                check_device(class_outputs, "处理后的class_outputs")
                check_device(mask_preds, "处理后的mask_preds")
                
                # 确保张量在同一设备上
                main_device = self.device
                logger.info(f"训练器主设备: {main_device}")
                
                logger.info("将所有输入移至同一设备...")
                try:
                    # 将输入张量明确移至指定设备
                    if not isinstance(labels, torch.Tensor):
                        logger.warning(f"labels不是张量，尝试转换: {type(labels)}")
                        labels = torch.tensor(labels, device=main_device)
                    else:
                        labels = labels.to(main_device)
                    
                    if masks is not None and isinstance(masks, torch.Tensor):
                        masks = masks.to(main_device)
                    
                    if isinstance(class_outputs, torch.Tensor):
                        class_outputs = class_outputs.to(main_device)
                    
                    if mask_preds is not None and isinstance(mask_preds, torch.Tensor):
                        mask_preds = mask_preds.to(main_device)
                    
                    # 再次检查设备
                    logger.info("转移后的设备:")
                    check_device(class_outputs, "class_outputs")
                    check_device(labels, "labels")
                    if masks is not None:
                        check_device(masks, "masks")
                    if mask_preds is not None:
                        check_device(mask_preds, "mask_preds")
                    
                except Exception as e:
                    logger.error(f"设备转移错误: {e}")
                    traceback.print_exc()
                
                # 计算分类损失
                try:
                    logger.info("\n尝试计算分类损失...")
                    logger.info(f"分类损失函数: {self.cls_loss_fn}")
                    
                    # 检查分类损失函数预期的输入格式和设备
                    check_device(class_outputs, "class_outputs最终")
                    check_device(labels, "labels最终")
                    
                    # 输出实际输入到loss函数中的形状
                    logger.info(f"class_outputs形状: {class_outputs.shape}")
                    logger.info(f"labels形状: {labels.shape}")
                    
                    # 尝试调用损失函数
                    cls_loss = self.cls_loss_fn(class_outputs, labels)
                    logger.info(f"分类损失计算成功: {cls_loss.item()}")
                    check_device(cls_loss, "cls_loss")
                except Exception as e:
                    logger.error(f"计算分类损失错误: {e}")
                    traceback.print_exc()
                    cls_loss = torch.tensor(0.0, device=main_device)
                
                # 如果有掩码预测和真实掩码，计算掩码损失
                if mask_preds is not None and masks is not None:
                    try:
                        logger.info("\n尝试计算掩码损失...")
                        
                        # 确保掩码形状匹配
                        logger.info(f"mask_preds形状: {mask_preds.shape}")
                        logger.info(f"masks形状: {masks.shape}")
                        
                        if mask_preds.shape != masks.shape:
                            logger.info("掩码形状不匹配，尝试插值...")
                            import torch.nn.functional as F
                            mask_preds = F.interpolate(mask_preds, size=masks.shape[2:], 
                                                    mode='bilinear', align_corners=False)
                            logger.info(f"插值后mask_preds形状: {mask_preds.shape}")
                        
                        # 应用掩码损失函数
                        mask_loss = self.mask_loss_fn(mask_preds, masks)
                        logger.info(f"掩码损失计算成功: {mask_loss.item()}")
                        check_device(mask_loss, "mask_loss")
                    except Exception as e:
                        logger.error(f"计算掩码损失错误: {e}")
                        traceback.print_exc()
                        mask_loss = torch.tensor(0.0, device=main_device)
                else:
                    mask_loss = torch.tensor(0.0, device=main_device)
                
                # 计算总损失
                logger.info("\n计算总损失...")
                total_loss = self.cls_weight * cls_loss
                
                if mask_loss is not None:
                    total_loss += self.mask_weight * mask_loss
                
                check_device(total_loss, "total_loss")
                logger.info(f"总损失计算成功: {total_loss.item()}")
                
                return total_loss, cls_loss, mask_loss, torch.tensor(0.0, device=main_device)
        
        # 创建简化训练器
        trainer = SimplifiedTrainer()
        
        # 生成测试数据
        logger.info("\n生成测试数据...")
        batch_size = 8
        num_classes = 2
        
        # 单输出场景
        dummy_outputs = torch.randn(batch_size, num_classes, device=device)
        dummy_labels = torch.randint(0, num_classes, (batch_size,), device=device)
        
        logger.info("\n测试单输出损失计算...")
        try:
            total_loss, cls_loss, mask_loss, freq_loss = trainer.compute_losses(dummy_outputs, dummy_labels)
            logger.info("单输出损失计算成功!")
        except Exception as e:
            logger.error(f"单输出损失计算失败: {e}")
            traceback.print_exc()
        
        # 双输出场景（掩码+分类）
        dummy_mask = torch.sigmoid(torch.randn(batch_size, 1, 64, 64, device=device))
        dummy_masks = torch.randint(0, 2, (batch_size, 1, 64, 64), dtype=torch.float, device=device)
        
        logger.info("\n测试双输出损失计算...")
        try:
            total_loss, cls_loss, mask_loss, freq_loss = trainer.compute_losses(
                (dummy_mask, dummy_outputs), dummy_labels, dummy_masks
            )
            logger.info("双输出损失计算成功!")
        except Exception as e:
            logger.error(f"双输出损失计算失败: {e}")
            traceback.print_exc()
        
        # 检查实际trainer.py中的compute_losses函数
        logger.info("\n检查实际trainer中的compute_losses函数...")
        try:
            logger.info("实际EnhancedTrainer.compute_losses函数代码:\n")
            import inspect
            compute_losses_code = inspect.getsource(EnhancedTrainer.compute_losses)
            logger.info(compute_losses_code)
            
            # 分析可能的设备不一致问题
            logger.info("\n分析可能的设备不一致问题:")
            logger.info("1. 检查是否所有张量都明确移至相同设备")
            logger.info("2. 检查是否处理了所有可能的异常情况")
            logger.info("3. 检查损失函数的实例化和使用方式")
            
            # 实际分析EnhancedTrainer中compute_losses的设备问题
            logger.info("\n实际分析EnhancedTrainer实例:")
            
            # 创建一个简单配置用于实例化EnhancedTrainer
            simple_config = {
                "OPTIMIZER": {"NAME": "adam", "LR": 0.0001},
                "MULTI_TASK": {"CLASSIFICATION": {"ENABLED": True}, 
                               "MASK": {"ENABLED": True}}
            }
            simple_config = easydict.EasyDict(simple_config)
            
            try:
                # 尝试直接创建EnhancedTrainer实例
                # 注意：这可能会失败，因为可能需要更多的参数
                logger.info("尝试创建EnhancedTrainer实例...")
                
                # 创建一个简单模型供测试
                class DummyModel(torch.nn.Module):
                    def __init__(self):
                        super().__init__()
                        self.conv = torch.nn.Conv2d(3, 64, 3, 1, 1)
                        self.fc = torch.nn.Linear(64, 2)
                        
                    def forward(self, x):
                        x = self.conv(x)
                        x = torch.mean(x, dim=(2, 3))
                        x = self.fc(x)
                        return x
                
                dummy_model = DummyModel().to(device)
                
                # 检查是否有create_trainer函数
                if hasattr(trainer_module, 'create_trainer'):
                    logger.info("使用create_trainer函数创建训练器...")
                    trainer_instance = trainer_module.create_trainer(
                        config=simple_config,
                        gpu_ids=[0] if torch.cuda.is_available() else [],
                        model_type='forensics',
                        mode='Both'
                    )
                else:
                    # 否则直接实例化
                    trainer_instance = EnhancedTrainer(
                        config=simple_config,
                        model=dummy_model,
                        device=device
                    )
                
                logger.info("成功创建训练器实例")
                
                # 检查训练器的compute_losses方法
                if hasattr(trainer_instance, 'compute_losses'):
                    logger.info("测试真实训练器的compute_losses方法...")
                    
                    # 创建测试数据
                    test_input = torch.randn(2, 3, 224, 224).to(device)
                    test_labels = torch.randint(0, 2, (2,)).to(device)
                    test_masks = torch.randint(0, 2, (2, 1, 224, 224), dtype=torch.float).to(device)
                    
                    # 先尝试前向传播
                    try:
                        logger.info("尝试前向传播...")
                        with torch.no_grad():
                            outputs = trainer_instance.model(test_input)
                        
                        logger.info(f"前向传播成功，输出类型: {type(outputs)}")
                        if isinstance(outputs, tuple):
                            logger.info(f"输出是一个元组，包含 {len(outputs)} 个元素")
                            for i, out in enumerate(outputs):
                                if torch.is_tensor(out):
                                    logger.info(f"输出[{i}] 形状: {out.shape}, 设备: {out.device}")
                        else:
                            logger.info(f"输出形状: {outputs.shape}, 设备: {outputs.device}")
                        
                        # 测试compute_losses
                        logger.info("测试真实compute_losses...")
                        losses = trainer_instance.compute_losses(outputs, test_labels, test_masks)
                        
                        if isinstance(losses, tuple):
                            for i, loss in enumerate(losses):
                                if torch.is_tensor(loss):
                                    logger.info(f"损失[{i}] 值: {loss.item()}, 设备: {loss.device}")
                                else:
                                    logger.info(f"损失[{i}] 不是张量: {loss}")
                    except Exception as e:
                        logger.error(f"测试真实compute_losses失败: {e}")
                        traceback.print_exc()
            except Exception as e:
                logger.error(f"创建EnhancedTrainer实例失败: {e}")
                traceback.print_exc()
            
        except Exception as e:
            logger.error(f"检查实际trainer.compute_losses失败: {e}")
            traceback.print_exc()
        
        logger.info("=== 设备不一致问题调试完成 ===")
        
    except Exception as e:
        logger.error(f"调试过程发生错误: {e}")
        traceback.print_exc()

if __name__ == "__main__":
    debug_loss_computation()
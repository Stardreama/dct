# 优化人脸伪造检测模型的文件修改路线图

为了全面提高模型性能，而不仅仅针对换脸类型的伪造，以下路线图按照实施顺序列出了需要修改的文件及其优化方向。这种综合优化将帮助模型应对各种类型的伪造技术。

---

## 1. 修改 `config.yaml`

### 修改内容:

- 更新学习率策略参数（添加余弦退火调度）
- 增加权重衰减控制参数
- 添加数据增强配置选项
- 配置多尺度训练参数
- 添加掩码监督权重参数
- 设置模型架构选择参数

### 优化目的:

- 提供更灵活的训练配置
- 支持不同的学习率策略
- 统一配置管理

---

## 2. 修改 `FaceDataset` 类（在 `train.py` 中）

### 修改内容:

- 增强数据增强策略（模拟压缩伪影、随机噪声、色彩变换）
- 优化掩码加载逻辑
- 添加更多图像变换（旋转、透视、剪切等）
- 实现混合增强策略
- 支持多尺度训练

### 优化目的:

- 提高模型泛化能力
- 增强对不同伪造类型的抵抗力
- 模拟真实场景中的各种图像失真

---

## 3. 修改 `trainer.py`

### 修改内容:

- 增强模型架构（添加注意力机制、特征融合模块）
- 改进损失函数（结合 Focal Loss、掩码损失）
- 优化优化器选择（采用 AdamW 而非 Adam）
- 增加学习率调度策略
- 实现混合精度训练
- 添加 EMA（指数移动平均）模型跟踪

### 优化目的:

- 提高模型对细微伪造痕迹的感知能力
- 改进模型对难例的学习效果
- 加速训练并提高稳定性

---

## 4. 修改 `network/models` 目录下的模型文件

### 修改内容:

- 整合先进的骨干网络（如 EfficientNet、RegNet）
- 添加多尺度特征融合模块
- 增强注意力机制设计
- 改进掩码生成模块
- 集成其他开源伪造检测模型的关键设计

### 优化目的:

- 增强特征提取能力
- 提高对各种伪造类型的检测精度
- 借鉴先进模型架构的优点

---

## 5. 修改 `utils.py`

### 修改内容:

- 增强模型评估工具（添加 AUC、精确率、召回率等指标）
- 添加按伪造类型分析性能的函数
- 实现模型预测可视化工具
- 添加混淆矩阵分析
- 增加错误案例分析功能

### 优化目的:

- 提供更全面的性能评估
- 深入了解模型在不同伪造类型上的表现
- 帮助发现改进方向

---

## 6. 修改 `train.py` 主训练流程

### 修改内容:

- 优化训练循环（添加验证策略）
- 实现模型集成训练
- 增加早停机制
- 添加模型检查点保存逻辑
- 实现学习率 warmup 策略
- 支持混合精度训练
- 添加训练过程可视化

### 优化目的:

- 提高训练效率和稳定性
- 防止过拟合
- 确保选择最优模型

---

## 7. 添加 `inference.py` 优化

### 修改内容:

- 改进预测流程
- 增强掩码可视化效果
- 添加模型集成预测功能
- 优化后处理逻辑
- 提高推理速度

### 优化目的:

- 提高模型实际使用时的性能
- 提供更直观的可视化结果
- 增强解释性

---

## 优先级实施顺序

1. **首先：** 修改 `config.yaml` 和 `FaceDataset`，建立增强的数据流水线。

   - ✅ **已完成**

   **完成成果：**

   - 在 `config.yaml` 添加了完整的数据增强配置模块：

     - 色彩变换参数（亮度、对比度、饱和度、色调）
     - 几何变换参数（旋转、裁剪、翻转）
     - 噪声和模糊效果控制
     - JPEG 压缩模拟参数
     - Cutout 遮挡配置

   - 增强了 `EnhancedFaceDataset` 类，实现以下功能：
     - 全面的数据增强管道（包括颜色调整、噪声添加、模糊效果等）
     - MixUp 数据混合策略，提高边界区分能力
     - 优化了掩码加载和处理逻辑，提取为通用方法
     - 实现数据集统计分析，便于了解数据分布
     - 支持配置驱动的增强策略控制
     - 模拟真实世界中的各种图像失真（压缩伪影、随机噪声等）
   - 更新了训练流程，确保正确使用增强的数据集和配置

2. **其次：** 优化 `network/models` 中的模型架构，结合先进的开源方法。

   - ✅ **已完成**

   **完成成果：**

   - **改进频域分析能力**：

     - 整合 F3-Net 的多频段分解技术，增强频域特征提取
     - 添加自适应频率过滤器，动态调整不同频段权重
     - 优化 DCT 变换效率和梯度传递
     - 对不同频率区域实现差异化处理

   - **增强特征融合机制**：

     - 实现动态特征融合模块，平衡 RGB 和频域特征的贡献
     - 添加多尺度特征聚合，提高对不同大小伪造区域的检测能力
     - 使用边界感知模块，专注于伪造边界检测
     - 增加特征重用机制，提高计算效率

   - **整合先进注意力机制**：

     - 添加 ECA 通道注意力，优化特征通道权重分配
     - 实现空间金字塔注意力，处理多尺度空间信息
     - 引入自注意力机制，捕捉全局上下文依赖关系
     - 集成 Vision Transformer 模块，增强长距离特征关联

   - **改进掩码生成模块**：

     - 重新设计掩码生成器，提高边界准确性
     - 使用多层特征融合进行掩码预测
     - 优化上采样策略，保持边界细节
     - 添加边界增强机制，改善掩码质量

   - **借鉴开源模型技术**：

     - 应用 Face X-ray 的边界检测思想，聚焦于图像拼接区域
     - 融合 SBI 的边界一致性原则
     - 整合 RECCE 的频谱分析技术
     - 采用 HRNet 的高分辨率特征维持能力

   - **提高模型效率**：

     - 优化特征提取路径，减少冗余计算
     - 精心设计跳跃连接，促进特征重用
     - 使用组卷积减少参数量
     - 改进前向传播路径，提高推理速度

   - **增强模型鲁棒性**：
     - 添加特征一致性约束，提高对未知伪造类型的抵抗力
     - 实现特征解耦，分离内容特征和伪造痕迹
     - 增加抗压缩和抗噪声机制
     - 优化梯度流，提高训练稳定性

   主要新增/修改的模块：

   - `EnhancedFilter` - 增强的频域过滤器
   - `ECAAttention` - 高效通道注意力
   - `SpatialPyramidAttention` - 多尺度空间注意力
   - `SelfAttention` - 全局上下文捕捉
   - `BoundaryAwareModule` - 边界感知模块
   - `DynamicFeatureFusion` - 动态特征融合
   - `TransformerBlock` - Transformer 自注意力块
   - `EnhancedFADHead` - 改进的频域分析头
   - `EnhancedF3Net` - 综合优化的检测网络

3. **再次：** 修改 `trainer.py`，实现改进的训练策略和损失函数。

   - ✅ **已完成**

   **完成成果：**

   - **改进损失函数体系**：
     - 实现 `FocalLoss` 聚焦难分类样本，提高对边界案例的敏感度
     - 设计 `EdgeAwareLoss` 增强掩码边缘准确性，使用 Sobel 算子检测和优化边界
     - 添加 `FrequencyConsistencyLoss` 促进频域特征一致性，提高对压缩伪影的鲁棒性
     - 实现损失函数加权组合策略，平衡不同学习目标

   - **优化训练策略**：
     - 引入混合精度训练（AMP），提升训练速度和内存效率
     - 实现 EMA（指数移动平均）模型追踪，增强模型稳定性和泛化能力
     - 添加梯度裁剪机制，提高训练过程稳定性
     - 设计更完善的模型保存和加载逻辑，支持训练断点续训

   - **增强优化器配置**：
     - 替换 Adam 为 AdamW 优化器，更好地处理权重衰减
     - 添加多种学习率调度策略（余弦退火、OneCycle等）
     - 实现学习率预热（Warmup）和自适应调整
     - 支持配置驱动的优化器参数设置

   - **完善验证机制**：
     - 增加专门的验证方法，支持模型性能评估
     - 使用 EMA 模型进行验证，获得更稳定的评估结果
     - 返回细粒度评估指标，包括分类准确率和掩码质量
     - 支持验证集上的损失分解监控

   - **增强训练监控**：
     - 实现详细的训练状态报告，包括各组件损失和学习率
     - 添加训练过程中的性能指标追踪
     - 支持训练过程中的早期错误检测
     - 优化日志输出格式，便于分析和调试

   - **提高代码质量**：
     - 重构 `Trainer` 类为 `EnhancedTrainer`，保留原版以向后兼容
     - 优化代码结构，提高可读性和可维护性
     - 添加详细文档和注释，说明各组件功能
     - 实现异常处理，提高训练过程健壮性

   主要新增/修改的类和方法：
   - `FocalLoss` - 关注难分类样本的损失函数
   - `EdgeAwareLoss` - 增强掩码边界准确性的损失函数
   - `FrequencyConsistencyLoss` - 促进频域特征一致性的损失函数
   - `ModelEMA` - 模型指数移动平均追踪器
   - `EnhancedTrainer` - 增强版训练器，整合所有优化

4. **然后：** 更新 `train.py` 主训练流程，加入验证和早停机制。

   - ✅ **已完成**

   **完成成果：**

   - **增强验证流程**：
     - 实现周期性模型验证，准确评估训练进度
     - 添加验证集性能监控，追踪关键指标
     - 支持使用EMA模型进行验证，获得更稳定的结果
     - 根据验证性能动态保存最佳模型

   - **实现早停机制**：
     - 开发 `EarlyStopping` 类，监控验证指标并优化训练长度
     - 支持可配置的早停参数（耐心值、最小改进阈值）
     - 增加早停状态报告，记录最佳模型出现的时期
     - 避免训练过拟合，节省计算资源

   - **优化训练循环**：
     - 改进批处理逻辑，增加训练稳定性
     - 实现渐进式训练策略，分阶段调整训练参数
     - 增强训练监控，提供实时性能指标
     - 优化GPU内存使用，支持更大的批量大小

   - **增强训练历史跟踪**：
     - 创建 `TrainingHistory` 类，全面记录训练过程
     - 实现训练曲线自动绘制（损失、准确率、学习率等）
     - 保存训练指标到CSV文件，便于后续分析
     - 生成训练过程摘要报告，便于比较不同实验

   - **提高训练可重复性**：
     - 添加随机种子控制，确保实验可重现
     - 实现训练配置自动保存，记录完整训练环境
     - 按时间戳组织实验，便于管理多个训练运行
     - 完善日志系统，记录关键训练事件

   - **增强错误处理**：
     - 完善异常捕获和日志记录，提高训练稳定性
     - 添加训练恢复机制，支持从中断点继续训练
     - 实现训练过程监控，检测异常训练状态
     - 优化资源释放，确保训练终止时正确释放GPU资源

   主要新增/修改的类和方法：
   - `EarlyStopping` - 早停监控器，防止过拟合
   - `TrainingHistory` - 训练历史记录器，提供可视化分析
   - `create_lr_scheduler` - 增强的学习率调度器工厂
   - `main` - 重构的主训练流程，整合所有优化功能

5. **再然后：** 增强 `utils.py` 和 `test.py`，提供更全面的评估工具。

   - ✅ **已完成**

   **完成成果：**

   - **增强评估指标**：
     - 实现 `ForensicsEvaluator` 评估类，提供全面的指标计算
     - 添加 ROC 曲线和 PR 曲线分析，突出模型在不同阈值下的表现
     - 集成混淆矩阵和详细分类报告，展示各类指标（精确率、召回率、F1值）
     - 添加掩码质量评估指标（IoU、Dice系数、像素准确率）
     - 实现按伪造类型分析性能，识别模型的优势和弱点

   - **改进测试数据集**：
     - 创建 `EnhancedTestDataset` 类，提供更丰富的样本信息
     - 自动提取和分类不同伪造类型
     - 实现数据集统计分析，了解测试集分布情况
     - 支持返回文件路径和伪造类型，便于深入分析

   - **添加模型集成功能**：
     - 设计 `ModelEnsemble` 类，支持多种集成策略
     - 实现平均集成、加权集成和最大值集成方法
     - 添加权重配置功能，调整不同模型的贡献度
     - 优化集成预测流程，提高准确率和鲁棒性

   - **可视化评估结果**：
     - 实现预测样本可视化，直观展示模型效果
     - 生成掩码叠加显示，突出伪造区域
     - 添加伪造类型性能对比图表，对比不同类型的检测难度
     - 创建精美的评估结果摘要报告，便于比较不同模型

   - **优化测试流程**：
     - 改进命令行参数解析，支持灵活的测试配置
     - 添加详细的日志记录，追踪测试过程
     - 实现推理速度评估，计算每秒处理帧数
     - 集成错误分析功能，识别模型的失误模式

   - **增强实用功能**：
     - 添加配置保存机制，记录完整测试环境
     - 实现时间戳命名的结果目录，便于多次测试比较
     - 提供详细的结果保存功能，导出数据用于后续分析
     - 添加灵活的阈值调整选项，优化决策边界

   主要新增/修改的类和方法：
   - `EnhancedTestDataset` - 增强的测试数据集类，提供更详细的样本信息
   - `ModelEnsemble` - 模型集成类，支持多种集成策略
   - `test_model` - 全面的模型评估函数，计算多种指标
   - `analyze_by_forgery_type` - 按伪造类型分析模型性能
   - `visualize_predictions` - 预测结果可视化工具
   - `save_type_analysis` - 伪造类型分析结果保存函数

6. **最后：** 修改`inference.py`,处理前端用户输入，判断是真实图像还是伪造图像。

---

 

```markdown
# 解决代码重复问题方案

## 1. 创建核心模块文件

建议创建一个名为 `core` 的新包目录，用于存放抽取出的共用功能。项目目录结构示例如下：

```

DCT_RGB_HRNet/
├── core/
│   ├── dataset.py         # 数据集相关功能
│   ├── augmentation.py    # 数据增强相关功能
│   ├── evaluation.py      # 评估相关功能
│   └── visualization.py   # 可视化相关功能
├── utils.py               # 保留基础工具函数
├── train.py
├── test.py
└── ...
```


## 2. 具体需要移动和重构的代码

### A. dataset.py - 统一数据集处理逻辑

将 `train.py` 中的 `EnhancedFaceDataset` 和 `test.py` 中的 `EnhancedTestDataset` 合并为一个基类，代码示例如下：

```python
import os
import torch
import numpy as np
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms

class BaseForensicDataset(Dataset):
    """
    基础伪造检测数据集
    提供共享的数据加载和掩码处理功能
    """
    def __init__(self, img_paths, dataset_type, 
                 aug_transform=None, tensor_transform=None, 
                 config=None, return_path=False):
        self.img_paths = img_paths
        self.dataset_type = dataset_type
        self.aug_transform = aug_transform
        self.tensor_transform = tensor_transform
        self.config = config
        self.return_path = return_path
        
        # 是否为训练模式
        self.is_train = dataset_type == 'train'
        
        # 灰度图像转换
        self.tensor_transform_gray = transforms.Compose([
            transforms.Resize(256),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5])])
            
        # 读取索引文件
        self.sample_list = []
        self.typepath = os.path.join(img_paths, f"{self.dataset_type}.txt")
        with open(self.typepath) as f:
            lines = f.readlines()
            for line in lines:
                self.sample_list.append(line.strip())
        
        # 分析数据集
        if hasattr(self, 'analyze_dataset'):
            self.analyze_dataset()
    
    def _load_mask_for_path(self, img_path, img_size):
        """
        通用掩码加载方法
        """
        try:
            if "fake" in img_path:
                # 提取文件路径组件
                path_components = img_path.split(os.path.sep)
                
                # 找出关键部分
                fake_idx = path_components.index("fake") if "fake" in path_components else -1
                
                if fake_idx != -1:
                    fake_type = path_components[fake_idx + 1]
                    video_id = path_components[fake_idx + 2]
                    frame_name = path_components[-1]
                    
                    # 构建掩码路径
                    base_dir = os.path.join(*path_components[:path_components.index("dataset") + 1])
                    mask_path = os.path.join(base_dir, "mask", fake_type, video_id, frame_name)
                    
                    if os.path.exists(mask_path):
                        return Image.open(mask_path).convert('L')
                    else:
                        # 掩码不存在，创建空白掩码
                        return Image.new('L', img_size, 0)
                else:
                    return Image.new('L', img_size, 0)
            else:
                # 真实图像，使用空白掩码
                return Image.new('L', img_size, 0)
        except Exception as e:
            print(f"警告: 无法加载掩码 {img_path}, {e}")
            # 创建空白掩码
            return Image.new('L', img_size, 0)
    
    def _analyze_dataset_stats(self):
        """通用数据集统计分析"""
        real_count = 0
        fake_count = 0
        fake_types = {}
        
        for item in self.sample_list:
            parts = item.split(' ')
            img_path = parts[0]
            label = int(parts[1])
            
            if label == 0:  # 真实图像
                real_count += 1
            else:  # 伪造图像
                fake_count += 1
                
                # 尝试识别伪造类型
                if "deepfakes" in img_path.lower():
                    fake_type = "deepfakes"
                elif "face2face" in img_path.lower() or "f2f" in img_path.lower():
                    fake_type = "face2face"
                elif "faceswap" in img_path.lower():
                    fake_type = "faceswap"
                elif "neuraltextures" in img_path.lower() or "nt" in img_path.lower():
                    fake_type = "neuraltextures"
                else:
                    fake_type = "unknown"
                
                fake_types[fake_type] = fake_types.get(fake_type, 0) + 1
        
        stats = {
            "total": len(self.sample_list),
            "real": real_count,
            "fake": fake_count,
            "forgery_types": fake_types
        }
        
        return stats
    
    def __len__(self):
        return len(self.sample_list)


class TrainingForensicDataset(BaseForensicDataset):
    """训练用的增强数据集"""
    def __init__(self, img_paths, type, dataset_type,
                 aug_transform=None, tensor_transform=None, config=None):
        super().__init__(img_paths, dataset_type, aug_transform, tensor_transform, config)
        self.type = type
        
        # 引入增强器
        self.augmenter = None
        if self.is_train and config is not None and hasattr(config, 'DATA_AUGMENTATION'):
            from core.augmentation import ForensicAugmenter
            self.augmenter = ForensicAugmenter(config)
    
    def analyze_dataset(self):
        """分析数据集统计信息并打印"""
        stats = self._analyze_dataset_stats()
        
        print(f"数据集 '{self.dataset_type}' 统计:")
        print(f"- 总样本数: {stats['total']}")
        print(f"- 真实图像: {stats['real']}")
        print(f"- 伪造图像: {stats['fake']}")
        print(f"- 伪造类型分布: {stats['forgery_types']}")
    
    def __getitem__(self, index):
        # 训练数据集的 __getitem__ 实现
        # (保留原有实现，调用增强器)
        # ...


class TestForensicDataset(BaseForensicDataset):
    """测试用的数据集"""
    def __init__(self, dataset_dir, split="test", transform=None, return_path=False):
        super().__init__(dataset_dir, split, None, transform, None, return_path)
        
        # 收集伪造类型
        self.forgery_types = set()
        self.samples = []
        
        for item in self.sample_list:
            img_path, label = item.split(' ')
            
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
        """测试集专用分析方法"""
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
    
    def __getitem__(self, index):
        # 测试数据集的 __getitem__ 实现
        # ...
```



### B. augmentation.py - 统一数据增强逻辑


将 `train.py` 中的所有增强方法提取到统一类中，代码示例如下：


```python
import torch
import numpy as np
from PIL import Image, ImageFilter, ImageEnhance, ImageOps
import cv2
from io import BytesIO
from random import random

class ForensicAugmenter:
    """
    伪造检测数据增强器
    提供各种数据增强方法
    """
    def __init__(self, config=None):
        self.config = config
        self.enabled = config is not None and hasattr(config, 'DATA_AUGMENTATION') and config.DATA_AUGMENTATION.ENABLED
    
    def apply_color_jitter(self, img):
        """应用颜色抖动"""
        if not self.enabled:
            return img
            
        cfg = self.config.DATA_AUGMENTATION.COLOR_JITTER
        
        if random() < 0.7:  # 70%概率应用颜色变换
            # 亮度调整
            enhancer = ImageEnhance.Brightness(img)
            img = enhancer.enhance(random() * cfg.BRIGHTNESS * 2 + (1 - cfg.BRIGHTNESS))
            
            # 对比度调整
            enhancer = ImageEnhance.Contrast(img)
            img = enhancer.enhance(random() * cfg.CONTRAST * 2 + (1 - cfg.CONTRAST))
            
            # 饱和度调整
            enhancer = ImageEnhance.Color(img)
            img = enhancer.enhance(random() * cfg.SATURATION * 2 + (1 - cfg.SATURATION))
            
            # 色调调整 (通过 PIL 的 ImageOps)
            if random() < 0.3:
                img = ImageOps.posterize(img, int(random() * 4) + 4)
                
        return img

    def apply_random_noise(self, img):
        """应用随机噪声"""
        # (相同逻辑...)

    def apply_blur(self, img):
        """应用模糊效果"""
        # (相同逻辑...)
        
    # 其余增强方法...

    def apply_mixup(self, img1, mask1, label1, get_another_sample_fn):
        """应用 MixUp 数据增强"""
        if not self.enabled or random() > 0.2:  # 20%概率应用
            return img1, mask1, label1
            
        # 随机选择另一个样本
        img2, mask2, label2 = get_another_sample_fn()
                
        # 确保尺寸一致
        img2 = img2.resize(img1.size, Image.BILINEAR)
        if mask2 is not None:
            mask2 = mask2.resize(mask1.size, Image.BILINEAR)
        else:
            mask2 = Image.new('L', mask1.size, 0)
            
        # 生成混合系数
        lam = random() * 0.4 + 0.3  # 0.3-0.7之间
        
        # 混合图像
        img1_np = np.array(img1).astype(np.float32)
        img2_np = np.array(img2).astype(np.float32)
        mixed_img = (lam * img1_np + (1 - lam) * img2_np).astype(np.uint8)
        
        # 混合掩码
        mask1_np = np.array(mask1).astype(np.float32)
        mask2_np = np.array(mask2).astype(np.float32)
        mixed_mask = (lam * mask1_np + (1 - lam) * mask2_np).astype(np.uint8)
        
        # 混合标签 (转换为 soft label)
        mixed_label = lam * label1 + (1 - lam) * label2
        
        return Image.fromarray(mixed_img), Image.fromarray(mixed_mask), mixed_label
```



### C. evaluation.py - 统一评估逻辑


将 `utils.py` 和 `test.py` 中重复的评估代码统一到一个工具类中，代码示例如下：


```python
import torch
import numpy as np
from sklearn.metrics import (
    roc_auc_score, precision_recall_curve, average_precision_score,
    roc_curve, confusion_matrix, classification_report, accuracy_score
)
from pathlib import Path
from tqdm import tqdm

class ModelEvaluator:
    """
    模型评估工具类
    统一评估逻辑
    """
    @staticmethod
    def evaluate_model(model, data_loader, device, threshold=0.5, return_predictions=False):
        """通用模型评估方法"""
        model.eval()
        
        # 初始化结果收集器
        all_preds = []
        all_probs = []
        all_labels = []
        all_paths = []
        all_types = []
        all_masks_pred = []
        all_masks_true = []
        
        with torch.no_grad():
            for batch in tqdm(data_loader, desc="评估中"):
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
        
        # 计算评估指标
        metrics = ModelEvaluator.calculate_metrics(all_labels, all_preds, all_probs)
        
        # 计算掩码评估指标
        mask_metrics = ModelEvaluator.evaluate_masks(all_masks_pred, all_masks_true)
        metrics['mask_metrics'] = mask_metrics
        
        # 添加伪造类型分析
        if len(all_types) > 0:
            type_analysis = ModelEvaluator.analyze_by_forgery_type(all_types, all_labels, all_preds, all_probs)
            metrics['type_analysis'] = type_analysis
        
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
    
    @staticmethod
    def calculate_metrics(labels, preds, probs):
        """计算各种分类评估指标"""
        accuracy = accuracy_score(labels, preds)
        auc_score = roc_auc_score(labels, probs)
        
        # ROC 曲线数据
        fpr, tpr, _ = roc_curve(labels, probs)
        
        # PR 曲线数据
        precision, recall, _ = precision_recall_curve(labels, probs)
        ap = average_precision_score(labels, probs)
        
        # 混淆矩阵
        cm = confusion_matrix(labels, preds)
        
        # 分类报告
        report = classification_report(labels, preds, output_dict=True)
        
        return {
            'accuracy': accuracy,
            'auc': auc_score,
            'ap': ap,
            'confusion_matrix': cm,
            'classification_report': report,
            'roc_data': {'fpr': fpr, 'tpr': tpr},
            'pr_data': {'precision': precision, 'recall': recall}
        }
    
    @staticmethod
    def evaluate_masks(pred_masks, true_masks):
        """评估掩码预测质量"""
        # (从 utils.py 移动...)
    
    @staticmethod  
    def analyze_by_forgery_type(types, labels, preds, probs):
        """按伪造类型分析性能"""
        # (从 test.py 移动...)
```



### D. visualization.py - 统一可视化逻辑


将各处散落的可视化代码统一到一个工具类中，代码示例如下：


```python
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import pandas as pd
from PIL import Image
import cv2
from pathlib import Path
import torch

class ForensicVisualizer:
    """
    可视化工具类
    提供各种可视化方法
    """
    @staticmethod
    def plot_training_curves(history, save_dir):
        """绘制训练曲线"""
        # (从 train.py 移动...)
    
    @staticmethod
    def plot_roc_curve(fpr, tpr, auc_score, save_path=None):
        """绘制 ROC 曲线"""
        # (从 utils.py 移动...)
    
    @staticmethod
    def plot_pr_curve(precision, recall, ap_score, save_path=None):
        """绘制 Precision-Recall 曲线"""
        # (从 utils.py 移动...)
    
    @staticmethod 
    def plot_confusion_matrix(cm, classes=['真实', '伪造'], save_path=None):
        """绘制混淆矩阵"""
        # (从 utils.py 移动...)
    
    @staticmethod
    def visualize_predictions(img_paths, mask_preds, true_labels, pred_labels, probabilities, save_dir=None):
        """可视化预测结果"""
        # (从 test.py 移动...)
        
    @staticmethod
    def plot_forgery_type_performance(type_analysis, save_dir=None):
        """绘制不同伪造类型的性能对比"""
        # (从 test.py 移动...)
```



## 3. 需要对现有文件的修改


在重构过程中，还需要对以下文件进行相应的修改：


- **修改 train.py**
- **修改 test.py**
- **修改 utils.py**

通过以上方案，可以有效解决代码重复问题，并将各模块功能清晰分离，从而提高代码的可维护性与复用性。


```

```

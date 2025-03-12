models.py 修改后的影响及其他文件调整分析
整合并优化后的 models.py 为整个项目提供了更强大、更灵活的模型架构，这会对其他文件产生一定影响。以下是对需要调整的文件分析：

1. train.py 需要的调整
   需要调整，主要是以下几点：

导入更新：

需要从新的 models.py 中导入 create_model 函数，而不是直接使用 Trainer 或 EnhancedTrainer
可能需要调整损失函数的计算方式，以适应新模型返回的多任务输出
训练流程调整：

处理 RGB 和 DCT 双输入的数据加载
添加对应于新增注意力机制和频域特征的损失项
更新训练评估指标以包含掩码预测准确率
配置适配：

需要更新配置文件，添加新的超参数如使用哪种注意力模块、DCT 特征维度等
可能需要调整学习率调度，因为增强的模型可能需要不同的优化策略 2. test.py 需要的调整
需要调整，主要是以下几点：

模型加载方式：

使用 create_model 工厂函数代替直接实例化模型
处理新模型架构的权重加载
评估流程：

添加对注意力图和中间特征的可视化支持
利用 extract_features 和 get_attention_maps 等新方法进行深入分析
更新评估指标计算方法，以包含掩码评估
预测展示：

添加新的可视化函数，以展示边界检测和频域关注区域 3. trainer.py 需要的调整
需要较大调整：

创建新的训练器类：

实现适配新 DeepForensicsNet 和 EnhancedF3Net 的训练器
添加对 DCT 特征的处理逻辑
损失函数扩展：

添加适合边界检测的损失函数
可能增加特征对比损失，以增强真假区分能力
优化策略更新：

可能需要分层学习率策略，让注意力模块有更高的学习率
实现渐进式训练策略，先训练主干网络，再微调注意力模块 4. transform.py 的使用调整
已经更新，但需要适配使用：

数据加载器更新：
需要在 dataset.py 中使用新的转换函数
特别是 get_complete_transform 和 get_dual_transform 5. config.yaml 配置文件更新
需要添加新的配置项：

模型配置：

添加 MODEL.TYPE 选项 ('enhanced', 'f3net', 'forensics')
添加注意力模块配置项如 MODEL.ATTENTION.TYPE
训练配置：

添加 DCT 转换参数
添加多任务损失的权重配置 6. 数据加载逻辑调整
需要调整 dataset.py：

数据加载方式：
支持同时加载图像和掩码
支持双输入格式 (RGB+DCT)
使用新的数据增强策略
文件修改依赖顺序分析

根据整个系统的依赖关系，以下是需要修改的文件，按照自底向上的依赖顺序排列：

1. 基础层文件
   network/transform.py（已完成）

实现了面向伪造检测的数据增强
提供了各种图像变换类和函数
network/dct_transform.py（已完成）

实现了 DCT 变换和频域特征提取
提供了多尺度频率分析功能
attention/ForensicAttentionFusion.py（已完成）

实现了多种注意力机制
提供了特征融合函数
network/enhanced_hrnet.py（已完成）

基于 HRNet 的增强版实现
已适配 DCT 输入和多任务输出 2. 模型定义层
models.py（已完成）
整合了所有组件
提供了多种模型架构和工厂函数 3. 配置系统层
config.yaml（已修改）
添加新模型类型和组件配置选项
更新训练参数和损失权重设置 4. 数据处理层
core/dataset.py（需要修改）

更新数据加载逻辑，支持掩码和 DCT 双输入
使用新的 transform.py 中的转换函数
添加新的数据集类支持边界标注
core/augmentation.py（可选修改）

可能需要调整以配合新的 transform.py
确保增强策略与模型期望一致 5. 训练评估层
trainer.py（需要修改）

创建适配新模型的训练器类
实现多任务损失函数
添加边界检测和频域特征的评估指标
core/evaluation.py（需要修改）

添加新的评估指标（边界 IoU、频域准确率）
更新模型评估方法以处理多任务输出
core/visualization.py（需要修改）

添加对注意力图的可视化
添加边界检测结果的可视化
实现频域特征的可视化方法 6. 应用入口层
train.py（需要修改）

使用 create_model 函数替代直接实例化
更新训练流程处理双输入
适配多任务损失计算
test.py（需要修改）

使用 create_model 函数加载模型
扩展评估逻辑以支持边界检测
添加多种可视化选项
修改优先级建议
基于依赖关系和影响范围，建议按以下优先级顺序进行修改：

最高优先级：config.yaml 和 dataset.py

这些是其他文件依赖的基础配置和数据提供者
高优先级：trainer.py 和 evaluation.py

训练和评估流程需要适配新模型架构
中等优先级：core/visualization.py 和 augmentation.py

视觉化和数据增强对整体功能有支持作用
标准优先级：train.py 和 test.py

作为应用入口，需要在其他组件就绪后调整

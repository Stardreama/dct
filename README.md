# DCT-RGB-HRNet: 深度学习视频伪造检测框架

## 项目介绍

DCT-RGB-HRNet 是一个基于深度学习的视频伪造检测框架，集成了频域特征分析和空域特征分析，结合注意力机制增强检测性能。该框架在 FaceForensics++ 数据集上进行了测试，能够有效检测多种类型的视频伪造，包括 Deepfakes、Face2Face、FaceSwap、NeuralTextures 和 FaceShifter 等。

## 功能特点

- 集成 DCT 频域分析和 RGB 空域分析的多路径架构
- 多级注意力机制增强特征表示
- 掩码预测支持（伪造区域定位）
- 支持多种预训练主干网络 (Xception, HRNet)
- 简化的训练和推理流程
- 易于扩展的模块化设计

## 环境配置

### 必要依赖

```bash
pip install torch torchvision numpy scikit-learn pillow opencv-python matplotlib seaborn tqdm easydict pyyaml
```


推荐环境：


- Python 3.8+
- PyTorch 1.10+
- CUDA 11.3+ (如使用 GPU)

### 数据集准备


本项目支持 FaceForensics++ 数据集格式。处理后的数据结构应如下所示：


```pgsql
FaceForensics++
├── original_sequences
│   ├── youtube
│   │   └── c23
│   │       └── frames
│   │           ├── 000
│   │           │   ├── 000.png
│   │           │   ├── 012.png
│   │           │   └── *.png
│   │           └── 001
│   └── actors
│       └── c23
│           └── frames
│               ├── 01_exit_phone_room
│               │   ├── 009.png
│               │   ├── 058.png
│               │   └── *.png
│               └── 01_hugging_happy
├── manipulated_sequences
│   ├── Deepfakes
│   │   └── c23
│   │       ├── frames
│   │       └── masks
│   ├── Face2Face
│   ├── FaceSwap
│   ├── NeuralTextures
│   ├── FaceShifter
│   └── DeepFakeDetection
├── train.json
├── val.json
└── test.json
```


其中 `train.json`、`val.json` 和 `test.json` 包含了训练、验证和测试集的文件路径和标签信息。


### 配置文件


在 `config.yaml` 文件中设置训练和模型参数：


```yaml
# 基本配置
GPUS: 1
LOG_DIR: "log/"
OUTPUT_DIR: "output/"
WORKERS: 8
PRINT_FREQ: 100
SEED: 42

# 数据集配置
DATASET:
  TYPE: "FaceForensics"
  TRAIN_PATH: "path/to/FaceForensics++/train.json"
  VAL_PATH: "path/to/FaceForensics++/val.json"
  TEST_PATH: "path/to/FaceForensics++/test.json"
  BATCH_SIZE: 32
  NUM_WORKERS: 4

# 模型配置
MODEL_CONFIG:
  TYPE: "forensics"  # 可选: "enhanced", "f3net", "forensics"
  MODE: "Both"       # 可选: "RGB", "FAD", "Both"
  IMG_SIZE: 256
  NUM_CLASSES: 2
  PRETRAINED:
    HRNET: "pretrained/hrnetv2_w48_imagenet_pretrained.pth"
    XCEPTION: "pretrained/xception-b5690688.pth"

# 训练配置
EPOCHS: 50
OPTIMIZER:
  NAME: "adam"
  LR: 0.0001
  WEIGHT_DECAY: 0.0001

# 学习率调度
LR_SCHEDULER:
  NAME: "cosine"
  COSINE:
    T_MAX: 50
    ETA_MIN: 0.00001
  WARMUP:
    ENABLED: true
    EPOCHS: 5
    MULTIPLIER: 1.0
```


## 训练模型


### 基本训练命令


```bash
python train.py --config config.yaml
```


### 自定义训练参数


```bash
python train.py --config config.yaml --model_type forensics --mode Both --batch_size 32 --epochs 50 --lr 0.0001
```


### 恢复训练


```bash
python train.py --config config.yaml --resume path/to/checkpoint.pth
```


### 使用混合精度训练（提升速度）


```bash
python train.py --config config.yaml --mixed_precision
```


### 评估模型


```bash
python train.py --eval_only --resume path/to/model_best.pth --config config.yaml
```


### 推理与可视化


```bash
python inference.py --model_path path/to/model_best.pth --input path/to/image.jpg --output output/visualization
```


### Web 界面


启动基于 Web 的推理界面：


```bash
python app.py --model_path path/to/model_best.pth
```


然后在浏览器中访问 [http://localhost:5000](http://localhost:5000/) 即可使用交互式界面进行视频伪造检测。


## 项目结构


```csharp
DCT_RGB_HRNet/
├── config.yaml          # 配置文件
├── train.py             # 训练脚本
├── inference.py         # 推理脚本
├── app.py               # Web 应用入口
├── models.py            # 模型定义
├── trainer.py           # 训练器实现
├── utils.py             # 通用工具函数
├── core/                # 核心功能模块
│   ├── dataset.py       # 数据集加载
│   ├── augmentation.py  # 数据增强
│   ├── evaluation.py    # 评估工具
│   └── visualization.py # 可视化工具
├── network/             # 网络组件
│   ├── transform.py     # 图像变换
│   ├── attention.py     # 注意力模块
│   └── dct_transform.py # DCT变换
├── attention/           # 注意力机制实现
│   └── ForensicAttentionFusion.py
├── templates/           # Web界面模板
└── static/              # Web界面静态文件
```


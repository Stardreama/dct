import torch
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
osenvs = len(os.environ['CUDA_VISIBLE_DEVICES'].split(','))
import torch.nn as nn
import torchvision
import yaml
import easydict
from network.transform import mesonet_data_transforms
from trainer import Trainer
from tqdm import tqdm
from sklearn.metrics import roc_auc_score, confusion_matrix, roc_curve
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from datetime import datetime
from PIL import Image

# 创建结果目录
RESULTS_DIR = './test_results'
os.makedirs(RESULTS_DIR, exist_ok=True)

# 自定义测试数据集类，使用索引文件
class IndexFileDataset(torch.utils.data.Dataset):
    def __init__(self, dataset_dir, split="test", transform=None):
        self.transform = transform
        self.samples = []
        
        # 读取索引文件
        index_file = os.path.join(dataset_dir, f"{split}.txt")
        with open(index_file, "r") as f:
            for line in f:
                img_path, label = line.strip().split()
                self.samples.append((img_path, int(label)))
    
    def __getitem__(self, index):
        img_path, label = self.samples[index]
        img = Image.open(img_path).convert('RGB')
        
        if self.transform:
            img = self.transform(img)
            
        return img, label
    
    def __len__(self):
        return len(self.samples)

def plot_confusion_matrix(y_true, y_pred, save_path):
    """绘制并保存混淆矩阵"""
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
               xticklabels=['Real', 'Fake'],
               yticklabels=['Real', 'Fake'])
    plt.xlabel('预测标签')
    plt.ylabel('真实标签')
    plt.title('混淆矩阵')
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"混淆矩阵已保存至: {save_path}")

def plot_roc_curve(y_true, y_score, save_path, auc_value):
    """绘制并保存ROC曲线"""
    fpr, tpr, _ = roc_curve(y_true, y_score)
    plt.figure(figsize=(10, 8))
    plt.plot(fpr, tpr, lw=2, label=f'ROC curve (AUC = {auc_value:.4f})')
    plt.plot([0, 1], [0, 1], 'k--', lw=2)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('假阳性率 (False Positive Rate)')
    plt.ylabel('真阳性率 (True Positive Rate)')
    plt.title('接收者操作特征曲线 (ROC)')
    plt.legend(loc="lower right")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"ROC曲线已保存至: {save_path}")

def calculate_metrics(y_true, y_pred, y_score):
    """计算各种评估指标"""
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
    
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1
    }

def main(config):
    # 创建一个唯一的时间戳目录来存储结果
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    result_dir = os.path.join(RESULTS_DIR, timestamp)
    os.makedirs(result_dir, exist_ok=True)
    print(f"测试结果将保存在: {result_dir}")
    
    test_path = './dataset'
    batch_size = 8
    mode = 'FAD'
    pretrained_path = './pretrained/xception-b5690688.pth'
    ckpt_dir = './checkpoints'
    ckpt_name = 'FAD_RGB_F2Fc0'
    torch.backends.cudnn.benchmark = True
    gpu_ids = [*range(osenvs)]
    model_path = os.path.join(ckpt_dir, ckpt_name, 'best.pkl')

    # 使用自定义数据集
    test_dataset = IndexFileDataset(test_path, split="test", transform=mesonet_data_transforms['val'])
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False, 
                                             drop_last=False, num_workers=8)
    test_dataset_size = len(test_dataset)
    corrects = 0
    acc = 0
    prod_all = []
    label_all = []
    pred_all = []
    
    # 记录测试时间
    start_time = datetime.now()
    
    print(f"加载模型: {model_path}")
    model = Trainer(config, gpu_ids, mode, pretrained_path)
    model.load_state_dict(torch.load(model_path))
    model = model.cuda()
    model.eval()
    
    print("开始测试...")
    with torch.no_grad():
        for (image, labels) in tqdm(test_loader, desc="测试进度"):
            image = image.cuda()
            labels = labels.cuda()
            mask, outputs = model(image)
            _, preds = torch.max(outputs.data, 1)
            corrects += torch.sum(preds == labels.data).to(torch.float32)
            
            # 保存预测结果
            m = nn.Sigmoid()
            output = m(outputs)
            prod_all.extend(output[:, 1].cpu().numpy())
            label_all.extend(labels.cpu().numpy())
            pred_all.extend(preds.cpu().numpy())
            
        # 计算准确率和AUC
        acc = (corrects / test_dataset_size).item()
        auc = roc_auc_score(label_all, prod_all)
        
        # 打印结果
        print('=================== 测试结果 ===================')
        print(f'测试数据集大小: {test_dataset_size}')
        print(f'准确率 (Accuracy): {acc:.4f}')
        print(f'AUC: {auc:.4f}')
        
        # 计算其他指标
        metrics = calculate_metrics(label_all, pred_all, prod_all)
        print(f'精确度 (Precision): {metrics["precision"]:.4f}')
        print(f'召回率 (Recall): {metrics["recall"]:.4f}')
        print(f'F1分数: {metrics["f1_score"]:.4f}')
        
        # 保存结果到文本文件
        result_file = os.path.join(result_dir, 'test_results.txt')
        with open(result_file, 'w') as f:
            f.write('=================== 测试结果 ===================\n')
            f.write(f'测试时间: {start_time}\n')
            f.write(f'模型路径: {model_path}\n')
            f.write(f'测试数据集: {test_path}\n')
            f.write(f'测试数据集大小: {test_dataset_size}\n\n')
            f.write(f'准确率 (Accuracy): {acc:.4f}\n')
            f.write(f'AUC: {auc:.4f}\n')
            f.write(f'精确度 (Precision): {metrics["precision"]:.4f}\n')
            f.write(f'召回率 (Recall): {metrics["recall"]:.4f}\n')
            f.write(f'F1分数: {metrics["f1_score"]:.4f}\n')
            
            # 添加真实/伪造的分类准确率
            real_indices = [i for i, label in enumerate(label_all) if label == 0]
            fake_indices = [i for i, label in enumerate(label_all) if label == 1]
            
            real_correct = sum(1 for i in real_indices if pred_all[i] == label_all[i])
            fake_correct = sum(1 for i in fake_indices if pred_all[i] == label_all[i])
            
            real_acc = real_correct / len(real_indices) if real_indices else 0
            fake_acc = fake_correct / len(fake_indices) if fake_indices else 0
            
            f.write(f'\n真实图像准确率: {real_acc:.4f} ({real_correct}/{len(real_indices)})\n')
            f.write(f'伪造图像准确率: {fake_acc:.4f} ({fake_correct}/{len(fake_indices)})\n')
            
        print(f'结果已保存至: {result_file}')
        
        # 绘制混淆矩阵
        cm_path = os.path.join(result_dir, 'confusion_matrix.png')
        plot_confusion_matrix(label_all, pred_all, cm_path)
        
        # 绘制ROC曲线
        roc_path = os.path.join(result_dir, 'roc_curve.png')
        plot_roc_curve(label_all, prod_all, roc_path, auc)
        
        # 输出测试耗时
        end_time = datetime.now()
        time_elapsed = end_time - start_time
        print(f'测试完成! 耗时: {time_elapsed.total_seconds():.2f}秒')
        
        # 在结果文件中添加测试耗时
        with open(result_file, 'a') as f:
            f.write(f'\n测试耗时: {time_elapsed.total_seconds():.2f}秒\n')
        
    return {
        'accuracy': acc,
        'auc': auc,
        'predictions': pred_all,
        'true_labels': label_all,
        'probabilities': prod_all,
        'result_dir': result_dir,
        'metrics': metrics
    }

if __name__ == '__main__':
    with open("./config.yaml", 'r') as stream:
        config = yaml.safe_load(stream)
    config = easydict.EasyDict(config)
    results = main(config)
    
    print("\n测试完成! 结果已保存到目录:", results['result_dir'])
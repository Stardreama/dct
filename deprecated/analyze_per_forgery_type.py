import os
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from tqdm import tqdm
from newTest import IndexFileDataset, calculate_metrics
from trainer import Trainer
from sklearn.metrics import roc_auc_score
import yaml
import easydict
from datetime import datetime
from network.transform import mesonet_data_transforms

def test_forgery_type(config, forgery_type, split_name):
    """为特定伪造类型运行测试"""
    # 创建结果目录
    result_dir = os.path.join('./test_results', f'{forgery_type}_{datetime.now().strftime("%Y%m%d_%H%M%S")}')
    os.makedirs(result_dir, exist_ok=True)
    
    test_path = "./dataset"  # 指向您的数据集路径
    batch_size = 8
    mode = 'FAD'
    pretrained_path = './pretrained/xception-b5690688.pth'
    ckpt_dir = './checkpoints'
    ckpt_name = 'FAD_RGB_F2Fc0'
    torch.backends.cudnn.benchmark = True
    
    osenvs = len(os.environ.get('CUDA_VISIBLE_DEVICES', '0').split(','))
    gpu_ids = [*range(osenvs)]
    model_path = os.path.join(ckpt_dir, ckpt_name, 'best.pkl')

    # 使用指定的split加载测试数据集
    test_dataset = IndexFileDataset(test_path, split=split_name, transform=mesonet_data_transforms['val'])
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False, 
                                            drop_last=False, num_workers=8)
    test_dataset_size = len(test_dataset)
    corrects = 0
    prod_all = []
    label_all = []
    pred_all = []
    
    print(f"加载模型: {model_path} (用于 {forgery_type} 测试)")
    model = Trainer(config, gpu_ids, mode, pretrained_path)
    model.load_state_dict(torch.load(model_path))
    model = model.cuda()
    model.eval()
    
    print(f"开始 {forgery_type} 测试 (数据集大小: {test_dataset_size})...")
    with torch.no_grad():
        for (image, labels) in tqdm(test_loader, desc=f"{forgery_type} 测试"):
            image = image.cuda()
            labels = labels.cuda()
            mask, outputs = model(image)
            _, preds = torch.max(outputs.data, 1)
            corrects += torch.sum(preds == labels.data).to(torch.float32)
            
            # 收集预测结果
            m = nn.Sigmoid()
            output = m(outputs)
            prod_all.extend(output[:, 1].cpu().numpy())
            label_all.extend(labels.cpu().numpy())
            pred_all.extend(preds.cpu().numpy())
    
    # 计算指标
    acc = (corrects / test_dataset_size).item()
    auc = roc_auc_score(label_all, prod_all)
    metrics = calculate_metrics(label_all, pred_all, prod_all)
    
    # 保存结果
    result_file = os.path.join(result_dir, f'{forgery_type}_results.txt')
    with open(result_file, 'w') as f:
        f.write(f'=================== {forgery_type} 测试结果 ===================\n')
        f.write(f'测试时间: {datetime.now()}\n')
        f.write(f'模型路径: {model_path}\n')
        f.write(f'测试数据集: {test_path} ({split_name}.txt)\n')
        f.write(f'测试数据集大小: {test_dataset_size}\n\n')
        f.write(f'准确率 (Accuracy): {acc:.4f}\n')
        f.write(f'AUC: {auc:.4f}\n')
        f.write(f'精确度 (Precision): {metrics["precision"]:.4f}\n')
        f.write(f'召回率 (Recall): {metrics["recall"]:.4f}\n')
        f.write(f'F1分数: {metrics["f1_score"]:.4f}\n')

    print(f"{forgery_type} 测试完成:")
    print(f'准确率: {acc:.4f}, AUC: {auc:.4f}, F1: {metrics["f1_score"]:.4f}')
    
    return {
        'accuracy': acc,
        'auc': auc,
        'predictions': pred_all,
        'true_labels': label_all,
        'probabilities': prod_all,
        'metrics': metrics,
        'result_dir': result_dir
    }

def analyze_per_forgery_type(config):
    """分析模型在不同伪造类型上的表现"""
    # 伪造类型列表
    forgery_types = ['Deepfakes', 'Face2Face', 'FaceSwap', 'NeuralTextures']
    
    results = {}
    
    for forgery_type in forgery_types:
        print(f"\n===== 分析 {forgery_type} 类型 =====")
        
        # 创建该类型的测试索引文件
        temp_file = create_forgery_type_test_file('./dataset', forgery_type, f"test_{forgery_type.lower()}.txt")
        
        # 使用自定义函数进行测试
        split_name = os.path.basename(temp_file).split('.')[0]
        forgery_results = test_forgery_type(config, forgery_type, split_name)
        
        # 保存结果
        results[forgery_type] = {
            'accuracy': forgery_results['accuracy'],
            'auc': forgery_results['auc'],
            'precision': forgery_results['metrics']['precision'],
            'recall': forgery_results['metrics']['recall'],
            'f1': forgery_results['metrics']['f1_score']
        }
    
    # 创建比较表格
    df = pd.DataFrame(results).T
    df.reset_index(inplace=True)
    df.rename(columns={'index': 'Forgery Type'}, inplace=True)
    
    # 保存到CSV
    csv_path = './test_results/forgery_type_comparison.csv'
    df.to_csv(csv_path, index=False)
    print(f"各伪造类型比较结果已保存至: {csv_path}")
    
    # 绘制条形图
    plt.figure(figsize=(12, 8))
    x = df['Forgery Type']
    width = 0.15
    multiplier = 0
    
    for metric in ['accuracy', 'auc', 'precision', 'recall', 'f1']:
        offset = width * multiplier
        rects = plt.bar(x + offset, df[metric], width, label=metric.capitalize())
        multiplier += 1
    
    plt.xlabel('伪造类型')
    plt.ylabel('得分')
    plt.title('不同伪造类型的检测性能')
    plt.legend(loc='lower right')
    plt.tight_layout()
    
    # 保存图表
    plt.savefig('./test_results/forgery_type_comparison.png')
    plt.close()
    print("比较图表已保存至: ./test_results/forgery_type_comparison.png")
    
    return df

def create_forgery_type_test_file(dataset_dir, forgery_type, output_filename):
    """创建特定伪造类型的测试文件"""
    # 读取原始测试文件
    samples = []
    real_count = 0
    fake_count = 0
    
    with open(os.path.join(dataset_dir, "test.txt"), "r") as f:
        for line in f:
            img_path, label = line.strip().split()
            label = int(label)
            
            # 添加所有真实样本
            if label == 0:
                samples.append(line.strip())
                real_count += 1
            # 只添加指定类型的伪造样本
            elif forgery_type.lower() in img_path.lower():
                samples.append(line.strip())
                fake_count += 1
    
    # 写入新的索引文件
    output_path = os.path.join(dataset_dir, output_filename)
    with open(output_path, "w") as f:
        for sample in samples:
            f.write(f"{sample}\n")
    
    print(f"创建了针对 {forgery_type} 的测试文件，包含 {len(samples)} 个样本 (真实: {real_count}, 伪造: {fake_count})")
    return output_path

if __name__ == "__main__":
    with open("./config.yaml", 'r') as stream:
        config = yaml.safe_load(stream)
    config = easydict.EasyDict(config)
    
    results_df = analyze_per_forgery_type(config)
    print("\n各伪造类型性能比较:")
    print(results_df)
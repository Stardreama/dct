import os
import torch
import yaml
import easydict
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from PIL import Image

class FaceDataset(Dataset):
    def __init__(self, img_paths, dataset_type, transform=None):
        self.img_paths = img_paths
        self.sample_list = list()
        self.dataset_type = dataset_type
        self.transform = transform
        
        # 读取索引文件
        self.typepath = os.path.join(img_paths, f"{self.dataset_type}.txt")
        with open(self.typepath) as f:
            lines = f.readlines()
            for line in lines:
                self.sample_list.append(line.strip())

    def __getitem__(self, index):
        item = self.sample_list[index]
        parts = item.split(' ')
        img_path = parts[0]
        label = int(parts[1])
        
        img = Image.open(img_path).convert('RGB')
        
        if self.transform:
            img = self.transform(img)
            
        return img, label

    def __len__(self):
        return len(self.sample_list)

def test_dataloader():
    # 定义数据变换
    tensor_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])
    
    # 创建数据集
    test_dataset = FaceDataset(
        img_paths="D:/project/DCT_RGB_HRNet/dataset",
        dataset_type='test',
        transform=tensor_transform
    )
    
    # 打印数据集大小
    print(f"数据集包含 {len(test_dataset)} 个样本")
    
    # 创建数据加载器
    batch_size = 4
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    
    # 测试数据加载
    for i, (images, labels) in enumerate(test_loader):
        print(f"批次 {i+1}: 图像形状 {images.shape}, 标签 {labels}")
        if i >= 2:  # 只显示前3批
            break
    
    print("数据加载器测试完成")
    return True

if __name__ == "__main__":
    test_dataloader()
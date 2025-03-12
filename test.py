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
from sklearn.metrics import roc_auc_score
from PIL import Image

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

def main(config):
    test_path = "D:/project/DCT_RGB_HRNet/dataset"  # 您的数据集路径
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
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=True, drop_last=False, num_workers=8)
    test_dataset_size = len(test_dataset)
    corrects = 0
    acc = 0
    prod_all = []
    label_all = []
    model = Trainer(config, gpu_ids, mode, pretrained_path)
    model.load_state_dict(torch.load(model_path))
    model = model.cuda()
    model.eval()
    with torch.no_grad():
        for (image, labels) in tqdm(test_loader):
            image = image.cuda()
            labels = labels.cuda()
            mask,outputs = model(image)
            _, preds = torch.max(outputs.data, 1)
            corrects += torch.sum(preds == labels.data).to(torch.float32)
            m = nn.Sigmoid()
            output = m(outputs)
            prod_all.extend(output[:, 1].cpu().numpy())
            label_all.extend(labels.cpu().numpy())
            # print('Iteration Acc {:.4f}'.format(torch.sum(preds == labels.data).to(torch.float32)/batch_size))
        acc = corrects / test_dataset_size
        auc = roc_auc_score(label_all, prod_all)
        print('Test Acc: {:.4f}, Test AUC: {:.4f}'.format(acc, auc))


if __name__ == '__main__':
    with open("./config.yaml", 'r') as stream:
        config = yaml.safe_load(stream)
    config = easydict.EasyDict(config)
    main(config)

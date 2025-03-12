import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
osenvs = len(os.environ['CUDA_VISIBLE_DEVICES'].split(','))
import time
import torch
import torch.nn as nn
import torchvision
import yaml
import easydict
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from tqdm import tqdm
from utils import evaluate, setup_logger
from trainer import Trainer
import numpy as np
import random
from random import random
from network.transform import mesonet_data_transforms

# 修改 FaceDataset 类
class FaceDataset(Dataset):
    def __init__(self, img_paths, type, dataset_type,
                 aug_transform=None,
                 tensor_transform=None):
        self.img_paths = img_paths
        self.sample_list = list()
        self.dataset_type = dataset_type
        self.type = type
        self.aug_transform = aug_transform
        self.tensor_transform = tensor_transform

        self.tensor_transform_gray = transforms.Compose([
            transforms.Resize(256),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5])])

        # 读取索引文件
        self.typepath = os.path.join(img_paths, f"{self.dataset_type}.txt")
        f = open(self.typepath)
        lines = f.readlines()
        for line in lines:
            self.sample_list.append(line.strip())
        f.close()

    def __getitem__(self, index):
        item = self.sample_list[index]
        parts = item.split(' ')
        img_path = parts[0]
        label = int(parts[1])
        
        try:
            img = Image.open(img_path).convert('RGB')
        except Exception as e:
            print(f"错误: 无法加载图像 {img_path}, {e}")
            # 返回替代图像
            return self.__getitem__((index + 1) % len(self.sample_list))
        
        # 从图像路径推导掩码路径
        try:
            if "fake" in img_path:
                # 提取文件路径组件
                path_components = img_path.split(os.path.sep)
                
                # 找出关键部分
                fake_idx = path_components.index("fake")
                fake_type = path_components[fake_idx + 1]
                video_id = path_components[fake_idx + 2]
                frame_name = path_components[-1]
                
                # 构建掩码路径
                base_dir = os.path.join(*path_components[:path_components.index("dataset") + 1])
                mask_path = os.path.join(base_dir, "mask", fake_type, video_id, frame_name)
                
                if os.path.exists(mask_path):
                    img_mask = Image.open(mask_path).convert('L')
                else:
                    # 掩码不存在，创建空白掩码
                    img_mask = Image.new('L', img.size, 0)
            else:
                # 真实图像，使用空白掩码
                img_mask = Image.new('L', img.size, 0)
        except Exception as e:
            print(f"警告: 无法加载掩码 {img_path}, {e}")
            # 创建空白掩码
            img_mask = Image.new('L', img.size, 0)

        # 数据增强
        if random() < 0.3:
            img = img.transpose(Image.FLIP_LEFT_RIGHT)
            img_mask = img_mask.transpose(Image.FLIP_LEFT_RIGHT)

        if 0.3 < random() < 0.6:
            img = img.transpose(Image.FLIP_TOP_BOTTOM)
            img_mask = img_mask.transpose(Image.FLIP_TOP_BOTTOM)

        if self.aug_transform is not None:
            img = self.aug_transform(img)
            img_mask = self.aug_transform(img_mask)

        if self.tensor_transform is not None:
            img = self.tensor_transform(img)
            img_mask = self.tensor_transform_gray(img_mask)

        return img, img_mask, label

    def __len__(self):
        return len(self.sample_list)


def main(config):
    # config
    train_path = config.TRAIN_PATH
    val_path = config.VAL_PATH
    test_path = config.VAL_PATH
    epoches = config.EPOCHES

    pretrained_path = './pretrained/xception-b5690688.pth'
    batch_size = 16
    gpu_ids = [*range(osenvs)]
    loss_freq = 100
    mode = 'FAD'
    ckpt_dir = './checkpoints'
    ckpt_name = 'FAD_RGB_F2Fc0'

    # transform and dataloader
    tensor_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),
        transforms.RandomGrayscale(p=0.2),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])
    test_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])
    train_dataset = FaceDataset(img_paths=train_path,
                                type=config.TYPE,
                                dataset_type='train',
                                aug_transform=None,
                                tensor_transform=tensor_transform)

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        num_workers=config.WORKERS,
        pin_memory=True
    )

    # init checkpoint and logger
    ckpt_path = os.path.join(ckpt_dir, ckpt_name)
    logger = setup_logger(ckpt_path, 'result.log', 'logger')
    best_val = 0.
    ckpt_model_name = 'best.pkl'

    # train
    model = Trainer(config, gpu_ids, mode, pretrained_path)
    model.total_steps = 0
    epoch = 0

    # **************************added********************************
    best_model_wts = model.state_dict()
    best_acc = 0.0
    iteration = 0
    for epoch in range(epoches):
        # 添加进度条
        progress_bar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{epoches}')
        for i, (image, mask, labels) in enumerate(train_loader):
            # for (image, labels) in train_loader:
            model.total_steps += 1
            # image = image.cuda()
            # labels = labels.cuda()
            image = image.detach()
            labels = labels.detach()
            model.set_input(image, mask, labels)
            loss = model.optimize_weight(config)
            if model.total_steps % loss_freq == 0:
                logger.debug(f'total_steps: {model.total_steps} loss: {loss}')
            txt_path = os.path.join(ckpt_path, 'logger.txt')
            if model.total_steps % 100 == 0:
                model.model.eval()
                # auc, r_acc, f_acc = evaluate(model, val_path, mode='valid')
                # logger.debug(f'(Val @ epoch {epoch}) auc: {auc}, r_acc: {r_acc}, f_acc:{f_acc}')
                acc = evaluate(model, config, test_path, batch_size)
                logger.debug(f'(Val @ epoch {epoch + 1}) acc: {acc}')
                if acc > best_acc:
                    best_acc = acc
                    best_model_wts = model.state_dict()

                model.load_state_dict(best_model_wts)
                # torch.save(model.module.state_dict(), os.path.join(output_path, "best.pkl"))
                torch.save(model.state_dict(), os.path.join(ckpt_path, "best.pkl"))

                with open(txt_path, "a") as f:
                    f.write(
                        "Epoch: {epoch:d} \n Val ACC: {acc:.4f} \n Loss: {loss:.4f} \n Best ACC: {best_acc:.4f} \n"
                        .format(epoch=epoch, acc=acc, loss=loss, best_acc=best_acc))

                model.model.train()
            # outputs = model(image)
            # _, preds = torch.max(outputs.data, 1)
            # loss = criterion(outputs, labels)
            # loss.backward()
        # 更新进度条信息
        progress_bar.set_postfix(loss=f'{loss:.4f}')
    model.model.eval()
    acc = evaluate(model, config, test_path, batch_size)
    logger.debug(f'(Test @ epoch {epoch + 1}) acc: {acc}')


if __name__ == '__main__':
    with open("./config.yaml", 'r') as stream:
        config = yaml.safe_load(stream)
    config = easydict.EasyDict(config)
    main(config)


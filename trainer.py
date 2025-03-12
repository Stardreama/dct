import torch
import torch.nn as nn
from torch.nn import parameter
from models import F3Net
import torch.nn.functional as F
import numpy as np
import os


def initModel(mod, gpu_ids):

    mod = mod.to(f'cuda:{gpu_ids[0]}')
    mod = nn.DataParallel(mod, gpu_ids)
    return mod


class Trainer(nn.Module): 
    def __init__(self, config, gpu_ids, mode, pretrained_path):
        super(Trainer, self).__init__()
        self.device = torch.device('cuda:{}'.format(gpu_ids[0])) if gpu_ids else torch.device('cpu')
        self.model = F3Net(config, mode=mode, device=self.device)
        # self.model = initModel(self.model, gpu_ids)
        self.loss_fn = nn.BCEWithLogitsLoss()
        self.loss_fn = nn.CrossEntropyLoss()
        self.loss_mse = nn.MSELoss(reduction='mean')
        self.optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, self.model.parameters()),
                                              lr=0.0002, betas=(0.9, 0.999))
        # self.optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, self.model.parameters()),
        #                                         lr=0.002, momentum=0.9, weight_decay=0)

    def set_input(self, input, mask, label):
        self.input = input.to(self.device)
        self.label = label.to(self.device)
        self.mask = mask.to(self.device)

    def forward(self, x):
        mask_pred, out = self.model(x)
        return mask_pred, out

    def optimize_weight(self, config):
        mask_pred, out = self.model(self.input)
        # print(stu_cla)
        # print(stu_cla.squeeze(1))
        self.loss_cla = self.loss_fn(out.squeeze(1), self.label)  # classify loss
        self.loss_mask = self.loss_mse(mask_pred, self.mask)
        self.loss = self.loss_cla + self.loss_mask
#        print("*******************")
#        print(self.loss_cla)
#        print(self.loss_mask)

        self.optimizer.zero_grad()
        self.loss.backward()
        self.optimizer.step()
        return self.loss

    def save(self, path):
        torch.save(self.model.state_dict(), path)

    def load(self, path):
        state_dict = torch.load(path)
        self.model.load_state_dict(state_dict)

from xception import Xception
from xception1 import Xception1
#from xception2 import Xception2
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
import numpy as np
import types
from attention.SEAttention import SEAttention
from attention.SKAttention import SKAttention
from network.cls_hrnet import *


# Filter Module
class Filter(nn.Module):
    def __init__(self, size, band_start, band_end, use_learnable=True, norm=False):
        super(Filter, self).__init__()
        self.use_learnable = use_learnable

        self.base = nn.Parameter(torch.tensor(generate_filter(band_start, band_end, size)), requires_grad=False).cuda()
        if self.use_learnable:
            self.learnable = nn.Parameter(torch.randn(size, size), requires_grad=True)
            self.learnable.data.normal_(0., 0.1)
            # Todo
            # self.learnable = nn.Parameter(torch.rand((size, size)) * 0.2 - 0.1, requires_grad=True)

        self.norm = norm
        if norm:
            self.ft_num = nn.Parameter(torch.sum(torch.tensor(generate_filter(band_start, band_end, size))), requires_grad=False)

    def forward(self, x):
        if self.use_learnable:
            filt = self.base + norm_sigma(self.learnable).cuda()
        else:
            filt = self.base.cuda()

        if self.norm:
            y = x * filt / self.ft_num
        else:
            y = x * filt
        return y


# FAD Module
class FAD_Head(nn.Module):
    def __init__(self, size):
        super(FAD_Head, self).__init__()

        # init DCT matrix
        self._DCT_all = nn.Parameter(torch.tensor(DCT_mat(size)).float(), requires_grad=False).cuda()
        self._DCT_all_T = nn.Parameter(torch.transpose(torch.tensor(DCT_mat(size)).float(), 0, 1), requires_grad=False).cuda()

        # define base filters and learnable
        # 0 - 1/16 || 1/16 - 1/8 || 1/8 - 1
        low_filter = Filter(size, 0, size // 2.82)
        middle_filter = Filter(size, size // 2.82, size // 2)
        high_filter = Filter(size, size // 2, size * 2)
        all_filter = Filter(size, 0, size * 2)

        self.filters = nn.ModuleList([low_filter, middle_filter, high_filter, all_filter])

    def forward(self, x):
        # DCT
        x_freq = self._DCT_all @ x @ self._DCT_all_T    # [N, 3, 299, 299]

        # 4 kernel
        y_list = []
        for i in range(4):
            x_pass = self.filters[i](x_freq)  # [N, 3, 299, 299]
            y = self._DCT_all_T @ x_pass @ self._DCT_all    # [N, 3, 299, 299]
            y_list.append(y)
        out = torch.cat(y_list, dim=1)    # [N, 12, 299, 299]
        # out_add = torch.add(y_list[0], y_list[1], y_list[2], y_list[3])
        # se = SEAttention(channel=12, reduction=4)
        # output = se(input)

        # [32, 12, 256, 256]
        # print(out.shape)
        return out


class F3Net(nn.Module):
    def __init__(self, config, num_classes=2, img_width=256, img_height=256, mode='FAD', device=None):
        super(F3Net, self).__init__()
        assert img_width == img_height
        img_size = img_width
        self.num_classes = num_classes
        self.mode = mode

        # init branches
        self.FAD_head = FAD_Head(img_size)
        self.init_xcep_branch(config)
        self.init_xcep_fuse()

        # classifier
        self.relu = nn.ReLU(inplace=True)
        self.fc = nn.Linear(4096 if self.mode == 'Both' or self.mode == 'Mix' else 2048, num_classes).cuda()
        self.dp = nn.Dropout(p=0.2)

    # first
    def init_xcep_branch(self, config):
        # Model initialization
        self.FAD_xcep1 = Xception1(self.num_classes)
        self.RGB_xcep1 = get_cls_net(config)  # RGB->HRNet
        state_dict, state_dict_hr = get_state_dict()

        self.FAD_xcep1.load_state_dict(state_dict, strict=False)
        self.RGB_xcep1.load_state_dict(state_dict_hr, strict=False)
        conv1_data = state_dict['conv1.weight'].data
        # copy on conv1
        # let new conv1 use old param to balance the network
        self.FAD_xcep1.conv1 = nn.Conv2d(12, 32, 3, 2, 0, bias=False)
        for i in range(4):
            self.FAD_xcep1.conv1.weight.data[:, i*3:(i+1)*3, :, :] = conv1_data / 4.0

    def init_xcep_fuse(self):
        self.fuse_xcep = Xception(self.num_classes)
        state_dict, state_dict_hr = get_state_dict()
        self.fuse_xcep.load_state_dict(state_dict, strict=False)

    def forward(self, x):
        fea_FAD = self.FAD_head(x)
        self.fuse_xcep = self.fuse_xcep.cuda()
        self.FAD_xcep1 = self.FAD_xcep1.cuda()
        self.RGB_xcep1 = self.RGB_xcep1.cuda()
        fea_RGB, mask_pred =self.RGB_xcep1(x)
        fea_Fre = self.FAD_xcep1.features(fea_FAD)
        # [32,728,16,16]
        # se = SKAttention(channel=728, reduction=8).cuda()
        # fea_Fre1, fea_RGB1 = se(fea_Fre, fea_RGB)
        output = torch.cat((fea_Fre, fea_RGB), dim=1)
        fea_final = self.fuse_xcep.features(output)
        # output = torch.add(fea_Fre, fea_RGB)
        # fea_final = self.fuse_xcep.features(output)
        fea_final = self._norm_fea(fea_final)
        y = fea_final

        f = self.dp(y)
        f = self.fc(f)
        return mask_pred, f

    def _norm_fea(self, fea):
        f = self.relu(fea)
        f = F.adaptive_avg_pool2d(f, (1,1))
        f = f.view(f.size(0), -1)
        return f


# utils
def DCT_mat(size):
    m = [[ (np.sqrt(1./size) if i == 0 else np.sqrt(2./size)) * np.cos((j + 0.5) * np.pi * i / size) for j in range(size)] for i in range(size)]
    return m


def generate_filter(start, end, size):
    return [[0. if i + j > end or i + j < start else 1. for j in range(size)] for i in range(size)]


def norm_sigma(x):
    return 2. * torch.sigmoid(x) - 1.


def get_state_dict(pretrained_path='pretrained/xception-b5690688.pth', pretrained_path2='pretrained/hrnet_w32.pth'):
    # load Xception
    state_dict = torch.load(pretrained_path)
    for name, weights in state_dict.items():
        if 'pointwise' in name:
            state_dict[name] = weights.unsqueeze(-1).unsqueeze(-1)
    state_dict1 = {k:v for k, v in state_dict.items() if 'fc' not in k}

    # load HRNet
    state_dict = torch.load(pretrained_path2)
    for name, weights in state_dict.items():
        if 'pointwise' in name:
            state_dict[name] = weights.unsqueeze(-1).unsqueeze(-1)
    state_dict2 = {k:v for k, v in state_dict.items() if 'fc' not in k}

    return state_dict1, state_dict2



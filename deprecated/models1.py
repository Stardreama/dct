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
from attention.SKAttention1 import SKAttention
from network.resnet_baseline1 import *


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


# LFS Module
class LFS_Head(nn.Module):
    def __init__(self, size, window_size, M):
        super(LFS_Head, self).__init__()

        self.window_size = window_size
        self._M = M

        # init DCT matrix
        self._DCT_patch = nn.Parameter(torch.tensor(DCT_mat(window_size)).float(), requires_grad=False)
        self._DCT_patch_T = nn.Parameter(torch.transpose(torch.tensor(DCT_mat(window_size)).float(), 0, 1), requires_grad=False)

        self.unfold = nn.Unfold(kernel_size=(window_size, window_size), stride=8, padding=0)

        # init filters
        self.filters = nn.ModuleList([Filter(window_size, window_size * 2. / M * i, window_size * 2. / M * (i+1), norm=True) for i in range(M)])
    
    def forward(self, x):
        # turn RGB into Gray
        x_gray = 0.299*x[:,0,:,:] + 0.587*x[:,1,:,:] + 0.114*x[:,2,:,:]
        x = x_gray.unsqueeze(1)

        # rescale to 0 - 255
        x = (x + 1.) * 122.5

        # calculate size
        N, C, W, H = x.size()
        # print(x.size())
        S = self.window_size
        # print(S)
        size_after = int((W - S + 8)/2)
        assert size_after == 128

        # sliding window unfold and DCT
        x_unfold = self.unfold(x)   # [N, C * S * S, L]   L:block num
        # print('x_unfold',x_unfold.shape)
        L = x_unfold.size()[2]
        # print(L)
        x_unfold = x_unfold.transpose(1, 2).reshape(N, L, C, S, S)  # [N, L, C, S, S]
        x_dct = self._DCT_patch @ x_unfold @ self._DCT_patch_T

        # M kernels filtering
        y_list = []
        for i in range(self._M):
            # y = self.filters[i](x_dct)    # [N, L, C, S, S]
            # y = torch.abs(y)
            # y = torch.sum(y, dim=[2,3,4])   # [N, L]
            # y = torch.log10(y + 1e-15)
            y = torch.abs(x_dct)
            # print(y.shape)
            y = torch.log10(y + 1e-15)
            # print(y.shape)
            y = self.filters[i](y)
            # print(y.shape)
            y = torch.sum(y, dim=[2,3,4])
            # print(y.shape)
            y = y.reshape(N, 32, 32)
            y = y.unsqueeze(dim=1)   # [N, 1, 149, 149]
            y_list.append(y)
        out = torch.cat(y_list, dim=1)  # [N, M, 149, 149]
        # print(out.shape)
        return out


class F3Net(nn.Module):
    def __init__(self, num_classes=2, img_width=256, img_height=256, LFS_window_size=8, LFS_stride=2, LFS_M = 6, mode='FAD', device=None):
        super(F3Net, self).__init__()
        assert img_width == img_height
        img_size = img_width
        self.num_classes = num_classes
        self.mode = mode
        self.window_size = LFS_window_size
        self._LFS_M = LFS_M

        # init branches
        if mode == 'FAD' or mode == 'Both':
            self.FAD_head = FAD_Head(img_size)
            self.init_xcep_FAD()
            self.init_xcep_FAD1()
            self.init_resnet50_RGB()
            # self.init_xcep()

        if mode == 'LFS' or mode == 'Both':
            self.LFS_head = LFS_Head(img_size, LFS_window_size, LFS_M)
            self.init_xcep_LFS()

        if mode == 'Original':
            self.init_xcep()

        # classifier
        self.relu = nn.ReLU(inplace=True)
        self.fc = nn.Linear(4096 if self.mode == 'Both' or self.mode == 'Mix' else 2048, num_classes).cuda()
        self.dp = nn.Dropout(p=0.2)

    # def init_xcep_FAD(self):
    #     self.FAD_xcep = Xception(self.num_classes)
    #
    #     # To get a good performance, using ImageNet-pretrained Xception model is recommended
    #     state_dict = get_xcep_state_dict()
    #     conv1_data = state_dict['conv1.weight'].data
    #
    #     self.FAD_xcep.load_state_dict(state_dict, False)
    #
    #     # copy on conv1
    #     # let new conv1 use old param to balance the network
    #     self.FAD_xcep.conv1 = nn.Conv2d(12, 32, 3, 2, 0, bias=False)
    #     for i in range(4):
    #         self.FAD_xcep.conv1.weight.data[:, i*3:(i+1)*3, :, :] = conv1_data / 4.0

    def init_xcep_FAD(self):
        self.FAD_xcep = Xception(self.num_classes)
        state_dict = get_xcep_state_dict()
        self.FAD_xcep.load_state_dict(state_dict, strict=False)

    def init_xcep_FAD1(self):
        self.FAD_xcep1 = Xception1(self.num_classes)
        self.RGB_xcep1 = Xception1(self.num_classes)
        state_dict = get_xcep_state_dict()
        self.FAD_xcep1.load_state_dict(state_dict, strict=False)
        self.RGB_xcep1.load_state_dict(state_dict, strict=False)
        conv1_data = state_dict['conv1.weight'].data
        # copy on conv1
        # let new conv1 use old param to balance the network
        self.FAD_xcep1.conv1 = nn.Conv2d(12, 32, 3, 2, 0, bias=False)
        for i in range(4):
            self.FAD_xcep1.conv1.weight.data[:, i*3:(i+1)*3, :, :] = conv1_data / 4.0

    def init_xcep_LFS(self):
        self.LFS_xcep = Xception(self.num_classes)
        
        # To get a good performance, using ImageNet-pretrained Xception model is recommended
        state_dict = get_xcep_state_dict()
        conv1_data = state_dict['conv1.weight'].data

        self.LFS_xcep.load_state_dict(state_dict, False)

        # copy on conv1
        # let new conv1 use old param to balance the network
        self.LFS_xcep.conv1 = nn.Conv2d(self._LFS_M, 32, 3, 1, 0, bias=False)
        for i in range(int(self._LFS_M / 3)):
            self.LFS_xcep.conv1.weight.data[:, i*3:(i+1)*3, :, :] = conv1_data / float(self._LFS_M / 3.0)

    def init_resnet50_RGB(self):
        self.RGB_Res = resnet50(pretrained=True,  num_classes=2)

    def init_xcep(self):
        self.xcep = Xception(self.num_classes)

        # To get a good performance, using ImageNet-pretrained Xception model is recommended
        state_dict = get_xcep_state_dict()
        self.xcep.load_state_dict(state_dict, strict=False)

    def forward(self, x):
        if self.mode == 'FAD':
            fea_FAD = self.FAD_head(x)
            self.FAD_xcep = self.FAD_xcep.cuda()
            self.FAD_xcep1 = self.FAD_xcep1.cuda()
            # self.RGB_xcep1 = self.RGB_xcep1.cuda()
            self.RGB_Res = self.RGB_Res.cuda()
            # fea_RGB =self.RGB_xcep1.features(x)
            fea_RGB = self.RGB_Res(x)
            fea_Fre = self.FAD_xcep1.features(fea_FAD)
            # [32,728,16,16]
            # se = SKAttention(channel=728, reduction=8).cuda()
            # fea_Fre1,fea_RGB1 = se(fea_Fre, fea_RGB)
            # output = torch.cat((fea_Fre1, fea_RGB1), dim=1)
            output = torch.cat((fea_Fre, fea_RGB), dim=1)

            fea_final = self.FAD_xcep.features(output)
            # output = torch.add(fea_Fre, fea_RGB)
            # fea_final = self.FAD_xcep.features(output)
            fea_final = self._norm_fea(fea_final)
            y = fea_final

        if self.mode == 'LFS':
            fea_LFS = self.LFS_head(x)
            fea_LFS = self.LFS_xcep.features(fea_LFS)
            fea_LFS = self._norm_fea(fea_LFS)
            y = fea_LFS

        if self.mode == 'Original':
            fea = self.xcep.features(x)
            fea = self._norm_fea(fea)
            y = fea

        if self.mode == 'Both':
            fea_FAD = self.FAD_head(x)
            fea_FAD = self.FAD_xcep.features(fea_FAD)
            fea_FAD = self._norm_fea(fea_FAD)
            fea_LFS = self.LFS_head(x)
            fea_LFS = self.LFS_xcep.features(fea_LFS)
            fea_LFS = self._norm_fea(fea_LFS)
            y = torch.cat((fea_FAD, fea_LFS), dim=1)

        f = self.dp(y)
        f = self.fc(f)
        return y,f

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

def get_xcep_state_dict(pretrained_path='pretrained/xception-b5690688.pth'):
    # load Xception
    state_dict = torch.load(pretrained_path)
    for name, weights in state_dict.items():
        if 'pointwise' in name:
            state_dict[name] = weights.unsqueeze(-1).unsqueeze(-1)
    state_dict = {k:v for k, v in state_dict.items() if 'fc' not in k}
    return state_dict
    

# overwrite method for xception in LFS branch
# plan A

def new_xcep_features(self, input):
    # x = self.conv1(input)
    # x = self.bn1(x)
    # x = self.relu(x)

    x = self.conv2(input)   # input :[149, 149, 6]  conv2:[in_filter:32]
    x = self.bn2(x)
    x = self.relu(x)

    x = self.block1(x)
    x = self.block2(x)
    x = self.block3(x)
    x = self.block4(x)
    x = self.block5(x)
    x = self.block6(x)
    x = self.block7(x)
    x = self.block8(x)
    x = self.block9(x)
    x = self.block10(x)
    x = self.block11(x)
    x = self.block12(x)

    x = self.conv3(x)
    x = self.bn3(x)
    x = self.relu(x)

    x = self.conv4(x)
    x = self.bn4(x)
    return x

# function for mix block

def fea_0_7(self, x):
    x = self.conv1(x)
    x = self.bn1(x)
    x = self.relu(x)

    x = self.conv2(x)
    x = self.bn2(x)
    x = self.relu(x)

    x = self.block1(x)
    x = self.block2(x)
    x = self.block3(x)
    x = self.block4(x)
    x = self.block5(x)
    x = self.block6(x)
    x = self.block7(x)
    return x

def fea_8_12(self, x):
    x = self.block8(x)
    x = self.block9(x)
    x = self.block10(x)
    x = self.block11(x)
    x = self.block12(x)

    x = self.conv3(x)
    x = self.bn3(x)
    x = self.relu(x)

    x = self.conv4(x)
    x = self.bn4(x)
    return x

class MixBlock(nn.Module):
    # An implementation of the cross attention module in F3-Net
    # Haven't added into the whole network yet
    def __init__(self, c_in, width, height):
        super(MixBlock, self).__init__()
        self.FAD_query = nn.Conv2d(c_in, c_in, (1,1))
        self.LFS_query = nn.Conv2d(c_in, c_in, (1,1))

        self.FAD_key = nn.Conv2d(c_in, c_in, (1,1))
        self.LFS_key = nn.Conv2d(c_in, c_in, (1,1))

        self.softmax = nn.Softmax(dim=-1)
        self.relu = nn.ReLU()

        self.FAD_gamma = nn.Parameter(torch.zeros(1))
        self.LFS_gamma = nn.Parameter(torch.zeros(1))

        self.FAD_conv = nn.Conv2d(c_in, c_in, (1,1), groups=c_in)
        self.FAD_bn = nn.BatchNorm2d(c_in)
        self.LFS_conv = nn.Conv2d(c_in, c_in, (1,1), groups=c_in)
        self.LFS_bn = nn.BatchNorm2d(c_in)

    def forward(self, x_FAD, x_LFS):
        B, C, W, H = x_FAD.size()
        assert W == H

        q_FAD = self.FAD_query(x_FAD).view(-1, W, H)    # [BC, W, H]
        q_LFS = self.LFS_query(x_LFS).view(-1, W, H)
        M_query = torch.cat([q_FAD, q_LFS], dim=2)  # [BC, W, 2H]

        k_FAD = self.FAD_key(x_FAD).view(-1, W, H).transpose(1, 2)  # [BC, H, W]
        k_LFS = self.LFS_key(x_LFS).view(-1, W, H).transpose(1, 2)
        M_key = torch.cat([k_FAD, k_LFS], dim=1)    # [BC, 2H, W]

        energy = torch.bmm(M_query, M_key)  #[BC, W, W]
        attention = self.softmax(energy).view(B, C, W, W)

        att_LFS = x_LFS * attention * (torch.sigmoid(self.LFS_gamma) * 2.0 - 1.0)
        y_FAD = x_FAD + self.FAD_bn(self.FAD_conv(att_LFS))

        att_FAD = x_FAD * attention * (torch.sigmoid(self.FAD_gamma) * 2.0 - 1.0)
        y_LFS = x_LFS + self.LFS_bn(self.LFS_conv(att_FAD))
        return y_FAD, y_LFS

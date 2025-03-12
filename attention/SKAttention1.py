import numpy as np
import torch
from torch import nn
from torch.nn import init
from collections import OrderedDict



class SKAttention(nn.Module):

    def __init__(self, channel=512,kernels=[1,3],reduction=16,group=1,L=32):
        super().__init__()
        self.d=max(L,channel//reduction)
        self.convs=nn.ModuleList([])
        for k in kernels:
            self.convs.append(
                nn.Sequential(OrderedDict([
                    ('conv',nn.Conv2d(channel,channel,kernel_size=k,padding=k//2,groups=group)),
                    ('bn',nn.BatchNorm2d(channel)),
                    ('relu',nn.ReLU())
                ]))
            )
        self.fc=nn.Linear(channel,self.d)
        self.fcs=nn.ModuleList([])
        for i in range(len(kernels)):
            self.fcs.append(nn.Linear(self.d,channel))
        self.softmax=nn.Softmax(dim=0)

    def split(self, tensor):
        a = torch.split(tensor, 4, dim=3)  # [bz,728,16,4]×4
        b1 = torch.split(a[0], 4, dim=2)
        b2 = torch.split(a[1], 4, dim=2)
        b3 = torch.split(a[2], 4, dim=2)
        b4 = torch.split(a[3], 4, dim=2)
        patch = []
        patch.append(b1[0])
        patch.append(b2[0])
        patch.append(b3[0])
        patch.append(b4[0])

        patch.append(b1[1])
        patch.append(b2[1])
        patch.append(b3[1])
        patch.append(b4[1])

        patch.append(b1[2])
        patch.append(b2[2])
        patch.append(b3[2])
        patch.append(b4[2])

        patch.append(b1[3])
        patch.append(b2[3])
        patch.append(b3[3])
        patch.append(b4[3])

        return patch

    def cat_tensor(self, tensor):
        x1 = tensor[:4]
        x2 = tensor[4:8]
        x3 = tensor[8:12]
        x4 = tensor[12:]

        x1_cat = torch.cat(x1, dim=3)
        x2_cat = torch.cat(x2, dim=3)
        x3_cat = torch.cat(x3, dim=3)
        x4_cat = torch.cat(x4, dim=3)

        x1_final = []
        x1_final.append(x2_cat)
        x1_final.append(x3_cat)
        x1_final.append(x4_cat)

        x_cat_final = torch.cat(x1_final, dim=2)

        return x_cat_final

    def forward(self, x, x1):
        bs, c, _, _ = x.size()
        conv_outs = []
        conv_outs.append(x)
        conv_outs.append(x1)
        out_rgb = self.split(x)
        out_freq = self.split(x1)
        rgb_final = []
        freq_final = []

        # fuse
        U = sum(conv_outs)  # bs,c,h,w[bz,728,16,16]
        patch1 = self.split(U)

        for i in range(0, 16):
            patch_rgb = out_rgb[i]
            patch_freq = out_freq[i]
            conv_outs1 = []
            conv_outs1.append(patch_rgb)
            conv_outs1.append(patch_freq)
            feats = torch.stack(conv_outs1, 0)  # k,bs,channel,h,w
            # reduction channel
            S = patch1[i].mean(-1).mean(-1)  # bs,c
            Z = self.fc(S)  # bs,d
            weights = []
            for fc in self.fcs:
                weight = fc(Z)
                weights.append(weight.view(bs,c,1,1))  # bs,channel
            attention_weughts=torch.stack(weights,0)  # k,bs,channel,1,1
            attention_weughts=self.softmax(attention_weughts)  # k,bs,channel,1,1
            # fuse
            V = attention_weughts*feats
            v1, v2 = torch.chunk(V, 2, dim=0)
            v1 = v1.squeeze(0) + patch_rgb[i]
            v2 = v2.squeeze(0) + patch_freq[i]

            rgb_final.append(v1)
            freq_final.append(v2)

        v1_out = self.cat_tensor(rgb_final)
        v2_out = self.cat_tensor(freq_final)

        return v1_out, v2_out


if __name__ == '__main__':
    input2 = torch.randn(50,512,7,7)
    input1 = torch.randn(50,512,7,7)
    se = SKAttention(channel=512,reduction=8)
    out1, out2 = se(input2, input1)
    print(out1.shape)
    print(out2.shape)

    
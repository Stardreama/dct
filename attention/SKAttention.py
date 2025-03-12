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

    def forward(self, x, x1):
        bs, c, _, _ = x.size()
        conv_outs=[]
        conv_outs.append(x)
        conv_outs.append(x1)
        # ### split
        # for conv in self.convs:
        #     conv_outs.append(conv(x))
        feats=torch.stack(conv_outs,0)#k,bs,channel,h,w
        # print(feats.shape) [2, 50, 512, 7, 7]

        ### fuse
        U=sum(conv_outs) #bs,c,h,w

        ### reduction channel
        S=U.mean(-1).mean(-1) #bs,c
        Z=self.fc(S) #bs,d
        #print(U.shape) [50, 512, 7, 7]
        #print(S.shape) [50, 512]
        #print(Z.shape) [50, 64]

        ### calculate attention weight
        weights=[]
        for fc in self.fcs:
            weight=fc(Z)
            weights.append(weight.view(bs,c,1,1)) #bs,channel
        attention_weughts=torch.stack(weights,0)#k,bs,channel,1,1
        attention_weughts=self.softmax(attention_weughts)#k,bs,channel,1,1

        ### fuse
        V = attention_weughts*feats
        v1, v2 = torch.chunk(V, 2, dim=0)
        v1 = v1.squeeze(0)
        v1 = torch.add(x, v1)
        v2 = v2.squeeze(0)
        v2 = torch.add(x1,v2)

        return v1, v2


if __name__ == '__main__':
    input2 = torch.randn(50,512,7,7)
    input1 = torch.randn(50,512,7,7)
    se = SKAttention(channel=512,reduction=8)
    out1, out2 = se(input2, input1)
    print(out1.shape)
    print(out2.shape)

    
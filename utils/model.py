import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F

class Convolutional_Attention(nn.Module):
    def __init__(self,
                 channels,
                 num_heads,
                 proj_drop=0.0,
                 kernel_size=3,
                 stride_kv=1,
                 stride_q=1,
                 padding_kv="same",
                 padding_q="same",
                 attention_bias=True
                 ):
        super().__init__()
        self.stride_kv = stride_kv
        self.stride_q = stride_q
        self.num_heads = num_heads
        self.proj_drop = proj_drop
        
        self.layer_q = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size, stride_q, padding_q, bias=attention_bias, groups=channels),
            nn.ReLU()
        )
        self.layernorm_q = nn.LayerNorm(channels, eps=1e-5)

        self.layer_k = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size, stride_kv, padding_kv, bias=attention_bias, groups=channels),
            nn.ReLU()
        )
        self.layernorm_k = nn.LayerNorm(channels, eps=1e-5)

        self.layer_v = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size, stride_kv, padding_kv, bias=attention_bias, groups=channels),
            nn.ReLU()
        )
        self.layernorm_v = nn.LayerNorm(channels, eps=1e-5)
        
        self.attention = nn.MultiheadAttention(embed_dim=channels, 
                                               bias=attention_bias, 
                                               batch_first=True,
                                               dropout = 0,# self.proj_drop,
                                               num_heads=self.num_heads)

    def _build_projection(self, x, mode):
        # x shape [batch,channel,size,size]
        # mode:0->q,1->k,2->v,for torch.script can not script str
        
        if mode == 0:
            x1 = self.layer_q(x)
            x1 = x1.permute(0, 2, 3, 1)
            x1 = self.layernorm_q(x1)
            proj = x1.permute(0, 3, 1, 2)
        elif mode == 1:
            x1 = self.layer_k(x)
            x1 = x1.permute(0, 2, 3, 1)
            x1 = self.layernorm_k(x1)
            proj = x1.permute(0, 3, 1, 2)            
        elif mode == 2:
            x1 = self.layer_v(x)
            x1 = x1.permute(0, 2, 3, 1)
            x1 = self.layernorm_v(x1)
            proj = x1.permute(0, 3, 1, 2)        

        return proj

    def get_qkv(self, x):
        q = self._build_projection(x, 0)
        k = self._build_projection(x, 1)
        v = self._build_projection(x, 2)

        return q, k, v

    def forward(self, x):
        q, k, v = self.get_qkv(x)
        q = q.view(x.shape[0], x.shape[1], x.shape[2]*x.shape[3])
        k = k.view(x.shape[0], x.shape[1], x.shape[2]*x.shape[3])
        v = v.view(x.shape[0], x.shape[1], x.shape[2]*x.shape[3])
        q = q.permute(0, 2, 1)
        k = k.permute(0, 2, 1)
        v = v.permute(0, 2, 1)
        x1 = self.attention(query=q, value=v, key=k, need_weights=False)
        
        x1 = x1[0].permute(0, 2, 1)
        x1 = x1.view(x1.shape[0], x1.shape[1], np.sqrt(x1.shape[2]).astype(int), np.sqrt(x1.shape[2]).astype(int))
        x1 = F.dropout(x1, self.proj_drop)

        return x1
 
class Transformer(nn.Module):

    def __init__(self,
                 # in_channels,
                 out_channels,
                 num_heads,
                 dpr = 0.0,
                 proj_drop=0.5,
                 attention_bias=True,
                 padding_q="same",
                 padding_kv="same",
                 stride_kv=1,
                 stride_q=1):
        super().__init__()
        
        self.attention_output = Convolutional_Attention(channels=out_channels,
                                         num_heads=num_heads,
                                         proj_drop=proj_drop,
                                         padding_q=padding_q,
                                         padding_kv=padding_kv,
                                         stride_kv=stride_kv,
                                         stride_q=stride_q,
                                         attention_bias=attention_bias,
                                         )

        self.conv1 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding="same")
        self.layernorm = nn.LayerNorm(self.conv1.out_channels, eps=1e-5)
        self.wide_focus = Wide_Focus(out_channels, out_channels)

    def forward(self, x):
        x1 = self.attention_output(x)
        x2 = self.conv1(x1) + x

        x3 = x2.permute(0, 2, 3, 1)
        x3 = self.layernorm(x3)
        x3 = x3.permute(0, 3, 1, 2)

        out = self.wide_focus(x3) + x2
        return out
    
class Wide_Focus(nn.Module): 
    """
    Wide-Focus module.
    """
    def __init__(self,
                 in_channels,
                 out_channels):
        super().__init__()
        
        self.layer1 = nn.Sequential(
            nn.Conv2d(in_channels,out_channels, kernel_size=3, stride=1, padding="same"),
            nn.GELU(),
            nn.Dropout(0.1)
        )
        self.layer_dilation2 = nn.Sequential(
            nn.Conv2d(in_channels,out_channels, kernel_size=3, stride=1, padding="same", dilation=2),
            nn.GELU(),
            nn.Dropout(0.1)
        )
        self.layer_dilation3 = nn.Sequential(
            nn.Conv2d(in_channels,out_channels, kernel_size=3, stride=1, padding="same", dilation=3),
            nn.GELU(),
            nn.Dropout(0.1)
        )
        self.layer4 = nn.Sequential(
            nn.Conv2d(in_channels,out_channels, kernel_size=3, stride=1, padding="same"),
            nn.GELU(),
            nn.Dropout(0.1)
        )


    def forward(self, x):
        x1 = self.layer1(x)
        x2 = self.layer_dilation2(x)
        x3 = self.layer_dilation3(x)
        added = x1 + x2 + x3
        x_out = self.layer4(added)
        return x_out

class Block_decoder(nn.Module):
    def __init__(self, in_channels, out_channels, att_heads, dpr):
        super().__init__()
        self.layernorm = nn.LayerNorm(in_channels, eps=1e-5)
        self.upsample = nn.Upsample(scale_factor=2)
        self.layer1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding="same"),
            nn.ReLU()
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(out_channels*2, out_channels, kernel_size=3, stride=1, padding="same"),
            nn.ReLU()
        )
        self.layer3 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding="same"),
            nn.ReLU(),
            nn.Dropout(0.3)
        )
        self.trans = Transformer(out_channels, att_heads, dpr)
        
    def forward(self, x, skip):
        x1 = x.permute(0, 2, 3, 1)
        x1 = self.layernorm(x1)
        x1 = x1.permute(0, 3, 1, 2)
        x1 = self.upsample(x1)
        x1 = self.layer1(x1)
        x1 = torch.cat((skip, x1), axis=1)
        x1 = self.layer2(x1)
        x1 = self.layer3(x1)
        out = self.trans(x1)
        return out

class DS_out(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.upsample = nn.Upsample(scale_factor=2)
        self.layernorm = nn.LayerNorm(in_channels, eps=1e-5)
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding="same"),
            nn.ReLU()
        ) 
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding="same"),
            nn.ReLU()
        ) 
        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding="same"),
            nn.Sigmoid()
        ) 

    def forward(self, x):
        x1 = self.upsample(x)
        x1 = x1.permute(0, 2, 3, 1)
        x1 = self.layernorm(x1)
        x1 = x1.permute(0, 3, 1, 2)
        x1 = self.conv1(x1)
        x1 = self.conv2(x1)
        out = self.conv3(x1)
        
        return out

class Block_encoder_without_skip(nn.Module):
    def __init__(self, in_channels, out_channels, att_heads, dpr):
        super().__init__()
        self.layernorm = nn.LayerNorm(in_channels, eps=1e-5)
        self.layer1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding="same"),
            nn.ReLU()
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding="same"),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.MaxPool2d((2,2))
        )
        self.trans = Transformer(out_channels, att_heads, dpr)

    def forward(self, x):
        x1 = x.permute(0, 2, 3, 1)
        x1 = self.layernorm(x1)
        x1 = x1.permute(0, 3, 1, 2)
        x1 = self.layer1(x1)
        x1 = self.layer2(x1)
        x1 = self.trans(x1)
        return x1

class Block_encoder_with_skip(nn.Module):
    def __init__(self, in_channels, out_channels, att_heads, dpr):
        super().__init__()
        self.layernorm = nn.LayerNorm(in_channels, eps=1e-5)
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, in_channels, kernel_size=3, stride=1, padding="same"),
            nn.ReLU()
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding="same"),
            nn.ReLU()
        )
        self.layer3 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding="same"),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.MaxPool2d((2,2))
        )
        self.trans = Transformer(out_channels, att_heads, dpr)
    
    def forward(self, x, scale_img="none"):
        x1 = x.permute(0, 2, 3, 1)
        x1 = self.layernorm(x1)
        x1 = x1.permute(0, 3, 1, 2)
        x1 = torch.cat((self.layer1(scale_img), x1), axis=1)
        x1 = self.layer2(x1)
        x1 = self.layer3(x1)
        x1 = self.trans(x1)
        return x1



class FCT(nn.Module):
    def __init__(self,args):
        super().__init__()

        # attention heads and filters per block
        att_heads = [2, 4, 8, 8, 16, 8, 8, 4, 2]
        filters = [16, 32, 64, 128, 384, 128, 64, 32, 16]

        # number of blocks used in the model
        blocks = len(filters)

        stochastic_depth_rate = 0.0

        #probability for each block
        dpr = [x for x in np.linspace(0, stochastic_depth_rate, blocks)]

        self.drp_out = 0.3

        # Multi-scale input
        self.scale_img = nn.AvgPool2d(2,2)   

        # model
        self.block_1 = Block_encoder_without_skip(1, filters[0], att_heads[0], dpr[0])
        self.block_2 = Block_encoder_with_skip(filters[0], filters[1], att_heads[1], dpr[1])
        self.block_3 = Block_encoder_with_skip(filters[1], filters[2], att_heads[2], dpr[2])
        self.block_4 = Block_encoder_with_skip(filters[2], filters[3], att_heads[3], dpr[3])
        self.block_5 = Block_encoder_without_skip(filters[3], filters[4], att_heads[4], dpr[4])
        self.block_6 = Block_decoder(filters[4], filters[5], att_heads[5], dpr[5])
        self.block_7 = Block_decoder(filters[5], filters[6], att_heads[6], dpr[6])
        self.block_8 = Block_decoder(filters[6], filters[7], att_heads[7], dpr[7])
        self.block_9 = Block_decoder(filters[7], filters[8], att_heads[8], dpr[8])

        self.ds7 = DS_out(filters[6], 4)
        self.ds8 = DS_out(filters[7], 4)
        self.ds9 = DS_out(filters[8], 4)
        
    def forward(self,x):

        # Multi-scale input
        scale_img_2 = self.scale_img(x) # x shape[batch_size,channel(1),224,224]
        scale_img_3 = self.scale_img(scale_img_2) # shape[batch,1,56,56]
        scale_img_4 = self.scale_img(scale_img_3) # shape[batch,1,28,28] 

        x = self.block_1(x)
        # print(f"Block 1 out -> {list(x.size())}")
        skip1 = x
        x = self.block_2(x, scale_img_2)
        # print(f"Block 2 out -> {list(x.size())}")
        skip2 = x
        x = self.block_3(x, scale_img_3)
        # print(f"Block 3 out -> {list(x.size())}")
        skip3 = x
        x = self.block_4(x, scale_img_4)
        # print(f"Block 4 out -> {list(x.size())}")
        skip4 = x

        
        x = self.block_5(x)
        # print(f"Block 5 out -> {list(x.size())}")
        x = self.block_6(x, skip4)
        # print(f"Block 6 out -> {list(x.size())}")
        x = self.block_7(x, skip3)
        # print(f"Block 7 out -> {list(x.size())}")
        skip7 = x
        x = self.block_8(x, skip2)
        # print(f"Block 8 out -> {list(x.size())}")
        skip8 = x
        x = self.block_9(x, skip1)
        # print(f"Block 9 out -> {list(x.size())}")
        skip9 = x

        out7 = self.ds7(skip7)
        # print(f"DS 7 out -> {list(out7.size())}")
        out8 = self.ds8(skip8)
        # print(f"DS 8 out -> {list(out8.size())}")
        out9 = self.ds9(skip9)
        # print(f"DS 9 out -> {list(out9.size())}")

        return out7, out8, out9
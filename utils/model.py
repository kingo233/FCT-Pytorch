import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
import lightning as L
from torchvision.ops.stochastic_depth import StochasticDepth


class Convolutional_Attention(nn.Module):
    def __init__(self,
                 channels,
                 num_heads,
                 img_size,
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
            nn.ReLU(),
            nn.Dropout(0.3)
        )
        self.layernorm_q = nn.LayerNorm([channels,img_size,img_size], eps=1e-5)

        self.layer_k = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size, stride_kv, padding_kv, bias=attention_bias, groups=channels),
            nn.ReLU(),
            nn.Dropout(0.3)
        )
        self.layernorm_k = nn.LayerNorm([channels,img_size,img_size], eps=1e-5)

        self.layer_v = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size, stride_kv, padding_kv, bias=attention_bias, groups=channels),
            nn.ReLU(),
            nn.Dropout(0.3)
        )
        self.layernorm_v = nn.LayerNorm([channels,img_size,img_size], eps=1e-5)
        
        self.attention = nn.MultiheadAttention(embed_dim=channels, 
                                               bias=attention_bias, 
                                               batch_first=True,
                                               dropout=self.proj_drop,
                                               num_heads=self.num_heads)

    def _build_projection(self, x, mode):
        # x shape [batch,channel,size,size]
        # mode:0->q,1->k,2->v,for torch.script can not script str

        if mode == 0:
            x1 = self.layer_q(x)
            proj = self.layernorm_q(x1)
        elif mode == 1:
            x1 = self.layer_k(x)
            proj = self.layernorm_k(x1)
        elif mode == 2:
            x1 = self.layer_v(x)
            proj = self.layernorm_v(x1)

        return proj

    def get_qkv(self, x):
        q = self._build_projection(x, 0)
        k = self._build_projection(x, 1)
        v = self._build_projection(x, 2)

        return q, k, v

    def forward(self, x):
        q, k, v = self.get_qkv(x)
        q = q.view(q.shape[0], q.shape[1], q.shape[2]*q.shape[3])
        k = k.view(k.shape[0], k.shape[1], k.shape[2]*k.shape[3])
        v = v.view(v.shape[0], v.shape[1], v.shape[2]*v.shape[3])
        q = q.permute(0, 2, 1)
        k = k.permute(0, 2, 1)
        v = v.permute(0, 2, 1)
        x1 = self.attention(query=q, value=v, key=k, need_weights=False)

        x1 = x1[0].permute(0, 2, 1)
        x1 = x1.view(x1.shape[0], x1.shape[1], np.sqrt(
            x1.shape[2]).astype(int), np.sqrt(x1.shape[2]).astype(int))

        return x1


class Transformer(nn.Module):

    def __init__(self,
                 # in_channels,
                 out_channels,
                 num_heads,
                 dpr,
                 img_size,
                 proj_drop=0.5,
                 attention_bias=True,
                 padding_q="same",
                 padding_kv="same",
                 stride_kv=1,
                 stride_q=1):
        super().__init__()

        self.attention_output = Convolutional_Attention(channels=out_channels,
                                                        num_heads=num_heads,
                                                        img_size=img_size,
                                                        proj_drop=proj_drop,
                                                        padding_q=padding_q,
                                                        padding_kv=padding_kv,
                                                        stride_kv=stride_kv,
                                                        stride_q=stride_q,
                                                        attention_bias=attention_bias,
                                                        )

        self.stochastic_depth = StochasticDepth(dpr,mode='batch')
        self.conv1 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding="same")
        self.layernorm = nn.LayerNorm([out_channels, img_size, img_size])
        self.wide_focus = Wide_Focus(out_channels, out_channels)

    def forward(self, x):
        x1 = self.attention_output(x)
        x1 = self.stochastic_depth(x1)
        x2 = self.conv1(x1) + x

        x3 = self.layernorm(x2)
        x3 = self.wide_focus(x3)
        x3 = self.stochastic_depth(x3)

        out = x3 + x2
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
            nn.Dropout(0.3)
        )
        self.layer_dilation2 = nn.Sequential(
            nn.Conv2d(in_channels,out_channels, kernel_size=3, stride=1, padding="same", dilation=2),
            nn.GELU(),
            nn.Dropout(0.3)
        )
        self.layer_dilation3 = nn.Sequential(
            nn.Conv2d(in_channels,out_channels, kernel_size=3, stride=1, padding="same", dilation=3),
            nn.GELU(),
            nn.Dropout(0.3)
        )
        self.layer4 = nn.Sequential(
            nn.Conv2d(in_channels,out_channels, kernel_size=3, stride=1, padding="same"),
            nn.GELU(),
            nn.Dropout(0.3)
        )

    def forward(self, x):
        x1 = self.layer1(x)
        x2 = self.layer_dilation2(x)
        x3 = self.layer_dilation3(x)
        added = x1 + x2 + x3
        x_out = self.layer4(added)
        return x_out


class Block_decoder(nn.Module):
    def __init__(self, in_channels, out_channels, att_heads, dpr, img_size):
        super().__init__()
        self.layernorm = nn.LayerNorm([in_channels, img_size, img_size])
        # img size *= 2
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
        self.trans = Transformer(out_channels, att_heads, dpr, img_size * 2)

    def forward(self, x, skip):
        x1 = self.layernorm(x)
        x1 = self.upsample(x1)
        x1 = self.layer1(x1)
        x1 = torch.cat((skip, x1), axis=1)
        x1 = self.layer2(x1)
        x1 = self.layer3(x1)
        out = self.trans(x1)
        return out


class DS_out(nn.Module):
    def __init__(self, in_channels, out_channels, img_size):
        super().__init__()
        # img size *= 2
        self.upsample = nn.Upsample(scale_factor=2)
        self.layernorm = nn.LayerNorm([in_channels,img_size*2,img_size*2], eps=1e-5)
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding="same"),
            nn.ReLU()
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding="same"),
            nn.ReLU()
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding="same"),
        )

    def forward(self, x):
        x1 = self.upsample(x)
        x1 = self.layernorm(x1)
        x1 = self.conv1(x1)
        x1 = self.conv2(x1)
        out = self.conv3(x1)

        return out


class Block_encoder_without_skip(nn.Module):
    def __init__(self, in_channels, out_channels, att_heads, dpr, img_size):
        super().__init__()
        """LayerNorm Example
        >>> # Image Example
        >>> N, C, H, W = 20, 5, 10, 10
        >>> input = torch.randn(N, C, H, W)
        >>> # Normalize over the last three dimensions (i.e. the channel and spatial dimensions)
        >>> # as shown in the image below
        >>> layer_norm = nn.LayerNorm([C, H, W])
        >>> output = layer_norm(input)
        """
        self.layernorm = nn.LayerNorm([in_channels, img_size, img_size])
        self.layer1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding="same"),
            nn.ReLU()
        )
        # img_size => img_size // 2
        self.layer2 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding="same"),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.MaxPool2d((2, 2))
        )
        self.trans = Transformer(out_channels, att_heads, dpr, img_size // 2)

    def forward(self, x):
        x = self.layernorm(x)
        x1 = self.layer1(x)
        x1 = self.layer2(x1)
        x1 = self.trans(x1)
        return x1


class Block_encoder_with_skip(nn.Module):
    def __init__(self, in_channels, out_channels, att_heads, dpr, img_size):
        super().__init__()
        self.layernorm = nn.LayerNorm([in_channels, img_size, img_size])
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, in_channels, kernel_size=3, stride=1, padding="same"),
            nn.ReLU()
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(in_channels * 2, out_channels, kernel_size=3, stride=1, padding="same"),
            nn.ReLU()
        )
        # image size /= 2
        self.layer3 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding="same"),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.MaxPool2d((2, 2))
        )
        self.trans = Transformer(out_channels, att_heads, dpr, img_size // 2)

    def forward(self, x, scale_img):
        x1 = self.layernorm(x)
        x1 = torch.cat((self.layer1(scale_img), x1), axis=1)
        x1 = self.layer2(x1)
        x1 = self.layer3(x1)
        x1 = self.trans(x1)
        return x1


class FCT(L.LightningModule):
    def __init__(self, args):
        super().__init__()

        self.drp_out = 0.3
        self.img_size = args.img_size
        self.loss_fn = nn.BCEWithLogitsLoss()
        self.args = args

        # attention heads and filters per block
        att_heads = [2, 4, 8, 16, 32, 16, 8, 4, 2]
        filters = [32, 64, 128, 256, 512, 256, 128, 64, 32]

        # number of blocks used in the model
        blocks = len(filters)

        stochastic_depth_rate = 0.5

        # probability for each block
        dpr = [x for x in np.linspace(0, stochastic_depth_rate, blocks)]

        # Multi-scale input
        self.scale_img = nn.AvgPool2d(2, 2)

        # model
        # [N,1,img_size,img_size] => [N,filters[0],img_size // 2,img_size // 2]
        self.block_1 = Block_encoder_without_skip(1, filters[0], att_heads[0], dpr[0], self.img_size)
        self.block_2 = Block_encoder_with_skip(filters[0], filters[1], att_heads[1], dpr[1], self.img_size // 2)
        self.block_3 = Block_encoder_with_skip(filters[1], filters[2], att_heads[2], dpr[2], self.img_size // 4)
        self.block_4 = Block_encoder_with_skip(filters[2], filters[3], att_heads[3], dpr[3], self.img_size // 8)
        self.block_5 = Block_encoder_without_skip(filters[3], filters[4], att_heads[4], dpr[4], self.img_size // 16)
        self.block_6 = Block_decoder(filters[4], filters[5], att_heads[5], dpr[5], self.img_size // 32)
        self.block_7 = Block_decoder(filters[5], filters[6], att_heads[6], dpr[6], self.img_size // 16)
        self.block_8 = Block_decoder(filters[6], filters[7], att_heads[7], dpr[7], self.img_size // 8)
        self.block_9 = Block_decoder(filters[7], filters[8], att_heads[8], dpr[8], self.img_size // 4)

        self.ds7 = DS_out(filters[6], 4, self.img_size // 8)
        self.ds8 = DS_out(filters[7], 4, self.img_size // 4)
        self.ds9 = DS_out(filters[8], 4, self.img_size // 2)

    def forward(self, x):

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

    def training_step(self, batch, batch_idx):
        x, y = batch
        pred_y = self(x)
        # dsn
        down1 = F.interpolate(y, self.img_size // 2)
        down2 = F.interpolate(y, self.img_size // 4)
        loss = (self.loss_fn(pred_y[2], y) * 0.57 + self.loss_fn(pred_y[1], down1) * 0.29 + self.loss_fn(pred_y[0], down2) * 0.14)
        self.log("loss/train_loss", loss,on_epoch=True)

        # train dice
        y_pred = torch.argmax(pred_y[2], axis=1)
        y_pred_onehot = F.one_hot(y_pred, 4).permute(0, 3, 1, 2)
        dice = self.compute_dice(y_pred_onehot, y)
        dice_LV = dice[3]
        dice_RV = dice[1]
        dice_MYO = dice[2]
        self.log('dice/all_train_dice', dice[1:].mean(), on_epoch=True)
        self.log('dice/train_LV_dice', dice_LV, on_epoch=True)
        self.log('dice/train_RV_dice', dice_RV, on_epoch=True)
        self.log('dice/train_MYO_dice', dice_MYO, on_epoch=True)
        # save grad
        for name, params in self.named_parameters():
            if params.grad is not None:
                self.log(f'abs_{name}',params.grad.abs().mean(), on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        pred_y = self(x)
        # dsn
        down1 = F.interpolate(y, self.img_size // 2)
        down2 = F.interpolate(y, self.img_size // 4)
        loss = (self.loss_fn(pred_y[2], y) * 0.57 + self.loss_fn(pred_y[1], down1) * 0.29 + self.loss_fn(pred_y[0], down2) * 0.14)
        self.log("loss/validation_loss", loss,on_epoch=True)

        # train dice
        y_pred = torch.argmax(pred_y[2], axis=1)
        y_pred_onehot = F.one_hot(y_pred, 4).permute(0, 3, 1, 2)
        dice = self.compute_dice(y_pred_onehot, y)
        dice_LV = dice[3]
        dice_RV = dice[1]
        dice_MYO = dice[2]
        self.log('dice/all_validate_dice', dice[1:].mean(), on_epoch=True)
        self.log('dice/LV_dice', dice_LV, on_epoch=True)
        self.log('dice/RV_dice', dice_RV, on_epoch=True)
        self.log('dice/MYO_dice', dice_MYO, on_epoch=True)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.args.lr)
        return {
            "optimizer":    optimizer,
            "lr_scheduler": {
                "scheduler": torch.optim.lr_scheduler.ReduceLROnPlateau(
                                optimizer, mode='min', factor=self.args.lr_factor, min_lr=self.args.min_lr,patience=5),
                "monitor": "loss/train_loss",
                "name": "lr"
            }
        }
    

    @torch.no_grad()
    def compute_dice(self, pred_y, y):
        """
        Computes the Dice coefficient for each class in the ACDC dataset.
        Assumes binary masks with shape (num_masks, num_classes, height, width).
        """
        epsilon = 1e-6
        num_masks = pred_y.shape[0]
        num_classes = pred_y.shape[1]
        dice_scores = torch.zeros((num_classes,), device=self.device)

        for c in range(num_classes):
            intersection = torch.sum(pred_y[:, c] * y[:, c])
            sum_masks = torch.sum(pred_y[:, c]) + torch.sum(y[:, c])
            dice_scores[c] = (2. * intersection + epsilon) / (sum_masks + epsilon)

        return dice_scores

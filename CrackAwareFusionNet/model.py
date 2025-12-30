"""
REFERENCES:
1. Hybrid Encoder (MiT & ResNet): 
   - Inspired by "Hybrid-Segmentor" (https://github.com/junegoo94/Hybrid-Segmentor)
   - Utilizes Mix-Transformer (MiT) for global context and ResNet for local features.
2. Fusion & Decoder Path:
   - Based on "TransFuse" (https://github.com/Rayicer/TransFuse)
   - Implements Bi-directional Fusion and Attention-Gate based skip connections.
3. Crack Attention Modules (CrackAM):
   - Logic adapted from "HACNetV2" (https://github.com/hanshenchen/HACNetV2)
   - Designed to capture elongated and directional crack structures.
"""
from math import sqrt

import torch
from torch import nn, optim
import torch.nn.functional as F
from torch.optim import lr_scheduler
from torchvision.models import resnet18, ResNet18_Weights
import pytorch_lightning as pl
from torchmetrics.classification import BinaryAccuracy, BinaryF1Score, BinaryJaccardIndex, BinaryPrecision, BinaryRecall
from einops import rearrange

import config
from metric import DiceBCELoss, DiceLoss

DEVICE = config.DEVICE

class LayerNorm2d(nn.LayerNorm):
    def forward(self, x):
        x = rearrange(x, "b c h w -> b h w c")
        x = super().forward(x)
        x = rearrange(x, "b h w c -> b c h w")
        return x
    
class DepthWiseConv(nn.Module):
    def __init__(self, in_dim, out_dim, kernel, padding, stride=1, bias=True):
        super(DepthWiseConv, self).__init__()
        self.DW_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim,
                                 kernel_size=kernel, stride=stride, 
                                 padding=padding, groups=in_dim, bias=bias)
        self.PW_conv = nn.Conv2d(in_channels=in_dim, out_channels=out_dim,
                                 kernel_size=1, bias=bias)
    
    def forward(self, x):
        x = self.DW_conv(x)
        x = self.PW_conv(x)
        return x

class OverlapPatchEmbedding(nn.Module):
    def __init__(self, kernel, stride, padding, in_dim, out_dim):
        super(OverlapPatchEmbedding, self).__init__()
        self.overlap_patches = nn.Unfold(kernel_size=kernel, stride=stride, padding=padding)
        self.embedding = nn.Conv2d(in_dim * kernel**2, out_dim, 1)

    def forward(self, x):
        h, w = x.shape[-2:]
        x = self.overlap_patches(x)
        n_patches = x.shape[-1]
        divider = int(sqrt(h * w / n_patches))
        x = rearrange(x, 'b c (h w) -> b c h w', h=h // divider)
        x = self.embedding(x)
        return x

class EfficientMSA(nn.Module):
    def __init__(self, dim, n_heads, reduction_ratio):
        super(EfficientMSA, self).__init__()
        self.ln = LayerNorm2d(dim)
        self.reshaping_k = nn.Conv2d(dim, dim, kernel_size=reduction_ratio, stride=reduction_ratio)
        self.reshaping_v = nn.Conv2d(dim, dim, kernel_size=reduction_ratio, stride=reduction_ratio)
        self.attention = nn.MultiheadAttention(embed_dim=dim, num_heads=n_heads, batch_first=True)

    def forward(self, x):
        n, c, h, w = x.shape
        x = self.ln(x)
        reshaped_k = self.reshaping_k(x)
        reshaped_v = self.reshaping_v(x)
        reshaped_k = rearrange(reshaped_k, "b c h w -> b (h w) c")
        reshaped_v = rearrange(reshaped_v, "b c h w -> b (h w) c")
        q = rearrange(x, "b c h w -> b (h w) c")
        output, _ = self.attention(q, reshaped_k, reshaped_v)
        output = rearrange(output, "b (h w) c -> b c h w", h=h, w=w)
        return output

class MixFFN(nn.Module):
    def __init__(self, dim, expansion_factor):
        super(MixFFN, self).__init__()
        latent_dim = dim * expansion_factor
        self.ln = LayerNorm2d(dim)
        self.mixffn = nn.Sequential(
            nn.Conv2d(dim, latent_dim, 1),
            DepthWiseConv(latent_dim, latent_dim, kernel=3, padding=1),
            nn.GELU(),
            nn.Conv2d(latent_dim, dim, 1)
        )

    def forward(self, x):
        x = self.ln(x)
        x = self.mixffn(x)
        return x

class MiT(nn.Module):
    def __init__(self, channels, dims, n_heads, expansion, reduction_ratio, n_layers):
        super(MiT, self).__init__()
        kernel_stride_pad = ((3, 2, 1), (3, 2, 1), (3, 2, 1), (3, 2, 1), (3, 2, 1))
        dims = (channels, *dims)
        dim_pairs = list(zip(dims[:-1], dims[1:]))

        self.stages = nn.ModuleList([])
        for (in_dim, out_dim), (kernel, stride, padding), n_layers, expansion, n_heads, reduction_ratio in zip(
            dim_pairs, kernel_stride_pad, n_layers, expansion, n_heads, reduction_ratio
        ):
            overlapping = OverlapPatchEmbedding(kernel, stride, padding, in_dim, out_dim)
            layers = nn.ModuleList([])
            for _ in range(n_layers):
                layers.append(nn.ModuleList([
                    EfficientMSA(dim=out_dim, n_heads=n_heads, reduction_ratio=reduction_ratio),
                    MixFFN(dim=out_dim, expansion_factor=expansion)
                ]))
            self.stages.append(nn.ModuleList([overlapping, layers]))

    def forward(self, x):
        layer_outputs = []
        for overlapping, layers in self.stages:
            x = overlapping(x)
            for (attention, ffn) in layers:
                x = attention(x) + x
                x = ffn(x) + x
            layer_outputs.append(x)
        return layer_outputs

resnet_encoder = resnet18(weights=ResNet18_Weights.DEFAULT)

class ResNetEncoder(nn.Module):
    def __init__(self, encoder=resnet_encoder):
        super(ResNetEncoder, self).__init__()
        self.encoder1 = nn.Sequential(encoder.conv1, encoder.bn1, encoder.relu)
        self.mp = encoder.maxpool
        self.encoder2 = encoder.layer1
        self.encoder3 = encoder.layer2
        self.encoder4 = encoder.layer3
        self.encoder5 = encoder.layer4

    def forward(self, x):
        output1 = self.encoder1(x)
        output2 = self.mp(output1)
        output2 = self.encoder2(output2)
        output3 = self.encoder3(output2)
        output4 = self.encoder4(output3)
        output5 = self.encoder5(output4)
        return output1, output2, output3, output4, output5

class Conv(nn.Module):
    def __init__(self, inp_dim, out_dim, kernel_size=3, stride=1, bn=False, relu=True, bias=True):
        super(Conv, self).__init__()
        self.inp_dim = inp_dim
        self.conv = nn.Conv2d(inp_dim, out_dim, kernel_size, stride, padding=(kernel_size - 1) // 2, bias=bias)
        self.relu = None
        self.bn = None
        if relu:
            self.relu = nn.ReLU(inplace=True)
        if bn:
            self.bn = nn.BatchNorm2d(out_dim)

    def forward(self, x):
        assert x.size()[1] == self.inp_dim, "{} {}".format(x.size()[1], self.inp_dim)
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x

class CrackAM(nn.Module):
    def __init__(self, channels, rate=1, add_maxpool=False, **_):
        super(CrackAM, self).__init__()
        self.fc = nn.Conv2d(int(channels), channels, kernel_size=1, padding=0)
        self.gate = nn.Sigmoid()

    def forward(self, x):
        max_pool_h = torch.max(x, dim=3)[0]
        max_pool_v = torch.max(x, dim=2)[0]
        xtmp = torch.concat((max_pool_h, max_pool_v), dim=2)
        x_se = xtmp.mean((2), keepdim=True).unsqueeze(-1)
        x_se = self.fc(x_se)
        return x * self.gate(x_se)

class CrackSPAM(nn.Module):
    def __init__(self, kernel_size=7):
        super().__init__()
        self.compress = lambda x: torch.cat((torch.mean(x, dim=1, keepdim=True), torch.max(x, dim=1, keepdim=True)[0]), dim=1)
        self.conv_h = nn.Conv2d(2, 1, kernel_size=(kernel_size, 1), padding=(kernel_size // 2, 0))
        self.conv_w = nn.Conv2d(2, 1, kernel_size=(1, kernel_size), padding=(0, kernel_size // 2))
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x_compress = self.compress(x)
        h_attn = self.conv_h(x_compress)
        w_attn = self.conv_w(x_compress)
        attn = self.sigmoid(h_attn + w_attn)
        return x * attn

class CrackAwareBiFusionModule(nn.Module):
    def __init__(self, ch_1, ch_2, r_2, ch_int, ch_out, drop_rate=0.):
        super(CrackAwareBiFusionModule, self).__init__()
        self.crack_spm = CrackSPAM(kernel_size=7)
        self.crack_am = CrackAM(channels=ch_2)
        self.W_g = Conv(ch_1, ch_int, 1, bn=True, relu=False)
        self.W_x = Conv(ch_2, ch_int, 1, bn=True, relu=False)
        self.W = Conv(ch_int, ch_int, 3, bn=True, relu=True)
        self.relu = nn.ReLU(inplace=True)
        self.sigmoid = nn.Sigmoid()
        self.residual = Residual(ch_1 + ch_2 + ch_int, ch_out)
        self.dropout = nn.Dropout2d(drop_rate)
        self.drop_rate = drop_rate

    def forward(self, g, x):
        W_g = self.W_g(g)
        W_x = self.W_x(x)
        bp = self.W(W_g * W_x)
        g = self.crack_spm(g)
        x = self.crack_am(x)
        fuse = self.residual(torch.cat([g, x, bp], 1))
        return self.dropout(fuse) if self.drop_rate > 0 else fuse

class Attention_block(nn.Module):
    def __init__(self, F_g, F_l, F_int):
        super(Attention_block, self).__init__()
        self.W_g = nn.Sequential(nn.Conv2d(F_g, F_int, kernel_size=1, stride=1, padding=0, bias=True), nn.BatchNorm2d(F_int))
        self.W_x = nn.Sequential(nn.Conv2d(F_l, F_int, kernel_size=1, stride=1, padding=0, bias=True), nn.BatchNorm2d(F_int))
        self.psi = nn.Sequential(nn.Conv2d(F_int, 1, kernel_size=1, stride=1, padding=0, bias=True), nn.BatchNorm2d(1), nn.Sigmoid())
        self.relu = nn.ReLU(inplace=True)

    def forward(self, g, x):
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)
        return x * psi

class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels)
        )
        self.identity = nn.Sequential(nn.Conv2d(in_channels, out_channels, kernel_size=1, padding=0), nn.BatchNorm2d(out_channels))
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.double_conv(x) + self.identity(x))

class Up(nn.Module):
    def __init__(self, in_ch1, out_ch, in_ch2=0, attn=False):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        if in_ch2 > 0:
            self.conv = DoubleConv(in_ch1 + in_ch2, out_ch)
        else:
            self.conv = DoubleConv(in_ch1, out_ch)
        if attn:
            self.attn_block = Attention_block(F_g=in_ch1, F_l=in_ch2, F_int=min(in_ch1, in_ch2) if in_ch2 > 0 else in_ch1)
        else:
            self.attn_block = None

    def forward(self, x1, x2=None):
        x1 = self.up(x1)
        if x2 is not None:
            diffY = x2.size()[2] - x1.size()[2]
            diffX = x2.size()[3] - x1.size()[3]
            x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2])
            if self.attn_block is not None:
                x2 = self.attn_block(x1, x2)
            x1 = torch.cat([x2, x1], dim=1)
        return self.conv(x1)

class Residual(nn.Module):
    def __init__(self, inp_dim, out_dim):
        super(Residual, self).__init__()
        self.relu = nn.ReLU(inplace=True)
        self.bn1 = nn.BatchNorm2d(inp_dim)
        self.conv1 = Conv(inp_dim, int(out_dim / 2), 1, relu=False)
        self.bn2 = nn.BatchNorm2d(int(out_dim / 2))
        self.conv2 = Conv(int(out_dim / 2), int(out_dim / 2), 3, relu=False)
        self.bn3 = nn.BatchNorm2d(int(out_dim / 2))
        self.conv3 = Conv(int(out_dim / 2), out_dim, 1, relu=False)
        self.skip_layer = Conv(inp_dim, out_dim, 1, relu=False)
        self.need_skip = (inp_dim != out_dim)

    def forward(self, x):
        residual = self.skip_layer(x) if self.need_skip else x
        out = self.bn1(x)
        out = self.relu(out)
        out = self.conv1(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn3(out)
        out = self.relu(out)
        out = self.conv3(out)
        out += residual
        return out

class CrackAwareFusionNet(pl.LightningModule):
    def __init__(self, channels=3, dims=(64, 128, 256, 512), n_heads=(1, 2, 8, 8), expansion=(8, 8, 4, 4), reduction_ratio=(8, 4, 2, 1), n_layers=(2, 2, 2, 2), learning_rate=1e-4):
        super(CrackAwareFusionNet, self).__init__()
        self.save_hyperparameters()

        self.mix_transformer = MiT(channels, dims, n_heads, expansion, reduction_ratio, n_layers)
        self.cnn_encoder = ResNetEncoder()

        self.fusion1 = CrackAwareBiFusionModule(ch_1=64, ch_2=64, r_2=4, ch_int=32, ch_out=32)
        self.fusion2 = CrackAwareBiFusionModule(ch_1=64, ch_2=128, r_2=4, ch_int=64, ch_out=64)
        self.fusion3 = CrackAwareBiFusionModule(ch_1=128, ch_2=256, r_2=4, ch_int=128, ch_out=128)
        self.fusion4 = CrackAwareBiFusionModule(ch_1=256, ch_2=512, r_2=4, ch_int=256, ch_out=256)

        self.up5 = Up(512, 256, 256, attn=True)
        self.up4 = Up(256, 128, 128, attn=True)
        self.up3 = Up(128, 64, 64, attn=True)
        self.up2 = Up(64, 64, 32, attn=True)
        self.up1 = Up(64, 64, attn=False)

        self.final = nn.Sequential(Conv(64, 8, 3, bn=True, relu=True), Conv(8, 1, 1, bn=False, relu=False))

        self.loss_fn = DiceBCELoss()

        self.accuracy = BinaryAccuracy()
        self.f1_score = BinaryF1Score()
        self.recall = BinaryRecall()
        self.precision = BinaryPrecision()

        self.jaccard_ind = BinaryJaccardIndex()
        self.dice_loss_fn = DiceLoss()

        self.lr = learning_rate

    def forward(self, x):
        mit_features = self.mix_transformer(x)
        cnn_features = self.cnn_encoder(x)

        fused1 = self.fusion1(cnn_features[0], mit_features[0])
        fused2 = self.fusion2(cnn_features[1], mit_features[1])
        fused3 = self.fusion3(cnn_features[2], mit_features[2])
        fused4 = self.fusion4(cnn_features[3], mit_features[3])

        d5 = self.up5(cnn_features[4], fused4)
        d4 = self.up4(d5, fused3)
        d3 = self.up3(d4, fused2)
        d2 = self.up2(d3, fused1)
        d1 = self.up1(d2)

        out = self.final(d1)
        return out, d1, d2, d3, d4, d5

    def training_step(self, batch, batch_idx):
        loss, pred, y = self._common_step(batch, batch_idx)
        accuracy = self.accuracy(pred, y)
        f1_score = self.f1_score(pred, y)
        re = self.recall(pred, y)
        precision = self.precision(pred, y)
        jaccard = self.jaccard_ind(pred, y)
        dice_loss = self.dice_loss_fn(pred, y)
        dice = 1.0 - dice_loss

        self.log_dict({
            'train_loss': loss,
            'train_accuracy': accuracy,
            'train_f1_score': f1_score,
            'train_precision': precision,
            'train_recall': re,
            'train_IOU': jaccard,
            'train_dice': dice
        }, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)

        return loss

    def validation_step(self, batch, batch_idx):
        loss, pred, y = self._common_step(batch, batch_idx)
        accuracy = self.accuracy(pred, y)
        f1_score = self.f1_score(pred, y)
        re = self.recall(pred, y)
        precision = self.precision(pred, y)
        jaccard = self.jaccard_ind(pred, y)
        dice_loss = self.dice_loss_fn(pred, y)
        dice = 1.0 - dice_loss

        self.log_dict({
            'val_loss': loss,
            'val_accuracy': accuracy,
            'val_f1_score': f1_score,
            'val_precision': precision,
            'val_recall': re,
            'val_IOU': jaccard,
            'val_dice': dice
        }, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)

        return loss

    def test_step(self, batch, batch_idx):
        loss, pred, y = self._common_step(batch, batch_idx)
        accuracy = self.accuracy(pred, y)
        f1_score = self.f1_score(pred, y)
        re = self.recall(pred, y)
        precision = self.precision(pred, y)
        jaccard = self.jaccard_ind(pred, y)
        dice_loss = self.dice_loss_fn(pred, y)
        dice = 1.0 - dice_loss

        self.log_dict({
            'test_loss': loss,
            'test_accuracy': accuracy,
            'test_f1_score': f1_score,
            'test_precision': precision,
            'test_recall': re,
            'test_IOU': jaccard,
            'test_dice': dice
        }, on_step=False, on_epoch=True, prog_bar=False, sync_dist=True)

        return loss

    def _common_step(self, batch, batch_idx):
        x, y = batch
        x = x.to(config.DEVICE)
        y = y.float().unsqueeze(1).to(config.DEVICE)
        pred_lst = self.forward(x)
        logits = pred_lst[0]

        loss = self.loss_fn(logits, y, weight=0.5)
        pred = torch.sigmoid(logits)
        pred = (pred > 0.5).float()
        return loss, pred, y

    def predict_step(self, batch, batch_idx):
        x, y = batch
        x = x.to(config.DEVICE)
        logits = self.forward(x)[0]
        preds = torch.sigmoid(logits)
        preds = (preds > 0.5).float()
        return preds

    def configure_optimizers(self):
        optimizer = optim.AdamW(self.parameters(), lr=self.lr, weight_decay=1e-5)
        lr_schedule = lr_scheduler.ReduceLROnPlateau(optimizer, patience=3)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": lr_schedule,
                "monitor": "val_loss",
                "interval": "epoch",
                "frequency": 1
            },

        }

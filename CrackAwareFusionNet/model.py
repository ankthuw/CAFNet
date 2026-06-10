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
import torch
from torch import nn, optim
import torch.nn.functional as F
from torch.optim import lr_scheduler
from torchvision.models import resnet18, ResNet18_Weights
import pytorch_lightning as pl
from torchmetrics.classification import BinaryAccuracy, BinaryF1Score, BinaryJaccardIndex, BinaryPrecision, BinaryRecall
from einops import rearrange

from metric import DiceBCELoss


class LayerNorm2d(nn.LayerNorm):
    def forward(self, x):
        x = rearrange(x, "b c h w -> b h w c")
        x = super().forward(x)
        x = rearrange(x, "b h w c -> b c h w")
        return x


class DWConv(nn.Module):
    def __init__(self, dim, kernel_size=3, padding=1):
        super().__init__()
        self.dw = nn.Conv2d(dim, dim, kernel_size, padding=padding, groups=dim)
        self.pw = nn.Conv2d(dim, dim, 1)

    def forward(self, x):
        return self.pw(self.dw(x))


class OverLapPatchEmbedding(nn.Module):
    def __init__(self, in_dim, out_dim, kernel_size, stride, padding):
        super().__init__()
        self.proj = nn.Conv2d(
            in_dim, out_dim,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding
        )
        self.norm = LayerNorm2d(out_dim)

    def forward(self, x):
        x = self.proj(x)
        x = self.norm(x)
        return x


class EfficientMSA(nn.Module):
    def __init__(self, dim, num_heads, reduction_ratio):
        super().__init__()
        self.norm = LayerNorm2d(dim)
        self.sr = nn.Conv2d(dim, dim, kernel_size=reduction_ratio, stride=reduction_ratio)
        self.sr_norm = LayerNorm2d(dim)
        self.attn = nn.MultiheadAttention(
            embed_dim=dim,
            num_heads=num_heads,
            batch_first=True
        )
        self.proj = nn.Conv2d(dim, dim, 1)

    def forward(self, x):
        b, c, h, w = x.shape
        x_norm = self.norm(x)
        q = rearrange(x_norm, "b c h w -> b (h w) c")
        kv = self.sr(x_norm)
        kv = self.sr_norm(kv)
        kv = rearrange(kv, "b c h w -> b (h w) c")
        out, _ = self.attn(q, kv, kv)
        out = rearrange(out, "b (h w) c -> b c h w", h=h, w=w)
        out = self.proj(out)
        return out


class MixFFN(nn.Module):
    def __init__(self, dim, expansion):
        super().__init__()
        hidden_dim = dim * expansion
        self.norm = LayerNorm2d(dim)
        self.ffn = nn.Sequential(
            nn.Conv2d(dim, hidden_dim, 1),
            DWConv(hidden_dim),
            nn.GELU(),
            nn.Conv2d(hidden_dim, dim, 1)
        )

    def forward(self, x):
        return self.ffn(self.norm(x))


class MiTBlock(nn.Module):
    def __init__(self, dim, num_heads, expansion, reduction_ratio):
        super().__init__()
        self.attn = EfficientMSA(dim, num_heads, reduction_ratio)
        self.ffn = MixFFN(dim, expansion)

    def forward(self, x):
        x = x + self.attn(x)
        x = x + self.ffn(x)
        return x


class MiT(nn.Module):
    def __init__(
        self,
        in_channels=3,
        embed_dims=(64, 64, 128, 256),
        num_heads=(1, 2, 4, 8),
        mlp_ratios=(4, 4, 4, 4),
        reduction_ratios=(8, 4, 2, 1),
        depths=(2, 2, 2, 2)
    ):
        super().__init__()
        self.stages = nn.ModuleList()
        prev_dim = in_channels

        for i in range(len(embed_dims)):
            stage = nn.ModuleList()
            stage.append(
                OverLapPatchEmbedding(
                    in_dim=prev_dim,
                    out_dim=embed_dims[i],
                    kernel_size=3,
                    stride=2,
                    padding=1
                )
            )
            blocks = nn.Sequential(*[
                MiTBlock(
                    dim=embed_dims[i],
                    num_heads=num_heads[i],
                    expansion=mlp_ratios[i],
                    reduction_ratio=reduction_ratios[i]
                )
                for _ in range(depths[i])
            ])
            stage.append(blocks)
            self.stages.append(stage)
            prev_dim = embed_dims[i]

    def forward(self, x):
        features = []
        for patch_embed, blocks in self.stages:
            x = patch_embed(x)
            x = blocks(x)
            features.append(x)
        return features


class ResNetEncoder(nn.Module):
    def __init__(self, backbone=resnet18(weights=ResNet18_Weights.DEFAULT)):
        super().__init__()
        self.stem = nn.Sequential(
            backbone.conv1,
            backbone.bn1,
            backbone.relu
        )
        self.maxpool = backbone.maxpool
        self.layer1 = backbone.layer1
        self.layer2 = backbone.layer2
        self.layer3 = backbone.layer3
        self.layer4 = backbone.layer4

    def forward(self, x):
        c1 = self.stem(x)
        c2 = self.maxpool(c1)
        c2 = self.layer1(c2)
        c3 = self.layer2(c2)
        c4 = self.layer3(c3)
        c5 = self.layer4(c4)
        return [c1, c2, c3, c4, c5]


class Conv(nn.Module):
    def __init__(self, in_dim, out_dim, kernel_size=3, stride=1, padding=1, bn=True, relu=True):
        super().__init__()
        self.conv = nn.Conv2d(in_dim, out_dim, kernel_size, stride, padding, bias=not bn)
        self.bn = nn.BatchNorm2d(out_dim) if bn else nn.Identity()
        self.relu = nn.ReLU(inplace=True) if relu else nn.Identity()

    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))


class DoubleConv(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.double_conv = nn.Sequential(
            Conv(in_dim, out_dim),
            Conv(out_dim, out_dim, relu=False)
        )
        self.skip = Conv(in_dim, out_dim, kernel_size=1, padding=0, relu=False)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.double_conv(x) + self.skip(x))


class ResidualBlock(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        mid_dim = out_dim // 2
        self.block = nn.Sequential(
            Conv(in_dim, mid_dim, 1, stride=1, padding=0, bn=True, relu=True),
            Conv(mid_dim, mid_dim, 3, stride=1, padding=1, bn=True, relu=True),
            Conv(mid_dim, out_dim, 1, stride=1, padding=0, bn=True, relu=False)
        )
        if in_dim != out_dim:
            self.skip = nn.Sequential(
                nn.Conv2d(in_dim, out_dim, 1, bias=False),
                nn.BatchNorm2d(out_dim)
            )
        else:
            self.skip = nn.Identity()
        self.final_relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.final_relu(self.block(x) + self.skip(x))


class CrackAM(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.fc = nn.Conv2d(dim, dim, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        h_pool = torch.max(x, dim=3, keepdim=True)[0]
        v_pool = torch.max(x, dim=2, keepdim=True)[0]
        se = h_pool.mean(2, keepdim=True) + v_pool.mean(3, keepdim=True)
        se = self.fc(se)
        return x * self.sigmoid(se)


class CrackSPAM(nn.Module):
    def __init__(self, kernel_size=7):
        super().__init__()
        self.compress = lambda x: torch.cat(
            (torch.mean(x, dim=1, keepdim=True), torch.max(x, dim=1, keepdim=True)[0]), dim=1
        )
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
    def __init__(self, cnn_dim, trans_dim, bi_dim, out_dim, drop=0., crackam=True, crackspam=True):
        super().__init__()
        self.am = CrackAM(trans_dim) if crackam else nn.Identity()
        self.spam = CrackSPAM() if crackspam else nn.Identity()
        self.cnn_proj = Conv(cnn_dim, bi_dim, kernel_size=1, padding=0, bn=True, relu=False)
        self.trans_proj = Conv(trans_dim, bi_dim, kernel_size=1, padding=0, bn=True, relu=False)
        self.conv = Conv(bi_dim, bi_dim, kernel_size=3, bn=True, relu=True)
        self.refine = ResidualBlock(cnn_dim + trans_dim + bi_dim, out_dim)
        self.drop = nn.Dropout2d(drop) if drop > 0 else nn.Identity()

    def forward(self, cnn, trans):
        cnn_feat = self.spam(cnn)
        trans_feat = self.am(trans)
        bi = self.conv(self.cnn_proj(cnn) * self.trans_proj(trans))
        fuse = torch.cat([cnn_feat, trans_feat, bi], dim=1)
        return self.drop(self.refine(fuse))


class AttnGate(nn.Module):
    def __init__(self, gate_dim, skip_dim, inter_dim):
        super().__init__()
        self.conv_gate = Conv(gate_dim, inter_dim, kernel_size=1, padding=0, bn=True, relu=False)
        self.conv_skip = Conv(skip_dim, inter_dim, kernel_size=1, padding=0, bn=True, relu=False)
        self.psi = nn.Sequential(
            nn.Conv2d(inter_dim, 1, kernel_size=1),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, gate, skip):
        g1 = self.conv_gate(gate)
        s1 = self.conv_skip(skip)
        combine = self.relu(g1 + s1)
        attn = self.psi(combine)
        return skip * attn


class Upsample(nn.Module):
    def __init__(self, in_dim, out_dim, skip_dim=0, attn_gate=True):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        if skip_dim > 0:
            self.conv = DoubleConv(in_dim + skip_dim, out_dim)
        else:
            self.conv = DoubleConv(in_dim, out_dim)
        if attn_gate:
            self.attn_block = AttnGate(
                gate_dim=in_dim,
                skip_dim=skip_dim,
                inter_dim=min(in_dim, skip_dim) if skip_dim > 0 else in_dim
            )
        else:
            self.attn_block = None

    def forward(self, x_dec, x_skip=None):
        x_dec = self.up(x_dec)
        if x_skip is not None:
            diff_h = x_skip.size(2) - x_dec.size(2)
            diff_w = x_skip.size(3) - x_dec.size(3)
            x_dec = F.pad(
                x_dec,
                [
                    diff_w // 2, diff_w - diff_w // 2,
                    diff_h // 2, diff_h - diff_h // 2
                ]
            )
            if self.attn_block is not None:
                x_skip = self.attn_block(x_dec, x_skip)
            x_dec = torch.cat([x_skip, x_dec], dim=1)
        return self.conv(x_dec)

class CrackAwareFusionNet(pl.LightningModule):
    def __init__(
        self,
        in_channels=3,
        embed_dims=(64, 64, 128, 256),
        num_heads=(1, 2, 4, 8),
        mlp_ratios=(4, 4, 4, 4),
        reduction_ratios=(8, 4, 2, 1),
        depths=(2, 2, 2, 2),
        crackam=True,
        crackspam=True,
        attn_gate=False,
        learning_rate=1e-4,
        weight_decay=1e-5,
    ):
        super(CrackAwareFusionNet, self).__init__()
        self.save_hyperparameters()

        self.mit = MiT(
            in_channels=in_channels,
            embed_dims=embed_dims,
            num_heads=num_heads,
            mlp_ratios=mlp_ratios,
            reduction_ratios=reduction_ratios,
            depths=depths
        )
        self.cnn = ResNetEncoder()

        self.fusion1 = CrackAwareBiFusionModule(cnn_dim=64, trans_dim=64, bi_dim=64, out_dim=64, crackspam=crackspam, crackam=crackam)
        self.fusion2 = CrackAwareBiFusionModule(cnn_dim=64, trans_dim=64, bi_dim=64, out_dim=64, crackspam=crackspam, crackam=crackam)
        self.fusion3 = CrackAwareBiFusionModule(cnn_dim=128, trans_dim=128, bi_dim=128, out_dim=128, crackspam=crackspam, crackam=crackam)
        self.fusion4 = CrackAwareBiFusionModule(cnn_dim=256, trans_dim=256, bi_dim=256, out_dim=256, crackspam=crackspam, crackam=crackam)

        self.up5 = Upsample(in_dim=512, skip_dim=256, out_dim=256, attn_gate=attn_gate)
        self.up4 = Upsample(in_dim=256, skip_dim=128, out_dim=128, attn_gate=attn_gate)
        self.up3 = Upsample(in_dim=128, skip_dim=64, out_dim=64, attn_gate=attn_gate)
        self.up2 = Upsample(in_dim=64, skip_dim=64, out_dim=64, attn_gate=attn_gate)
        self.up1 = Upsample(in_dim=64, skip_dim=0, out_dim=64, attn_gate=False)

        self.final = nn.Sequential(
            Conv(in_dim=64, out_dim=8, kernel_size=3, bn=True, relu=True),
            Conv(in_dim=8, out_dim=1, kernel_size=1, padding=0, bn=False, relu=False)
        )

        self.loss_fn = DiceBCELoss()

        self.acc = BinaryAccuracy()
        self.f1 = BinaryF1Score()
        self.re = BinaryRecall()
        self.pre = BinaryPrecision()
        self.iou = BinaryJaccardIndex()

        self.lr = learning_rate
        self.weight_decay = weight_decay

    def forward(self, x):
        mit_f = self.mit(x)
        cnn_f = self.cnn(x)

        fused1 = self.fusion1(cnn_f[0], mit_f[0])
        fused2 = self.fusion2(cnn_f[1], mit_f[1])
        fused3 = self.fusion3(cnn_f[2], mit_f[2])
        fused4 = self.fusion4(cnn_f[3], mit_f[3])

        d5 = self.up5(cnn_f[4], fused4)
        d4 = self.up4(d5, fused3)
        d3 = self.up3(d4, fused2)
        d2 = self.up2(d3, fused1)
        d1 = self.up1(d2, x_skip=None)

        out = self.final(d1)
        return out, d1, d2, d3, d4, d5

    def training_step(self, batch, batch_idx):
        loss, preds, masks = self._common_step(batch, batch_idx)

        f1 = self.f1(preds, masks)
        iou = self.iou(preds, masks)

        self.log_dict(
            {
                "train_loss": loss,
                "train_f1": f1,
                "train_iou": iou,
            },
            on_epoch=True,
            on_step=False,
            prog_bar=False,
            sync_dist=True,
        )

        return loss

    def validation_step(self, batch, batch_idx):
        loss, preds, masks = self._common_step(batch, batch_idx)

        f1 = self.f1(preds, masks)
        iou = self.iou(preds, masks)

        self.log(
            "val_loss",
            loss,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            sync_dist=True,
            rank_zero_only=True,
        )
        self.log_dict(
            {
                "val_f1": f1,
                "val_iou": iou,
            },
            on_step=False,
            on_epoch=True,
            prog_bar=False,
            sync_dist=True,
        )

    def test_step(self, batch, batch_idx):
        loss, preds, masks = self._common_step(batch, batch_idx)

        acc = self.acc(preds, masks)
        pre = self.pre(preds, masks)
        re = self.re(preds, masks)
        f1 = self.f1(preds, masks)
        iou = self.iou(preds, masks)

        self.log_dict(
            {
                "test_loss": loss,
                "test_acc": acc,
                "test_precision": pre,
                "test_recall": re,
                "test_f1": f1,
                "test_iou": iou,
            },
            on_epoch=True,
            prog_bar=True,
            sync_dist=True,
        )

    def _common_step(self, batch, batch_idx):
        imgs, masks, _ = batch
        masks = masks.float().unsqueeze(1)

        logits = self.forward(imgs)[0]
        loss = self.loss_fn(logits, masks)

        probs = torch.sigmoid(logits)
        preds = (probs > 0.5).float()

        return loss, preds, masks

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        imgs, _, _ = batch
        logits = self.forward(imgs)[0]
        probs = torch.sigmoid(logits)
        preds = (probs > 0.5).float()
        return preds

    def configure_optimizers(self):
        optimizer = optim.AdamW(
            self.parameters(),
            lr=self.lr,
            weight_decay=self.weight_decay,
        )
        scheduler = lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=0.1,
            patience=5,
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val_loss",
                "interval": "epoch",
                "frequency": 1,
            },
        }


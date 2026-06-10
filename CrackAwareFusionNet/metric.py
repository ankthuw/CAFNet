import torch
import torch.nn as nn
import torch.nn.functional as F


class DiceBCELoss(nn.Module):
    def __init__(self, dice_weight=0.5, smooth=1e-5):
        super().__init__()
        self.dice_weight = dice_weight
        self.smooth = smooth

    def forward(self, logits, targets):
        bce = F.binary_cross_entropy_with_logits(logits, targets)

        probs = torch.sigmoid(logits)
        dims = (1, 2, 3)

        intersection = (probs * targets).sum(dims)
        union = probs.sum(dims) + targets.sum(dims)

        dice_loss = 1.0 - (2.0 * intersection + self.smooth) / (union + self.smooth)
        dice_loss = dice_loss.mean()

        return self.dice_weight * dice_loss + (1.0 - self.dice_weight) * bce

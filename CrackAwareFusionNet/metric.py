import torch
import torch.nn as nn
import torch.nn.functional as F


class DiceLoss(nn.Module):
    """Dice loss for binary segmentation.

    Note: if your model already applies sigmoid, comment out the sigmoid in forward.
    """

    def __init__(self, weight=None, size_average=True):
        super().__init__()

    def forward(self, inputs, targets, smooth: float = 1.0, weight=None):
        inputs = torch.sigmoid(inputs)
        inputs = inputs.view(-1)
        targets = targets.view(-1)

        intersection = (inputs * targets).sum()
        dice = (2.0 * intersection + smooth) / (inputs.sum() + targets.sum() + smooth)
        return 1.0 - dice


class DiceBCELoss(nn.Module):
    """Combination of Dice loss and BCE (with logits).

    Use `set_debug_mode(True)` to enable printed debug info.
    """

    def __init__(self, weight=None, size_average=True):
        super().__init__()
        self.debug_mode = False

    def set_debug_mode(self, mode: bool):
        self.debug_mode = bool(mode)

    def forward(self, inputs, targets, smooth: float = 1.0, weight: float = 0.5):
        inputs_sigmoid = torch.sigmoid(inputs)

        inputs = inputs.view(-1)
        inputs_sigmoid = inputs_sigmoid.view(-1)
        targets = targets.view(-1)

        intersection = (inputs_sigmoid * targets).sum()
        dice_loss = 1.0 - (2.0 * intersection + smooth) / (inputs_sigmoid.sum() + targets.sum() + smooth)
        bce = F.binary_cross_entropy_with_logits(inputs, targets, reduction="mean")



        return weight * bce + (1.0 - weight) * dice_loss





from typing import List

from torch import nn

from src.losses.ssim import SSIM


class SSIM_Loss(SSIM):
    def forward(self, img1, img2):
        return 100 * (1 - super(SSIM_Loss, self).forward(img1, img2))


class CustomLoss(nn.Module):
    def __init__(self, losses: List[nn.Module], weights: List[float]):
        super(CustomLoss, self).__init__()
        self.losses = losses
        self.weights = weights
        assert len(losses) == len(weights)

    def forward(self, output, target):
        total = 0
        for w, f_loss in zip(self.weights, self.losses):
            loss = w * f_loss(output, target)
            total += loss

        return total

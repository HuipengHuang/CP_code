from .base_loss import BaseLoss
import torch
from torch.nn import BCEWithLogitsLoss
class BinaryCrossEntropyLoss(BaseLoss):
    def __init__(self):
        super().__init__()
    def forward(self, logit, target):
        prob = torch.sigmoid(logit)
        loss = torch.mean(- (target * torch.log(prob[:, 1] + 1e-4) + (1 - target) * torch.log(1 - prob[:, 1] + 1e-4)))
        return loss

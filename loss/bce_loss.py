from .base_loss import BaseLoss
import torch
from torch.nn import BCEWithLogitsLoss
class BinaryCrossEntropyLoss(BaseLoss):
    def __init__(self):
        super().__init__()
    def forward(self, logit, target):
        prob = torch.sigmoid(logit)
        prob = torch.clamp(prob, min=1e-5, max=1 - 1e-5)
        loss = torch.mean(- (target * torch.log(prob[:, 1]) + (1 - target) * torch.log(1 - prob[:, 1])))
        return loss

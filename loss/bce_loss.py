from .base_loss import BaseLoss
import torch
from torch.nn import BCEWithLogitsLoss
class BinaryCrossEntropyLoss(BaseLoss):
    def __init__(self):
        super().__init__()

    def forward(self, logit, target):
        loss = BCEWithLogitsLoss()(logit, target)
        return loss

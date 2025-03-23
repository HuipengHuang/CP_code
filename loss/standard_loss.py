import torch.nn as nn
from .base_loss import BaseLoss

class StandardLoss(BaseLoss):
    def __init__(self, args):
        super().__init__()
        if args.temperature:
            self.T = args.temperature
        else:
            self.T = 1
    def forward(self, logits, target):
        return nn.CrossEntropyLoss()(logits / self.T, target)
from .base_loss import BaseLoss
import torch
class AttentionMilLoss(BaseLoss):
    def __init__(self):
        super().__init__()

    def forward(self, logit, target):
        """Logit is a scaler."""
        prob = torch.sigmoid(logit)
        target = target.float()
        loss = torch.mean(-1. * (target * torch.log(prob[:,1] + 1e-8) + (1. - target) * torch.log(1. - prob[:, 1] + 1e-8)))
        return loss

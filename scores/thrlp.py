import torch
from .base_score import BaseScore


class THRLP(BaseScore):
    def __call__(self, prob):
        return 1 - torch.log(prob)

    def compute_target_score(self, prob, target):
        """It will return a tensor with dimension 1."""
        target_prob = prob[torch.arange(len(target)).to(target.device), target]
        return 1 - torch.log(target_prob)

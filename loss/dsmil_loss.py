import torch
import torch.nn as nn
from .base_loss import BaseLoss


class DSMilLoss(BaseLoss):
    def __init__(self, loss_function="standard"):
        super().__init__()
        if loss_function == "standard":
            self.criterion = nn.CrossEntropyLoss()
        elif loss_function == "bce":
            self.criterion = torch.nn.BCEWithLogitsLoss(reduction='mean')
        else:
            raise NotImplementedError

    def forward(self, input_list, target):
        ins_prediction, bag_prediction, _, _ = input_list
        max_prediction, _ = torch.max(ins_prediction, 0)
        print(bag_prediction.shape)
        print(max_prediction.shape)
        print("haha")
        bag_loss = self.criterion(bag_prediction.view(1, -1), target.view(1, -1))
        max_loss = self.criterion(max_prediction.view(1, -1), target.view(1, -1))
        loss = 0.5 * bag_loss + 0.5 * max_loss
        return loss
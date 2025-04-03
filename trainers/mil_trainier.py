from overrides import overrides
from .trainer import Trainer


class MilTrainer(Trainer):
    """Design for some specific model that the output is not a single logit."""
    def __init__(self,args, num_classes):
        super().__init__(args, num_classes)

    @overrides
    def train_batch(self, data, target):
        output_list = self.net(data)
        loss = self.loss_function(output_list, target)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

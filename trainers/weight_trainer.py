import torch
from .trainer import Trainer
import torch.nn as nn
from tqdm import tqdm
from loss import contr_loss

class WeightTrainer(Trainer):
    def __init__(self, args, num_classes):
        super().__init__(args, num_classes)
        weight = nn.Sequential(nn.Linear(args.input_dimension, 256), nn.ReLU(), nn.Linear(256, 64), nn.ReLU(), nn.Linear(64, 1)).to(self.device)
        self.weight = weight
        self.predictor.weight = weight
        self.weight_optimizer = torch.optim.Adam(weight.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
        self.weight_loss = contr_loss.ConftrLoss(args, self.predictor)

    def train(self, train_loader, epochs, val_loader=None):
        super().train(train_loader, epochs, val_loader)
        self.train_weight(train_loader, 1)

    def train_weight(self, dataloader, epochs):
        for param in self.net.parameters():
            param.requires_grad = False
        self.weight.train()

        for epoch in range(epochs):
            weight_score_list = []
            target_list = []
            for data, target in tqdm(dataloader, desc=f"{epoch+1} / {epochs}"):
                instance_weight = self.weight(data).squeeze(0)

                instance_weight = torch.softmax(instance_weight, dim=0)
                target_list.append(target)
                instance_prob_list = []
                for instance in data.squeeze(0):
                    instance_logits = self.net(instance.view(1, 1, -1))
                    instance_prob = self.activation_function(instance_logits).squeeze(dim=0)
                    instance_prob_list.append(instance_prob)

                all_instance_prob = torch.stack(instance_prob_list, dim=0)

                bag_score = self.predictor.score(all_instance_prob)
                weighted_score = (bag_score * instance_weight).sum(dim=0)

                weight_score_list.append(weighted_score)
                if(len(weight_score_list) == 2):
                    batch_score = torch.stack(weight_score_list, dim=0)
                    batch_target = torch.cat(target_list, dim=0)
                    loss = self.weight_loss.weight_forward(batch_score, batch_target)
                    self.weight_optimizer.zero_grad()
                    loss.backward()
                    self.weight_optimizer.step()
                    weight_score_list = []
                    target_list = []
from .base_loss import BaseLoss
import torch
import torch.nn.functional as F
import torch.nn as nn

class ConftrLoss(BaseLoss):
    def __init__(self, args, predictor):
        super().__init__()
        self.alpha = args.alpha
        self.predictor = predictor
        self.batch_size = args.batch_size
        if args.temperature is None:
            self.T = 1
        else:
            self.T = args.temperature

        if args.size_loss_weight is None:
            raise ValueError("Please specify a size_loss_weight")
        else:
            self.size_loss_weight = args.size_loss_weight

        if args.tau is None:
            raise ValueError("Please specify a tau.")
        else:
            self.tau = args.tau

    def forward(self, logits, target) -> torch.Tensor:
        shuffled_indices = torch.randperm(logits.size(0))
        shuffled_logit = logits[shuffled_indices]
        shuffled_target = target[shuffled_indices]

        pred_size = int(shuffled_target.size(0) * 0.5)
        pred_logit, cal_logit = shuffled_logit[:pred_size], shuffled_logit[pred_size:]
        pred_target, cal_target = shuffled_target[:pred_size], shuffled_target[pred_size:]

        threshold = self.predictor.smooth_calibrate_batch_logit(cal_logit, cal_target, self.alpha)
        pred_prob = self.predictor.final_activation_function(pred_logit)
        pred_score = self.predictor.score(pred_prob)
        smooth_pred = torch.sigmoid((threshold - pred_score) / self.T)

        size_loss = self.compute_size_loss(smooth_pred)

        class_loss = self.compute_classification_loss(smooth_pred, pred_target)
        loss = torch.log(class_loss + self.size_loss_weight * size_loss + 1e-8)
        return loss

    def compute_size_loss(self, smooth_pred) -> torch.Tensor:
        size_loss = torch.maximum(torch.sum(smooth_pred,dim=-1) - self.tau, torch.tensor([0], device=smooth_pred.device)).mean()
        return size_loss

    def compute_classification_loss(self, smooth_pred, target):
        one_hot_labels = F.one_hot(target, num_classes=smooth_pred.shape[1]).float()
        loss_matrix = torch.eye(smooth_pred.shape[1], device=smooth_pred.device)
        l1 = (1 - smooth_pred) * one_hot_labels * loss_matrix[target]
        l2 = smooth_pred * (1 - one_hot_labels) * loss_matrix[target]

        loss = torch.sum(torch.maximum(l1 + l2, torch.zeros_like(l1)), dim=1)
        return torch.mean(loss)

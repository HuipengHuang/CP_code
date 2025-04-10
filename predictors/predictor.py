from scores.utils import get_score, get_train_score
import torch
import torch.nn as nn
import math
import torchsort
from trainers.utils import five_scores


class Predictor:
    def __init__(self, args, net, num_classes, final_activation_function, adapter=None):
        self.args = args
        self.test_score = get_score(args)
        if args.train_score is None:
            self.train_score = self.test_score
        else:
            self.train_score = get_train_score(args)
        self.adapter = adapter
        self.compute_auc = (args.compute_auc == "True")
        self.score = None
        self.threshold = None
        self.alpha = args.alpha
        self.net = net
        self.num_classes = num_classes
        if final_activation_function == "softmax":
            self.final_activation_function = nn.Softmax(dim=-1)
        elif final_activation_function == "sigmoid":
            self.final_activation_function = nn.Sigmoid()
        else:
            raise NotImplementedError(f"activation function {final_activation_function} is not implemented.")
        self.device = torch.device(f"cuda:{args.gpu}")

    def calibrate(self, cal_loader, alpha=None):
        """ Input calibration dataloader.
            Compute scores for all the calibration data and take the (1 - alpha) quantile."""
        self.set_mode("test")
        if alpha is None:
            alpha = self.alpha

        if self.adapter:
            threshold = self.calibrate_with_adapter(cal_loader, alpha)
        else:
            threshold = self.calibrate_without_adapter(cal_loader, alpha)
        self.threshold = threshold
        return threshold

    def calibrate_with_adapter(self, cal_loader, alpha):
        with torch.no_grad():
            if alpha is None:
                alpha = self.alpha
            cal_score = torch.tensor([], device=self.device)
            for data, target in cal_loader:
                data = data.to(self.device)
                target = target.to(self.device)

                logits = self.adapter(self.net(data))
                prob = self.final_activation_function(logits)

                batch_score = self.score.compute_target_score(prob, target)

                cal_score = torch.cat((cal_score, batch_score), 0)

            N = cal_score.shape[0]
            threshold = torch.quantile(cal_score, math.ceil((1 - alpha) * (N + 1)) / N, dim=0)
            return threshold

    def calibrate_without_adapter(self, cal_loader, alpha=None):
        with torch.no_grad():
            if alpha is None:
                alpha = self.alpha
            cal_score = torch.tensor([], device=self.device)
            for data, target in cal_loader:
                data = data.to(self.device)
                target = target.to(self.device)

                logits = self.net(data)
                prob = self.final_activation_function(logits)

                batch_score = self.score.compute_target_score(prob, target)

                cal_score = torch.cat((cal_score, batch_score), 0)

            N = cal_score.shape[0]
            threshold = torch.quantile(cal_score, math.ceil((1 - alpha) * (N + 1)) / N, dim=0)
            return threshold

    def calibrate_batch_logit(self, logits, target, alpha):
        """Design for conformal training, which needs to compute threshold in every batch"""
        prob = self.final_activation_function(logits)
        batch_score = self.score.compute_target_score(prob, target)
        N = target.shape[0]
        return torch.quantile(batch_score, math.ceil((1 - alpha) * (N + 1)) / N, dim=0)

    def smooth_calibrate_batch_logit(self, logits, target, alpha):
        prob = self.final_activation_function(logits)
        batch_score = self.score.compute_target_score(prob, target)
        N = target.shape[0]
        sorted_score = torchsort.soft_sort(batch_score.unsqueeze(0), regularization_strength=0.1)
        threshold = sorted_score[0, math.ceil((1 - alpha) * (N + 1)) - 1]
        return threshold

    def evaluate(self, test_loader):
        """Must be called after calibration.
        Output a dictionary containing Top1 Accuracy, Coverage and Average Prediction Set Size."""
        self.set_mode("test")
        self.net.eval()
        if self.threshold is not None:
            bag_prob, bag_labels = [], []
            average_set_size = 0
            coverage = 0
            with torch.no_grad():
                for i, (data, target) in enumerate(test_loader):
                    bag_labels.append(target.item())
                    data = data.to(self.device)
                    target = target.to(self.device)

                    test_logits = self.net(data)

                    prob = torch.softmax(test_logits, dim=-1)
                    bag_prob.append(prob[:, 1].cpu().squeeze().numpy())
                    score_tensor = self.score(prob)
                    average_set_size += (score_tensor < self.threshold).sum().item()
                    coverage += (
                                score_tensor[torch.arange(score_tensor.shape[0]), target] < self.threshold).sum().item()

                coverage = coverage / len(test_loader)
                average_set_size = average_set_size / len(test_loader)
                accuracy, auc_value, precision, recall, fscore = five_scores(bag_labels, bag_prob, )
                print(
                    f"average set size: {average_set_size}, coverage: {coverage}, accuracy:{accuracy}, auc:{auc_value}, precision:{precision}, recall:{recall}, fscore:{fscore}")
                result_dict = {"Coverage": coverage, "Average Set Size": average_set_size, "Accuracy": accuracy,
                               "AUC": auc_value, "Precision": precision, "Recall": recall, "Fscore": fscore}
                return result_dict
        else:
            bag_prob, bag_labels = [], []

            with torch.no_grad():
                for i, (data, target) in enumerate(test_loader):
                    bag_labels.append(target.item())
                    data = data.to(self.device)
                    target = target.to(self.device)

                    test_logits = self.net(data)

                    bag_prob.append(torch.softmax(test_logits, dim=-1)[:, 1].cpu().squeeze().numpy())

                accuracy, auc_value, precision, recall, fscore = five_scores(bag_labels, bag_prob, )
                print(f"accuracy:{accuracy}, auc:{auc_value}, precision:{precision}, recall:{recall}, fscore:{fscore}")
                result_dict = {"Accuracy": accuracy, "AUC": auc_value, "Precision": precision, "Recall": recall,
                               "Fscore": fscore}
                return result_dict

    def set_mode(self, mode="train"):
        if mode == "train":
            self.score = self.train_score
        elif mode == "test":
            self.score = self.test_score
        else:
            raise ValueError(f"mode {mode} is not supported. Mode could only be train or test")



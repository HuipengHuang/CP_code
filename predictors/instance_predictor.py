from scores.utils import get_score, get_train_score
import torch
import torch.nn as nn
import math
import torchsort
from trainers.utils import five_scores


class Instance_Predictor:
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
        with torch.no_grad():
            if alpha is None:
                alpha = self.alpha
            cal_list = []
            j = 0
            for i, (data, target) in enumerate(cal_loader):

                data = data.to(self.device)
                target = target.to(self.device)
                if target == 0:
                    j += 1
                    if j==3:
                        break
                    for instance in data:
                        if self.args.model == "dsmil":
                            instance_logits = self.net(instance)[1]
                        else:
                            instance_logits = self.net(instance)
                        instance_prob = self.final_activation_function(instance_logits)
                        instance_batch_score = self.score.compute_target_score(instance_prob, torch.zeros(size=(instance_prob.shape[0],), dtype=torch.int32, device=self.device))
                        cal_list.append(instance_batch_score)
            cal_score = torch.cat(cal_list, dim=0)
            print(cal_score.shape)
            N = cal_score.shape[0]
            threshold = torch.quantile(cal_score, math.ceil((1 - alpha) * (N + 1)) / N, dim=0)
            self.threshold = threshold
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
        self.set_mode("test")
        self.net.eval()
        """Use conformal prediction when threshold is not None."""
        if self.threshold is not None:
            bag_prob, bag_labels = [], []
            average_set_size = 0
            coverage = 0
            num_instance = 0
            j = 0
            with torch.no_grad():
                for i, (data, target) in enumerate(test_loader):

                    bag_labels.append(target.item())
                    data = data.to(self.device)
                    target = target.to(self.device)
                    if self.args.model == "dsmil":
                        test_logits = self.net(data)[1]
                    else:
                        test_logits = self.net(data)

                    prob = self.final_activation_function(test_logits)
                    bag_prob.append(prob[:, 1].cpu().squeeze().numpy())
                    if target == 1:
                        continue
                    else:
                        j +=1
                        if j==3:
                            break
                        for instance in data:
                            num_instance += 1
                            if self.args.model == "dsmil":
                                instance_logits = self.net(instance)[1]
                            else:
                                instance_logits = self.net(instance)
                            instance_prob = self.final_activation_function(instance_logits)
                            instance_batch_score = self.score(instance_prob)
                            average_set_size += (instance_batch_score <= self.threshold).sum().item()
                            if instance_batch_score[0, 0] <= self.threshold:
                                coverage += 1

                coverage = coverage / num_instance
                average_set_size = average_set_size / num_instance
                accuracy, auc_value, precision, recall, fscore = five_scores(bag_labels, bag_prob,)
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

                    if self.args.model == "dsmil":
                        test_logits = self.net(data)[1]
                    else:
                        test_logits = self.net(data)

                    bag_prob.append(self.final_activation_function(test_logits, dim=-1)[:, 1].cpu().squeeze().numpy())

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



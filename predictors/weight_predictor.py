from scores.utils import get_score, get_train_score
import torch
import torch.nn as nn
import math
import torchsort
from trainers.utils import five_scores


class WeightPredictor:
    def __init__(self, args, net, num_classes, final_activation_function ,adapter=None):
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
        self.weight = None

    def calibrate(self, cal_loader, alpha=None):
        """ Input calibration dataloader.
            Compute scores for all the calibration data and take the (1 - alpha) quantile."""
        with torch.no_grad():
            if alpha is None:
                alpha = self.alpha
            cal_list = []
            for i, (data, target) in enumerate(cal_loader):
                data = data.to(self.device)
                target = target.to(self.device)
                instance_weight = self.weight(data)
                instance_weight = torch.softmax(instance_weight, dim=-1)
                instance_prob_list = []
                for instance in data.squeeze(0):
                    if self.args.model == "dsmil":
                        instance_logits = self.net(instance.view(1, -1))[1]
                    else:
                        instance_logits = self.net(instance.view(1, -1))
                    instance_prob = self.final_activation_function(instance_logits)
                    instance_prob_list.append(instance_prob[target])
                all_instance_probs = torch.stack(instance_prob_list, dim=0)
                all_instances_score = self.score(all_instance_probs)
                bag_socre = (all_instances_score * instance_weight).sum(dim=0)
                cal_list.append(bag_socre)

            cal_score = torch.cat(cal_list, dim=0)
            N = cal_score.shape[0]
            threshold = torch.quantile(cal_score, math.ceil((1 - alpha) * (N + 1)) / N, dim=0)
            self.threshold = threshold
            return threshold


    def evaluate(self, test_loader):
        self.set_mode("test")
        self.net.eval()
        """Use conformal prediction when threshold is not None."""
        if self.threshold is not None:
            bag_prob, bag_labels = [], []
            average_set_size = 0
            coverage = 0

            with torch.no_grad():
                for i, (data, target) in enumerate(test_loader):
                    w = self.weight(data)
                    w = torch.softmax(w, dim=-1)
                    bag_labels.append(target.item())
                    data = data.to(self.device)
                    target = target.to(self.device)
                    if self.args.model == "dsmil":
                        test_logits = self.net(data)[1]
                    else:
                        test_logits = self.net(data)

                    prob = self.final_activation_function(test_logits)
                    bag_prob.append(prob[:, 1].cpu().squeeze().numpy())
                    instance_prob_list = []
                    for instance in data.squeeze(0):
                        if self.args.model == "dsmil":
                            instance_logits = self.net(instance.view(1, -1))[1]
                        else:
                            instance_logits = self.net(instance.view(1, -1))
                        instance_prob = self.final_activation_function(instance_logits)
                        instance_prob_list.append(instance_prob)
                    all_instance_prob = torch.stack(instance_prob_list, dim=0)
                    all_instance_score = self.score(all_instance_score)
                    bag_score = (all_instance_score * w).sum(dim=0)

                    average_set_size += (bag_score < self.threshold).sum().item()
                    coverage += (bag_score[target] < self.threshold).item()

                coverage = coverage / len(test_loader.dataset)
                average_set_size = average_set_size / len(test_loader.dataset)
                print(coverage)
                print(average_set_size)
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



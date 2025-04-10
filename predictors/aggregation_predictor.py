from predictors.predictor import Predictor
import torch
from trainers.utils import five_scores

class AggPredictor(Predictor):
    def __init__(self, args, net, num_classes, final_activation_function, adapter=None):
        super(AggPredictor, self).__init__(args, net, num_classes, final_activation_function, adapter)
    def get_prob(self, data):
        raise NotImplementedError

    def evaluate(self, test_loader):
        agg_result = self.evaluate_with_aggregation(test_loader)
        standard_result = self.standard_aggregation(test_loader)
        result_dict = {**standard_result, **agg_result}
        return result_dict

    def evaluate_with_aggregation(self, test_loader):
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

                    prob = self.get_prob(data)

                    bag_prob.append(prob[:, 1].cpu().squeeze().numpy())
                    score_tensor = self.score(prob)
                    average_set_size += (score_tensor < self.threshold).sum().item()
                    coverage += (
                            score_tensor[torch.arange(score_tensor.shape[0]), target] < self.threshold).sum().item()

                coverage = coverage / len(test_loader)
                average_set_size = average_set_size / len(test_loader)
                accuracy, auc_value, precision, recall, fscore = five_scores(bag_labels, bag_prob, )
                print(f"Aggregation Method: {self.args.aggregation}")
                print(
                    f"average set size: {average_set_size}, coverage: {coverage}, accuracy:{accuracy}, auc:{auc_value}, precision:{precision}, recall:{recall}, fscore:{fscore}")
                result_dict = {"agg-Coverage": coverage, "agg-Average Set Size": average_set_size, "agg-Accuracy": accuracy,
                               "agg-AUC": auc_value, "agg-Precision": precision, "agg-Recall": recall, "agg-Fscore": fscore}
                print("")
                return result_dict
        else:
            bag_prob, bag_labels = [], []

            with torch.no_grad():
                for i, (data, target) in enumerate(test_loader):
                    bag_labels.append(target.item())
                    data = data.to(self.device)

                    prob = self.get_prob(data)

                    bag_prob.append(prob[:, 1].cpu().squeeze().numpy())

                accuracy, auc_value, precision, recall, fscore = five_scores(bag_labels, bag_prob, )
                print(f"Aggregation Method: {self.args.aggregation}")
                print(f"accuracy:{accuracy}, auc:{auc_value}, precision:{precision}, recall:{recall}, fscore:{fscore}")
                result_dict = {"Accuracy": accuracy, "AUC": auc_value, "Precision": precision, "Recall": recall,
                               "Fscore": fscore}
                print("")
                return result_dict

    def standard_aggregation(self, test_loader):
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
                print("Standard Method")
                print(
                    f"average set size: {average_set_size}, coverage: {coverage}, accuracy:{accuracy}, auc:{auc_value}, precision:{precision}, recall:{recall}, fscore:{fscore}")
                print("")
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
                print("Standard Method")
                print(f"accuracy:{accuracy}, auc:{auc_value}, precision:{precision}, recall:{recall}, fscore:{fscore}")
                print("")
                result_dict = {"Accuracy": accuracy, "AUC": auc_value, "Precision": precision, "Recall": recall,
                               "Fscore": fscore}
                return result_dict


class MaxPredictor(AggPredictor):
    def __init__(self, args, net, num_classes, final_activation_function, adapter=None):
        super(MaxPredictor, self).__init__(args, net, num_classes, final_activation_function, adapter)

    def get_prob(self, data):
        prob = torch.zeros(size=(1, self.num_classes), device=data.device)

        for i in range(data.shape[1]):
            test_logits = self.net(data[0][i])
            instance_prob = self.final_activation_function(test_logits)
            if prob[:, 1] < instance_prob[:, 1]:
                prob = instance_prob
        return prob
from predictors.predictor import Predictor
import torch
from trainers.utils import five_scores
from kmeans_pytorch import kmeans

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
        j = 0
        for i in range(int(data.shape[1] / 100)):
            instance_prob = self.final_activation_function(self.net(data[:, i * 100 : i * 100 + 100, :]))
            if prob[:, 1] < instance_prob[:, 1]:
                prob = instance_prob
            j = i + 1
        if data.shape[1] % 100 != 0:
            instance_prob = self.final_activation_function(self.net(data[:, j * 100: , :]))
            if prob[0, 1] < instance_prob[0, 1]:
                prob = instance_prob
        return prob


class KMeanPredictor(AggPredictor):
    def __init__(self, args, net, num_classes, final_activation_function, adapter=None, n_cluster=3):
        super(KMeanPredictor, self).__init__(args, net, num_classes, final_activation_function, adapter)
        self.n_cluster = n_cluster

    def get_prob(self, data):
        """
        Args:
            data: Input tensor of shape (1, N, 1024) on GPU.
        Returns:
            prob: Aggregated probability (1, num_classes) on GPU.
        """
        # Extract data (assume batch_size=1)
        data_tensor = data[0]  # Shape: (N, 1024)

        # Run K-Means on GPU
        cluster_ids, _ = kmeans(
            X=data_tensor,
            num_clusters=self.n_cluster,
            device=data_tensor.device  # Use same device as input
        )

        # Initialize probabilities
        prob = torch.zeros(size=(1, self.num_classes), device=data_tensor.device)

        # Process each cluster
        for i in range(self.n_cluster):
            mask = (cluster_ids == i)
            cluster_data = data_tensor[mask]  # Shape: (M, 1024), M = cluster size

            # Forward pass through network
            instance_prob = self.final_activation_function(self.net(cluster_data))

            # Update prob (keep min probability for class 1)
            if instance_prob[0, 1] < prob[0, 1]:
                prob = instance_prob

        return prob
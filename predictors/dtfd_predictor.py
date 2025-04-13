import random

import numpy as np
from torch.utils.data import RandomSampler

from scores.utils import get_score, get_train_score
import torch
import torch.nn as nn
import math
import torchsort
from trainers.utils import five_scores, get_cam_1d


class DTFDPredictor:
    def __init__(self, args, net_list, num_classes, final_activation_function, adapter=None):
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
        self.classifier, self.attention, self.dimReduction, self.attCls = net_list
        self.num_classes = num_classes
        if final_activation_function == "softmax":
            self.final_activation_function = nn.Softmax(dim=-1)
        elif final_activation_function == "sigmoid":
            self.final_activation_function = nn.Sigmoid()
        else:
            raise NotImplementedError(f"activation function {final_activation_function} is not implemented.")
        self.device = torch.device(f"cuda:{args.gpu}")
        self.bag_size = args.bag_size
        self.distill = args.distill

    def calibrate(self, cal_loader, alpha=None):
        """ Input calibration dataloader.
            Compute scores for all the calibration data and take the (1 - alpha) quantile."""
        self.classifier.eval()
        self.attention.eval()
        self.dimReduction.eval()

        with torch.no_grad():
            ds = cal_loader.dataset
            num_bag = len(ds)

            numIter = num_bag // self.bag_size
            tIDX = list(RandomSampler(range(num_bag)))

            for idx in range(numIter):
                tidx_slide = tIDX[idx * self.bag_size: (idx + 1) * self.bag_size]
                tlabel = []
                batch_feat = []
                for i in tidx_slide:
                    data, target = ds[i]
                    tlabel.append(target)
                    batch_feat.append(data)

                label_tensor = torch.LongTensor(tlabel).to(self.device)

                for tidx, tfeat in enumerate(batch_feat):
                    tslideLabel = label_tensor[tidx].unsqueeze(0)
                    midFeat = self.dimReduction(tfeat)

                    AA = self.attention(midFeat, isNorm=False).squeeze(0)  ## N

                    cal_score = torch.tensor([], device=self.device)

                    for jj in range(2):

                        feat_index = list(range(tfeat.shape[0]))
                        random.shuffle(feat_index)
                        index_chunk_list = np.array_split(np.array(feat_index), self.numgroup)
                        index_chunk_list = [sst.tolist() for sst in index_chunk_list]

                        slide_d_feat = []
                        slide_sub_preds = []
                        slide_sub_labels = []

                        for tindex in index_chunk_list:
                            slide_sub_labels.append(tslideLabel)
                            idx_tensor = torch.LongTensor(tindex).to(self.device)
                            tmidFeat = midFeat.index_select(dim=0, index=idx_tensor)

                            tAA = AA.index_select(dim=0, index=idx_tensor)
                            tAA = torch.softmax(tAA, dim=0)
                            tattFeats = torch.einsum('ns,n->ns', tmidFeat, tAA)  ### n x fs
                            tattFeat_tensor = torch.sum(tattFeats, dim=0).unsqueeze(0)  ## 1 x fs

                            tPredict = self.classifier(tattFeat_tensor)  ### 1 x 2
                            slide_sub_preds.append(tPredict)

                            patch_pred_logits = get_cam_1d(self.classifier, tattFeats.unsqueeze(0)).squeeze(
                                0)  ###  cls x n
                            patch_pred_logits = torch.transpose(patch_pred_logits, 0, 1)  ## n x cls
                            patch_pred_softmax = torch.softmax(patch_pred_logits, dim=1)  ## n x cls

                            _, sort_idx = torch.sort(patch_pred_softmax[:, -1], descending=True)
                            instance_per_group = 1

                            if self.distill == 'MaxMinS':
                                topk_idx_max = sort_idx[:instance_per_group].long()
                                topk_idx_min = sort_idx[-instance_per_group:].long()
                                topk_idx = torch.cat([topk_idx_max, topk_idx_min], dim=0)
                                d_inst_feat = tmidFeat.index_select(dim=0, index=topk_idx)
                                slide_d_feat.append(d_inst_feat)
                            elif self.distill == 'MaxS':
                                topk_idx_max = sort_idx[:instance_per_group].long()
                                topk_idx = topk_idx_max
                                d_inst_feat = tmidFeat.index_select(dim=0, index=topk_idx)
                                slide_d_feat.append(d_inst_feat)
                            elif self.distill == 'AFS':
                                slide_d_feat.append(tattFeat_tensor)

                        slide_d_feat = torch.cat(slide_d_feat, dim=0)
                        slide_sub_preds = torch.cat(slide_sub_preds, dim=0)
                        slide_sub_labels = torch.cat(slide_sub_labels, dim=0)

                        gPred_0 = torch.cat([gPred_0, slide_sub_preds], dim=0)

                        gSlidePred = self.attCls(slide_d_feat)
                        prob = torch.softmax(gSlidePred, dim=1)
                        score = self.score(prob, tslideLabel)
                        cal_score = torch.cat((cal_score, score), dim=0)

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
            with torch.no_grad():
                for i, (data, target) in enumerate(test_loader):
                    bag_labels.append(target.item())
                    data = data.to(self.device)
                    target = target.to(self.device)
                    if self.args.model == "dsmil":
                        test_logits = self.net(data)[1]
                    else:
                        test_logits = self.net(data)

                    prob = self.final_activation_function(test_logits, dim=-1)
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



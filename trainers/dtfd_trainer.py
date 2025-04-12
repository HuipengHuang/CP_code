import random

import numpy as np
import torch
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import RandomSampler
from .utils import get_cam_1d, eval_metric
from tqdm import tqdm
import models.DTFD as DTFD
import torch.nn.functional as F
from .early_stopping import EarlyStopping
from loss.utils import get_loss_function
from predictors.utils import get_predictor
import torch.nn as nn


class DFDT_Trainer:
    """
    Trainer class that implement all the functions regarding training.
    All the arguments are passed through args."""
    def __init__(self, args, num_classes):
        self.device = torch.device(f"cuda:{args.gpu}")
        final_activation_function = args.final_activation_function
        if final_activation_function == "softmax":
            self.activation_function = nn.Softmax(dim=-1)
        elif final_activation_function == "sigmoid":
            self.activation_function = nn.Sigmoid()
        else:
            raise NotImplementedError(f"activation function {final_activation_function} is not implemented.")
        self.predictor = get_predictor(args, None, num_classes=num_classes,
                                       adapter=None,
                                       final_activation_function=final_activation_function)
        self.predictor.set_mode("train")
        self.num_classes = num_classes
        self.loss_function = get_loss_function(args, self.predictor)
        if args.patience:
            self.early_stopping = EarlyStopping(patience=args.patience)
        else:
            self.early_stopping = None
        self.args = args

        self.numgroup = args.numgroup
        self.distill = args.distill
        self.bag_size = args.bag_size
        if args.shuffle == "True":
            self.shuffle = True
        else:
            self.shuffle = False
        assert self.numgroup is not None, print("numgroup could not be None.")
        assert self.distill is not None, print("distill could not be None.")

        self.classifier = DTFD.network.Classifier_1fc(512, num_classes, args.dropout).to(self.device)
        self.attention = DTFD.attention.Attention_Gated(512).to(self.device)
        self.dimReduction = DTFD.network.DimReduction(args.input_dimension, 512, dropout=args.dropout).to(self.device)
        self.attCls = DTFD.attention.Attention_with_Classifier(L=512, num_cls=num_classes, droprate=args.dropout).to(self.device)
        trainable_parameter = []
        trainable_parameter += list(self.classifier.parameters())
        trainable_parameter += list(self.attention.parameters())
        trainable_parameter += list(self.dimReduction.parameters())

        self.optimizer = None
        if args.optimizer == 'sgd':
            self.optimizer0 = torch.optim.SGD(trainable_parameter, lr=args.learning_rate, momentum=args.momentum,
                                             weight_decay=args.weight_decay, nesterov=(args.nesterov == "True"))
            self.optimizer1 = torch.optim.SGD(self.attCls.parameters(), lr=args.learning_rate, momentum=args.momentum,
                                             weight_decay=args.weight_decay, nesterov=(args.nesterov == "True"))
        elif args.optimizer == 'adam':
            self.optimizer0 = torch.optim.Adam(trainable_parameter, lr=args.learning_rate,
                                              weight_decay=args.weight_decay)
            self.optimizer1 = torch.optim.Adam(self.attCls.parameters(), lr=args.learning_rate,
                                              weight_decay=args.weight_decay)

        if args.learning_rate_scheduler == 'cosine':
            self.scheduler0 = CosineAnnealingLR(self.optimizer0, T_max=args.epoch, eta_min=0)
            self.scheduler1 = CosineAnnealingLR(self.optimizer1, T_max=args.epoch, eta_min=0)
        else:
            self.scheduler0 = None
            self.scheduler1 = None

    def train_loop(self, train_loader, epoch):
        self.classifier.train()
        self.attention.train()
        self.dimReduction.train()

        ds = train_loader.dataset
        num_bag = len(ds)

        numIter = num_bag // self.bag_size
        tIDX = list(RandomSampler(range(num_bag)))

        for idx in tqdm(range(numIter)):
            tidx_slide = tIDX[idx * self.bag_size: (idx + 1) * self.bag_size]

            for tidx, bag_idx in enumerate(tidx_slide):
                slide_pseudo_feat = []
                slide_sub_preds = []
                slide_sub_labels = []

                tfeat_tensor, tslideLabel = ds[bag_idx]
                tfeat_tensor, tslideLabel = tfeat_tensor.to(self.device), tslideLabel.to(self.device)

                feat_index = list(range(tfeat_tensor.shape[0]))
                if self.shuffle:
                    random.shuffle(feat_index)
                index_chunk_list = np.array_split(feat_index, num_bag)

                index_chunk_list = [sst.tolist() for sst in index_chunk_list]

                for tindex in index_chunk_list:
                    slide_sub_labels.append(tslideLabel)
                    subFeat_tensor = torch.index_select(tfeat_tensor, dim=0,
                                                        index=torch.LongTensor(tindex).to(self.device))
                    tmidFeat = self.dimReduction(subFeat_tensor)
                    tAA = self.attention(tmidFeat).squeeze(0)

                    tattFeats = torch.einsum('ns,n->ns', tmidFeat, tAA)
                    tattFeat_tensor = torch.sum(tattFeats, dim=0).squeeze(0)
                    tPredict = self.classifier(tattFeat_tensor)
                    slide_sub_preds.append(tPredict)

                    patch_pred_logits = get_cam_1d(self.classifier, tattFeats.unsqueeze(0)).squeeze(0)  ###  cls x n
                    patch_pred_logits = torch.transpose(patch_pred_logits, 0, 1)  ## n x cls
                    patch_pred_prob = self.activation_function(patch_pred_logits)  ## n x cls

                    _, sort_idx = torch.sort(patch_pred_prob[:, -1], descending=True)
                    topk_idx_max = sort_idx[:1].long()
                    topk_idx_min = sort_idx[-1:].long()
                    topk_idx = torch.cat([topk_idx_max, topk_idx_min], dim=0)

                    if self.distill == 'MaxMinS':

                        MaxMin_inst_feat = tmidFeat.index_select(dim=0, index=topk_idx)  ##########################
                        slide_pseudo_feat.append(MaxMin_inst_feat)
                    elif self.distill == 'MaxS':
                        max_inst_feat = tmidFeat.index_select(dim=0, index=topk_idx_max)
                        slide_pseudo_feat.append(max_inst_feat)
                    elif self.distill == 'AFS':
                        af_inst_feat = tattFeat_tensor
                        slide_pseudo_feat.append(af_inst_feat)

                slide_sub_preds = torch.cat(slide_sub_preds, dim=0)  ### numGroup x fs
                slide_sub_labels = torch.tensor(slide_sub_labels, device=slide_sub_preds.device)  ### numGroup
                loss0 = self.loss_function(slide_sub_preds, slide_sub_labels).mean()
                self.optimizer0.zero_grad()
                loss0.backward(retain_graph=True)

                ## optimization for the second tier
                gSlidePred = self.attCls(slide_pseudo_feat)
                loss1 = self.loss_function(gSlidePred, tslideLabel).mean()
                self.optimizer1.zero_grad()
                loss1.backward()
                self.optimizer0.step()
                self.optimizer1.step()
        self.scheduler0.step()
        self.scheduler1.step()

    def val_loop(self, val_loader):
        self.classifier.eval()
        self.attention.eval()
        self.dimReduction.eval()
        gPred_0 = torch.FloatTensor().to(self.device)
        gt_0 = torch.LongTensor().to(self.device)
        gPred_1 = torch.FloatTensor().to(self.device)
        gt_1 = torch.LongTensor().to(self.device)
        test_loss0 = 0
        test_loss1 = 0
        with torch.no_grad():
            ds = val_loader.dataset
            num_bag = len(ds)

            numIter = num_bag // self.bag_size
            tIDX = list(RandomSampler(range(num_bag), numIter))

            for idx in range(numIter):
                tidx_slide = tIDX[idx * self.bag_size: (idx + 1) * self.bag_size]

                for tidx, bag_idx in enumerate(tidx_slide):
                    slide_pseudo_feat = []
                    slide_sub_preds = []
                    slide_sub_labels = []

                    tfeat_tensor, tslideLabel = ds[bag_idx].to(self.device)

                    feat_index = list(range(tfeat_tensor.shape[0]))
                    if self.shuffle:
                        random.shuffle(feat_index)
                    index_chunk_list = np.array_split(feat_index, num_bag)

                    index_chunk_list = [sst.tolist() for sst in index_chunk_list]

                    all_slide_pred_prob = []

                    for tindex in index_chunk_list:
                        slide_sub_labels.append(tslideLabel)
                        subFeat_tensor = torch.index_select(tfeat_tensor, dim=0,
                                                            index=torch.LongTensor(tindex).to(self.device))
                        tmidFeat = self.dimReduction(subFeat_tensor)
                        tAA = self.attention(tmidFeat).squeeze(0)

                        tattFeats = torch.einsum('ns,n->ns', tmidFeat, tAA)
                        tattFeat_tensor = torch.sum(tattFeats, dim=0).squeeze(0)
                        tPredict = self.classifier(tattFeat_tensor)
                        slide_sub_preds.append(tPredict)

                        patch_pred_logits = get_cam_1d(self.classifier, tattFeats.unsqueeze(0)).squeeze(0)  ###  cls x n
                        patch_pred_logits = torch.transpose(patch_pred_logits, 0, 1)  ## n x cls
                        patch_pred_prob = self.activation_function(patch_pred_logits)  ## n x cls

                        _, sort_idx = torch.sort(patch_pred_prob[:, -1], descending=True)
                        topk_idx_max = sort_idx[:1].long()
                        topk_idx_min = sort_idx[-1:].long()
                        topk_idx = torch.cat([topk_idx_max, topk_idx_min], dim=0)

                        if self.distill == 'MaxMinS':

                            MaxMin_inst_feat = tmidFeat.index_select(dim=0, index=topk_idx)  ##########################
                            slide_pseudo_feat.append(MaxMin_inst_feat)
                        elif self.distill == 'MaxS':
                            max_inst_feat = tmidFeat.index_select(dim=0, index=topk_idx_max)
                            slide_pseudo_feat.append(max_inst_feat)
                        elif self.distill == 'AFS':
                            af_inst_feat = tattFeat_tensor
                            slide_pseudo_feat.append(af_inst_feat)

                    slide_d_feat = torch.cat(slide_d_feat, dim=0)
                    slide_sub_preds = torch.cat(slide_sub_preds, dim=0)
                    slide_sub_labels = torch.cat(slide_sub_labels, dim=0)

                    gPred_0 = torch.cat([gPred_0, slide_sub_preds], dim=0)
                    gt_0 = torch.cat([gt_0, slide_sub_labels], dim=0)
                    loss0 = self.loss_function(slide_sub_preds, slide_sub_labels)
                    test_loss0 += loss0.item()

                    gSlidePred = self.attCls(slide_d_feat)
                    allSlide_pred_prob.append(self.activation_function(gSlidePred, dim=1))

                allSlide_pred_prob = torch.cat(allSlide_pred_prob, dim=0)
                allSlide_pred_prob = torch.mean(allSlide_pred_prob, dim=0).unsqueeze(0)
                gPred_1 = torch.cat([gPred_1, allSlide_pred_prob], dim=0)
                gt_1 = torch.cat([gt_1, tslideLabel], dim=0)

                loss1 = F.nll_loss(allSlide_pred_prob, tslideLabel)
                test_loss1 += loss1.item()

        gPred_0 = torch.softmax(gPred_0, dim=1)
        gPred_0 = gPred_0[:, -1]
        gPred_1 = gPred_1[:, -1]

        test_loss0 /= len(ds)
        test_loss1 /= len(ds)

        accuracy0, auc0, precision0, recall0, F10 = eval_metric(gPred_0, gt_0)
        accuracy1, auc1, precision1, recall1, F11 = eval_metric(gPred_1, gt_1)

        print(
            f'  First-Tier accuracy:{accuracy0}, auc:{auc0}, precision:{precision0}, recall:{recall0}, F1:{F10}')
        print(
            f'  Second-Tier accuracy:{accuracy1}, auc:{auc1}, precision:{precision1}, recall:{recall1}, F1:{F11}')


        return accuracy1, auc1, precision1, recall1, F11, test_loss1

    def train(self, train_loader, epochs, val_loader=None):
        if val_loader is None or self.early_stopping is None:
            for epoch in range(epochs):
                self.train_loop(train_loader, epoch)
        else:
            for epoch in range(epochs):
                self.train_loop(train_loader, epoch)
                accuracy, auc_value, precision, recall, fscore, loss = self.val_loop(val_loader)
                stop = self.early_stopping(loss, epoch)
                if stop:
                    break



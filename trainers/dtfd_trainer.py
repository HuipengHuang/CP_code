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
from models import dtfd
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

        if args.numgroup:
            self.numgroup = args.numgroup
        else:
            self.numgroup = 4
        if args.distill:
            self.distill = args.distill
        else:
            self.distill = "MaxMinS"
        if args.shuffle == "True":
            self.shuffle = True
        else:
            self.shuffle = False

        self.classifier = DTFD.network.Classifier_1fc(512, num_classes, args.dropout if args.dropout else 0).to(self.device)
        self.attention = DTFD.attention.Attention_Gated(512).to(self.device)
        self.dimReduction = DTFD.network.DimReduction(args.input_dimension, 512, dropout=args.dropout if args.dropout else 0).to(self.device)
        self.attCls = DTFD.attention.Attention_with_Classifier(L=512, num_cls=num_classes, droprate=args.dropout if args.dropout else 0).to(
            self.device)
        self.dtfdmil = dtfd.DTFDMIL(self.device, self.classifier, self.attention, self.dimReduction, self.attCls, self.numgroup,
                                    final_activation_function, self.distill, self.shuffle)
        trainable_parameter = []
        trainable_parameter += list(self.classifier.parameters())
        trainable_parameter += list(self.attention.parameters())
        trainable_parameter += list(self.dimReduction.parameters())

        self.predictor = get_predictor(args, self.dtfdmil, num_classes=num_classes,
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
        if args.bag_size:
            self.bag_size = args.bag_size
        else:
            self.bag_size = 4



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
            self.scheduler0 = CosineAnnealingLR(self.optimizer0, T_max=args.epochs, eta_min=0)
            self.scheduler1 = CosineAnnealingLR(self.optimizer1, T_max=args.epochs, eta_min=0)
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
                tfeat_tensor, tslideLabel = ds[bag_idx]
                tfeat_tensor, tslideLabel = tfeat_tensor.to(self.device), tslideLabel.to(self.device)

                slide_sub_preds, gSlidePred = self.dtfdmil(tfeat_tensor)
                slide_sub_labels = torch.zeros(size=(slide_sub_preds.shape[0],), device=self.device, dtype=torch.int64) + tslideLabel

                loss0 = self.loss_function(slide_sub_preds, slide_sub_labels).mean()
                self.optimizer0.zero_grad()
                loss0.backward(retain_graph=True)

                loss1 = self.loss_function(gSlidePred, tslideLabel.view(-1)).mean()
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

                    allSlide_pred_softmax = []

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

                            patch_pred_logits = get_cam_1d(self.classifier, tattFeats.unsqueeze(0)).squeeze(0)  ###  cls x n
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
                        gt_0 = torch.cat([gt_0, slide_sub_labels], dim=0)
                        loss0 = self.loss_function(slide_sub_preds, slide_sub_labels).mean()
                        test_loss0 += loss0.item()

                        gSlidePred = self.attCls(slide_d_feat)
                        allSlide_pred_softmax.append(torch.softmax(gSlidePred, dim=1))

                    allSlide_pred_softmax = torch.cat(allSlide_pred_softmax, dim=0)
                    allSlide_pred_softmax = torch.mean(allSlide_pred_softmax, dim=0).unsqueeze(0)
                    gPred_1 = torch.cat([gPred_1, allSlide_pred_softmax], dim=0)
                    gt_1 = torch.cat([gt_1, tslideLabel], dim=0)

                    loss1 = F.nll_loss(allSlide_pred_softmax, tslideLabel)
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



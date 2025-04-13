import torch
import torch.nn as nn
import random
import numpy as np

from trainers.utils import get_cam_1d


class DTFDMIL(nn.Module):
    def __init__(self, device, classifier, attention, dimReduction, attCls, numgroup=4, final_activation_function="softmax" ,distill="MaxMinS", shuffle=True):
        super(DTFDMIL, self).__init__()
        self.classifier = classifier
        self.attention = attention
        self.dimReduction = dimReduction
        self.attCls = attCls
        self.distill = distill
        self.numgroup = numgroup
        self.shuffle = shuffle
        self.device = device
        if final_activation_function == "softmax":
            self.activation_function = nn.Softmax(dim=-1)
        elif final_activation_function == "sigmoid":
            self.activation_function = nn.Sigmoid()
        else:
            raise NotImplementedError(f"activation function {final_activation_function} is not implemented.")

    def forward(self, tfeat_tensor):
        slide_pseudo_feat = []
        slide_sub_preds = []


        feat_index = list(range(tfeat_tensor.shape[0]))
        if self.shuffle:
            random.shuffle(feat_index)
        index_chunk_list = np.array_split(feat_index, self.numgroup)

        index_chunk_list = [sst.tolist() for sst in index_chunk_list]

        for tindex in index_chunk_list:
            subFeat_tensor = torch.index_select(tfeat_tensor, dim=0,
                                                index=torch.LongTensor(tindex).to(self.device))
            tmidFeat = self.dimReduction(subFeat_tensor)
            tAA = self.attention(tmidFeat).squeeze(0)

            tattFeats = torch.einsum('ns,n->ns', tmidFeat, tAA)
            tattFeat_tensor = torch.sum(tattFeats, dim=0).unsqueeze(0)
            print("haha")
            print(tattFeat_tensor.shape)
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
        slide_pseudo_feat = torch.cat(slide_pseudo_feat, dim=0)
        slide_sub_preds = torch.cat(slide_sub_preds, dim=0)  ### numGroup x fs
        print(slide_sub_preds.shape)
        print("--")
        gSlidePred = self.attCls(slide_pseudo_feat)
        """return instance logits and bag logits"""
        return slide_sub_preds, gSlidePred

""" This two model's architecture is specially designed for bag-mnist.
    It randomly group data into bags.
    For more details, see Paper: Attention-based Deep Multiple Instance Learning
    Link: https://arxiv.org/pdf/1802.04712
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class AttentionModel(nn.Module):
    def __init__(self):
        super(AttentionModel, self).__init__()
        self.M = 500
        self.L = 128
        self.ATTENTION_BRANCHES = 1



        self.feature_extractor_part2 = nn.Sequential(
            nn.Linear(1024, self.M),
            nn.ReLU(),
        )

        self.attention = nn.Sequential(
            nn.Linear(self.M, self.L), # matrix V
            nn.Tanh(),
            nn.Linear(self.L, self.ATTENTION_BRANCHES) # matrix w (or vector w if self.ATTENTION_BRANCHES==1)
        )

        self.classifier = nn.Sequential(
            nn.Linear(self.M*self.ATTENTION_BRANCHES, 1),
        )

    def forward(self, data):
        if data.shape[0] == 1:
            return self.forward_without_batch(data)
        else:
            return self.forward_with_batch(data)
    def forward_without_batch(self, H):
        H = self.feature_extractor_part2(H)  # KxM
        A = self.attention(H)
        A = torch.transpose(A, 1, 0)  # ATTENTION_BRANCHESxK
        A = F.softmax(A, dim=1)  # softmax over K

        Z = A.reshape(1, -1) @ H.squeeze(0) # ATTENTION_BRANCHESxM

        logits = self.classifier(Z)
        logits = torch.cat((1 - logits, logits), dim=-1)
        return logits


    def forward_with_batch(self, data):

        batch_size =data.shape[0]
        num_img = data.shape[1]
        data = data.reshape(batch_size * num_img, data.shape[2], data.shape[3], data.shape[4])
        H = self.feature_extractor_part1(data)
        H = H.view(batch_size, -1, 800)
        H = self.feature_extractor_part2(H)  # Shape: [batch_size, M]

        # Compute attention for the entire batch
        A = self.attention(H)  # Shape: [batch_size, K, ATTENTION_BRANCHES]
        A = A.transpose(1, 2)  # Transpose to [batch_size, ATTENTION_BRANCHES, K]
        A = F.softmax(A, dim=-1)

        Z = torch.bmm(A, H)

        # Classify Z
        Z = Z.squeeze(1)  # Shape: [batch_size, M] (if ATTENTION_BRANCHES == 1)
        logits = self.classifier(Z)  # Shape: [batch_size, 1]

        # Concatenate logits with (1 - logits) for binary classification
        logits = torch.cat((1 - logits, logits), dim=-1)  # Shape: [batch_size, 2]

        return logits

    # AUXILIARY METHODS
    def calculate_classification_error(self, X, Y):
        Y = Y.float()
        _, Y_hat, _ = self.forward(X)
        error = 1. - Y_hat.eq(Y).cpu().float().mean().data.item()

        return error, Y_hat

    def calculate_objective(self, X, Y):
        Y = Y.float()
        Y_prob, _, A = self.forward(X)
        Y_prob = torch.clamp(Y_prob, min=1e-5, max=1. - 1e-5)
        neg_log_likelihood = -1. * (Y * torch.log(Y_prob) + (1. - Y) * torch.log(1. - Y_prob))  # negative log bernoulli

        return neg_log_likelihood, A

class GatedAttentionModel(nn.Module):
    def __init__(self):
        super(GatedAttentionModel, self).__init__()
        self.M = 500
        self.L = 128
        self.ATTENTION_BRANCHES = 1


        self.feature_extractor_part2 = nn.Sequential(
            nn.Linear(1024, self.M),
            nn.ReLU(),
        )

        self.attention_V = nn.Sequential(
            nn.Linear(self.M, self.L), # matrix V
            nn.Tanh()
        )

        self.attention_U = nn.Sequential(
            nn.Linear(self.M, self.L), # matrix U
            nn.Sigmoid()
        )

        self.attention_w = nn.Linear(self.L, self.ATTENTION_BRANCHES) # matrix w (or vector w if self.ATTENTION_BRANCHES==1)

        self.classifier = nn.Sequential(
            nn.Linear(self.M*self.ATTENTION_BRANCHES, 1),
        )

    def forward(self, data):
        return self.forward_without_batch(data)

    def forward_without_batch(self, H):
        H = self.feature_extractor_part2(H)  # KxM
        A_V = self.attention_V(H)  # KxL
        A_U = self.attention_U(H)  # KxL
        A = self.attention_w(A_V * A_U)  # element wise multiplication # KxATTENTION_BRANCHES
        A = torch.transpose(A, 1, 0)  # ATTENTION_BRANCHESxK
        A = F.softmax(A, dim=-1)  # softmax over K
        Z = A.reshape(1, -1) @ H.squeeze(0)  # ATTENTION_BRANCHESxM
        logits = self.classifier(Z)
        logits = torch.cat((1 - logits, logits), dim=1)
        return logits

"""    def forward(self, data):
        if data.shape[0] == 1:
            return self.forward_without_batch(data)
        else:
            #return self.forward_with_batch(data)"""

"""  def forward_with_batch(self, H):
        H = H.view(batch_size, -1, 800)
        H = self.feature_extractor_part2(H)  # KxM

        A_V = self.attention_V(H)  # KxL
        A_U = self.attention_U(H)  # KxL
        A = self.attention_w(A_V * A_U) # element wise multiplication # KxATTENTION_BRANCHES
        A = torch.transpose(A, 1, 2)  # ATTENTION_BRANCHESxK
        A = F.softmax(A, dim=-1)  # softmax over K

        Z = torch.bmm(A, H)  # ATTENTION_BRANCHESxM

        logits = self.classifier(Z)

        logits = torch.cat((1 - logits, logits), dim=1)
        logits = logits.squeeze(2)
        return logits"""


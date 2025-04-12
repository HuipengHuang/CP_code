import torch.nn as nn
import torch
import numpy as np
import torch.nn.functional as F
from .embed_position import EPEG

class MultiHeadAttention(nn.Module):

    def __init__(self, d_model=512, n_head=8, dropout=0.1):
        super().__init__()
        assert d_model % n_head == 0, "d_model must be divisible by n_head"

        self.d_model = d_model
        self.n_head = n_head
        self.d_k = d_model // n_head  # Dimension per head

        # Linear projections for Q, K, V
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)

        self.dropout = nn.Dropout(dropout)
        self.out = nn.Linear(d_model, d_model)  # Output projection

    def forward(self, x):
        B, L2, M, M2, D = x.shape
        assert D == self.d_model

        q = self.w_q(x)
        k = self.w_k(x)
        v = self.w_v(x)

        q = q.view(B, L2, M * M2, self.n_head, self.d_k)
        k = k.view(B, L2, M * M2, self.n_head, self.d_k)
        v = v.view(B, L2, M * M2, self.n_head, self.d_k)

        q = q.transpose(1, 3)
        k = k.transpose(1, 3)
        v = v.transpose(1, 3)

        scores = torch.matmul(q, k.transpose(-1, -2)) / (self.d_k ** 0.5)


        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        attn_output = torch.matmul(attn_weights, v)
        attn_output = attn_output.transpose(1, 3)
        attn_output = attn_output.contiguous().view(B, L2, M, M2, self.d_model)

        output = self.out(attn_output)
        output = self.dropout(output)

        return output



class RMSA(nn.Module):
    def __init__(self, region_size):
        super(RMSA, self).__init__()
        self.region_size = region_size
        self.norm = nn.LayerNorm(self.region_size, eps=1e-6)
        self.ps_embed = EPEG()

    def forward(self, x):
        #  x is the logit after offline feature extractor
        #  x:[B, I, D]
        x = self.norm(x)
        B, I, D = x.shape
        H, W = int(np.ceil(np.sqrt(I))), int(np.ceil(np.sqrt(I)))

        delta = H % self.region_size
        H += delta
        W += delta
        add_length = H * W - I

        x = torch.cat((x, x[:, :add_length, :]), dim=1)

        # X is reshaped to B L**2, M, M, D
        x = x.reshape(B, -1, self.region_size, self.region_size, D)
        x = self.ps_embed(x)
        return x

class CrMSA(nn.Module):
    """Cross-region multi-head self-attention"""
    def __init__(self, D, K):
        super(CrMSA, self).__init__()
        self.D = D
        self.K = K
        self.w = nn.Linear(D, K)
        self.attn = MultiHeadAttention()

    def forward(self, x):
        # Input shape: (B, L2, M, M2, D)
        B, L2, M, M2, D = x.shape

        # Step 1: Aggregate representative features (R_l)
        # Compute logits for attention weights: (B, L2, M, M2, k)
        logits = self.w(x)
        # Softmax to get aggregation weights W_l^a: (B, L2, M, M2, K)
        W_a = torch.softmax(logits, dim=-1)
        # Transpose for matrix multiplication: (B, L2, M, K, M2)
        W_a_t = W_a.transpose(-1, -2)
        # B, L2, M, K, D
        R = torch.matmul(W_a_t, x)
        R_hat = self.attn(R)

        logits = logits.reshape(B, L2, M * M2, logits.shape[-1])

        minmax_w = (logits - logits.min(dim=-1, keepdim=True)) / (logits.max(dim=-1, keepdim=True) - logits.min(dim=-1, keepdim=True) + 1e-8)
        minmax_w = torch.softmax(minmax_w, dim=-1)
        print(minmax_w.transpose(-1, -2).shape)
        print(R_hat.shape)
        result = torch.matmul(minmax_w.transpose(-1, -2), R_hat)
        result = torch.matmul(result, minmax_w)

        return result

"""
        logits_d = torch.sum(logits, dim=-2)  # (B, L2, K)
        W_d = (logits_d - logits_d.min(dim=-1, keepdim=True)[0]) / 
              (logits_d.max(dim=-1, keepdim=True)[0] - logits_d.min(dim=-1, keepdim=True)[0] + 1e-8)
        W_d = W_d.unsqueeze(-1)  # (B, L2, K, 1)
        W_d_t = W_d.transpose(-1, -2)  # (B, L2, 1, K)

        # Distribute: Z_l = (W_l^d)^T * R_hat * W_l^d
        # First matmul: (B, L2, 1, K) @ (B, L2, K, 1, D) -> (B, L2, 1, 1, D)
        Z = torch.matmul(W_d_t, R_hat)
        # Second matmul: (B, L2, 1, 1, D) @ (B, L2, K, 1) -> (B, L2, M, M, D)
        Z = torch.matmul(Z.squeeze(-2), W_d).view(B, L2, M, M, D)"""


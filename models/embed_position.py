import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import einops

class PPEG(nn.Module):
    def __init__(self, dim=512):
        super(PPEG, self).__init__()
        self.proj = nn.Conv2d(dim, dim, 7, 1, 7 // 2, groups=dim)
        self.proj1 = nn.Conv2d(dim, dim, 5, 1, 5 // 2, groups=dim)
        self.proj2 = nn.Conv2d(dim, dim, 3, 1, 3 // 2, groups=dim)

    def forward(self, x):
        """class token should be appended at the beginning of x before input to forward function"""
        cls_token, feat_token = x[:, 0], x[:, 1:]

        B, N, C = feat_token.shape
        #N-1 because
        H, W = int(np.ceil(np.sqrt(N))), int(np.ceil(np.sqrt(N)))

        add_length = H * W - N
        feat_token = torch.cat((feat_token, feat_token[:, :add_length, :]), dim=1)

        cnn_feat = feat_token.transpose(1, 2).view(B, C, H, W)
        feat = self.proj(cnn_feat) + cnn_feat + self.proj1(cnn_feat) + self.proj2(cnn_feat)
        feat = feat.flatten(2).transpose(1, 2)

        out = torch.cat((cls_token.unsqueeze(1), feat), dim=1)
        return out

class EPEG(nn.Module):
    def __init__(self, d_model=512, n_head=8, dropout=0.1, conv_kernel_size=7):
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
        self.conv1d = nn.Conv1d(
            in_channels=1,
            out_channels=1,
            kernel_size=conv_kernel_size,
            padding=(conv_kernel_size - 1) // 2
        )

    def forward(self, x):
        B, L2, M, M, D = x.shape
        assert D == self.d_model

        q = self.w_q(x)  # (B, L2, M, M, D)
        k = self.w_k(x)  # (B, L2, M, M, D)
        v = self.w_v(x)  # (B, L2, M, M, D)

        # Reshape for multi-head over M x M grid
        q = q.view(B * L2, M * M, self.n_head, self.d_k).transpose(1, 2)  # (B * L2, n_head, M * M, d_k)
        k = k.view(B * L2, M * M, self.n_head, self.d_k).transpose(1, 2)  # (B * L2, n_head, M * M, d_k)
        v = v.view(B * L2, M * M, self.n_head, self.d_k).transpose(1, 2)  # (B * L2, n_head, M * M, d_k)

        scores = torch.matmul(q, k.transpose(-1, -2)) / (self.d_k ** 0.5)  # (B * L2, n_head, M * M, M * M)

        scores_reshaped = scores.view(B * L2 * self.n_head, 1, M * M * M * M)
        conv_scores = self.conv1d(scores_reshaped)
        conv_scores = conv_scores.view(B * L2, self.n_head, M * M, M * M)

        attn_weights = F.softmax(scores + conv_scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        attn_output = torch.matmul(attn_weights, v)  # (B * L2, n_head, M * M, d_k)
        attn_output = attn_output.transpose(1, 2).contiguous().view(B, L2, M, M, self.d_model)

        output = self.out(attn_output)  # (B, L2, M, M, D)
        output = self.dropout(output)

        return output



class PositionEmbedding(nn.Module):
    def __init__(self, size, dim=512):
        super().__init__()
        self.size = size
        self.pe = nn.Embedding(size + 1, dim, padding_idx=0)
        self.pos_ids = torch.arange(1, size + 1, dtype=torch.long).cuda()

    def forward(self, emb):
        device = emb.device
        b, n, *_ = emb.shape
        pos_ids = self.pos_ids
        if n > self.size:
            zeros = torch.zeros(n - self.size, dtype=torch.long, device=device)
            pos_ids = torch.cat([pos_ids, zeros])
        pos_ids = einops.repeat(pos_ids, 'n -> b n', b=b)
        pos_emb = self.pe(pos_ids)  # [b n pe_dim]
        embeddings = torch.cat([emb, pos_emb], dim=-1)
        return embeddings
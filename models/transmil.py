import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from nystrom_attention import NystromAttention
import torchvision.models as models
from .embed_position import PPEG

class TransLayer(nn.Module):

    def __init__(self, device, norm_layer=nn.LayerNorm, dim=512):
        super().__init__()
        self.norm = norm_layer(dim).to(device)
        self.attn = NystromAttention(
            dim=dim,
            dim_head=dim // 8,
            heads=8,
            num_landmarks=dim // 2,  # number of landmarks
            pinv_iterations=6,
            # number of moore-penrose iterations for approximating pinverse. 6 was recommended by the paper
            residual=True,
            # whether to do an extra residual with the value or not. supposedly faster convergence if turned on
            dropout=0.1
        ).to(device)

    def forward(self, x):
        x = x + self.attn(self.norm(x))

        return x


class TransMIL(nn.Module):
    def __init__(self, device, n_classes=2):
        super(TransMIL, self).__init__()

        self.pos_layer = PPEG(dim=512).to(device)
        self._fc1 = nn.Sequential(nn.Linear(1024, 512), nn.ReLU()).to(device)
        self.cls_token = nn.Parameter(torch.randn(1, 1, 512)).to(device)
        self.n_classes = n_classes
        self.layer1 = TransLayer(dim=512, device=device)
        self.layer2 = TransLayer(dim=512, device=device)
        self.norm = nn.LayerNorm(512).to(device)
        self._fc2 = nn.Linear(512, self.n_classes).to(device)
        self.device = device

    def forward(self, h):
        h = h.to(self.device)
        h = self._fc1(h)  # [B, n, 512]
        # ---->cls_token
        B = h.shape[0]
        H = h.shape[1]

        _H, _W = int(np.ceil(np.sqrt(H))), int(np.ceil(np.sqrt(H)))
        add_length = _H * _W - H
        h = torch.cat([h, h[:, :add_length, :]], dim=1)

        cls_tokens = self.cls_token.expand(B, -1, -1)
        h = torch.cat((cls_tokens, h), dim=1)

        # ---->Translayer x1
        h = self.layer1(h)  # [B, N, 512]

        # ---->PPEG
        h = self.pos_layer(h)  # [B, N, 512]

        # ---->Translayer x2
        h = self.layer2(h)  # [B, N, 512]

        # ---->cls_token
        h = self.norm(h)[:, 0]

        # ---->predict
        logits = self._fc2(h)
        return logits

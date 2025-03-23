import torch
import torch.nn as nn
from overrides import overrides


class Adapter(nn.Module):
    def __init__(self, input_dim, num_classes, device):
        super().__init__()
        self.num_classes = num_classes
        self.adapter_net = nn.Sequential(nn.Linear(in_features=input_dim, out_features=128, bias=True, device=device),
                                        nn.ReLU(),
                                         nn.Linear(in_features=128, out_features=128, bias=True, device=device),
                                         nn.ReLU(),
                                        nn.Linear(in_features=128, out_features=num_classes, bias=True, device=device))

    def forward(self, logits):
        return self.adapter_net(logits)



class CAdapter(Adapter):
    def __init__(self, input_dim, num_classes, device):
        super(CAdapter, self).__init__(input_dim, num_classes, device)

    @overrides
    def forward(self, logits):
        prob = torch.softmax(logits, dim=-1)
        U = torch.triu(torch.ones(self.num_classes, self.num_classes).to(logits.device))
        sorted_prob, sorted_indices = torch.sort(prob, descending=True)

        #  Caluculate r_i - r_(i+1), r is sorted_prob
        #  A small problem. when i set diffs to sqrt(r_i - r_(i+1)),
        #  I suffer from gradient vanishing, the logits will become nan.
        diffs = sorted_prob[:, :-1] - sorted_prob[:, 1:]

        diffs = torch.cat((diffs, torch.ones((diffs.shape[0], 1),
                                             dtype=diffs.dtype,
                                             device=diffs.device)), dim=1)
        calibrated_logits = self.adapter_net(prob)
        calibrated_logits[:, :-1] = torch.sigmoid(calibrated_logits[:, :-1])
        calibrated_logits = diffs * calibrated_logits

        fitted_logits = torch.zeros_like(logits)
        for i in range(logits.shape[0]):
            sorted_row_prob, row_sorted_indices = sorted_prob[i], sorted_indices[i]

            S = torch.zeros(size=(self.num_classes, self.num_classes), device=logits.device)
            S[torch.arange(logits.shape[1]), row_sorted_indices] = 1
            #  S.T = inverse of S
            fitted_logits[i] = S.T.to(torch.float) @ U @ calibrated_logits[i]
        #  Residual connection. It would not affect ranking.
        fitted_logits = fitted_logits + logits
        return fitted_logits

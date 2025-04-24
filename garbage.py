import torch
import torch.nn as nn
x = torch.ones(size=(32, 3))
y = torch.ones(size=(32, 1))
print((x * y).sum(dim=0).shape)
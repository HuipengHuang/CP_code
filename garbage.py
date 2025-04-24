import torch
import torch.nn as nn
x = torch.ones(size=(10,))
print(x.unsqueeze(dim=0).shape)
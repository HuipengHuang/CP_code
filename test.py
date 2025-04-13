import torch
import torch.nn as nn
x  = torch.ones(size=(1,2,3))
print(x.squeeze(0).shape)
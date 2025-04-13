import torch
import torch.nn as nn

z = nn.CrossEntropyLoss()
x = torch.ones(size=(1, 2))
y = torch.tensor(1)
print(z(x, y.view(1)))
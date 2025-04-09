import torch
import torch.nn as nn
loss = nn.CrossEntropyLoss()
x = torch.load(r"E:\PycharmProjects\conftr\normal_001.pt")
print(x.shape)
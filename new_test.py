from models import attention_base_model
import torch
x  =  torch.ones(size=(1, 3, 512))
model = attention_base_model.GatedAttentionModel(input_dim=512)
print(model(x).shape)
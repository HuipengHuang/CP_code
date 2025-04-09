import torch
from loss import dsmil_loss
import models.dsmil as dsmil

i_classifier = dsmil.FCLayer(in_size=1024, out_size=2)
b_classifier = dsmil.BClassifier(input_size=1024, mDim=512, output_class=2,
                                       passing_v=False)
milnet = dsmil.MILNet(i_classifier, b_classifier)
x = torch.ones(size=(1,3,1024))
out = milnet(x)
y = torch.tensor([1])
losss = dsmil_loss.DSMilLoss()


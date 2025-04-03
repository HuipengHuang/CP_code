from .mil_predictor import MilPredictor
from .predictor import Predictor
import torch
def get_predictor(args, net, num_classes, adapter=None, final_activation_function="softmax"):
    if args.model == "dsmil":
        predictor = MilPredictor(args, net, num_classes, final_activation_function, adapter)
    else:
        predictor = Predictor(args, net, num_classes, final_activation_function, adapter)
    return predictor

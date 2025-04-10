from .mil_predictor import MilPredictor
from .predictor import Predictor
from .aggregation_predictor import MaxPredictor
import torch
def get_predictor(args, net, num_classes, adapter=None, final_activation_function="softmax"):
    if args.model == "dsmil":
        predictor = MilPredictor(args, net, num_classes, final_activation_function, adapter)
    elif args.aggregation is not None:
        if args.aggregation == "max":
            predictor = MaxPredictor(args, net, num_classes, final_activation_function, adapter)
        else:
            raise NotImplementedError
    else:
        predictor = Predictor(args, net, num_classes, final_activation_function, adapter)
    return predictor

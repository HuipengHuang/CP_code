from .predictor import Predictor
from .instance_predictor import Instance_Predictor
from .dtfd_predictor import DTFDPredictor
from .weight_predictor import WeightPredictor
from .aggregation_predictor import MaxPredictor, KMeanPredictor
import torch
def get_predictor(args, net, num_classes, adapter=None, final_activation_function="softmax"):
    if args.model == "dtfdmil":
        predictor = DTFDPredictor(args, net, num_classes, final_activation_function, adapter)
    elif args.model == "weight":
        predictor = WeightPredictor(args, net, num_classes, final_activation_function, adapter)
    elif args.aggregation is not None:
        if args.aggregation == "max":
            predictor = MaxPredictor(args, net, num_classes, final_activation_function, adapter)
        elif args.aggregation == "kmean":
            predictor = KMeanPredictor(args, net, num_classes, final_activation_function, adapter)
        else:
            raise NotImplementedError
    elif args.instance_level == "True":
        predictor = Instance_Predictor(args, net, num_classes, final_activation_function)
    else:
        predictor = Predictor(args, net, num_classes, final_activation_function, adapter)
    return predictor

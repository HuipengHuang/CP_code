from .bce_loss import BinaryCrossEntropyLoss
from .contr_loss import ConftrLoss
from .uncertainty_aware_loss import UncertaintyAwareLoss
from .cadapter_loss import CAdapterLoss
from .standard_loss import StandardLoss
import torch.nn as nn


def get_loss_function(args, predictor):
    if args.loss == "conftr":
        assert args.size_loss_weight is not None, print("Please specify a size_loss_weight")
        assert args.tau is not None, print("Please specify a tau.")
        assert args.size_loss_weight >= 0, print("size_loss_weight should be greater than or equal to 0.")
        assert args.tau >= 0, print("Tau should be greater than or equal to 0.")

        return ConftrLoss(args, predictor)

    elif args.loss == "ua":
        return UncertaintyAwareLoss(args, predictor)
    elif args.loss == "cadapter":
        return CAdapterLoss(args, predictor)
    elif args.loss == "bce":
        return BinaryCrossEntropyLoss()
    elif args.loss == "standard":
        return StandardLoss(args)

from .trainer import Trainer
from .uncertainty_aware_trainer import UncertaintyAwareTrainer
from .mil_trainier import MilTrainer

def get_trainer(args, num_classes):
    if args.algorithm == "uatr":
        trainer = UncertaintyAwareTrainer(args, num_classes)
    elif args.model == "dsmil":
        trainer = MilTrainer(args, num_classes)
    else:
        trainer = Trainer(args, num_classes)
    return trainer
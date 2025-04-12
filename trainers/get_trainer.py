from .trainer import Trainer
from .uncertainty_aware_trainer import UncertaintyAwareTrainer
from .dtfd_trainer import DFDT_Trainer
def get_trainer(args, num_classes):
    if args.algorithm == "uatr":
        trainer = UncertaintyAwareTrainer(args, num_classes)
    elif args.model == "dtfdmil":
        trainer = DFDT_Trainer(args, num_classes)
    else:
        trainer = Trainer(args, num_classes)
    return trainer
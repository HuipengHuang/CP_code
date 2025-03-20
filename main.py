import argparse
from torch.utils.data import DataLoader
from common.utils import build_dataset, set_seed, save_exp_result
from trainers.utils import get_trainer


parser = argparse.ArgumentParser()

parser.add_argument("--model", type=str, default="resnet50", help='Choose neural network architecture.')
parser.add_argument("--dataset", type=str, default="cifar100", choices=["cifar10", "cifar100", "imagenet"], help="Choose dataset for training.")
parser.add_argument('--seed', type=int, default=None)
parser.add_argument("--pretrained", default="False", type=str, choices=["True", "False"])
parser.add_argument("--save", default="False", choices=["True", "False"], type=str)
parser.add_argument("--algorithm",'-alg', default="standard", choices=["standard", "uatr"], help="Uncertainty aware training use uatr. Otherwise use standard")

#  Training configuration
parser.add_argument("--optimizer", type=str, default="sgd", choices=["sgd", "adam"], help="Choose optimizer.")
parser.add_argument("--learning_rate", "-lr", type=float, default=0.1, help="Initial learning rate for optimizer")
parser.add_argument("--epochs", '-e', type=int, default=1, help='Number of epochs to train')
parser.add_argument("--batch_size",'-bsz', type=int, default=32)
parser.add_argument("--momentum", type=float, default=0, help='Momentum')
parser.add_argument("--weight_decay", type=float, default=0, help='Weight decay')
parser.add_argument("--loss", type=str,default='standard', choices=['standard','conftr','ua'], help='Loss function you want to use. ua means uncertainty aware')
parser.add_argument("--temperature",'-T', type=float, default=None, help='Temperature scaling')

#  Hyperpatameters for Conformal Prediction
parser.add_argument("--alpha", type=float, default=0.1, help="Error Rate")
parser.add_argument("--score", type=str, default="thr", choices=["thr", "aps", "raps", "saps"])
parser.add_argument("--cal_ratio", type=float, default=0.5, help="Ratio of calibration data's size. (1 - cal_ratio) means ratio of test data's size")

#  Hyperparameters for ConfTr
parser.add_argument("--size_loss_weight", type=float, default=None, help='Weight for size loss in ConfTr')
parser.add_argument("--tau", type=float, default=None, help='Hyperparameter for ConfTr. Soft predicted Size larger than tau will be penalized in the size loss.')

#  Hyperparameter for aps, raps and saps
parser.add_argument("--random",type=str,default=None,choices=["True","False"])
parser.add_argument("--raps_size_regularization",type=float, default=None, help='K_reg for raps loss')
parser.add_argument("--raps_weight",type=float, default=None ,help="lambda for size regularization in raps.")
parser.add_argument("--saps_weight",type=float, default=None ,help="lambda for size regularization in saps.")

#  Hyperparameter for uncertainty aware loss
parser.add_argument("--mu", type=float, default=None, help="Weight of train_loss_score in the uncertainty_aware_loss function")
parser.add_argument("--mu_size", type=float, default=None, help="Weight of train_loss_size in the unvertainty_aware_loss function")

#  Hyperparameter for c-adapter
parser.add_argument("--adapter", type=str, default="False", choices=["True", "False"], help="Add Adapter or not.")
parser.add_argument("--train_net", type=str, default=None, choices=["True", "False"], help="Train the original newural network or not.")
parser.add_argument("--train_adapter", type=str, default=None, choices=["True", "False"], help="Train the adapter or not.")

args = parser.parse_args()
seed = args.seed
if seed:
    set_seed(seed)

train_dataset, cal_dataset, test_dataset, num_classes = build_dataset(args)
train_loader = DataLoader(train_dataset, batch_size=args.batch_size, drop_last=True)
cal_loader = DataLoader(cal_dataset, batch_size=args.batch_size)
test_loader = DataLoader(test_dataset, batch_size=args.batch_size)

trainer = get_trainer(args, num_classes)
trainer.train(train_loader, args.epochs)
trainer.predictor.calibrate(cal_loader)
result_dict = trainer.predictor.evaluate(test_loader)

for key, value in result_dict.items():
    print(f'{key}: {value}')

if args.save == "True":
    save_exp_result(args, trainer, result_dict)

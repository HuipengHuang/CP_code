import argparse
from torch.utils.data import DataLoader, ConcatDataset
from common.utils import set_seed, save_exp_result
from datasets.utils import build_dataloader, get_dataset_list
from trainers.utils import get_trainer

parser = argparse.ArgumentParser()

parser.add_argument("--model", type=str, default="resnet50", help='Choose neural network architecture.')
parser.add_argument("--gpu", type=int,help="CUDA device ID (e.g., 0, 1, etc.)")
parser.add_argument("--dataset", type=str, default="cifar100", choices=["cifar10", "cifar100", "imagenet", "mnist_bag", "camelyon17", "camelyon16", "tcga_lung_cancer"],
                    help="Choose dataset for training.")
parser.add_argument('--seed', type=int, default=None)
parser.add_argument("--pretrained", default="False", type=str, choices=["True", "False"])
parser.add_argument("--save", default="False", choices=["True", "False"], type=str)
parser.add_argument("--save_model", default=None, choices=["True", "False"])
parser.add_argument("--algorithm",'-alg', default="cp", choices=["standard", "cp", "uatr"],
                    help="standard means only evaluate top1 accuracy."
                         "cp means use conformal prediction at evaluation stage. "
                         "Uncertainty aware training use uatr. Otherwise use standard")
parser.add_argument("--final_activation_function",default="softmax", choices=["softmax", "sigmoid"])
parser.add_argument("--save_feature", default=None, choices=["True", "False"])
parser.add_argument("--save_result", default=None, choices=["True", "False"])
parser.add_argument("--extract_feature_model", default=None, choices=["resnet18", "resnet50"])
parser.add_argument("--input_dimension", default=None, type=int, choices=[512, 1024])
parser.add_argument("--cross_validation", default=None, type=int)

#  Training configuration
parser.add_argument("--optimizer", type=str, default="sgd", choices=["sgd", "adam"], help="Choose optimizer.")
parser.add_argument("--learning_rate", "-lr", type=float, default=1e-1, help="Initial learning rate for optimizer")
parser.add_argument("--epochs", '-e', type=int, default=1, help='Number of epochs to train')
parser.add_argument("--batch_size",'-bsz', type=int, default=32)
parser.add_argument("--momentum", type=float, default=0, help='Momentum')
parser.add_argument("--weight_decay", type=float, default=0, help='Weight decay')
parser.add_argument("--nesterov", default=False, choices=["True", "False"], type=str)
parser.add_argument("--learning_rate_scheduler", default=None, type=str, choices=["cosine"])
parser.add_argument("--loss", type=str,default='standard', choices=['standard', 'conftr', 'ua', "cadapter", "bce"],
                    help='Loss function you want to use. standard loss is Cross Entropy Loss.')

#  Hyperparameters for Multi-instance learning
parser.add_argument("--multi_instance_learning", "-mil", default=None, type=str, choices=["True", "False"],)
parser.add_argument("--compute_auc","-auc", default=None, type=str, choices=["True", "False"],help="Compute AUC or not.")

#  Hyperparameters for Conformal Prediction
parser.add_argument("--alpha", type=float, default=0.1, help="Error Rate")
parser.add_argument("--train_score", type=str, default=None, choices=["thr", "thrlp"],
                    help="train_score is set to be the same as test_score when --train_score is None.")
parser.add_argument("--test_score", type=str, default="thr", choices=["thr", "aps", "raps", "saps", "thrlp"])
parser.add_argument("--cal_ratio", type=float, default=0.5,
                    help="Ratio of calibration data's size. (1 - cal_ratio) means ratio of test data's size")

#  Hyperparameters for ConfTr
parser.add_argument("--size_loss_weight", type=float, default=None, help='Weight for size loss in ConfTr')
parser.add_argument("--tau", type=float, default=None,
                    help='Hyperparameter for ConfTr.'
                         'Soft predicted Size larger than tau will be penalized in the size loss.')
parser.add_argument("--temperature",'-T', type=float, default=None,
                    help='Temperature scaling for ConfTr or C-adapter loss')

#  Hyperparameter for aps, raps and saps
parser.add_argument("--random",type=str,default=None,choices=["True","False"])
parser.add_argument("--raps_size_regularization",type=float, default=None, help='K_reg for raps loss')
parser.add_argument("--raps_weight",type=float, default=None ,help="lambda for size regularization in raps.")
parser.add_argument("--saps_weight",type=float, default=None ,help="lambda for size regularization in saps.")

#  Hyperparameter for uncertainty aware loss
parser.add_argument("--mu", type=float, default=None,
                    help="Weight of train_loss_score in the uncertainty_aware_loss function")
parser.add_argument("--mu_size", type=float, default=None,
                    help="Weight of train_loss_size in the uncertainty_aware_loss function")

#  Hyperparameter for c-adapter
parser.add_argument("--adapter", type=str, default="False", choices=["True", "False"],
                    help="Add Adapter or not.")
parser.add_argument("--cadapter", type=str, default="False", choices=["True", "False"],
                    help="Add CAdapter or not.")
parser.add_argument("--train_net", type=str, default=None, choices=["True", "False"],
                    help="Train the neural network or not.")
parser.add_argument("--train_adapter", type=str, default=None, choices=["True", "False"],
                    help="Train the adapter or not.")

args = parser.parse_args()
seed = args.seed
if seed:
    set_seed(seed)
if args.cross_validation is None:
    if args.algorithm == "cp" or args.algorithm == "uatr":
        train_loader, cal_loader, test_loader, num_classes = build_dataloader(args)

        trainer = get_trainer(args, num_classes)

        trainer.train(train_loader, args.epochs)

        trainer.predictor.calibrate(cal_loader)
        result_dict = trainer.predictor.evaluate(test_loader)
        for key, value in result_dict.items():
            print(f'{key}: {value}')
        if args.save == "True":
            save_exp_result(args, trainer, result_dict)

    else:

        train_loader, _, test_loader, num_classes = build_dataloader(args)
        trainer = get_trainer(args, num_classes)

        trainer.train(train_loader, args.epochs)
        result_dict = trainer.predictor.evaluate(test_loader)
        print(f"AUC: {trainer.predictor.compute_auc(test_loader)}")
        for key, value in result_dict.items():
            print(f'{key}: {value}')
        if args.save == "True":
            save_exp_result(args, trainer, result_dict)

else:
    num_validation = args.cross_validation
    dataset_list, num_classes = get_dataset_list(args)
    for i in range(num_validation):
        if i != 0:
            j = i - 1
        else:
            j = 1
        test_dataset = dataset_list[i]
        cal_dataset = dataset_list[j]
        train_dataset = ConcatDataset([ dataset_list[i] for k in range(num_validation) if k != i and k != j])
        train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
        cal_dataloader = DataLoader(cal_dataset, batch_size=args.batch_size, shuffle=True)
        test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

        trainer = get_trainer(args, num_classes)
        trainer.train(train_dataloader, args.epochs)
        trainer.predictor.calibrate(cal_dataloader)
        result_dict = trainer.predictor.evaluate(test_dataloader)
        for key, value in result_dict.items():
            print(f'{key}: {value}')
        if args.save == "True":
            save_exp_result(args, trainer, result_dict)



import random

import numpy as np
from sklearn.model_selection import KFold
import torch
from torch.utils.data import ConcatDataset, Subset, DataLoader

from datasets.utils import build_dataloader, build_dataset
from trainers.utils import get_trainer
from common.utils import save_exp_result


def cp(args):
    if args.algorithm == "cp" or args.algorithm == "uatr":
        train_loader, cal_loader, test_loader, num_classes = build_dataloader(args)

        trainer = get_trainer(args, num_classes)

        trainer.train(train_loader, args.epochs, val_loader=cal_loader)

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

def standard(args):
    train_loader, _, test_loader, num_classes = build_dataloader(args)
    trainer = get_trainer(args, num_classes)

    trainer.train(train_loader, args.epochs)
    result_dict = trainer.predictor.evaluate(test_loader)
    print(f"AUC: {trainer.predictor.compute_auc(test_loader)}")
    for key, value in result_dict.items():
        print(f'{key}: {value}')
    if args.save == "True":
        save_exp_result(args, trainer, result_dict)


def cp_with_cv(args):
    # Load datasets
    mil_train_dataset, mil_cal_dataset, mil_test_dataset, num_classes = build_dataset(args)

    # Combine all datasets for k-fold CV
    ds = ConcatDataset([mil_train_dataset, mil_cal_dataset, mil_test_dataset])
    n_samples = len(ds)

    # Initialize results storage
    all_results = []

    # Repeat k-fold CV for args.ktime times
    for time in range(args.ktime):
        # Initialize k-fold splitter
        kfold = KFold(n_splits=args.kfold, shuffle=True, random_state=args.seed + time if args.seed else None)

        # Perform k-fold CV
        for fold, (train_idx, val_idx) in enumerate(kfold.split(np.arange(n_samples))):
            print(len(train_idx), len(val_idx))
            print(f"\nTime {time + 1}/{args.ktime}, Fold {fold + 1}/{args.kfold}")

            # Create subsets
            train_subset = Subset(ds, train_idx)
            val_subset = Subset(ds, val_idx)

            # Create data loaders
            train_loader = DataLoader(train_subset, batch_size=args.batch_size, shuffle=True)
            val_loader = DataLoader(val_subset, batch_size=args.batch_size, shuffle=False)

            # Initialize and train model
            trainer = get_trainer(args, num_classes)
            trainer.train(train_loader, args.epochs)

            # Evaluate on validation set
            result_dict = trainer.predictor.evaluate(val_loader)

            all_results.append(result_dict)

            # Print fold results
            print(f"Fold {fold + 1} Results:")
            for key, value in result_dict.items():
                print(f'{key}: {value}')

    # Calculate and print average performance across all folds and times
    avg_results = {}
    for metric in all_results[0].keys():
        avg_results[metric] = torch.mean([r[metric] for r in all_results])

    print("\nFinal Average Performance:")
    for key, value in avg_results.items():
        print(f'Average {key}: {value:.4f}')

    if args.save == "True":
        save_exp_result(args, trainer, avg_results)

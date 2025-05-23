import numpy as np
from torch.utils.data import ConcatDataset, Subset, DataLoader
from sklearn.model_selection import StratifiedKFold, KFold
from datasets.utils import build_dataloader, build_dataset
from trainers.get_trainer import get_trainer
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


def standard(args):
    train_loader, _, test_loader, num_classes = build_dataloader(args)
    trainer = get_trainer(args, num_classes)

    trainer.train(train_loader, args.epochs, val_loader=test_loader)
    result_dict = trainer.predictor.evaluate(test_loader)
    for key, value in result_dict.items():
        print(f'{key}: {value}')
    if args.save == "True":
        save_exp_result(args, trainer, result_dict)


def cross_validation(args):
    # Load datasets
    mil_train_dataset, mil_cal_dataset, mil_test_dataset, num_classes = build_dataset(args)

    # Combine all datasets for k-fold CV
    if mil_cal_dataset is None:
        ds = ConcatDataset([mil_train_dataset, mil_test_dataset])
    else:
        ds = ConcatDataset([mil_train_dataset, mil_cal_dataset, mil_test_dataset])

    n_samples = len(ds)
    label = np.array([target.item() for _, target in ds])

    all_results = []
    if args.stratified:
        KFOLD = StratifiedKFold
    else:
        KFOLD = KFold
    # Repeat k-fold CV for args.ktime times
    for time in range(args.ktime):
        skf = KFOLD(n_splits=args.kfold, shuffle=True, random_state=args.seed+time if args.seed else None)
        # Perform k-fold CV
        for fold, (train_idx, test_idx) in enumerate(skf.split(np.arange(n_samples), label)):
            print(f"\nTime {time + 1}/{args.ktime}, Fold {fold + 1}/{args.kfold}")

            trainer = get_trainer(args, num_classes)

            train_subset = Subset(ds, train_idx)
            test_subset = Subset(ds, test_idx)

            # Create data loaders
            if args.algorithm == "cp":
                skf_cal = KFOLD(n_splits=args.kfold - 1, shuffle=True, random_state=args.seed+time+fold if args.seed else None)
                train_label = np.array([target.item() for _, target in train_subset])
                train_sub_idx, cal_idx = next(iter(skf_cal.split(np.arange(len(train_subset)), train_label)))
                cal_subset = Subset(train_subset,cal_idx)
                train_subset = Subset(train_subset,train_sub_idx)
            else:
                cal_subset = None

            if cal_subset is not None:
                cal_loader = DataLoader(cal_subset, batch_size=args.batch_size, shuffle=False)
                trainer.predictor.calibrate(cal_loader)
            else:
                cal_loader = None

            train_loader = DataLoader(train_subset, batch_size=args.batch_size, shuffle=True)
            test_loader = DataLoader(test_subset, batch_size=args.batch_size, shuffle=False)

            if args.algorithm == "standard":
                trainer.train(train_loader, args.epochs, test_loader)
            else:
                trainer.train(train_loader, args.epochs, cal_loader)

            # Evaluate on validation set
            result_dict = trainer.predictor.evaluate(test_loader)

            all_results.append(result_dict)

            # Print fold results
            print(f"Fold {fold + 1} Results:")
            for key, value in result_dict.items():
                print(f'{key}: {value}')

    # Calculate and print average performance across all folds and times
    avg_results = {}
    for metric in all_results[0].keys():
        avg_results[metric] = np.mean([r[metric] for r in all_results])

    print("\nFinal Average Performance:")
    for key, value in avg_results.items():
        print(f'Average {key}: {value:.4f}')

    if args.save == "True":
        save_exp_result(args, trainer, avg_results)

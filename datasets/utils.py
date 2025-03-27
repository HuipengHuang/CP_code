import torch
from torchvision.transforms import transforms
from torch.utils.data import DataLoader, Subset, random_split
from torch.utils.data import ConcatDataset
import wilds
from .camelyon17 import MILCamelyon17

def wsi_collate_fn(batch):
    """For handling camelyon17 dataset"""
    inputs = torch.stack([item[0] for item in batch], dim=0)
    labels = torch.stack([item[1] for item in batch])
    #metadata = torch.stack([item[2] for item in batch])
    return inputs, labels

def build_dataset(args):
    dataset_name = args.dataset

    if dataset_name == "cifar10":
        from torchvision.datasets import CIFAR10
        num_classes = 10
        train_dataset = CIFAR10(root='./data', train=True, download=True,transform=transforms.Compose([transforms.ToTensor()]))
        cal_test_dataset = CIFAR10(root='./data', train=False, download=True,
                                 transform=transforms.Compose([transforms.ToTensor()]))

    elif dataset_name == "cifar100":
        from torchvision.datasets import CIFAR100
        num_classes = 100
        train_dataset = CIFAR100(root='./data', train=True, download=True,transform=transforms.Compose([transforms.ToTensor()]))
        cal_test_dataset = CIFAR100(root='./data', train=False, download=True,
                                 transform=transforms.Compose([transforms.ToTensor()]))

    elif dataset_name == "imagenet":
        from torchvision.datasets import ImageNet
        num_classes = 1000
        train_dataset = ImageNet(root='./data', train=True, download=True,transform=transforms.Compose([transforms.ToTensor()]))
        cal_test_dataset = ImageNet(root='./data', train=False, download=True,
                                 transform=transforms.Compose([transforms.ToTensor()]))

    elif dataset_name == "mnist_bag":
        from . import mnist_bag
        if args.multi_instance_learning != "True":
            raise ValueError("Please set multi-instance-learning to true.")
        if args.batch_size != 1:
            print(f"Attention. Current batch size is {args.batch_size}, not 1.")
        # positive or negative
        num_classes = 2
        device = torch.device(f"cuda:{args.gpu}")
        train_dataset = mnist_bag.MnistBags(device, train=True)
        cal_test_dataset = mnist_bag.MnistBags(device, train=False)

    elif dataset_name == "camelyon17":
        if args.multi_instance_learning != "True":
            raise ValueError("Please set multi-instance-learning to true.")

        assert args.batch_size == 1, print("Batch size must be 1.")

        num_classes = 2
        dataset = wilds.get_dataset(dataset="camelyon17", download=True)

        train_dataset = dataset.get_subset("train")
        #  Validation (ID)
        id_cal_dataset = dataset.get_subset("id_val")
        #  Validation(OOD)
        od_cal_dataset = dataset.get_subset("val")
        test_dataset = dataset.get_subset("test")

        #  Make Sure the calibration data and the test data are exchangeable.
        concat_dataset = ConcatDataset((id_cal_dataset, od_cal_dataset, test_dataset))
        cal_size = int(len(concat_dataset) * args.cal_ratio)
        test_size = len(concat_dataset) - cal_size
        cal_dataset, test_dataset = random_split(concat_dataset, [cal_size, test_size])

        device = torch.device(f"cuda:{args.gpu}")
        mil_train_dataset = MILCamelyon17(train_dataset, device, transform=transforms.Compose([transforms.ToTensor()]))
        mil_test_dataset = MILCamelyon17(test_dataset, device, transform=transforms.Compose([transforms.ToTensor()]))
        mil_cal_dataset = MILCamelyon17(cal_dataset, device, transform=transforms.Compose([transforms.ToTensor()]))
        return mil_train_dataset, mil_cal_dataset, mil_test_dataset, num_classes

    if args.algorithm != "standard":
        cal_size = int(len(cal_test_dataset) * args.cal_ratio)
        test_size = len(cal_test_dataset) - cal_size
        cal_dataset, test_dataset = random_split(cal_test_dataset, [cal_size, test_size])
    else:
        cal_dataset = None
        test_dataset = cal_test_dataset

    return train_dataset, cal_dataset, test_dataset, num_classes

def build_dataloader(args):
    train_dataset, cal_dataset, test_dataset, num_classes = build_dataset(args)
    if args.dataset == "camelyon17":
        train_laoder = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, drop_last=True)
        if cal_dataset:
            cal_loader = DataLoader(cal_dataset, batch_size=args.batch_size, shuffle=True)
        else:
            cal_loader = None
        test_laoder = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)
    else:
        train_laoder = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, drop_last=True)
        if cal_dataset:
            cal_loader = DataLoader(cal_dataset, batch_size=args.batch_size, shuffle=True)
        else:
            cal_loader = None
        test_laoder = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    return train_laoder, cal_loader, test_laoder, num_classes

def split_dataloader(original_dataloader, split_ratio=0.5):
        """
        Splits a DataLoader into two Datasets

        Args:
            original_dataloader (DataLoader): The original DataLoader to split.
            split_ratio (float): The ratio of the first subset (default: 0.7).

        Returns:
            subset1: Training dataset
            subset2: Calibration dataset
        """
        dataset = original_dataloader.dataset
        total_size = len(dataset)

        split_size = int(split_ratio * total_size)

        indices = torch.randperm(total_size)
        indices_subset1 = indices[:split_size]
        indices_subset2 = indices[split_size:]

        subset1 = Subset(dataset, indices_subset1)
        subset2 = Subset(dataset, indices_subset2)

        return subset1, subset2
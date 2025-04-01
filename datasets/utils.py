import csv
import torch
import torchvision
from torchvision.transforms import transforms
from torch.utils.data import DataLoader, Subset, random_split
from torch.utils.data import ConcatDataset
import wilds
from .camelyon17 import MILCamelyon17
from .camelyon16 import MILCamelyon16, MILCamelyon16_rn18
import os
from torchvision import models
from data_preprocess import csv2pth


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

    elif dataset_name == "camelyon16":
        print("i am in 16")
        if args.multi_instance_learning != "True":
            raise ValueError("Please set multi-instance-learning to true.")

        assert args.batch_size == 1, print("Batch size must be 1.")
        num_classes = 2
        device = torch.device(f"cuda:{args.gpu}")
        save_path = "./data/camelyon16_rn18_feature"
        if  args.save_feature:
            csv2pth.csv2pth(data_path=r"./data/camelyon16_rn18_csv", save_path=save_path)

        mil_train_dataset = MILCamelyon16_rn18(device=device, path=save_path+"/train")
        mil_cal_test_dataset = MILCamelyon16_rn18(device, path=save_path+"/test")

        cal_size = int(args.cal_ratio * len(mil_cal_test_dataset))
        test_size = len(mil_cal_test_dataset) - cal_size
        mil_cal_dataset, mil_test_dataset = random_split(mil_cal_test_dataset, [cal_size, test_size])
        return mil_train_dataset, mil_cal_dataset, mil_test_dataset, num_classes

    elif dataset_name == "camelyon17":
        print("I am in 17")
        if args.multi_instance_learning != "True":
            raise ValueError("Please set multi-instance-learning to true.")

        assert args.batch_size == 1, print("Batch size must be 1.")

        num_classes = 2
        device = torch.device(f"cuda:{args.gpu}")

        if args.save_feature == "True":
            dataset = wilds.get_dataset(dataset="camelyon17", download=True)

            train_dataset = dataset.get_subset("train")
            #  Validation (ID)
            id_cal_dataset = dataset.get_subset("id_val")
            #  Validation(OOD)
            od_cal_dataset = dataset.get_subset("val")
            test_dataset = dataset.get_subset("test")

            #  Make Sure the calibration data and the test data are exchangeable.
            concat_dataset = ConcatDataset((id_cal_dataset, od_cal_dataset, test_dataset))

            save_features(device=device, path="./data/camelyon17_features/train", dataset=train_dataset)
            save_features(device=device, path="./data/camelyon17_features/test", dataset=concat_dataset)

        mil_train_dataset = MILCamelyon17(device=device,path="./data/camelyon17_features/train")
        mil_cal_test_dataset = MILCamelyon17(device, path="./data/camelyon17_features/test")
        cal_size = int(args.cal_ratio * len(mil_cal_test_dataset))
        test_size = len(mil_cal_test_dataset) - cal_size
        mil_cal_dataset, mil_test_dataset = random_split(mil_cal_test_dataset, [cal_size, test_size])
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
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, drop_last=True)
        if cal_dataset:
            cal_loader = DataLoader(cal_dataset, batch_size=args.batch_size, shuffle=False)
        else:
            cal_loader = None
        test_laoder = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)
    else:
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, drop_last=True)
        if cal_dataset:
            cal_loader = DataLoader(cal_dataset, batch_size=args.batch_size, shuffle=False)
        else:
            cal_loader = None
        test_laoder = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    return train_loader, cal_loader, test_laoder, num_classes



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


def save_features(device, path, dataset, transform=torchvision.transforms.Compose([transforms.ToTensor(),
                                                                                   transforms.Normalize(mean=[0.485, 0.456, 0.406],  std=[0.229, 0.224, 0.225]
    )])):
    if os.path.exists(path):
        return
    else:
        os.makedirs(path)
        os.mkdir(path+"/data")

    with torch.no_grad():
        feature_extractor_part = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1).to(device)
        feature_extractor_part.eval()

        dictionary = {}
        for i in range(len(dataset)):
            img, label, metadata = dataset[i]
            img_tensor = transform(img).to(device)
            label = label.to(device)
            metadata = tuple(metadata.tolist())

            if metadata not in dictionary.keys():

                dictionary[metadata] = [feature_extractor_part(img_tensor.unsqueeze(0)), label]
            else:
                dictionary[metadata][0] = torch.cat(
                    (dictionary[metadata][0], feature_extractor_part(img_tensor.unsqueeze(0))), dim=0)

        label_mapping = []
        for i, (metadata, value) in enumerate(dictionary.items()):
            data, label = value
            # Save the feature tensor
            save_path = os.path.join(path, f'data/data_{i}.pth')
            torch.save(data, save_path)

            # Store filename and label for CSV
            label_value = label.item()
            label_mapping.append((f'{i}', label_value))

        # Save labels to CSV
        csv_path = os.path.join(path, 'labels.csv')
        with open(csv_path, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(['filename', 'label'])  # Header
            writer.writerows(label_mapping)


def build_subset_dataloader(args, train=True):
    assert args.multi_instance_learning == "True", print("Not supported yet.")
    dataset_name = args.dataset

    if dataset_name == "camelyon16":
        assert args.multi_instance_learning == "True", print("Please set multi-instance-learning to true.")

        assert args.batch_size == 1, print("Batch size must be 1.")
        num_classes = 2
        device = torch.device(f"cuda:{args.gpu}")
        if train:
            mil_train_dataset = MILCamelyon16(device=device, path="./data/camelyon16_features", train=True)
            train_loader = DataLoader(mil_train_dataset, batch_size=args.batch_size, shuffle=True, drop_last=True)
            return train_loader, None, num_classes
        else:
            mil_cal_test_dataset = MILCamelyon16(device, path="./data/camelyon16_features", train=False)

            cal_size = int(args.cal_ratio * len(mil_cal_test_dataset))
            test_size = len(mil_cal_test_dataset) - cal_size
            mil_cal_dataset, mil_test_dataset = random_split(mil_cal_test_dataset, [cal_size, test_size])
            cal_loader = DataLoader(mil_cal_dataset, batch_size=args.batch_size, shuffle=False)
            test_loader = DataLoader(mil_test_dataset, batch_size=args.batch_size, shuffle=False)
            return cal_loader, test_loader, num_classes

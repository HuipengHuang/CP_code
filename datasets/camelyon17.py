import torch
from torch.utils.data import Dataset
from torchvision.transforms import transforms
from torchvision import models
from tqdm import tqdm

class MILCamelyon17(Dataset):
    def __init__(self, dataset, device, transform=transforms.ToTensor()):
        self.device = device
        self.feature_extractor_part = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1).to(self.device)
        for param in self.feature_extractor_part.parameters():
            param.requires_grad = False

        dict ={}
        for i in range(len(dataset)):
            img, label, metadata = dataset[i]
            img_tensor = transform(img).to(self.device)
            label = label.to(self.device)
            metadata = tuple(metadata.tolist())
            if metadata not in dict:
                dict[metadata] = [self.feature_extractor_part(img_tensor.unsqueeze(0)), label]
            else:
                dict[metadata][0] = torch.cat((dict[metadata][0], self.feature_extractor_part(img_tensor.unsqueeze(0))), dim=0)

        self.bag = []
        self.label = []
        for val in dict.values():
            self.bag.append(val[0])
            self.label.append(val[1])

    def __len__(self):
        return len(self.label)

    def __getitem__(self, idx):
        data = self.bag[idx]
        label = self.label[idx]
        return data, label


"""class MILCamelyon17(Dataset):
    def __init__(self, dataset, device, transform=transforms.ToTensor()):
        self.device = device
        self.dataset = dataset
        self.transform = transform
        self.metadata_dict = {}

        self.feature_extractor_part = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1).to(self.device)
        for param in self.feature_extractor_part.parameters():
            param.requires_grad = False

        for i in tqdm(range(len(dataset))):
            _, _, metadata = dataset[i]
            metadata = tuple(metadata.tolist())
            if metadata not in self.metadata_dict:
                self.metadata_dict[metadata] = []
            self.metadata_dict[metadata].append(i)
    def __len__(self):
        return len(list(self.metadata_dict.keys()))

    def __getitem__(self, idx):
        metadata = self.metadata_keys[idx]
        indices = self.metadata_dict[metadata]
        bag_tensor = []
        label = None
        for i in indices:
            img, lbl, _ = self.dataset[i]
            img_tensor = self.transform(img).to(self.device)
            logit = self.feature_extractor_part(img_tensor.unsqueeze(0))
            bag_tensor.append(logit)
            label = lbl

        bag = torch.cat(bag_tensor, dim=0)
        label = label.to(self.device)
        return bag, label"""

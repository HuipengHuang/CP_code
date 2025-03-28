import torch
from torch.utils.data import Dataset
from torchvision.transforms import transforms
from torchvision import models

class MILCamelyon17(Dataset):
    def __init__(self, dataset, device, transform=transforms.ToTensor()):
        self.device = device
        self.dataset = dataset
        self.transform = transform
        self.metadata_dict = {}

        self.feature_extractor_part = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1).to(self.device)
        for param in self.feature_extractor_part.parameters():
            param.requires_grad = False

        for i in range(len(dataset)):
            _, _, metadata = dataset[i]
            metadata = tuple(metadata.tolist())
            if metadata not in self.metadata_dict:
                self.metadata_dict[metadata] = []
            self.metadata_dict[metadata].append(i)
        self.metadata_keys = list(self.metadata_dict.keys())
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
        return bag, label

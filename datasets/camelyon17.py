import csv
import os
from tqdm import tqdm
import torch
from torch.utils.data import Dataset


class MILCamelyon17(Dataset):
    def __init__(self, device, path):
        self.device = device
        self.data_list = []
        self.label_list = []
        self.path = path


        # Correct CSV filename based on your save_features function
        csv_path = os.path.join(path, 'labels.csv')
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"Labels CSV not found at {csv_path}")

        with open(csv_path, 'r') as csvfile:
            reader = csv.DictReader(csvfile)
            for row in tqdm(reader,desc="Loading dataset"):
                # Extract filename and label from CSV
                file_idx = row['filename']  # This is just the index number
                label = eval(row['label'])  # Convert string to int or list

                # Construct the full filepath
                # Note: your save_features saves to 'data/data_{i}.pth'
                data_path = os.path.join(path, 'data', f'data_{file_idx}.pth')

                # Load data and move to device
                data = torch.load(data_path)

                label = torch.tensor(label)

                self.data_list.append(data)
                self.label_list.append(label)

    def __len__(self):
        return len(self.label_list)  # Use label_list, not label

    def __getitem__(self, idx):
        data = self.data_list[idx]
        label = self.label_list[idx]
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
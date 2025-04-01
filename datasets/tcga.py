import csv
import os
import torch
from torch.utils.data import Dataset
from tqdm import tqdm


class TCGA_rn18(Dataset):
    def __init__(self, device, path, train):
        self.device = device
        self.data_list = []
        self.label_list = []

        if train == True:
            path = path + "/train/"
        else:
            path = path + "/test/"
        self.path = path


        # Correct CSV filename based on your save_features function
        csv_path = os.path.join(path, 'label.csv')
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
                data = torch.load(data_path).to(self.device).to(torch.float32)

                label = torch.tensor(label, device=device)

                self.data_list.append(data)
                self.label_list.append(label)

    def __len__(self):
        return len(self.label_list)  # Use label_list, not label

    def __getitem__(self, idx):
        data = self.data_list[idx]
        label = self.label_list[idx]
        return data, label
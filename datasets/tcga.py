import csv
import os

import pandas as pd
import torch
from torch.utils.data import Dataset
from tqdm import tqdm


class TCGA_rn18(Dataset):
    def __init__(self, device, path):
        self.device = device
        self.data_list = []
        self.label_list = []
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


class TCGA_rn50(Dataset):
    def __init__(self, device, path):
        self.device = device
        self.data_list = []
        self.label_list = []
        self.path = path

        df = pd.read_csv(f"{path}/label.csv", header=None, index_col=0)
        df.rename(columns={df.columns[0]: "label"}, inplace=True)
        for filename in os.listdir(f"{path}/pt_files"):
            #get rid of .pt
            file_name = filename.split(".")[0]
            file_name_list = file_name.split("-")
            id = file_name_list[0] + "-" + file_name_list[1] + "-" + file_name_list[2]

            label = df.loc[id, "label"]
            laber = 0 if label == "LUAD" else 1
            data = torch.load(os.path.join(f"{path}/pt_files", f"{filename}")).to(self.device).to(torch.float32)
            label = torch.tensor(label, device=device)

            self.data_list.append(data)
            self.label_list.append(label)

    def __len__(self):
        return len(self.label_list)  # Use label_list, not label

    def __getitem__(self, idx):
        data = self.data_list[idx]
        label = self.label_list[idx]
        return data, label
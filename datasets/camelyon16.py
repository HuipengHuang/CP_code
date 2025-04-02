import csv
import os

import torch
from torch.utils.data import Dataset
import pickle
from tqdm import tqdm
import pandas as pd

class MILCamelyon16(Dataset):
    def __init__(self, device, path, train=True):
        self.device = device
        self.data_list = []
        self.label_list = []
        self.path = path

        #  It will return a dictionary. Every element(Every element is a bag) in the dictionary's values is a list.
        #  Every element in the list is a dictionary. {"feature": (1024,), "label": 0 or 1, 'file_name': ...}
        if train:
            with open(path + "/mDATA_train.pkl", "rb") as f:
                data_dict = pickle.load(f)
                for data in tqdm(data_dict.values(), desc='Loading dataset'):
                    bag_feature = torch.cat(
                        [torch.tensor(instance["feature"]).to(self.device).unsqueeze(0) for instance in data], dim=0)
                    bag_label = torch.tensor(1, device=device) if 1 in [instance["label"] for instance in data] else torch.tensor(0,device=device)

                    self.data_list.append(bag_feature)
                    self.label_list.append(bag_label)

        else:
            df = pd.read_csv(path + '/test_reference.csv', header=None,
                             names=['Slide_ID', 'Label', 'Subtype', 'Metastasis_Type'])
            df = df.set_index('Slide_ID')
            with open(path + "/mDATA_test.pkl", "rb") as f:
                data_dict = pickle.load(f)
                for key in tqdm(data_dict.keys(), desc='Loading dataset'):
                    bag_feature = torch.cat(
                        [torch.tensor(instance["feature"]).to(self.device).unsqueeze(0) for instance in data_dict[key]], dim=0)

                    bag_label = 1 if df.loc[key]["Label"] == "Tumor" else 0
                    self.data_list.append(bag_feature)
                    self.label_list.append(torch.tensor(bag_label, device=device))


    def __len__(self):
        return len(self.label_list)

    def __getitem__(self, idx):
        data = self.data_list[idx]
        label = self.label_list[idx]
        return data, label

class MILCamelyon16_rn18(Dataset):
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


class MILCamelyon16_rn50(Dataset):
    def __init__(self, device, path, train=True):
        self.device = device
        self.data_list = []
        self.label_list = []
        self.path = path

        df = pd.read_csv(path + '/test_reference.csv', header=None,
                         names=['Slide_ID', 'Label', 'Subtype', 'Metastasis_Type'])
        df = df.set_index('Slide_ID')
        for filename in os.listdir(path):
                if train:
                    if "normal" in filename:
                        label = 0
                    elif "tumor" in filename:
                        label = 1
                    else:
                        continue
                    data = torch.load(os.path.join(path, filename)).to(self.device).to(torch.float32)
                else:
                    if "test" in filename:
                        file_name = filename.split('.')[0]
                        label = 1 if df.loc[file_name]["Label"] == "Tumor" else 0
                        data = torch.load(os.path.join(path, filename)).to(self.device).to(torch.float32)
                    else:
                        continue
                label = torch.tensor(label, device=device)

                self.data_list.append(data)
                self.label_list.append(label)

    def __len__(self):
        return len(self.label_list)  # Use label_list, not label

    def __getitem__(self, idx):
        data = self.data_list[idx]
        label = self.label_list[idx]
        return data, label


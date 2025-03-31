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
                    bag_label = 1 if 1 in [instance["label"] for instance in data] else 0

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
                    bag_label = df.loc[key]["Label"]

                    self.data_list.append(bag_feature)
                    self.label_list.append(bag_label)


    def __len__(self):
        return len(self.label_list)  # Use label_list, not label

    def __getitem__(self, idx):
        data = self.data_list[idx]
        label = self.label_list[idx]
        return data, label



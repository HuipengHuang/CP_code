import torch
from torch.utils.data import Dataset
import pickle
from tqdm import tqdm

class MILCamelyon16(Dataset):
    def __init__(self, device, path):
        self.device = device
        self.data_list = []
        self.label_list = []
        self.path = path

        #  It will return a list. Every element(Every element is a bag) in the list is a list.
        #  Every element in the list is a dictionary. {"feature": (1024,), "label": 0 or 1, 'file_name': ...}
        data_list = pickle.load(path)
        for i in tqdm(range(len(data_list)), desc='Loading dataset'):
            bag_feature = torch.cat([torch.tensor(instance["feature"]).to(self.device).unsqueeze(0) for instance in data_list[i]], dim=0)
            bag_label = 1 if 1 in [instance["label"] for instance in data_list[i]] else 0
            self.data_list.append(bag_feature)
            self.label_list.append(bag_label)


    def __len__(self):
        return len(self.label_list)  # Use label_list, not label

    def __getitem__(self, idx):
        data = self.data_list[idx]
        label = self.label_list[idx]
        return data, label



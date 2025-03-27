import torch
from torch.utils.data import Dataset
from torchvision.transforms import transforms


class MILCamelyon17(Dataset):
    def __init__(self, dataset, device, transform=transforms.ToTensor()):
        self.device = device
        dict ={}
        for i in range(len(dataset)):
            if i == 2000:
                break
            img, label, metadata = dataset[i]
            img_tensor = transform(img)
            label = label
            metadata = metadata
            metadata = tuple(metadata.tolist())
            if metadata not in dict:
                dict[metadata] = [img_tensor.unsqueeze(0), label]
            else:
                dict[metadata][0] = torch.cat((dict[metadata][0], img_tensor.unsqueeze(0)), dim=0)

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

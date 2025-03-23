import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms
import numpy as np


class MnistBags(Dataset):
    """ Dataset used in paper: Attention-based Deep Multiple Instance Learning
        Link: https://arxiv.org/pdf/1802.04712
        Difference with standard MNIST dataset: It randomly samples a specific number of data to become a bag.
        Each data has its own label.
        Label of a bag is 1 if a data's label in the bag is 9 else 0.

        Args:
            mean_bag_length: Expected average number of data in every bag.
            (Due to randomness, the estimated average bag_length may not equal to mean_bag_length)
            var_bag_length: Variance of bag_length
            num_bag: Number of bags
            train: Get train data or test data.
        """
    def __init__(self, mean_bag_length=10, var_bag_length=0, num_bag=50, train=True):
        self.mean_bag_length = mean_bag_length
        self.var_bag_length = var_bag_length
        self.num_bag = num_bag
        self.train = train
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.num_in_train = 60000
        self.num_in_test = 10000

        if self.train:
            self.train_bags_list, self.train_labels_list = self._create_bags()
        else:
            self.test_bags_list, self.test_labels_list = self._create_bags()

    def _create_bags(self):
        #  Label of a bag is 1 if a data's label in the bag is 9.
        # Note that the batch_size is just the total length of the dataset.
        # Thus  for (batch_data, batch_labels) in loader:
        #             all_imgs = batch_data.to(self.device)
        #             all_labels = batch_labels.to(self.device)
        # This is correct. The author wrote it...
        if self.train:
            loader = DataLoader(datasets.MNIST('./data',
                                                          train=True,
                                                          download=True,
                                                          transform=transforms.Compose([
                                                              transforms.ToTensor(),
                                                              transforms.Normalize((0.1307,), (0.3081,))])),
                                           batch_size=self.num_in_train,
                                           shuffle=False)
        else:
            loader = DataLoader(datasets.MNIST('./data',
                                                          train=False,
                                                          download=True,
                                                          transform=transforms.Compose([
                                                              transforms.ToTensor(),
                                                              transforms.Normalize((0.1307,), (0.3081,))])),
                                           batch_size=self.num_in_test,
                                           shuffle=False)

        for (batch_data, batch_labels) in loader:
            all_imgs = batch_data.to(self.device)
            all_labels = batch_labels.to(self.device)

        bags_list = []
        labels_list = []

        for i in range(self.num_bag):
            #  bag_length: number of data in the bag.
            bag_length = int(np.random.normal(self.mean_bag_length, self.var_bag_length, 1))
            if bag_length < 1:
                bag_length = 1

            if self.train:
                #  get the bag data's indices
                #  Indices is of shape (bag_length,)
                indices = torch.LongTensor(np.random.randint(0, self.num_in_train, bag_length))
            else:
                indices = torch.LongTensor(np.random.randint(0, self.num_in_test, bag_length))

            labels_in_bag = all_labels[indices]
            labels_of_bag = 1 if ((labels_in_bag == 9).sum() > 0) else 0

            img_in_bag = all_imgs[indices]

            labels_list.append(labels_of_bag)
            bags_list.append(img_in_bag)

        return bags_list, labels_list

    def __len__(self):
        if self.train:
            return len(self.train_labels_list)
        else:
            return len(self.test_labels_list)

    def __getitem__(self, index):
        if self.train:
            bag = self.train_bags_list[index]
            label = self.train_labels_list[index]
        else:
            bag = self.test_bags_list[index]
            label = self.test_labels_list[index]

        return bag, label

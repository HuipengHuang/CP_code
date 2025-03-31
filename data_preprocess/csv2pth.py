import csv
import os
import torch
import pandas as pd


def csv2pth(data_path, save_path):
    if os.path.exists(save_path):
        return
    else:
        os.mkdir(save_path+"/train")
        os.mkdir(save_path+"/test")
        with open(save_path + "/train/label.csv", 'w') as f:
            pass
        with open(save_path + "/test/label.csv", 'w') as f:
            pass
    train_writer = csv.writer(open(save_path + "/train/label.csv", 'w', newline=""),)
    train_writer.writerow(['filename', 'label'])
    test_writer = csv.writer(open(save_path + "/test/label.csv", 'w', newline=""))
    test_writer.writerow(['filename', 'label'])
    i = 0
    j = 0
    for file in os.listdir(data_path + "/1-tumor/"):
        tumor_df = pd.read_csv(data_path + "/1-tumor/" + file, header=None)
        np_array = tumor_df.to_numpy()
        tensor_array = torch.from_numpy(np_array)
        if "test" in file:
            torch.save(tensor_array, save_path + f"/test/data/data_{j}.pth")
            test_writer.writerow([j, 1])
            j += 1
        else:
            torch.save(tensor_array, save_path + f"/train/data/data_{i}.pth")
            train_writer.writerow([i, 1])
            i += 1

    for file in os.listdir(data_path + "/0-normal/"):
        tumor_df = pd.read_csv(data_path + "/0-normal/" + file, header=None)
        np_array = tumor_df.to_numpy()
        tensor_array = torch.from_numpy(np_array)
        if "test" in file:
            torch.save(tensor_array, save_path + f"/test/data/data_{j}.pth")
            test_writer.writerow([j, 0])
            j += 1
        else:
            torch.save(tensor_array, save_path + f"/train/data/data_{i}.pth")
            train_writer.writerow([i, 0])
            i += 1

csv2pth(data_path=r".\data\camelyon16_r18", save_path=r".\data\camelyon16_18_torch")
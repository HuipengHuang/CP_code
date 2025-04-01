import csv
import os
import torch
import pandas as pd


def csv2pth(data_path, save_path):
    if os.path.exists(save_path):
        return
    else:
        os.makedirs(save_path+"/train", exist_ok=True)
        os.makedirs(save_path+"/test", exist_ok=True)
        os.makedirs(save_path + "/test/data", exist_ok=True)
        os.makedirs(save_path + "/train/data", exist_ok=True)
        with open(save_path + "/train/label.csv", 'w') as f:
            f.write("")
        with open(save_path + "/test/label.csv", 'w') as f:
            f.write("")

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


def tcga_rn18_csv2pth(data_path, save_path):
    if os.path.exists(save_path):
        return
    else:
        os.makedirs(save_path+"/train", exist_ok=True)
        os.makedirs(save_path+"/test", exist_ok=True)
        os.makedirs(save_path + "/test/data", exist_ok=True)
        os.makedirs(save_path + "/train/data", exist_ok=True)
        with open(save_path + "/train/label.csv", 'w') as f:
            f.write("")
        with open(save_path + "/test/label.csv", 'w') as f:
            f.write("")

    train_writer = csv.writer(open(save_path + "/train/label.csv", 'w', newline=""),)
    train_writer.writerow(['filename', 'label'])
    test_writer = csv.writer(open(save_path + "/test/label.csv", 'w', newline=""))
    test_writer.writerow(['filename', 'label'])

    df = pd.read_csv(data_path + "/TCGA.csv")
    train_size = int(df.shape[0] * 0.6)
    train_df = df[:train_size]
    test_df = df[train_size:]
    for i in range(train_df.shape[0]):
        file_name = train_df.iloc[i, 0]
        file_name = file_name.split("/")[1]

        label = train_df.iloc[i, 1]
        data_df = pd.read_csv(data_path + f"/tcga_lung_data_feats/{file_name}.csv", header=None)
        np_array = data_df.to_numpy()
        tensor_array = torch.from_numpy(np_array)
        torch.save(tensor_array, save_path + f"/train/data/data_{i}.pth")
        train_writer.writerow([i, label])
    for i in range(test_df.shape[0]):
        file_name = test_df.iloc[i][0]
        file_name = file_name.split("/")[1]

        label = test_df.iloc[i][1]
        data_df = pd.read_csv(data_path + f"/tcga_lung_data_feats/{file_name}.csv", header=None)
        np_array = data_df.to_numpy()
        tensor_array = torch.from_numpy(np_array)
        torch.save(tensor_array, save_path + f"/test/data/data_{i}.pth")
        test_writer.writerow([i, label])





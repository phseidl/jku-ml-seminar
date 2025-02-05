import json

import numpy as np
import pandas as pd
import torch
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader
from tqdm import tqdm


def read_json(filepath):
    if filepath is None:
        return None
    fd = open(filepath, "r")
    content = json.load(fd)
    fd.close()
    return content


def get_seizure_times(filepath):
    csv_lines = pd.read_csv(filepath, skiprows=5)
    start_stop_times = []
    for l in range(csv_lines.shape[0]):
        if (csv_lines.iloc[l, 3] == 'seiz'):
            start_stop_times.append([csv_lines.iloc[l, 1], csv_lines.iloc[l, 2]])
    return start_stop_times


def check_balance_of_dataset(dataset, subset = False, type='train'):
    if not subset:
        labels = dataset.get_labels()
    else:
        labels = [d[1] for d in dataset]
    label_list = []
    for label in tqdm(labels, 'Counting'):
        if not subset:
            label = np.load(label)
        label_list.append(label)
    # check balance of dataset
    label_list = np.array(label_list)
    label_list = label_list.max(axis=1)
    df = pd.DataFrame(label_list, columns=['labels'])
    label_list = label_list.flatten()
    unique, counts = np.unique(label_list, return_counts=True)
    print("stats for ", type)
    print("unique: ", unique)
    print("counts: ", counts)
    # print("df: ", df)

    return df, counts


def collate_fn(train_data):
    return train_data


if __name__ == "__main__":
    train_dir = "data/processed/train/labels"
    check_balance_of_dataset(train_dir)
    print("Done")

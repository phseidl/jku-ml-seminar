import glob
import json
import os

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


def check_balance_of_dataset(dataset, subset=False):
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
    df = pd.DataFrame(label_list.max(axis=1), columns=['labels'])
    label_list = label_list.flatten()
    unique, counts = np.unique(label_list, return_counts=True)
    print("unique: ", unique)
    print("counts: ", counts)
    # print("df: ", df)

    return df, counts


def collate_fn(train_data):
    return train_data


def plot_eeg_similarity_map(mma, sample, n_head):
    print("mma: ", mma)
    sample = sample.permute(1, 0)
    # plot_head_map(att_map[0].cpu().data.numpy(), label, label)
    label = ["fp1-f7", "fp2-f8", "f7-t3", "f8-t4", "t3-t5", "t4-t6", "t5-o1", "t6-o2", "t3-c3", "c4-t4", "c3-cz",
             "cz-c4", "fp1-f3", "fp2-f4", "f3-c3", "f4-c4", "c3-p3", "c4-p4", "p3-o1", "p4-o2"]

    plt.figure()
    for idx, label_name in enumerate(label):
        plt.subplot(20, 1, idx + 1)
        plt.plot(sample[idx].detach().cpu().numpy())
        plt.legend(label_name)
    plt.show()

    for i in range(n_head):
        # plt.subplots(4,1,i+1)
        fig, ax = plt.subplots()
        # ax[0][1].pcolor(mma, cmap=plt.cm.Blues)
        heatmap = ax.pcolor(mma[i], cmap=plt.cm.Blues)
        ax.set_xticks(np.arange(20) + 0.5, minor=False)
        ax.set_yticks(np.arange(20) + 0.5, minor=False)
        ax.set_xlim(0, 20)
        ax.set_ylim(0, 20)
        ax.invert_yaxis()
        ax.xaxis.tick_top()
        ax.set_xticklabels(label, minor=False)
        ax.set_yticklabels(label, minor=False)
        plt.xticks(rotation=45)
        plt.show()
    exit(1)


def evaluate_model(model: torch.nn.Module, dataloader: DataLoader, device: torch.device):
    model.eval()
    """Function for evaluation of a model `model` on the eeg_data in `dataloader` on device `device`"""
    with torch.no_grad():  # We do not need gradients for evaluation

        # Loop over all samples in `dataloader`
        prediction_list = list()
        target_list = list()

        for eeg_data in tqdm(dataloader, desc="scoring", position=0):
            # Get a sample and move inputs and targets to device
            inputs = eeg_data
            inputs = inputs.to(device)
            targets = targets.to(device)

            # Get outputs for network
            outputs = model(inputs)
            targets = targets.cpu()
            prediction_list.append(np.array(outputs))
            target_list.append(np.array(targets))
    model.train()
    return torch.tensor(torch.nn.CrossEntropyLoss(prediction_list, target_list))


if __name__ == "__main__":
    train_dir = "data/processed/train/labels"
    check_balance_of_dataset(train_dir)
    print("Done")

"""
+----------------------------------------------------------------------------------------------------------------------+
AMBULATORY SEIZURE FORECASTING USING WEARABLE DEVICES
Dataset and dataloader definition (datasets.py)

Johannes Kepler University, Linz
ML Seminar/Practical Work 2023/24
Author:  Jozsef Kovacs
Created: 17/02/2024

This file contains the dataset and dataloader class definitions.
+----------------------------------------------------------------------------------------------------------------------+
"""

from torch.utils.data import Dataset, DataLoader
import utils
from utils import console_msg


class MsgTrainDataset(Dataset):
    """
        Custom dataset class for the My Seizure Gauge data obtained from Epilepsyecosystem.org.
    """

    def __init__(self, train_dir, transform):
        self.LABEL_0, self.LABEL_1 = 0.18, 0.92
        self.train_dir = train_dir
        self.transform = transform

        # TODO: initialize dataset
        # TODO: load metadata files into a metadata structure
        # TODO: implement dataset iterator over parquet files : interictal/preictal segments

    def __len__(self):
        # TODO: dataset size calculcation
        return 0

    def __getitem__(self, idx):
        # TODO: implement data retrieval from segment
        features = None
        label = self.LABEL_1
        return features, label, idx


class MsgPredictionDataset(Dataset):
    """
        The prediction dataset is selected from the 2/3 to the end of the covered time-frame.
    """
    def __init__(self, config, transform):
        # TODO: initialize dataset
        # TODO: load metadata files into a metadata structure
        # TODO: implement dataset iterator over parquet files : interictal/preictal segments
        pass

    def __len__(self):
        # TODO: dataset size calculcation
        return 0

    def __getitem__(self, idx):
        # TODO: implement data retrieval from segment
        features = None
        label = self.LABEL_1
        return features, label, idx


class DevDataLoader(object):
    """
        Data loader helper - moving the data to the used device
    """
    def __init__(self, dl, num_samples, device):
        self.dl = dl
        self.device = device
        self.num_samples = num_samples

    def __iter__(self):
        for b in self.dl:
            yield utils.to_device(b, self.device)

    def __len__(self):
        return len(self.dl)

def init_datasets_and_loaders(mode, config):
    # TODO: dataset and loader initialization helper function
    pass

def init_forecasting_loaders(mode, config):
    # TODO: forecasting dataset initialization helper function
    pass
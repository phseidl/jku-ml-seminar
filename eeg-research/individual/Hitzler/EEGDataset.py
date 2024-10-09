import glob
import os
import re

import numpy as np
import mne
from torch.utils.data import Dataset
from tqdm import tqdm
from sklearn.preprocessing import normalize

class EEGDataset(Dataset):
    def __init__(self, data_dir, labels_dir):
        self.data = (glob.glob(os.path.join(data_dir, "*.fif")))
        self.labels = (glob.glob(os.path.join(labels_dir, "*labels.npy")))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        # get 30s epochs from edf file
        edf = np.squeeze(mne.read_epochs(self.data[index], verbose=40).get_data(copy=True))
        edf = normalize(edf)
        labels = np.load(self.labels[index])
        return edf.astype(np.float32), labels.astype(np.float32)

    def get_labels(self):
        return self.labels

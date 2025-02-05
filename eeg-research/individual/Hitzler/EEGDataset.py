import glob
import os

import mne
import numpy as np
from sklearn.preprocessing import normalize
from torch.utils.data import Dataset


class EEGDataset(Dataset):
    def __init__(self, data_dir, labels_dir, enc_model='raw'):
        # get list of edf files based on encoding model
        if enc_model == 'raw':
            self.data = (glob.glob(os.path.join(data_dir, "*.fif")))
        elif enc_model == 'stft':
            self.data = (glob.glob(os.path.join(data_dir, "*stft.npy")))
        else:
            raise ValueError("Invalid enc_model")
        # get list of labels
        self.labels = (glob.glob(os.path.join(labels_dir, "*labels.npy")))
        self.enc_model = enc_model

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        # get 30s epochs from edf file
        if self.enc_model == 'stft':
            edf = np.load(self.data[index])
        else:
            edf = np.squeeze(mne.read_epochs(self.data[index], verbose=40).get_data(copy=True))
        # normalize the data
        edf = normalize(edf)
        # get labels
        labels = np.load(self.labels[index])
        return edf.astype(np.float32), labels.astype(np.float32)

    def get_labels(self):
        return self.labels

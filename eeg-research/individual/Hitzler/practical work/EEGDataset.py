import glob
import os
import mne
import numpy as np
from sklearn.preprocessing import normalize
from torch.utils.data import Dataset


class EEGDataset(Dataset):
    def __init__(self, data_dir, labels_dir, enc_model='raw', dataset='TUH', montage_type='unipolar'):
        if dataset == 'TUH':
            if montage_type == 'unipolar':
                self.data = sorted(glob.glob(os.path.join(data_dir, "*unipolar-epo.fif")))
            elif montage_type == 'bipolar':
                self.data = sorted(glob.glob(os.path.join(data_dir, "*bipolar-epo.fif")))
        else:
            self.data = sorted(glob.glob(os.path.join(data_dir, "*epo.fif")))
        self.labels = sorted(glob.glob(os.path.join(labels_dir, "*labels.npy")))
        self.enc_model = enc_model

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        edf = np.squeeze(mne.read_epochs(self.data[index], verbose=40).get_data(copy=True))
        if self.enc_model == 'raw':
            edf = normalize(edf)
        labels = np.load(self.labels[index])
        return edf.astype(np.float32), labels.astype(np.float32)

    def get_labels(self):
        return self.labels

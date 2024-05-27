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
import torch
from torch.utils.data import Dataset, DataLoader, TensorDataset
import utils
from utils import console_msg
from msg_subject_data import MsgDataHelper, MsgSubjectData
from pathlib import Path
import pandas as pd
import numpy as np


class MsgTrainDataset(Dataset):
    """
        Custom dataset class for the My Seizure Gauge data obtained from Epilepsyecosystem.org.
    """
    def __init__(self, mdh: MsgDataHelper, subject_id: str, sequence_length: int = 60, sampling_freq: int = 128,
                 preictal_augment_factor : int = 6, batch_size : int = 15):

        self.sampling_freq = sampling_freq
        self.sequence_length = sequence_length
        self.preictal_augment_factor = preictal_augment_factor
        self.batch_size = batch_size
        self.input_size = mdh.input_size

        # label values
        self.LABEL_0, self.LABEL_1 = mdh.LABEL_0, mdh.LABEL_1

        # check if preprocessed DS descriptor file exists
        descfile = Path(mdh.preproc_dir / Path(subject_id) / Path('preproc_ds_desc.csv'))
        if descfile.exists() and descfile.is_file():
            self.ds_desc = pd.read_csv(descfile)
        else:
            # initialize dataset - training files
            self.filelist = list()
            for data_type in ['preictal_train', 'interictal_train']:
                path = mdh.preproc_dir / Path(subject_id) / Path(data_type)
                self.filelist += [(file, data_type, 0, 0, 0) for file in path.glob("*.h5")]
                if data_type == 'preictal_train':
                    # balance out classes by augmenting the number of preictal segments
                    self.filelist += (self.preictal_augment_factor - 1) * \
                                     [(file, data_type, 0, 0, 1) for file in path.glob("*.h5")]

            # calculate indices and size
            self.size = 0
            for i, (file, data_type, _, _, additive_noise) in enumerate(self.filelist):
                df = pd.read_hdf(file, key='df', mode='r')
                from_ix = self.size
                self.size += len(df) // (sequence_length * sampling_freq * batch_size)
                to_ix = self.size
                self.filelist[i] = (file, data_type, from_ix, to_ix, additive_noise)

            self.ds_desc = pd.DataFrame(self.filelist, columns=['filename', 'data_type', 'from_index', 'to_index', 'additive_noise'])
            self.ds_desc.to_csv(descfile)

        for i in self.ds_desc.index:
            console_msg(">>>", i, self.ds_desc['filename'][i], self.ds_desc['data_type'][i],
                        self.ds_desc['from_index'][i], self.ds_desc['to_index'][i], self.ds_desc['additive_noise'][i])

    def __len__(self):
        # TODO: dataset size calculcation
        return self.ds_desc['to_index'].iloc[-1]

    def find_file_index(self, idx):
        for i in self.ds_desc.index:
            if self.ds_desc['from_index'][i] <= idx < self.ds_desc['to_index'][i]:
                return i
        return None

    def __getitem__(self, idx):

        for i in self.ds_desc.index:
            if self.ds_desc['from_index'][i] <= idx < self.ds_desc['to_index'][i]:
                start = (idx - self.ds_desc['from_index'][i]) * self.sequence_length * self.sampling_freq
                stop = start + self.batch_size * self.sequence_length * self.sampling_freq
                df = pd.read_hdf(self.ds_desc['filename'][i], key='df', mode='r', start=start, stop=stop)
                data = np.asarray(df)

                # random additive noise - if augmented preictal segment
                if self.ds_desc['additive_noise'][i] == 1:
                    for j in range(0, data.shape[0], self.sampling_freq):
                        data[j:j+self.sampling_freq] = np.median(data[j:j+self.sampling_freq], axis=0) \
                                                       * np.random.uniform(1e-15, 1. - 1e-15, 1)

                return torch.from_numpy(data) \
                    .reshape(self.batch_size, -1, self.input_size), \
                    torch.ones((self.batch_size,), dtype=torch.float) * \
                    (self.LABEL_1 if self.ds_desc['data_type'][i] == 'preictal_train' else self.LABEL_0), \
                    self.ds_desc['additive_noise'][i], \
                    idx

        return None, None, None, idx


class MsgTensorDataset(TensorDataset):
    """
        Custom training dataset class for the My Seizure Gauge data obtained from Epilepsyecosystem.org.
    """
    def __init__(self, mdh: MsgDataHelper, subject_id: str, split_folder: str, sequence_length: int = 60, sampling_freq: int = 128):

        console_msg(f"STARTED LOADING THE DATASET... {subject_id}")
        self.sampling_freq = sampling_freq
        self.sequence_length = sequence_length
        self.input_size = mdh.input_size
        self.feature_list = []
        self.target_list = []

        # label values
        self.LABEL_0, self.LABEL_1 = mdh.LABEL_0, mdh.LABEL_1

        preictal_cnt = 0
        interictal_cnt = 0
        pp_meta = pd.read_csv(mdh.preproc_dir / Path(subject_id) / Path(split_folder) / 'pp_metadata.csv')
        for data_type, label in [('preictal_train', self.LABEL_1), ('interictal_train', self.LABEL_0)]:
            for file in pp_meta['filename'][pp_meta['type'] == data_type]:
                if data_type == 'preictal_train':
                    preictal_cnt += 1
                else:
                    interictal_cnt += 1
                console_msg(f"processing: {file}")
                df = pd.read_hdf(file, key='df', mode='r')
                a = np.array(df, dtype=np.float32)
                mult = self.sampling_freq // (mdh.config.stft.seg_size if mdh.config.stft.reduce else 1)
                a = np.nan_to_num(a.reshape(-1, mult * self.sequence_length, self.input_size))
                self.feature_list.append(a)
                self.target_list += [label] * a.shape[0]

        # initialize superclass with features & labels
        self.features = torch.from_numpy(np.vstack(self.feature_list))
        self.labels = torch.tensor(self.target_list, dtype=torch.float32)
        super().__init__(self.features, self.labels)
        console_msg(f"FINISHED LOADING THE DATASET... {subject_id}, number of segments: {len(self)}")


class MsgDatasetInMem(Dataset):
    """
        Custom dataset class for the My Seizure Gauge data obtained from Epilepsyecosystem.org.
    """
    def __init__(self, mdh: MsgDataHelper, subject_id: str, sequence_length: int = 60, sampling_freq: int = 128,
                 preictal_augment_factor : int = 6):

        console_msg(f"STARTED LOADING THE DATASET... {subject_id}")
        self.sampling_freq = sampling_freq
        self.sequence_length = sequence_length
        self.preictal_augment_factor = preictal_augment_factor
        self.input_size = mdh.input_size
        self.lookup_table = []
        self.targets = []

        # label values
        self.LABEL_0, self.LABEL_1 = mdh.LABEL_0, mdh.LABEL_1

        index = 0
        preictal_cnt = 0
        interictal_cnt = 0
        pp_meta = pd.read_csv(mdh.preproc_dir / Path(subject_id) / 'pp_metadata.csv')
        for data_type, label in [('preictal_train', self.LABEL_1), ('interictal_train', self.LABEL_0)]:
            for file in pp_meta['filename'][pp_meta['type'] == data_type]:
                if data_type == 'preictal_train':
                    preictal_cnt += 1
                else:
                    interictal_cnt += 1
                console_msg(f"processing: {file}")
                df = pd.read_hdf(file, key='df', mode='r')
                a = np.array(df, dtype=np.float32)
                mult = self.sampling_freq // (mdh.config.stft.seg_size if mdh.config.stft.reduce else 1)
                a = np.nan_to_num(a.reshape(-1, mult * self.sequence_length, self.input_size))
                self.lookup_table.append((index, index + a.shape[0], a, label))
                index += a.shape[0]
                self.targets += [label] * a.shape[0]

        console_msg(f"FINISHED LOADING THE DATASET... {subject_id}, number of segments: {len(self)}")

    def __len__(self):
        # TODO: dataset size calculcation
        return 0 if len(self.lookup_table) == 0 else self.lookup_table[-1][1]

    def __getsegment__(self, idx):
        ind_list = [idx] if not isinstance(idx, list) else idx
        if (ind_list >= len(self)).any():
            raise IndexError(f"at least one specified index is greater than the dataset size {len(self)}")

        seglist = []
        for ind in ind_list:
            for entry in self.lookup_table:
                if entry[0] <= ind < entry[1]:
                    seglist.append(entry[2][ind - entry[0], :, :], entry[3])
        return seglist

    def __getitem__(self, key):
        data = self.__getsegment__(key)
        if data is None:
            return None
        return torch.from_numpy(data[0]), torch.tensor(data[1], dtype=torch.float32)


class MsgPredictionDataset(Dataset):
    """
        Custom dataset class for the My Seizure Gauge data obtained from Epilepsyecosystem.org.
    """
    def __init__(self, mdh: MsgDataHelper, subject_id: str, split_folder: str, sequence_length: int = 60, sampling_freq: int = 128):

        self.sampling_freq = sampling_freq
        self.sequence_length = sequence_length
        self.input_size = mdh.input_size

        #statfile = Path(mdh.preproc_dir / Path(subject_id) / Path('preproc_ds_stat.csv'))
        #if statfile.exists() and statfile.is_file():
        #    self.ds_stat = pd.read_csv(statfile)
        #    dstat = self.ds_stat.to_numpy()
        #    self.dmin = dstat[0, 1:]
        #    self.dmax = dstat[1, 1:]
        #else:
        #    raise AttributeError("dmin/dmax not found for min-max-scaler")

        # label values
        self.LABEL_0, self.LABEL_1 = mdh.LABEL_0, mdh.LABEL_1

        self.mdh = mdh

        # initialize dataset - training files
        pp_meta = pd.read_csv(mdh.preproc_dir / Path(subject_id) / Path(split_folder) / 'pp_metadata.csv')
        self.file_meta = pp_meta[(pp_meta['type'] == 'preictal_test') | (pp_meta['type'] == 'interictal_test')].reset_index()
        #for data_type in ['preictal_test', 'interictal_test']:
        #    self.filelist = [(file, data_type) for file in pp_meta['filename'][pp_meta['type'] == data_type]]
        #for data_type in ['preictal_test', 'interictal_test']:
        #    path = mdh.preproc_dir / Path(subject_id) / Path(data_type)
        #    self.filelist += [(file, data_type) for file in path.glob("*.h5")]

    def __len__(self):
        return len(self.file_meta)

    def __getitem__(self, idx):
        #filename = self.filelist[idx][0]
        #data_type = self.filelist[idx][1]
        filename, data_type, start, end = self.file_meta[['filename', 'type', 'start', 'end']].iloc[idx]
        df = pd.read_hdf(filename, key='df', mode='r')
        data = np.asarray(df, dtype=np.float32)
        label = self.LABEL_1 if data_type == 'preictal_test' else self.LABEL_0
        mult = self.sampling_freq // (self.mdh.config.stft.seg_size if self.mdh.config.stft.reduce else 1)
        data = np.nan_to_num(data.reshape(-1, mult * self.sequence_length, self.input_size))
        #data -= self.dmin
        #data /= (self.dmax - self.dmin)
        #data *= 2.
        #data -= 1.

        return torch.from_numpy(data), torch.tensor(label, dtype=torch.float32), data_type, filename, start, end


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

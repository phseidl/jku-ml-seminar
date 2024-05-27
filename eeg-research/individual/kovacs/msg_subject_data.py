from pathlib import Path
import preproc_utils as ppu
from utils import console_msg
from sklearn.preprocessing import StandardScaler
import numpy as np
from pickle import dump, load
from itertools import combinations


class MsgDataHelper():

    def __init__(self, config, subjects: list = None):

        self.input_dir = Path(config.data_root) / Path(config.input_dir)
        self.preproc_dir = Path(config.data_root) / Path(config.preproc_dir)
        self.subject_list = subjects if subjects else config.wearable_data.subjects
        self.available_channels = config.wearable_data.sensor_channels.__dict__
        self.input_size = config.network_config.input_size

        self.preictal_setback = config.preictal_setback
        self.interictal_separation = config.interictal_separation
        self.lead_seizure_separation = config.lead_seizure_separation

        self.LABEL_0, self.LABEL_1 = tuple(config.target_labels)

        self.config = config

        self.subject_data = {subject_id: MsgSubjectData(self, subject_id) for subject_id in self.subject_list}


class MsgSubjectData(object):

    def __init__(self, mdh: MsgDataHelper, subject_id: str):

        # load metadata for subject
        self.subject_id = subject_id
        self.mdh = mdh
        self.metadata = dict()
        for folder, channels in mdh.available_channels.items():
            for ch in channels:
                self.metadata[(folder, ch)] = ppu.get_metadata(mdh.input_dir, subject_id, folder, ch)

        self.labels = ppu.get_labels(mdh.input_dir, subject_id)
        self.data = {}
        self.interictal_preictal_ratio = None
        self.load_scaler()
        self.cross_validation = mdh.config.cross_validation

    def split_train_test(self, split_ratio, interictal_preictal_ratio):
        # select training / test data segments
        mct = self.max_common_timestamp()

        # find suitable preictal (only lead seizures) and interictal (min. separation) segments in the dataset
        self.spi = ppu.find_suitable_preictals(self.labels, mct, separation_th=self.mdh.lead_seizure_separation)  # pre-ictal segments
        self.sii = ppu.find_suitable_interictals(self, self.labels, mct, separation=self.mdh.interictal_separation)  # inter-ictal segments

        # eliminate inadequate segments (low quality, no signal, device not worn)]
        self.spi = self.eliminate_low_quality_segments(self.spi)
        self.sii = self.eliminate_low_quality_segments(self.sii)

        # segment counts and train/test split (chronological split: train early stage, test later stage)
        self.num_spi = len(self.spi)
        self.num_spi_train = round(self.num_spi * split_ratio)  # training/test split
        self.num_sii = len(self.sii)
        self.num_sii_train = round(self.num_sii * split_ratio)  # training/test split

        # adjust data augmentation ratio if necessary (configured value not succifient or too high)
        self.augment_ratio = min(self.num_sii // self.num_spi, interictal_preictal_ratio)
        self.num_training_seg = self.num_spi_train * (1 + self.augment_ratio)

        # split 2:1 and separate interval descriptors into 'segment_type'_'dataset_purpose' (e.g. preictal_test)
        iict_step = max(1, self.num_sii_train // (self.num_spi_train * int(self.augment_ratio)))
        console_msg('interictal step size:', iict_step)

        # let's take all possible splits of the preictals (if cross-validation is requested)
        spi_indices = np.arange(self.num_spi)
        split_combinations = list(combinations(spi_indices, self.num_spi_train))
        # only take the first (original) split - if cross-validation is omitted
        self.num_split_combinations = len(split_combinations) if self.cross_validation else 1
        for split, comb in enumerate(split_combinations):
            self.data[split] = {'preictal_train': [], 'preictal_test': []}
            for i, segment in enumerate(self.spi):
                self.data[split]['preictal_train' if i in list(comb) else 'preictal_test'].append(segment)

        #self.data['preictal_train'] = self.spi[:self.num_spi_train]
        #self.data['preictal_test'] = self.spi[self.num_spi_train:]

        self.data['interictal_train'] = self.sii[:self.num_sii_train:iict_step][:self.num_training_seg]
        self.data['interictal_test'] = self.sii[self.num_sii_train::1]

    def print_preprocessing_summary(self):
        console_msg( '----------------------------------------------')
        console_msg(f' Total suitable PREICTAL segments.....:{self.num_spi:>7d}')
        console_msg(f'   - nr. of PREICTALs for TRAINING....:{self.num_spi_train:>7d}')
        console_msg(f'   - nr. of PREICTALs for TESTING.....:{self.num_spi - self.num_spi_train:>7d}')
        console_msg(f' Total suitable INTERICTAL segments...:{self.num_sii:>7d}')
        console_msg(f'   - nr. of INTERICTALs for TRAINING..:{self.num_sii_train:>7d}')
        console_msg(f'   - actual number of segments used...:{self.num_training_seg:>7d}')
        console_msg(f'   - nr. of INTERICTALs for TESTING...:{self.num_sii - self.num_sii_train:>7d}')
        console_msg(f' Separation of LEAD SEIZURES (min)....:{int(self.mdh.lead_seizure_separation):>7d}')
        console_msg(f' Setback for PREICTAL segments (min)..:{int(self.mdh.preictal_setback):>7d}')
        console_msg(f' Separation for INTERICTALs (min).....:{int(self.mdh.interictal_separation):>7d}')
        console_msg(f' Ratio of PREICTAL data augmentation..:{self.augment_ratio:>7d}')
        console_msg(f' Total (augm.) PREICTALS for training.:{self.num_training_seg:>7d}')
        console_msg('----------------------------------------------')

    def augmentation(self):
        self.augmented_preictal_train = {}
        console_msg(" > start data augmentation...")
        for split in range(self.num_split_combinations):
            for interval in self.data[split]['preictal_train']:
                self.augmented_preictal_train[split] = {}
                start_ts, end_ts = interval
                inp_data = self.get_input_data(start_ts, end_ts)
                for i in range(self.augment_ratio):
                    console_msg(f" > creating augmented copy {i+1} of preictal train interval {start_ts} - {end_ts}")
                    self.augmented_preictal_train[split][(start_ts, end_ts, i+1)] = ppu.add_signal_based_noise(inp_data)

    def calculate_zscore_params(self, config):
        self.scaler = {}
        for split in range(self.num_split_combinations):
            self.scaler[split] = StandardScaler()

            # partially fit preictal train and interictal train intervals
            for iv in self.data[split]['preictal_train'] + self.data['interictal_train']:
                start_ts, end_ts = iv
                inp_data = self.get_input_data(start_ts, end_ts)
                if inp_data is None or len(inp_data) == 0:
                    continue
                d = ppu.preprocess(config, inp_data)
                d = np.nan_to_num(d)
                self.scaler[split].partial_fit(d)

            # partially fit augmented preictal intervals
            if not self.augmented_preictal_train[split] is None:
                for key, data in self.augmented_preictal_train[split].items():
                    d = ppu.preprocess(config, data)
                    d = np.nan_to_num(d)
                    self.scaler[split].partial_fit(d)

        # save the scaler
        self.save_scaler()

    def max_common_timestamp(self):
        return min([df['segments.startTime'].iloc[-1] for _, df in self.metadata.items()])

    def get_input_data(self, start_ts, end_ts):
        return ppu.get_input_data(self.mdh.input_dir, self.subject_id, self.metadata, self.mdh.available_channels,
                                  start_ts, end_ts)

    def save_scaler(self):
        dirpath = Path(self.mdh.preproc_dir / self.subject_id)
        dirpath.mkdir(parents=True, exist_ok=True)
        for split in range(self.num_split_combinations):
            with open(Path(dirpath / f"stdscaler_{split:03d}.pkl"), "wb") as f:
                dump(self.scaler, f)

    def load_scaler(self):
        self.scaler = {}
        path = Path(self.mdh.preproc_dir / self.subject_id)
        scaler_files = list(path.glob(f"stdscaler_*.pkl"))
        #sc_file = Path(self.mdh.preproc_dir / self.subject_id / f"stdscaler_{split:03d}.pkl")
        for sc_file in scaler_files:
            if sc_file.exists() and sc_file.is_file():
                with open(sc_file, "rb") as f:
                    split = int(sc_file.name[10:13])
                    self.scaler[split] = load(f)

    def eliminate_low_quality_segments(self, segments):
        res = []
        for i, interval in enumerate(segments):
            start_ts, end_ts = interval
            df = self.get_input_data(start_ts, end_ts)
            if df is None:
                continue
            temp_variability = self.invalid_signal_pct(df['TEMP_TEMP'].to_numpy())
            if (sum(np.isnan(df.to_numpy())) / len(df) > self.mdh.config.dataset_config.exclude_threshold).any() \
                or temp_variability > self.mdh.config.dataset_config.exclude_threshold:
                continue
            res.append(interval)
        return res

    def invalid_signal_pct(self, signal, segment_length=7680, threshold=1e-5):
        std_signal = (signal - signal.mean()) / signal.std()
        num_segments = len(std_signal) // segment_length
        variabilities = []

        for i in range(num_segments):
            segment = std_signal[i * segment_length:(i + 1) * segment_length]
            variability = np.std(segment)
            variabilities.append(variability)
        return np.sum(np.array(variabilities) < threshold) / num_segments
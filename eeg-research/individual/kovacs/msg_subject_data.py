from pathlib import Path
import preproc_utils as ppu
from utils import console_msg


class MsgDataHelper():

    def __init__(self, config, subjects: list = None):

        self.input_dir = Path(config.data_root) / Path(config.input_dir)
        self.preproc_dir = Path(config.data_root) / Path(config.preproc_dir)
        self.subject_list = subjects if subjects else config.wearable_data.subjects
        self.available_channels = config.wearable_data.sensor_channels.__dict__
        self.input_size = config.network_config.input_size

        self.subject_data = {subject_id: MsgSubjectData(self, subject_id) for subject_id in self.subject_list}

        self.preictal_setback = config.preictal_setback
        self.interictal_separation = config.interictal_separation
        self.lead_seizure_separation = config.lead_seizure_separation

        self.LABEL_0, self.LABEL_1 = tuple(config.target_labels)

        self.config = config


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

    def split_train_test(self, split_ratio, interictal_preictal_ratio):
        # select training / test data segments
        mct = self.max_common_timestamp()
        self.spi = ppu.find_suitable_preictals(self.labels, mct, separation_th=self.mdh.lead_seizure_separation)  # pre-ictal segments
        self.sii = ppu.find_suitable_interictals(self, self.labels, mct, separation=self.mdh.interictal_separation)  # inter-ictal segments
        self.num_spi = len(self.spi)
        self.num_spi_train = int(self.num_spi * split_ratio)  # training/test split
        self.num_sii = len(self.sii)
        self.num_sii_train = int(self.num_sii * split_ratio)  # training/test split

        console_msg( '----------------------------------------------')
        console_msg(f' Total suitable PREICTAL segments.....:{self.num_spi:>7d}')
        console_msg(f' Nr. of PREICTALs for training........:{self.num_spi_train:>7d}')
        console_msg(f' Nr. of PREICTALs for testing.........:{self.num_spi - self.num_spi_train:>7d}')
        console_msg(f' Total suitable INTERICTAL segments...:{self.num_sii:>7d}')
        console_msg(f' Nr. of INTERICTALs for training......:{self.num_sii_train:>7d}')
        console_msg(f' Nr. of INTERICTALs for testing.......:{self.num_sii - self.num_sii_train:>7d}')
        console_msg(f' Setback for PREICTAL segments (min)..:{int(self.mdh.preictal_setback):>7d}')
        console_msg(f' Separation for INTERICTALs (min).....:{int(self.mdh.interictal_separation):>7d}')
        console_msg('----------------------------------------------')

        # split 2:1
        iict_step = self.num_sii_train // (self.num_spi_train * int(interictal_preictal_ratio))
        console_msg('interictal step size:', iict_step)
        self.data['preictal_train'], self.data['preictal_test'] = self.spi[:self.num_spi_train], self.spi[self.num_spi_train:]
        self.data['interictal_train'], self.data['interictal_test'] = self.sii[:self.num_sii_train:iict_step], self.sii[self.num_sii_train::iict_step]

    def calculate_zscore_params(self):
        self.means, self.stds = ppu.calc_z_score(self.mdh.config, self.mdh.input_dir, self.subject_id, self.metadata,
                                                 self.mdh.available_channels,
                                                 self.data['preictal_train'] + self.data['interictal_train'],
                                                 feature_dim=self.mdh.input_size)

    def max_common_timestamp(self):
        return min([df['segments.startTime'].iloc[-1] for _, df in self.metadata.items()])

    def get_input_data(self, start_ts, end_ts):
        return ppu.get_input_data(self.mdh.input_dir, self.subject_id, self.metadata, self.mdh.available_channels,
                                  start_ts, end_ts)

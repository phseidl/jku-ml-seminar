import torch
import os
import mne
import pandas as pd
import numpy as np
from scipy.signal import decimate
from torch import from_numpy as np2TT
from torch.utils.data import Dataset
import numpy as np
import mne
import os


class SparseEEGDataset_USZ(Dataset):
    def __init__(self, 
                 subjects, 
                 window_size, 
                 stride, 
                 num_inputs, 
                 num_outputs, 
                 cfg_dataset, 
                 eeg_channels_to_keep = None, 
                 ieeg_channels_to_keep = None, 
                 train=True, 
                 montage = 'standard_1020', 
                 use_montage='random',
                 downsample_ieeg = 200,
                 n_input_chs = 20, 
                 n_output_chs = 20,
                 ch_dropout = None,
                ):
        """
        Args:
            x (torch.Tensor): Input EEG data of shape [number of EEG segments, channels, sequence length].
            y (torch.Tensor): Target EEG data of shape [number of EEG segments, channels, sequence length].
            num_inputs (int): Number of input points to sample.
            num_outputs (int): Number of output points to sample.
            train (bool): If True, random sampling is applied; otherwise, deterministic sampling.
        """
        super(SparseEEGDataset_USZ, self).__init__()

        self.num_inputs = num_inputs
        self.num_outputs = num_outputs
        self.train = train
        self.downsample_ieeg = downsample_ieeg
        self.n_input_chs = n_input_chs
        self.n_output_chs = n_output_chs
        self.ch_dropout = ch_dropout
        self.use_montage = use_montage

        base_dir = cfg_dataset['dataset_dir']
        subjects_to_process = subjects

        # Initialize a dictionary to store the data
        all_data = {"x": [], "y": []}

        # Hyperparameters for segmentation
        window_size_sec = window_size
        stride_sec = stride

        # Iterate over subjects
        for subject in subjects_to_process:
            subject_dir = os.path.join(base_dir, f'sub-{subject}')
            if not os.path.exists(subject_dir):
                print(f"Directory for subject {subject} not found, skipping.")
                continue

            # Walk through subfolders
            for root, dirs, files in os.walk(subject_dir):
                # Identify iEEG or EEG based on folder structure
                if "ieeg" in root:
                    # Process iEEG data
                    for file in files:
                        if file.endswith(".edf"):
                            edf_path = os.path.join(root, file)
                            tsv_path = self.find_electrode_file(root)  # Find the electrode file
                            if tsv_path:
                                ieeg_data = self.process_edf_with_positions(edf_path, tsv_path)
                                filtered_data, filtered_channel_names, filtered_positions = self.filter_channels(ieeg_data['data'], ieeg_data['channel_names'], ieeg_channels_to_keep, ieeg_data['positions']) #only keep channels that are in ieeg_channels_to_keep
                                ieeg_data['data'] = filtered_data
                                ieeg_data['channel_names'] = filtered_channel_names
                                ieeg_data['positions'] = filtered_positions
                                # Segment the iEEG data
                                segments = self.segment_data(ieeg_data["data"], window_size_sec, stride_sec, ieeg_data["sampling_frequency"], 'ieeg')
                                # Store data along with metadata (channel_names, sampling_frequency, positions)
                                all_data["y"].extend([{
                                    "segment": segment,
                                    "channel_names": ieeg_data["channel_names"],
                                    "sampling_frequency": ieeg_data["sampling_frequency"],
                                    "positions": ieeg_data["positions"],
                                } for segment in segments])
                            else:
                                print(f"TSV file containing 'electrode' missing for {edf_path}, skipping.")
                elif "eeg" in root:
                    # Process EEG data
                    for file in files:
                        if file.endswith(".edf"):
                            edf_path = os.path.join(root, file)
                            eeg_data = self.process_edf_with_montage_positions(edf_path, montage)
                            filtered_data, filtered_channel_names, filtered_positions = self.filter_channels(eeg_data['data'], eeg_data['channel_names'], eeg_channels_to_keep, eeg_data['positions'])
                            eeg_data['data'] = filtered_data
                            eeg_data['channel_names'] = filtered_channel_names
                            eeg_data['positions'] = filtered_positions
                            # Segment the EEG data
                            segments = self.segment_data(eeg_data["data"], window_size_sec, stride_sec, eeg_data["sampling_frequency"], 'eeg')
                            # Store data along with metadata (channel_names, sampling_frequency, positions)
                            all_data["x"].extend([{
                                "segment": segment,
                                "channel_names": eeg_data["channel_names"],
                                "sampling_frequency": eeg_data["sampling_frequency"],
                                "positions": eeg_data["positions"],
                            } for segment in segments])

        print("Processing complete!")
        self.min_vals_pos, self.max_vals_pos = self.get_global_position_min_max(all_data)

        self.x = all_data["x"]     #scalp EEG data, list with segmented data, channel_names, sampling_freq and positions
        self.y = all_data["y"]     #iEEG data, list with segmented data, channel_names, sampling_freq and positions
        #print(self.x)
        #print(self.y)
        #print(len(self.y))
        #print(len(self.x))
        #assert len(self.x) == len(self.y)



    def find_electrode_file(self, directory):
        """
        Finds the .tsv file containing the word 'electrode' in the given directory.
        Returns the path to the file if found, otherwise None.
        """
        tsv_files = [f for f in os.listdir(directory) if 'electrode' in f and f.endswith('.tsv')]
        return os.path.join(directory, tsv_files[0]) if tsv_files else None

    def process_edf_with_positions(self, edf_path, tsv_path):
        """
        Processes an iEEG .edf file with positions.
        """
        raw = mne.io.read_raw_edf(edf_path, preload=True)
        data, _ = raw.get_data(return_times=True)

        if self.downsample_ieeg is None:
            sampling_frequency = raw.info['sfreq']
        else:
            data = decimate(data, int(raw.info['sfreq']/self.downsample_ieeg), ftype='iir')  #downsample
            sampling_frequency = self.downsample_ieeg

        channel_names = raw.info['ch_names']

        # Load electrode positions
        positions_df = pd.read_csv(tsv_path, sep='\t')
        positions = []
        for name in channel_names:
            match = positions_df[positions_df['name'] == name]
            if not match.empty:
                positions.append(match[['x', 'y', 'z']].values[0])  # Extract x, y, z
            else:
                positions.append([np.nan, np.nan, np.nan])  # Fallback
        positions = np.array(positions)

        return {
            "data": data,
            "sampling_frequency": sampling_frequency,
            "channel_names": channel_names,
            "positions": positions,
        }

    def process_edf_with_montage_positions(self, edf_path, montage_name):
        """
        Processes an EEG .edf file and assigns positions using a standard montage.
        """
        raw = mne.io.read_raw_edf(edf_path, preload=True)
        data, _ = raw.get_data(return_times=True)
        sampling_frequency = raw.info['sfreq']
        channel_names = raw.info['ch_names']

        # Load standard montage
        montage = mne.channels.make_standard_montage(montage_name)
        montage_positions = montage.get_positions()['ch_pos']

        # Assign positions to channels if found in the montage
        positions = []
        for name in channel_names:
            if name in montage_positions:
                positions.append(montage_positions[name] * 1e+3)    #convert to mm
            else:
                positions.append([np.nan, np.nan, np.nan])  # Fallback for missing channels
        positions = np.array(positions)

        return {
            "data": data,
            "sampling_frequency": sampling_frequency,
            "channel_names": channel_names,
            "positions": positions,
        }

    def normalize_segment(self, segment_data, eeg_type):
        """
        Normalize each channel of the segment by dividing by the 95th percentile of the absolute values.
        """
        normalized_data = []
        if eeg_type == 'eeg':
            # Iterate over each channel in the segment
            for channel_data in segment_data:
                # Calculate the 95th percentile of the absolute values of the channel
                percentile_95 = np.percentile(np.abs(channel_data), 95)
                # Normalize the channel by dividing by the 95th percentile
                normalized_channel = channel_data / percentile_95
                normalized_data.append(normalized_channel)
            return np.array(normalized_data)
        elif eeg_type == 'ieeg':
            # Iterate over each channel in the segment
            for channel_data in segment_data:
                # Calculate the 95th percentile of the absolute values of the channel
                percentile_95 = np.percentile(np.abs(channel_data), 95)
                # Normalize the channel by dividing by the 95th percentile
                normalized_channel = channel_data / percentile_95
                normalized_data.append(normalized_channel)
            return np.array(normalized_data)
            #return segment_data/100
        else:
            raise Exception("Wrong EEG type. Either ieeg or eeg.")
    
    def normalize_segments(self, eeg_data_x, eeg_data_y):
        percentiles_x = np.percentile(np.abs(eeg_data_x), 95, axis=-1, keepdims=True)
        percentiles_y = np.percentile(np.abs(eeg_data_y), 95, axis=-1, keepdims=True)
        if np.any(np.isnan(eeg_data_x / percentiles_x)):
            print('norm x is NaN detected')
        if np.any(np.isnan(eeg_data_y / percentiles_y)):
            print('norm y is NaN detected')    
        return eeg_data_x / percentiles_x, eeg_data_y / percentiles_y

    def segment_data(self, data, window_size_sec, stride_sec, sampling_frequency, eeg_type):
        """
        Segments the data into overlapping windows of a given size and stride.
        """
        # Convert window size and stride from seconds to samples
        window_size_samples = int(window_size_sec * sampling_frequency)
        stride_samples = int(stride_sec * sampling_frequency)

        segments = []
        for start in range(0, data.shape[1] - window_size_samples + 1, stride_samples):
            end = start + window_size_samples
            segment = data[:, start:end]
            #segment = self.normalize_segment(segment, eeg_type)   #95 percentile normalization
            segments.append(segment)

        return segments

    def get_global_position_min_max(self, all_data):
        """
        Computes the global minimum and maximum for positions (x, y, z)
        across all entries in all_data (both EEG and iEEG).
        """
        all_positions = []

        # Collect all position data from all entries in all_data
        for data_type in ["x", "y"]:  # EEG and iEEG data
            for entry in all_data[data_type]:
                positions = entry["positions"]  # Extract positions (x, y, z)
                all_positions.append(positions)

        # Convert the list of positions into a numpy array for easier min/max computation
        all_positions = np.concatenate(all_positions, axis=0)

        # Calculate the global min and max for each coordinate (x, y, z)
        min_vals = np.nanmin(all_positions, axis=0)  # Min values across all positions for each axis
        max_vals = np.nanmax(all_positions, axis=0)  # Max values across all positions for each axis

        return min_vals, max_vals

    def normalize_positions(self, positions, global_min, global_max):
        norm_positions = (positions - global_min) / (global_max - global_min)
        return norm_positions * 2 - 1  # Normalize to [-1, 1]

    def filter_channels(self, data, channel_names, channels_to_keep, positions):
        """
        Filters the data to include only the specified channels.
        """
        if channels_to_keep is not None:
            # Identify indices of channels to keep
            indices_to_keep = [i for i, name in enumerate(channel_names) if name in channels_to_keep]
            # Subset the data and channel names
            filtered_data = data[indices_to_keep, :]
            filtered_channel_names = [channel_names[i] for i in indices_to_keep]
            filtered_positions = [positions[i] for i in indices_to_keep]
        else:
            filtered_data = data
            filtered_channel_names = channel_names
            filtered_positions = positions
        return filtered_data, filtered_channel_names, filtered_positions
        
    def generate_random_pairs(self, channel_nr, N, ch_dropout):
        """
        Generate N random pairs of channel indices without repetition, with a specified percentage of dropout.
        
        Args:
        - channel_nr (int): Total number of channels.
        - N (int): Number of pairs to generate.
        - ch_dropout (float): factor of channels to drop (0 to 1).
        
        Returns:
        - List of random pairs with dropout applied.
        """
        if ch_dropout is None:
            all_pairs = [(i, j) for i in range(channel_nr) for j in range(channel_nr) if i != j]   
        else:
            # Determine the number of channels to keep after dropout
            dropout_count = int(channel_nr * ch_dropout)
            remaining_channels = np.random.choice(channel_nr, channel_nr - dropout_count, replace=False)
            # Generate all possible pairs from the remaining channels
            all_pairs = [(i, j) for i in remaining_channels for j in remaining_channels if i != j]
        np.random.shuffle(all_pairs)
        
        return all_pairs[:N]


    def create_montage(self, eeg_segment, channel_order, channel_pos, montage = 'tuh', montage_pairs=None, debug = False):
        bipolar_data = []
        bipolar_names = []
        bipolar_positions = []
        if montage == 'tuh':
            #TODO use self.channel_order
            montage_pairs = [
            ('FP1', 'F7'), ('F7', 'T3'), ('T3', 'T5'), ('T5', 'O1'),
            ('FP2', 'F8'), ('F8', 'T4'), ('T4', 'T6'), ('T6', 'O2'),
            ('A1', 'T3'), ('T3', 'C3'), ('C3', 'CZ'), ('CZ', 'C4'),
            ('C4', 'T4'), ('T4', 'A2'), ('FP1', 'F3'), ('F3', 'C3'),
            ('C3', 'P3'), ('P3', 'O1'), ('FP2', 'F4'), ('F4', 'C4'),
            ('C4', 'P4'), ('P4', 'O2')
            ]

            # Create the bipolar pairs
            for anode, cathode in montage_pairs:
                try:
                    anode_idx = self.channel_index_map[anode]
                    cathode_idx = self.channel_index_map[cathode]
                    bipolar_signal = eeg_segment[anode_idx] - eeg_segment[cathode_idx]
                    bipolar_data.append(bipolar_signal)
                    bipolar_names.append(f"{anode}-{cathode}")
                    # Combine 3D positions of anode and cathode into a (6,) array
                    combined_positions = np.concatenate([self.channel_pos[anode_idx], self.channel_pos[cathode_idx]])
                    bipolar_positions.append(combined_positions)
                except KeyError:
                  if debug:
                      print(f"Skipping pair {anode}-{cathode}: One or both channels are missing.")

            # Convert the bipolar data list to a numpy array
            bipolar_data = np.array(bipolar_data)
            bipolar_positions = np.array(bipolar_positions)  # Shape: (number_of_pairs, 6)

            if debug:
                # Output
                print("Bipolar Data Shape:", bipolar_data.shape)
                print("Bipolar Positions Shape:", bipolar_positions.shape)
                print("Bipolar Channels:", bipolar_names)
                print("Bipolar Data", bipolar_data)
                print("EEG segment", eeg_segment)


        elif montage == 'random':
            # Create the bipolar pairs
            for anode_idx, cathode_idx in montage_pairs:
                try:
                    bipolar_signal = eeg_segment[anode_idx] - eeg_segment[cathode_idx]
                    bipolar_data.append(bipolar_signal)
                    bipolar_names.append(f"{channel_order[anode_idx]}-{channel_order[cathode_idx]}")
                    # Combine 3D positions of anode and cathode into a (6,) array
                    combined_positions = np.concatenate([channel_pos[anode_idx], channel_pos[cathode_idx]])
                    bipolar_positions.append(combined_positions)
                except KeyError:
                  if debug:
                      print(f"Skipping pair {anode}-{cathode}: One or both channels are missing.")

            # Convert the bipolar data list to a numpy array
            bipolar_data = np.array(bipolar_data)
            bipolar_positions = np.array(bipolar_positions)  # Shape: (number_of_pairs, 6)
            

        elif montage == None or 'no_montage':
            bipolar_data = eeg_segment
            bipolar_positions =  self.channel_pos         # Shape: (number_of_pairs, 3)

        return bipolar_data, bipolar_positions



    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        # Retrieve the EEG segment and ground truth
        x = self.x[idx]    # keys: 'segment', 'channel_names', 'positions', 'sampling_frequency'
        y = self.y[idx]    # keys: 'segment', 'channel_names', 'positions', 'sampling_frequency'
        x_segment = np.array(x['segment']) # Shape: [channels, sequence_length]
        y_segment = np.array(y['segment']) # Shape: [channels, sequence_length]
                
        #x_segment = torch.tensor(x['segment'], dtype=torch.float32) # Shape: [channels, sequence_length]
        #y_segment = torch.tensor(y['segment'], dtype=torch.float32) # Shape: [channels, sequence_length]
        
        montage_pairs_x = self.generate_random_pairs(len(x['channel_names']), N=self.n_input_chs, ch_dropout=self.ch_dropout) if self.use_montage == 'random' else None
        montage_pairs_y = self.generate_random_pairs(len(y['channel_names']), N=self.n_output_chs, ch_dropout=None) if self.use_montage == 'random' else None
        x_segment, ch_positions_6d_x = self.create_montage(x_segment, montage=self.use_montage, channel_order=x['channel_names'], channel_pos=x['positions'], montage_pairs = montage_pairs_x) #montaged data
        y_segment, ch_positions_6d_y = self.create_montage(y_segment, montage=self.use_montage, channel_order=y['channel_names'], channel_pos=y['positions'], montage_pairs = montage_pairs_y)   #montaged data
        x_segment, y_segment = self.normalize_segments(x_segment, y_segment)

        x_segment = np2TT(x_segment)
        y_segment = np2TT(y_segment)
        
        x_n_chan, x_seq_len = x_segment.shape
        y_n_chan, y_seq_len = y_segment.shape
        
        x_channel_pos = ch_positions_6d_x
        y_channel_pos = ch_positions_6d_y
        #num_inputs = round(x_n_chan * x_seq_len * self.inputs_perc)
        #num_outputs = round(y_n_chan * y_seq_len * self.outputs_perc)
        num_inputs = self.num_inputs
        num_outputs = self.num_outputs

        #x_channel_pos = self.normalize_positions(x_channel_pos, self.min_vals_pos, self.max_vals_pos)*100  #x_channel_pos normalized to [-1, 1] and scaled to [-100, 100]
        x_channel_pos = (x_channel_pos - self.min_vals_pos.min())/(self.max_vals_pos.max() - self.min_vals_pos.min())*100   # rescale to [0,100]

        pos_channels_x = torch.tensor(np.repeat(x_channel_pos[:, np.newaxis, :], x_seq_len, axis=1), dtype=torch.float32)  # Shape: [channels, sequence_length, 6]
        pos_channels_x = pos_channels_x.permute(2,0,1)  # Shape: [6, channels, sequence_length]

        time_pos_x = (torch.arange(start=0, end=x_seq_len/x['sampling_frequency'], step=1/x['sampling_frequency'])).float().unsqueeze(0)*173  # Time positions: [1, sequence_length], scaled to [0,100]
        #time_pos_x = self.normalize_positions(time_pos_x, 0, x_seq_len/x['sampling_frequency']) * 100    #normalized to [-1, 1] and scaled to [-100, 100]
        time_pos_x = time_pos_x.repeat(x_n_chan, 1).unsqueeze(2)  # Shape: [channels, sequence_length, 1]
        time_pos_x = time_pos_x.permute(2,0,1) # Shape: [1, channels, sequence_length]

        #y_channel_pos = self.normalize_positions(y_channel_pos, self.min_vals_pos, self.max_vals_pos)*100   #normalized to [-1, 1] and scaled to [-100, 100]
        y_channel_pos = (y_channel_pos - self.min_vals_pos.min())/(self.max_vals_pos.max() - self.min_vals_pos.min())*100   # rescale to [0,100]
        pos_channels_y = torch.tensor(np.repeat(y_channel_pos[:, np.newaxis, :], y_seq_len, axis=1), dtype=torch.float32)
        pos_channels_y = pos_channels_y.permute(2,0,1)  # Shape: [3, channels, sequence_length]

        time_pos_y = (torch.arange(start=0, end=y_seq_len/y['sampling_frequency'], step=1/y['sampling_frequency'])).float().unsqueeze(0)*173  # Time positions: [1, sequence_length]
        #time_pos_y = self.normalize_positions(time_pos_y, 0, y_seq_len/y['sampling_frequency']) * 100    #normalized to [-1, 1] and scaled to [-100, 100]
        time_pos_y = time_pos_y.repeat(y_n_chan, 1).unsqueeze(2)  # Shape: [channels, sequence_length, 1]
        time_pos_y = time_pos_y.permute(2,0,1) # Shape: [1, channels, sequence_length]

        # Combine positions into a 3D tensor
        pos_encoding_x = torch.cat((pos_channels_x, time_pos_x), dim=0) # Shape: [7, channels, sequence_length]
        pos_encoding_y = torch.cat((pos_channels_y, time_pos_y), dim=0) # Shape: [7, channels, sequence_length]

        # Subsample random input points
        if num_inputs < x_n_chan * x_seq_len:
            if self.train:
                rng = None
            else:
                rng = torch.Generator().manual_seed(idx)
            input_indices = torch.randperm(x_n_chan * x_seq_len, generator=rng)[:num_inputs]
            input_indices = torch.stack(
                (input_indices // x_seq_len, input_indices % x_seq_len), dim=1
            )  # Shape: [num_inputs, 2]
            input_feat = x_segment[input_indices[:, 0], input_indices[:, 1]].unsqueeze(1)
            input_pos = pos_encoding_x[:, input_indices[:, 0], input_indices[:, 1]]   # (7, num_inputs), the 4 dimensions are 3d channel pos and 1d time
        else:
            input_feat = x_segment.reshape(-1, 1)
            input_pos = pos_encoding_x.reshape(pos_encoding_x.shape[0], -1)

        # Subsample random output points
        if num_outputs < y_n_chan * y_seq_len:
            if self.train:
                rng = None
            else:
                rng = torch.Generator().manual_seed(idx + 1)
            output_indices = torch.randperm(y_n_chan * y_seq_len, generator=rng)[:num_outputs]
            output_indices = torch.stack(
                (output_indices // y_seq_len, output_indices % y_seq_len), dim=1
            )  # Shape: [num_outputs, 2]
            target_feat = y_segment[output_indices[:, 0], output_indices[:, 1]].unsqueeze(1)
            output_pos = pos_encoding_y[:, output_indices[:, 0], output_indices[:, 1]]
        else:
            target_feat = y_segment.reshape(-1, 1)
            output_pos = pos_encoding_y.reshape(pos_encoding_y.shape[0], -1)

        return dict(
            index=idx,
            input_feat=input_feat.to(torch.float32),
            input_pos=input_pos.permute(1,0).to(torch.float32),
            target_feat=target_feat.to(torch.float32),
            output_pos=output_pos.permute(1,0).to(torch.float32),
        )